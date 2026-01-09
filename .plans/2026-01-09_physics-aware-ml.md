# Plan: Physics-Aware ML (v2.2.0)

**Date**: 2026-01-09
**Status**: APPROVED
**Task IDs**: #88-101

---

## Goal

Create PyTorch layers and utilities that track physical dimensions through neural network forward passes, enabling:
1. Unit-aware neural network layers
2. Dimensional loss functions
3. Automatic non-dimensionalization for training
4. Physics-informed training patterns

---

## Research Summary

### Key Findings

1. **PINNs** (Physics-Informed Neural Networks) focus on embedding PDE constraints into loss functions but don't track physical units through layer operations

2. **Non-dimensionalization is critical**:
   - Neural networks train best with inputs in range [-1, 1]
   - Physical quantities span many orders of magnitude
   - Need automatic scaling that preserves dimensional relationships

3. **Gap in existing tools**:
   - TorchPhysics (Bosch) - PDE solving, no unit tracking
   - pint/unyt - Unit tracking but no PyTorch integration
   - dimtensor can fill this gap

### Design Principles

1. **Dimension propagation**: Track dimensions through matrix operations
2. **Non-dimensionalization**: Scale to dimensionless for training, rescale on output
3. **Loss function units**: Ensure loss terms have compatible dimensions
4. **Minimal overhead**: Use DimTensor only at boundaries, raw tensors internally

---

## Implementation Approach

### Phase 1: DimLayer Base Class (Tasks #89)

```python
class DimLayer(nn.Module):
    """Base class for dimension-aware layers."""

    def __init__(self, input_dim: Dimension, output_dim: Dimension):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: DimTensor) -> DimTensor:
        # Validate input dimension
        if x.unit.dimension != self.input_dim:
            raise DimensionError(...)

        # Run computation
        result = self._forward_impl(x.tensor)

        # Return with output dimension
        return DimTensor(result, Unit(self.output_dim))
```

### Phase 2: DimLinear Layer (Task #90)

Linear transformation with dimension tracking:
- Weight matrix has dimensions: [output_dim / input_dim]
- y = Wx + b where W[i,j] has unit (output_i / input_j)

```python
class DimLinear(DimLayer):
    """Linear layer with physical dimensions.

    For physics: y = Wx where:
    - x has dimension D_in
    - W has dimension D_out / D_in
    - y has dimension D_out
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        bias: bool = True
    ):
        ...
```

### Phase 3: DimConv Layers (Task #91)

Convolution preserves spatial dimensions but can transform physical dimensions:

```python
class DimConv1d(DimLayer):
    """1D convolution with dimension tracking."""

class DimConv2d(DimLayer):
    """2D convolution with dimension tracking."""
```

### Phase 4: Dimensional Loss Functions (Tasks #92-94)

```python
class DimMSELoss(nn.Module):
    """MSE loss that checks dimensional compatibility."""

    def forward(self, pred: DimTensor, target: DimTensor) -> DimTensor:
        if pred.unit.dimension != target.unit.dimension:
            raise DimensionError(f"Cannot compute loss between {pred.unit} and {target.unit}")

        # Loss has dimension of (pred)^2
        diff = pred - target
        return (diff * diff).mean()

class PhysicsLoss(nn.Module):
    """Loss function for conservation laws."""

    def __init__(self, quantities: list[str], rtol: float = 1e-6):
        """
        Args:
            quantities: Names of conserved quantities ('energy', 'momentum', etc.)
            rtol: Relative tolerance for conservation check
        """
        ...
```

### Phase 5: Unit-Aware Normalization (Tasks #95-97)

BatchNorm and LayerNorm for dimensioned tensors:

```python
class DimBatchNorm1d(DimLayer):
    """Batch normalization that preserves dimensions.

    Normalizes to zero mean, unit variance but preserves the physical dimension.
    Running mean/var stored in the layer's physical units.
    """

class DimLayerNorm(DimLayer):
    """Layer normalization preserving dimensions."""
```

### Phase 6: Non-dimensionalization (Tasks #98-99)

The key utility for physics ML:

```python
class DimScaler:
    """Scale physical quantities to dimensionless values for training.

    Example:
        scaler = DimScaler()
        scaler.fit(velocity_data, temperature_data, pressure_data)

        # Training: convert to dimensionless
        v_scaled = scaler.transform(velocity_data)  # dimensionless tensor

        # Inference: convert back
        velocity = scaler.inverse_transform(v_scaled, units.m / units.s)
    """

    def __init__(self, method: str = "characteristic"):
        """
        Args:
            method: 'characteristic' - use characteristic scales
                    'minmax' - scale to [0, 1]
                    'standard' - scale to mean=0, std=1
        """

    def fit(self, *arrays: DimArray) -> "DimScaler":
        """Learn characteristic scales from data."""

    def transform(self, arr: DimArray) -> torch.Tensor:
        """Convert to dimensionless tensor."""

    def inverse_transform(self, tensor: torch.Tensor, unit: Unit) -> DimTensor:
        """Convert back to dimensional tensor."""
```

---

## File Structure

```
src/dimtensor/torch/
├── dimtensor.py        # Existing DimTensor class
├── layers.py           # NEW: DimLayer, DimLinear, DimConv
├── losses.py           # NEW: DimMSELoss, PhysicsLoss
├── normalization.py    # NEW: DimBatchNorm, DimLayerNorm
└── scaler.py           # NEW: DimScaler

tests/
├── test_dim_layers.py  # NEW
├── test_dim_losses.py  # NEW
└── test_dim_scaler.py  # NEW
```

---

## Implementation Steps

1. **Task #89**: Create `torch/layers.py` with DimLayer base class
2. **Task #90**: Implement DimLinear in `torch/layers.py`
3. **Task #91**: Implement DimConv1d, DimConv2d
4. **Task #92**: Create `torch/losses.py` with base loss utilities
5. **Task #93**: Implement DimMSELoss, DimL1Loss
6. **Task #94**: Implement PhysicsLoss for conservation laws
7. **Task #95**: Create `torch/normalization.py`
8. **Task #96**: Implement DimBatchNorm1d, DimBatchNorm2d
9. **Task #97**: Implement DimLayerNorm
10. **Task #98**: Create `torch/scaler.py` with DimScaler
11. **Task #99**: Add characteristic scale detection
12. **Task #100**: Create comprehensive tests
13. **Task #101**: Deploy v2.2.0

---

## Key Design Decisions

### 1. Dimension Propagation Through Linear Layers

For y = Wx:
- If x has dimension [L] (length) and we want y in [L/T] (velocity)
- W must have dimension [1/T]
- This is tracked but NOT enforced - user specifies output dimension

### 2. Internal vs. Boundary Computation

For performance, we track dimensions only at:
- Network input (validate DimTensor dimension)
- Network output (assign DimTensor dimension)

Internal layers use raw PyTorch tensors for speed.

### 3. Non-dimensionalization Strategy

Use characteristic scales:
- length_scale = max(|positions|)
- time_scale = characteristic period
- velocity_scale = length_scale / time_scale

This preserves physical relationships:
- v* = v / v_scale
- t* = t / t_scale
- x* = v* × t* (still works!)

---

## Test Cases

1. **DimLinear**: Input [m], output [m/s], verify dimensions
2. **DimConv2d**: Image with physical coordinates, verify spatial dims preserved
3. **DimMSELoss**: Error on mismatched dimensions
4. **PhysicsLoss**: Track energy conservation
5. **DimScaler**: Round-trip accuracy (scale → unscale)
6. **Integration**: Full forward pass with dimension tracking

---

## References

- [Physics-Informed Neural Networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)
- [TorchPhysics](https://github.com/boschresearch/torchphysics) - Bosch Research PINN library
- [Non-dimensionalization in ML](https://www.sciencedirect.com/science/article/pii/S2590054420300270)
