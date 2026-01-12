# Plan: Sensitivity Analysis Tools

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add comprehensive sensitivity analysis tools to dimtensor that allow users to perform local sensitivity analysis (partial derivatives), global sensitivity analysis (Sobol indices), and parameter importance ranking, all while preserving dimensional correctness.

---

## Background

Sensitivity analysis is critical for understanding how parameter changes affect model outputs in scientific computing. This is especially important for physics-based models where parameters have physical units. We need to:

1. Compute local sensitivities (gradients) with correct dimensional units
2. Perform global sensitivity analysis using variance decomposition methods
3. Rank parameter importance for model interpretation
4. Support all dimtensor backends (NumPy, PyTorch, JAX)

---

## Approach

### Option A: Full Integration with SALib

- Description: Build complete wrapper around SALib library with unit tracking
- Pros:
  - Leverages mature, well-tested algorithms (Sobol, Morris, FAST, etc.)
  - Provides wide range of sensitivity methods
  - Active development and community
- Cons:
  - External dependency
  - SALib API may need extensive wrapping to preserve units
  - Some methods may be difficult to adapt to unit-aware workflows

### Option B: Custom Implementation from Scratch

- Description: Implement sensitivity methods directly in dimtensor
- Pros:
  - Full control over dimensional correctness
  - No external dependencies
  - Tighter integration with existing code
- Cons:
  - Reinventing the wheel
  - More code to maintain and test
  - May have bugs that SALib has already solved

### Option C: Hybrid Approach

- Description: Implement local sensitivity natively, provide optional SALib integration for global methods
- Pros:
  - Local sensitivity is straightforward and benefits from native implementation
  - Global methods leverage SALib's proven implementations
  - Users can opt-in to SALib dependency
  - Best of both worlds
- Cons:
  - Need to maintain two code paths
  - Documentation must explain when SALib is needed

### Decision: Option C - Hybrid Approach

**Rationale**: Local sensitivity analysis is straightforward to implement and benefits from tight integration with dimtensor's gradient capabilities (especially PyTorch/JAX autograd). Global sensitivity methods are complex and benefit from SALib's mature implementations. Making SALib optional keeps the core library lightweight.

---

## Implementation Steps

### Phase 1: Core Infrastructure

1. [ ] Create `src/dimtensor/analysis/` package
2. [ ] Create `src/dimtensor/analysis/__init__.py` with public API
3. [ ] Create `src/dimtensor/analysis/sensitivity.py` for local methods
4. [ ] Create `src/dimtensor/analysis/global_sensitivity.py` for global methods

### Phase 2: Local Sensitivity Analysis

5. [ ] Implement `local_sensitivity()` function for numerical gradients
   - Finite differences with dimensional correctness
   - Support for NumPy DimArray
   - Proper unit tracking (∂output/∂param has units of output/param)
6. [ ] Implement `sensitivity_matrix()` for multi-parameter, multi-output case
   - Returns Jacobian matrix with proper units
7. [ ] Add PyTorch autograd-based sensitivity
   - Leverage DimTensor.backward() for exact gradients
   - Much faster than finite differences
8. [ ] Add JAX grad-based sensitivity
   - Use jax.grad with DimArray pytree support
   - Enable vmap for efficient multi-parameter analysis

### Phase 3: Parameter Importance Ranking

9. [ ] Implement `rank_parameters()` function
   - Normalized sensitivity coefficients
   - Handles multi-output cases
   - Returns ranked list with importance scores
10. [ ] Add visualization helpers
    - Tornado diagram (one-at-a-time sensitivity)
    - Bar charts for parameter ranking

### Phase 4: Global Sensitivity (SALib Integration)

11. [ ] Implement `check_salib()` helper to verify SALib availability
12. [ ] Create `sobol_indices()` function
    - Wraps SALib's Sobol analysis
    - Converts parameters to/from dimensionless for SALib
    - Preserves unit context for interpretation
13. [ ] Create `morris_screening()` function
    - Elementary effects method
    - Good for high-dimensional parameter spaces
14. [ ] Create `fast_sensitivity()` function
    - Fourier Amplitude Sensitivity Test
    - Efficient for many parameters

### Phase 5: Utilities and Helpers

15. [ ] Implement `normalize_parameters()` to convert DimArray parameters to [0,1] range
16. [ ] Implement `denormalize_results()` to restore units after global analysis
17. [ ] Add `SensitivityResult` dataclass to hold analysis results
18. [ ] Add plotting utilities for sensitivity visualizations

### Phase 6: Testing and Documentation

19. [ ] Write unit tests for local sensitivity
    - Test with known analytical functions
    - Verify dimensional correctness
    - Test all backends (NumPy, PyTorch, JAX)
20. [ ] Write integration tests for global sensitivity
    - Test Sobol with simple test functions
    - Verify SALib integration
21. [ ] Write examples and docstrings
22. [ ] Update documentation with sensitivity analysis guide

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/analysis/__init__.py | Create new file, export public API |
| src/dimtensor/analysis/sensitivity.py | Create new file, implement local sensitivity methods |
| src/dimtensor/analysis/global_sensitivity.py | Create new file, implement SALib integration |
| setup.py or pyproject.toml | Add optional SALib dependency: `analysis = ["SALib>=1.4"]` |
| tests/test_sensitivity.py | Create new file with comprehensive tests |
| tests/test_global_sensitivity.py | Create new file with SALib integration tests |
| docs/guides/sensitivity-analysis.md | Create new guide (if docs exist) |
| CONTINUITY.md | Update with task completion |

---

## Testing Strategy

### Unit Tests

- [ ] Test `local_sensitivity()` with quadratic function (analytical gradient)
  - f(x) = 0.5 * m * x^2 with units
  - Verify ∂f/∂m = 0.5 * x^2 has correct units
  - Verify ∂f/∂x = m * x has correct units
- [ ] Test `sensitivity_matrix()` with multi-parameter function
  - Verify shape (n_outputs, n_params)
  - Verify each element has correct units
- [ ] Test PyTorch autograd sensitivity
  - Compare with finite differences
  - Verify gradient flow
- [ ] Test JAX grad sensitivity
  - Compare with finite differences
  - Test with jit compilation
- [ ] Test parameter ranking
  - Verify sorting order
  - Test with different normalization methods

### Integration Tests

- [ ] Test full sensitivity workflow
  - Define physics model: kinetic energy E = 0.5 * m * v^2
  - Compute sensitivities ∂E/∂m and ∂E/∂v
  - Rank parameters by importance
- [ ] Test Sobol analysis (if SALib available)
  - Use Ishigami function (standard test function)
  - Verify first-order and total-order indices
- [ ] Test Morris screening
  - Verify elementary effects calculation
  - Test with high-dimensional input
- [ ] Test error handling
  - Incompatible dimensions
  - Missing SALib when needed
  - Invalid parameter ranges

### Manual Verification

- [ ] Compare results with published literature examples
- [ ] Test performance with large parameter sets
- [ ] Verify visualization outputs are interpretable

---

## Risks / Edge Cases

### Risk 1: Unit Handling in Global Methods

**Problem**: SALib expects dimensionless parameters in [0, 1] or defined ranges. How do we preserve unit information?

**Mitigation**:
- Store original units separately
- Normalize to [0, 1] for SALib computation
- Scale results back to dimensional form for interpretation
- Document clearly that Sobol indices are dimensionless ratios

### Risk 2: Computational Cost

**Problem**: Sobol analysis requires many model evaluations (typically 1000s). For expensive models, this is prohibitive.

**Mitigation**:
- Document computational requirements clearly
- Suggest Morris screening as faster alternative
- Provide progress callbacks
- Support parallel evaluation where possible

### Risk 3: Autograd Limitations

**Problem**: PyTorch/JAX autograd only works with differentiable operations. Some physics models may include non-differentiable operations (e.g., if/else, argmax).

**Mitigation**:
- Fall back to finite differences when autograd fails
- Document which operations break autograd
- Provide guidance on making models differentiable

### Risk 4: Multi-Dimensional Outputs

**Problem**: Sensitivity analysis is typically defined for scalar outputs. What about vector/tensor outputs?

**Mitigation**:
- Support sensitivity w.r.t. each output component separately
- Provide aggregation methods (sum of sensitivities, max sensitivity, etc.)
- Document how to interpret multi-output sensitivity

### Edge Case 1: Zero or Near-Zero Parameters

**Problem**: Computing relative sensitivity (∂f/∂x * x/f) fails when x ≈ 0 or f ≈ 0.

**Handling**:
- Detect near-zero values
- Use absolute sensitivity instead
- Add epsilon threshold parameter

### Edge Case 2: Dimensionless Parameters

**Problem**: Some parameters may be dimensionless (e.g., coefficients, ratios).

**Handling**:
- Treat naturally - they work the same way
- Ensure derivative units are still correct

### Edge Case 3: Constrained Parameters

**Problem**: Physical parameters often have constraints (mass > 0, probability in [0,1]).

**Handling**:
- Support bounded parameter ranges
- For global analysis, respect bounds during sampling
- Document how to specify constraints

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Local sensitivity works for NumPy, PyTorch, JAX
- [ ] Global sensitivity (Sobol, Morris) works with SALib
- [ ] All tests pass (unit + integration)
- [ ] Docstrings complete with examples
- [ ] Dimensional correctness verified
- [ ] No new type errors from mypy
- [ ] CONTINUITY.md updated
- [ ] Example notebook demonstrating usage (optional but recommended)

---

## Notes / Log

### API Design Sketch

```python
from dimtensor import DimArray, units
from dimtensor.analysis import (
    local_sensitivity,
    sensitivity_matrix,
    rank_parameters,
    sobol_indices,  # requires SALib
)

# Define model
def model(mass, velocity):
    return 0.5 * mass * velocity**2

# Parameters
m = DimArray(1.0, units.kg)
v = DimArray(10.0, units.m / units.s)

# Local sensitivity
dE_dm = local_sensitivity(model, m, args=(v,))
# Returns: DimArray with units J/kg

dE_dv = local_sensitivity(model, v, args=(m,))
# Returns: DimArray with units J·s/m

# Sensitivity matrix
params = {"mass": m, "velocity": v}
J = sensitivity_matrix(model, params)
# Returns: dict with units for each sensitivity

# Parameter ranking
ranking = rank_parameters(model, params)
# Returns: [("velocity", 0.8), ("mass", 0.2)]

# Global sensitivity (requires SALib)
param_ranges = {
    "mass": (0.5 * units.kg, 2.0 * units.kg),
    "velocity": (5.0 * units.m/units.s, 15.0 * units.m/units.s),
}
sobol_result = sobol_indices(model, param_ranges, n_samples=1000)
# Returns: SensitivityResult with S1, ST indices
```

### Implementation Notes

- Use finite differences by default: `(f(x + δ) - f(x - δ)) / (2δ)`
- Choose δ based on parameter magnitude and precision
- For autograd backends, provide `use_autograd=True` flag
- SALib integration is optional - check at runtime
- Consider adding `@jax.jit` decorator support for performance

---
