"""Tests for PyTorch DimTensor integration."""

import pytest

torch = pytest.importorskip("torch")

from dimtensor.torch import DimTensor
from dimtensor import units
from dimtensor.errors import DimensionError, UnitConversionError


class TestDimTensorCreation:
    """Tests for DimTensor creation."""

    def test_from_tensor(self):
        """Create DimTensor from torch.Tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        dt = DimTensor(t, units.m)
        assert dt.shape == (3,)
        assert dt.unit == units.m
        assert torch.allclose(dt.data, t)

    def test_from_list(self):
        """Create DimTensor from list."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        assert dt.shape == (3,)
        assert dt.unit == units.m

    def test_from_scalar(self):
        """Create DimTensor from scalar."""
        dt = DimTensor(5.0, units.kg)
        assert dt.numel == 1
        assert dt.unit == units.kg

    def test_default_dimensionless(self):
        """Default unit is dimensionless."""
        dt = DimTensor(torch.tensor([1.0]))
        assert dt.is_dimensionless

    def test_with_dtype(self):
        """Create with specific dtype."""
        dt = DimTensor([1.0, 2.0], units.m, dtype=torch.float64)
        assert dt.dtype == torch.float64

    def test_with_requires_grad(self):
        """Create with requires_grad=True."""
        dt = DimTensor([1.0, 2.0], units.m, requires_grad=True)
        assert dt.requires_grad


class TestDimTensorProperties:
    """Tests for DimTensor properties."""

    def test_shape(self):
        """Test shape property."""
        dt = DimTensor(torch.randn(2, 3, 4), units.m)
        assert dt.shape == torch.Size([2, 3, 4])

    def test_ndim(self):
        """Test ndim property."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        assert dt.ndim == 2

    def test_numel(self):
        """Test numel property."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        assert dt.numel == 6

    def test_dimension(self):
        """Test dimension property."""
        dt = DimTensor([1.0], units.m / units.s)
        assert dt.dimension == (units.m / units.s).dimension


class TestDimTensorArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_unit(self):
        """Add tensors with same unit."""
        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([3.0, 4.0], units.m)
        c = a + b
        assert torch.allclose(c.data, torch.tensor([4.0, 6.0]))
        assert c.unit == units.m

    def test_add_compatible_units(self):
        """Add tensors with compatible units (auto-convert)."""
        a = DimTensor([1000.0], units.m)
        b = DimTensor([1.0], units.km)
        c = a + b
        assert torch.allclose(c.data, torch.tensor([2000.0]))
        assert c.unit == units.m

    def test_add_incompatible_raises(self):
        """Adding incompatible dimensions raises error."""
        a = DimTensor([1.0], units.m)
        b = DimTensor([1.0], units.s)
        with pytest.raises(DimensionError):
            a + b

    def test_subtract(self):
        """Subtract tensors."""
        a = DimTensor([5.0, 6.0], units.m)
        b = DimTensor([1.0, 2.0], units.m)
        c = a - b
        assert torch.allclose(c.data, torch.tensor([4.0, 4.0]))

    def test_multiply_dimensions(self):
        """Multiply tensors - dimensions multiply."""
        length = DimTensor([2.0], units.m)
        width = DimTensor([3.0], units.m)
        area = length * width
        assert area.dimension == (units.m**2).dimension
        assert torch.allclose(area.data, torch.tensor([6.0]))

    def test_multiply_scalar(self):
        """Multiply by scalar."""
        dt = DimTensor([1.0, 2.0], units.m)
        result = dt * 3.0
        assert torch.allclose(result.data, torch.tensor([3.0, 6.0]))
        assert result.unit == units.m

    def test_divide_dimensions(self):
        """Divide tensors - dimensions divide."""
        distance = DimTensor([10.0], units.m)
        time = DimTensor([2.0], units.s)
        velocity = distance / time
        assert velocity.dimension == (units.m / units.s).dimension
        assert torch.allclose(velocity.data, torch.tensor([5.0]))

    def test_power(self):
        """Power operation scales dimension."""
        length = DimTensor([2.0], units.m)
        area = length ** 2
        assert area.dimension == (units.m**2).dimension
        assert torch.allclose(area.data, torch.tensor([4.0]))

    def test_sqrt(self):
        """Square root halves dimension."""
        area = DimTensor([4.0], units.m**2)
        length = area.sqrt()
        assert length.dimension == units.m.dimension
        assert torch.allclose(length.data, torch.tensor([2.0]))

    def test_negation(self):
        """Negation preserves unit."""
        dt = DimTensor([1.0, -2.0], units.m)
        neg = -dt
        assert torch.allclose(neg.data, torch.tensor([-1.0, 2.0]))
        assert neg.unit == units.m

    def test_abs(self):
        """Absolute value preserves unit."""
        dt = DimTensor([-1.0, 2.0, -3.0], units.m)
        result = abs(dt)
        assert torch.allclose(result.data, torch.tensor([1.0, 2.0, 3.0]))


class TestDimTensorAutograd:
    """Tests for autograd support."""

    def test_requires_grad(self):
        """Can set requires_grad."""
        dt = DimTensor([1.0, 2.0], units.m)
        dt.requires_grad_(True)
        assert dt.requires_grad

    def test_backward(self):
        """Gradients flow through operations."""
        x = DimTensor([2.0, 3.0], units.m, requires_grad=True)
        y = x ** 2
        loss = y.sum()
        loss.backward()
        # d/dx (x^2) = 2x
        expected_grad = torch.tensor([4.0, 6.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_detach(self):
        """Detach removes from computation graph."""
        dt = DimTensor([1.0], units.m, requires_grad=True)
        detached = dt.detach()
        assert not detached.requires_grad
        assert detached.unit == units.m


class TestDimTensorDevice:
    """Tests for device operations."""

    def test_default_device_cpu(self):
        """Default device is CPU."""
        dt = DimTensor([1.0], units.m)
        assert dt.device.type == "cpu"

    def test_to_device(self):
        """Move to different device."""
        dt = DimTensor([1.0], units.m)
        dt_moved = dt.to(device="cpu")  # Just test CPU since CUDA may not be available
        assert dt_moved.device.type == "cpu"
        assert dt_moved.unit == units.m

    def test_cpu_method(self):
        """CPU method returns CPU tensor."""
        dt = DimTensor([1.0], units.m)
        dt_cpu = dt.cpu()
        assert dt_cpu.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_method(self):
        """CUDA method moves to GPU."""
        dt = DimTensor([1.0], units.m)
        dt_cuda = dt.cuda()
        assert dt_cuda.device.type == "cuda"
        assert dt_cuda.unit == units.m


class TestDimTensorDtype:
    """Tests for dtype operations."""

    def test_float(self):
        """Cast to float32."""
        dt = DimTensor([1.0], units.m, dtype=torch.float64)
        dt_float = dt.float()
        assert dt_float.dtype == torch.float32

    def test_double(self):
        """Cast to float64."""
        dt = DimTensor([1.0], units.m)
        dt_double = dt.double()
        assert dt_double.dtype == torch.float64

    def test_half(self):
        """Cast to float16."""
        dt = DimTensor([1.0], units.m)
        dt_half = dt.half()
        assert dt_half.dtype == torch.float16


class TestDimTensorUnitConversion:
    """Tests for unit conversion."""

    def test_to_unit(self):
        """Convert to compatible unit."""
        dt = DimTensor([1000.0], units.m)
        dt_km = dt.to_unit(units.km)
        assert torch.allclose(dt_km.data, torch.tensor([1.0]))
        assert dt_km.unit == units.km

    def test_to_unit_incompatible_raises(self):
        """Converting to incompatible unit raises error."""
        dt = DimTensor([1.0], units.m)
        with pytest.raises(UnitConversionError):
            dt.to_unit(units.s)

    def test_magnitude(self):
        """Magnitude strips units."""
        dt = DimTensor([1.0, 2.0], units.m)
        mag = dt.magnitude()
        assert isinstance(mag, torch.Tensor)
        assert torch.allclose(mag, torch.tensor([1.0, 2.0]))


class TestDimTensorReductions:
    """Tests for reduction operations."""

    def test_sum(self):
        """Sum preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.sum()
        assert torch.allclose(result.data, torch.tensor(6.0))
        assert result.unit == units.m

    def test_mean(self):
        """Mean preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.mean()
        assert torch.allclose(result.data, torch.tensor(2.0))
        assert result.unit == units.m

    def test_var_squares_unit(self):
        """Variance squares unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.var()
        assert result.dimension == (units.m**2).dimension

    def test_min(self):
        """Min preserves unit."""
        dt = DimTensor([3.0, 1.0, 2.0], units.m)
        result = dt.min()
        assert torch.allclose(result.data, torch.tensor(1.0))
        assert result.unit == units.m

    def test_max(self):
        """Max preserves unit."""
        dt = DimTensor([3.0, 1.0, 2.0], units.m)
        result = dt.max()
        assert torch.allclose(result.data, torch.tensor(3.0))
        assert result.unit == units.m

    def test_norm(self):
        """Norm preserves unit."""
        dt = DimTensor([3.0, 4.0], units.m)
        result = dt.norm()
        assert torch.allclose(result.data, torch.tensor(5.0))
        assert result.unit == units.m


class TestDimTensorReshaping:
    """Tests for reshaping operations."""

    def test_reshape(self):
        """Reshape preserves unit."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        reshaped = dt.reshape(6)
        assert reshaped.shape == torch.Size([6])
        assert reshaped.unit == units.m

    def test_view(self):
        """View preserves unit."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        viewed = dt.view(3, 2)
        assert viewed.shape == torch.Size([3, 2])
        assert viewed.unit == units.m

    def test_transpose(self):
        """Transpose preserves unit."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        transposed = dt.transpose(0, 1)
        assert transposed.shape == torch.Size([3, 2])
        assert transposed.unit == units.m

    def test_flatten(self):
        """Flatten preserves unit."""
        dt = DimTensor(torch.randn(2, 3), units.m)
        flat = dt.flatten()
        assert flat.shape == torch.Size([6])
        assert flat.unit == units.m

    def test_squeeze(self):
        """Squeeze preserves unit."""
        dt = DimTensor(torch.randn(1, 3, 1), units.m)
        squeezed = dt.squeeze()
        assert squeezed.shape == torch.Size([3])

    def test_unsqueeze(self):
        """Unsqueeze preserves unit."""
        dt = DimTensor(torch.randn(3), units.m)
        unsqueezed = dt.unsqueeze(0)
        assert unsqueezed.shape == torch.Size([1, 3])


class TestDimTensorLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_matmul(self):
        """Matrix multiplication multiplies dimensions."""
        a = DimTensor(torch.randn(2, 3), units.m)
        b = DimTensor(torch.randn(3, 4), units.s)
        c = a.matmul(b)
        assert c.shape == torch.Size([2, 4])
        assert c.dimension == (units.m * units.s).dimension

    def test_matmul_operator(self):
        """@ operator works for matmul."""
        a = DimTensor(torch.randn(2, 3), units.m)
        b = DimTensor(torch.randn(3, 4), units.s)
        c = a @ b
        assert c.dimension == (units.m * units.s).dimension

    def test_dot(self):
        """Dot product multiplies dimensions."""
        a = DimTensor([1.0, 2.0, 3.0], units.m)
        b = DimTensor([4.0, 5.0, 6.0], units.s)
        c = a.dot(b)
        assert c.dimension == (units.m * units.s).dimension
        assert torch.allclose(c.data, torch.tensor(32.0))


class TestDimTensorComparison:
    """Tests for comparison operations."""

    def test_eq_same_unit(self):
        """Equality with same unit."""
        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([1.0, 3.0], units.m)
        result = a == b
        assert result[0].item() == True
        assert result[1].item() == False

    def test_lt(self):
        """Less than comparison."""
        a = DimTensor([1.0, 3.0], units.m)
        b = DimTensor([2.0, 2.0], units.m)
        result = a < b
        assert result[0].item() == True
        assert result[1].item() == False

    def test_compare_incompatible_raises(self):
        """Comparing incompatible dimensions raises error."""
        a = DimTensor([1.0], units.m)
        b = DimTensor([1.0], units.s)
        with pytest.raises(DimensionError):
            a < b


class TestDimTensorIndexing:
    """Tests for indexing operations."""

    def test_getitem(self):
        """Indexing preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        item = dt[1]
        assert torch.allclose(item.data, torch.tensor(2.0))
        assert item.unit == units.m

    def test_slice(self):
        """Slicing preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0, 4.0], units.m)
        sliced = dt[1:3]
        assert sliced.shape == torch.Size([2])
        assert sliced.unit == units.m

    def test_len(self):
        """Length returns first dimension size."""
        dt = DimTensor(torch.randn(5, 3), units.m)
        assert len(dt) == 5


class TestDimTensorStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes data and unit."""
        dt = DimTensor([1.0], units.m)
        r = repr(dt)
        assert "DimTensor" in r
        assert "m" in r

    def test_str_with_unit(self):
        """str shows data and unit."""
        dt = DimTensor([1.0], units.m)
        s = str(dt)
        assert "m" in s

    def test_str_dimensionless(self):
        """str for dimensionless shows only data."""
        dt = DimTensor([1.0])
        s = str(dt)
        # Should not show unit for dimensionless
        assert "m" not in s


class TestDimTensorConversion:
    """Tests for conversion methods."""

    def test_numpy(self):
        """Convert to numpy array."""
        dt = DimTensor([1.0, 2.0], units.m)
        arr = dt.numpy()
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(1.0)

    def test_item(self):
        """Get scalar value."""
        dt = DimTensor([3.14], units.m)
        assert dt.item() == pytest.approx(3.14)
