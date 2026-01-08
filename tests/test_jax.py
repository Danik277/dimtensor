"""Tests for JAX DimArray integration."""

import pytest

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except (ImportError, RuntimeError):
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")

if JAX_AVAILABLE:
    from dimtensor.jax import DimArray

from dimtensor import units
from dimtensor.errors import DimensionError, UnitConversionError


class TestDimArrayCreation:
    """Tests for JAX DimArray creation."""

    def test_from_jax_array(self):
        """Create DimArray from JAX array."""
        arr = jnp.array([1.0, 2.0, 3.0])
        da = DimArray(arr, units.m)
        assert da.shape == (3,)
        assert da.unit == units.m

    def test_from_list(self):
        """Create DimArray from list."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        assert da.shape == (3,)
        assert da.unit == units.m

    def test_from_scalar(self):
        """Create DimArray from scalar."""
        da = DimArray(5.0, units.kg)
        assert da.size == 1
        assert da.unit == units.kg

    def test_default_dimensionless(self):
        """Default unit is dimensionless."""
        da = DimArray(jnp.array([1.0]))
        assert da.is_dimensionless


class TestDimArrayProperties:
    """Tests for DimArray properties."""

    def test_shape(self):
        """Test shape property."""
        da = DimArray(jnp.zeros((2, 3, 4)), units.m)
        assert da.shape == (2, 3, 4)

    def test_ndim(self):
        """Test ndim property."""
        da = DimArray(jnp.zeros((2, 3)), units.m)
        assert da.ndim == 2

    def test_size(self):
        """Test size property."""
        da = DimArray(jnp.zeros((2, 3)), units.m)
        assert da.size == 6

    def test_dimension(self):
        """Test dimension property."""
        da = DimArray([1.0], units.m / units.s)
        assert da.dimension == (units.m / units.s).dimension


class TestDimArrayArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_unit(self):
        """Add arrays with same unit."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        c = a + b
        assert jnp.allclose(c.data, jnp.array([4.0, 6.0]))
        assert c.unit == units.m

    def test_add_compatible_units(self):
        """Add arrays with compatible units (auto-convert)."""
        a = DimArray([1000.0], units.m)
        b = DimArray([1.0], units.km)
        c = a + b
        assert jnp.allclose(c.data, jnp.array([2000.0]))
        assert c.unit == units.m

    def test_add_incompatible_raises(self):
        """Adding incompatible dimensions raises error."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a + b

    def test_subtract(self):
        """Subtract arrays."""
        a = DimArray([5.0, 6.0], units.m)
        b = DimArray([1.0, 2.0], units.m)
        c = a - b
        assert jnp.allclose(c.data, jnp.array([4.0, 4.0]))

    def test_multiply_dimensions(self):
        """Multiply arrays - dimensions multiply."""
        length = DimArray([2.0], units.m)
        width = DimArray([3.0], units.m)
        area = length * width
        assert area.dimension == (units.m**2).dimension
        assert jnp.allclose(area.data, jnp.array([6.0]))

    def test_multiply_scalar(self):
        """Multiply by scalar."""
        da = DimArray([1.0, 2.0], units.m)
        result = da * 3.0
        assert jnp.allclose(result.data, jnp.array([3.0, 6.0]))
        assert result.unit == units.m

    def test_divide_dimensions(self):
        """Divide arrays - dimensions divide."""
        distance = DimArray([10.0], units.m)
        time = DimArray([2.0], units.s)
        velocity = distance / time
        assert velocity.dimension == (units.m / units.s).dimension
        assert jnp.allclose(velocity.data, jnp.array([5.0]))

    def test_power(self):
        """Power operation scales dimension."""
        length = DimArray([2.0], units.m)
        area = length ** 2
        assert area.dimension == (units.m**2).dimension
        assert jnp.allclose(area.data, jnp.array([4.0]))

    def test_sqrt(self):
        """Square root halves dimension."""
        area = DimArray([4.0], units.m**2)
        length = area.sqrt()
        assert length.dimension == units.m.dimension
        assert jnp.allclose(length.data, jnp.array([2.0]))

    def test_negation(self):
        """Negation preserves unit."""
        da = DimArray([1.0, -2.0], units.m)
        neg = -da
        assert jnp.allclose(neg.data, jnp.array([-1.0, 2.0]))
        assert neg.unit == units.m


class TestJAXJit:
    """Tests for JAX JIT compilation."""

    def test_jit_basic(self):
        """JIT compilation preserves units."""
        @jax.jit
        def double(x):
            return x * 2.0

        da = DimArray([1.0, 2.0], units.m)
        result = double(da)
        assert jnp.allclose(result.data, jnp.array([2.0, 4.0]))
        assert result.unit == units.m

    def test_jit_multiply(self):
        """JIT multiplication preserves dimension algebra."""
        @jax.jit
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity ** 2

        m = DimArray([1.0, 2.0], units.kg)
        v = DimArray([3.0, 4.0], units.m / units.s)
        E = kinetic_energy(m, v)

        assert jnp.allclose(E.data, jnp.array([4.5, 16.0]))
        assert E.dimension == units.J.dimension

    def test_jit_add(self):
        """JIT addition preserves units."""
        @jax.jit
        def add_arrays(a, b):
            return a + b

        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        result = add_arrays(a, b)
        assert jnp.allclose(result.data, jnp.array([4.0, 6.0]))
        assert result.unit == units.m

    def test_jit_chain(self):
        """JIT with multiple operations."""
        @jax.jit
        def compute(x, y, t):
            return x + y * t

        x = DimArray([1.0], units.m)
        v = DimArray([2.0], units.m / units.s)
        t = DimArray([3.0], units.s)
        result = compute(x, v, t)
        assert jnp.allclose(result.data, jnp.array([7.0]))
        assert result.unit == units.m


class TestJAXVmap:
    """Tests for JAX vmap vectorization."""

    def test_vmap_basic(self):
        """vmap over batch dimension."""
        def square(x):
            return x ** 2

        # Create batched input
        batch = DimArray(jnp.array([[1.0], [2.0], [3.0]]), units.m)
        result = jax.vmap(square)(batch)
        assert jnp.allclose(result.data, jnp.array([[1.0], [4.0], [9.0]]))
        assert result.dimension == (units.m**2).dimension

    def test_vmap_binary(self):
        """vmap with two inputs."""
        def multiply(a, b):
            return a * b

        lengths = DimArray(jnp.array([[2.0], [3.0]]), units.m)
        widths = DimArray(jnp.array([[4.0], [5.0]]), units.m)
        areas = jax.vmap(multiply)(lengths, widths)
        assert jnp.allclose(areas.data, jnp.array([[8.0], [15.0]]))
        assert areas.dimension == (units.m**2).dimension


class TestJAXGrad:
    """Tests for JAX grad differentiation."""

    def test_grad_scalar(self):
        """Gradient of scalar function."""
        def f(x):
            return (x ** 2).sum()

        # Note: grad returns raw arrays for leaf gradients
        da = DimArray(jnp.array([3.0]), units.m)
        grad_f = jax.grad(lambda x: f(x).data.sum())
        grad_val = grad_f(da)

        # Gradient should be 2*x = 6.0
        assert jnp.allclose(grad_val.data, jnp.array([6.0]))

    def test_grad_preserves_structure(self):
        """Gradient preserves DimArray structure."""
        def f(x):
            return x ** 2

        da = DimArray(jnp.array([2.0, 3.0]), units.m)
        grad_f = jax.grad(lambda x: f(x).data.sum())
        grad_val = grad_f(da)

        assert isinstance(grad_val, DimArray)
        assert jnp.allclose(grad_val.data, jnp.array([4.0, 6.0]))


class TestDimArrayUnitConversion:
    """Tests for unit conversion."""

    def test_to_unit(self):
        """Convert to compatible unit."""
        da = DimArray([1000.0], units.m)
        da_km = da.to(units.km)
        assert jnp.allclose(da_km.data, jnp.array([1.0]))
        assert da_km.unit == units.km

    def test_to_unit_incompatible_raises(self):
        """Converting to incompatible unit raises error."""
        da = DimArray([1.0], units.m)
        with pytest.raises(UnitConversionError):
            da.to(units.s)


class TestDimArrayReductions:
    """Tests for reduction operations."""

    def test_sum(self):
        """Sum preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.sum()
        assert jnp.allclose(result.data, jnp.array(6.0))
        assert result.unit == units.m

    def test_mean(self):
        """Mean preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.mean()
        assert jnp.allclose(result.data, jnp.array(2.0))
        assert result.unit == units.m

    def test_var_squares_unit(self):
        """Variance squares unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.var()
        assert result.dimension == (units.m**2).dimension

    def test_min(self):
        """Min preserves unit."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.min()
        assert jnp.allclose(result.data, jnp.array(1.0))
        assert result.unit == units.m

    def test_max(self):
        """Max preserves unit."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.max()
        assert jnp.allclose(result.data, jnp.array(3.0))
        assert result.unit == units.m


class TestDimArrayReshaping:
    """Tests for reshaping operations."""

    def test_reshape(self):
        """Reshape preserves unit."""
        da = DimArray(jnp.zeros((2, 3)), units.m)
        reshaped = da.reshape((6,))
        assert reshaped.shape == (6,)
        assert reshaped.unit == units.m

    def test_transpose(self):
        """Transpose preserves unit."""
        da = DimArray(jnp.zeros((2, 3)), units.m)
        transposed = da.transpose()
        assert transposed.shape == (3, 2)
        assert transposed.unit == units.m

    def test_flatten(self):
        """Flatten preserves unit."""
        da = DimArray(jnp.zeros((2, 3)), units.m)
        flat = da.flatten()
        assert flat.shape == (6,)
        assert flat.unit == units.m


class TestDimArrayLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_dot(self):
        """Dot product multiplies dimensions."""
        a = DimArray([1.0, 2.0, 3.0], units.m)
        b = DimArray([4.0, 5.0, 6.0], units.s)
        c = a.dot(b)
        assert c.dimension == (units.m * units.s).dimension
        assert jnp.allclose(c.data, jnp.array(32.0))

    def test_matmul(self):
        """Matrix multiplication multiplies dimensions."""
        a = DimArray(jnp.zeros((2, 3)), units.m)
        b = DimArray(jnp.zeros((3, 4)), units.s)
        c = a.matmul(b)
        assert c.shape == (2, 4)
        assert c.dimension == (units.m * units.s).dimension

    def test_matmul_operator(self):
        """@ operator works for matmul."""
        a = DimArray(jnp.ones((2, 3)), units.m)
        b = DimArray(jnp.ones((3, 4)), units.s)
        c = a @ b
        assert c.dimension == (units.m * units.s).dimension


class TestDimArrayComparison:
    """Tests for comparison operations."""

    def test_eq_same_unit(self):
        """Equality with same unit."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([1.0, 3.0], units.m)
        result = a == b
        assert result[0].item() == True
        assert result[1].item() == False

    def test_lt(self):
        """Less than comparison."""
        a = DimArray([1.0, 3.0], units.m)
        b = DimArray([2.0, 2.0], units.m)
        result = a < b
        assert result[0].item() == True
        assert result[1].item() == False

    def test_compare_incompatible_raises(self):
        """Comparing incompatible dimensions raises error."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a < b


class TestDimArrayIndexing:
    """Tests for indexing operations."""

    def test_getitem(self):
        """Indexing preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        item = da[1]
        assert jnp.allclose(item.data, jnp.array(2.0))
        assert item.unit == units.m

    def test_slice(self):
        """Slicing preserves unit."""
        da = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        sliced = da[1:3]
        assert sliced.shape == (2,)
        assert sliced.unit == units.m


class TestDimArrayStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes data and unit."""
        da = DimArray([1.0], units.m)
        r = repr(da)
        assert "DimArray" in r
        assert "m" in r

    def test_str_with_unit(self):
        """str shows data and unit."""
        da = DimArray([1.0], units.m)
        s = str(da)
        assert "m" in s


class TestDimArrayConversion:
    """Tests for conversion methods."""

    def test_numpy(self):
        """Convert to numpy array."""
        da = DimArray([1.0, 2.0], units.m)
        arr = da.numpy()
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(1.0)
