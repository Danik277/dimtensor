"""Tests for module-level array functions."""

import numpy as np
import pytest

from dimtensor import (
    DimArray,
    DimensionError,
    concatenate,
    stack,
    split,
    dot,
    matmul,
    norm,
    units,
)


class TestConcatenate:
    """Test concatenate function."""

    def test_concatenate_same_units(self):
        """Concatenate arrays with same units."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        result = concatenate([a, b])
        np.testing.assert_array_equal(result.data, [1.0, 2.0, 3.0, 4.0])
        assert result.unit.dimension == units.m.dimension

    def test_concatenate_compatible_units(self):
        """Concatenate arrays with compatible units (converts)."""
        a = DimArray([1.0], units.km)  # 1 km
        b = DimArray([500.0], units.m)  # 500 m = 0.5 km
        result = concatenate([a, b])
        np.testing.assert_array_almost_equal(result.data, [1.0, 0.5])
        assert result.unit == units.km

    def test_concatenate_incompatible_raises(self):
        """Concatenate with incompatible dimensions raises."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            concatenate([a, b])

    def test_concatenate_along_axis(self):
        """Concatenate 2D arrays along axis."""
        a = DimArray([[1, 2], [3, 4]], units.m)
        b = DimArray([[5, 6]], units.m)
        result = concatenate([a, b], axis=0)
        assert result.shape == (3, 2)

    def test_concatenate_empty_raises(self):
        """Empty sequence raises ValueError."""
        with pytest.raises(ValueError):
            concatenate([])

    def test_concatenate_single_array(self):
        """Single array returns copy."""
        a = DimArray([1.0, 2.0], units.m)
        result = concatenate([a])
        np.testing.assert_array_equal(result.data, a.data)
        assert result.unit == a.unit


class TestStack:
    """Test stack function."""

    def test_stack_same_units(self):
        """Stack arrays with same units."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        result = stack([a, b])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_stack_incompatible_raises(self):
        """Stack with incompatible dimensions raises."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.kg)
        with pytest.raises(DimensionError):
            stack([a, b])

    def test_stack_along_axis(self):
        """Stack along specified axis."""
        a = DimArray([1, 2], units.m)
        b = DimArray([3, 4], units.m)
        result = stack([a, b], axis=1)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result.data, [[1, 3], [2, 4]])

    def test_stack_empty_raises(self):
        """Empty sequence raises ValueError."""
        with pytest.raises(ValueError):
            stack([])

    def test_stack_compatible_units(self):
        """Stack arrays with compatible units (converts)."""
        a = DimArray([1.0], units.km)
        b = DimArray([1000.0], units.m)  # 1 km
        result = stack([a, b])
        np.testing.assert_array_almost_equal(result.data, [[1.0], [1.0]])


class TestSplit:
    """Test split function."""

    def test_split_equal_parts(self):
        """Split into equal parts."""
        arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        result = split(arr, 2)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0].data, [1.0, 2.0])
        np.testing.assert_array_equal(result[1].data, [3.0, 4.0])
        assert result[0].unit == units.m
        assert result[1].unit == units.m

    def test_split_at_indices(self):
        """Split at specified indices."""
        arr = DimArray([1.0, 2.0, 3.0, 4.0, 5.0], units.m)
        result = split(arr, [2, 4])
        assert len(result) == 3
        np.testing.assert_array_equal(result[0].data, [1.0, 2.0])
        np.testing.assert_array_equal(result[1].data, [3.0, 4.0])
        np.testing.assert_array_equal(result[2].data, [5.0])

    def test_split_2d(self):
        """Split 2D array."""
        arr = DimArray([[1, 2], [3, 4], [5, 6], [7, 8]], units.kg)
        result = split(arr, 2, axis=0)
        assert len(result) == 2
        assert result[0].shape == (2, 2)
        assert result[0].unit == units.kg


class TestDot:
    """Test dot product function."""

    def test_dot_dimensions_multiply(self):
        """Dot product multiplies dimensions."""
        length = DimArray([1.0, 2.0, 3.0], units.m)
        force = DimArray([4.0, 5.0, 6.0], units.N)
        work = dot(length, force)
        # m * N = m * kg*m/s^2 = kg*m^2/s^2 = J
        np.testing.assert_array_almost_equal(work.data, [32.0])
        assert work.dimension == units.J.dimension

    def test_dot_same_dimension(self):
        """Dot product of same dimensions squares."""
        v = DimArray([3.0, 4.0], units.m / units.s)
        v_squared = dot(v, v)
        # (m/s)^2
        np.testing.assert_array_almost_equal(v_squared.data, [25.0])
        assert v_squared.dimension.length == 2
        assert v_squared.dimension.time == -2

    def test_dot_matrix_vector(self):
        """Dot product of matrix and vector."""
        matrix = DimArray([[1, 0], [0, 1]], units.m)
        vector = DimArray([3.0, 4.0], units.s)
        result = dot(matrix, vector)
        # m * s = m*s
        assert result.shape == (2,)
        assert result.dimension.length == 1
        assert result.dimension.time == 1

    def test_dot_scalar_result(self):
        """Dot product producing scalar is wrapped in 1D array."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.s)
        result = dot(a, b)
        assert result.ndim >= 1  # Not a scalar
        np.testing.assert_array_almost_equal(result.data, [11.0])


class TestMatmul:
    """Test matrix multiplication function."""

    def test_matmul_dimensions_multiply(self):
        """Matrix multiplication multiplies dimensions."""
        A = DimArray([[1, 2], [3, 4]], units.m)
        B = DimArray([[5, 6], [7, 8]], units.s)
        result = matmul(A, B)
        assert result.shape == (2, 2)
        assert result.dimension.length == 1
        assert result.dimension.time == 1

    def test_matmul_physics_example(self):
        """Physics example: rotation matrix times velocity."""
        # 90-degree rotation matrix (dimensionless)
        R = DimArray([[0, -1], [1, 0]], units.rad)  # rad is dimensionless
        v = DimArray([1.0, 0.0], units.m / units.s)
        v_rotated = matmul(R, v)
        np.testing.assert_array_almost_equal(v_rotated.data, [0.0, 1.0])

    def test_matmul_vector_vector(self):
        """Matrix multiplication of two vectors."""
        a = DimArray([1.0, 2.0, 3.0], units.m)
        b = DimArray([[1.0], [2.0], [3.0]], units.s)
        result = matmul(a, b)
        np.testing.assert_array_almost_equal(result.data, [14.0])


class TestNorm:
    """Test norm function."""

    def test_norm_preserves_unit(self):
        """Norm preserves original unit."""
        v = DimArray([3.0, 4.0], units.m)
        result = norm(v)
        np.testing.assert_array_almost_equal(result.data, [5.0])
        assert result.unit.dimension == units.m.dimension

    def test_norm_velocity_magnitude(self):
        """Physics: velocity magnitude."""
        velocity = DimArray([3.0, 4.0, 0.0], units.m / units.s)
        speed = norm(velocity)
        np.testing.assert_array_almost_equal(speed.data, [5.0])
        assert speed.dimension == (units.m / units.s).dimension

    def test_norm_with_axis(self):
        """Norm along axis."""
        arr = DimArray([[3.0, 4.0], [5.0, 12.0]], units.m)
        result = norm(arr, axis=1)
        np.testing.assert_array_almost_equal(result.data, [5.0, 13.0])

    def test_norm_frobenius(self):
        """Frobenius norm of matrix (default for 2D)."""
        arr = DimArray([[1, 2], [3, 4]], units.m)
        # Default for matrix is Frobenius norm
        result = norm(arr)
        expected = np.sqrt(1 + 4 + 9 + 16)
        np.testing.assert_array_almost_equal(result.data, [expected])

    def test_norm_keepdims(self):
        """Norm with keepdims."""
        arr = DimArray([[3.0, 4.0]], units.m)
        result = norm(arr, axis=1, keepdims=True)
        assert result.shape == (1, 1)
        np.testing.assert_array_almost_equal(result.data, [[5.0]])

    def test_norm_1d(self):
        """Norm of 1D array."""
        arr = DimArray([1.0, 2.0, 2.0], units.kg)
        result = norm(arr)
        np.testing.assert_array_almost_equal(result.data, [3.0])
        assert result.unit == units.kg
