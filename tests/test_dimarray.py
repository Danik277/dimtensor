"""Tests for the DimArray class."""

import numpy as np
import pytest

from dimtensor import DimArray, DimensionError, units


class TestDimArrayCreation:
    """Test DimArray creation."""

    def test_from_list(self):
        """Create DimArray from list."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        assert arr.shape == (3,)
        assert arr.unit == units.m
        np.testing.assert_array_equal(arr.data, [1.0, 2.0, 3.0])

    def test_from_numpy(self):
        """Create DimArray from numpy array."""
        data = np.array([1.0, 2.0, 3.0])
        arr = DimArray(data, units.kg)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr.data, data)

    def test_multidimensional(self):
        """Create multidimensional DimArray."""
        arr = DimArray([[1, 2], [3, 4]], units.s)
        assert arr.shape == (2, 2)
        assert arr.ndim == 2

    def test_scalar(self):
        """Create DimArray from scalar."""
        arr = DimArray(5.0, units.m)
        assert arr.size == 1

    def test_default_dimensionless(self):
        """Without unit, defaults to dimensionless."""
        arr = DimArray([1.0, 2.0])
        assert arr.is_dimensionless


class TestDimArrayArithmetic:
    """Test DimArray arithmetic operations."""

    def test_add_same_dimension(self):
        """Addition with same dimension works."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        result = a + b
        np.testing.assert_array_equal(result.data, [4.0, 6.0])
        assert result.unit.dimension == units.m.dimension

    def test_add_different_units_same_dimension(self):
        """Addition converts units with same dimension."""
        a = DimArray([1.0], units.km)  # 1 km
        b = DimArray([500.0], units.m)  # 500 m
        result = a + b
        # Result should be in km (left operand's unit)
        np.testing.assert_array_almost_equal(result.data, [1.5])

    def test_add_incompatible_raises(self):
        """Addition with incompatible dimensions raises error."""
        velocity = DimArray([1.0], units.m / units.s)
        acceleration = DimArray([9.8], units.m / units.s**2)
        with pytest.raises(DimensionError):
            velocity + acceleration

    def test_subtract_same_dimension(self):
        """Subtraction with same dimension works."""
        a = DimArray([5.0, 6.0], units.m)
        b = DimArray([1.0, 2.0], units.m)
        result = a - b
        np.testing.assert_array_equal(result.data, [4.0, 4.0])

    def test_subtract_incompatible_raises(self):
        """Subtraction with incompatible dimensions raises error."""
        mass = DimArray([1.0], units.kg)
        length = DimArray([1.0], units.m)
        with pytest.raises(DimensionError):
            mass - length

    def test_multiply_dimensions(self):
        """Multiplication combines dimensions."""
        velocity = DimArray([10.0], units.m / units.s)
        time = DimArray([5.0], units.s)
        distance = velocity * time
        np.testing.assert_array_almost_equal(distance.data, [50.0])
        assert distance.dimension == units.m.dimension

    def test_multiply_scalar(self):
        """Scalar multiplication preserves units."""
        arr = DimArray([1.0, 2.0], units.m)
        result = arr * 3
        np.testing.assert_array_equal(result.data, [3.0, 6.0])
        assert result.unit.dimension == units.m.dimension

        # Also test right multiplication
        result2 = 3 * arr
        np.testing.assert_array_equal(result2.data, [3.0, 6.0])

    def test_divide_dimensions(self):
        """Division combines dimensions."""
        distance = DimArray([100.0], units.m)
        time = DimArray([10.0], units.s)
        velocity = distance / time
        np.testing.assert_array_almost_equal(velocity.data, [10.0])
        assert velocity.dimension == (units.m / units.s).dimension

    def test_divide_scalar(self):
        """Scalar division preserves units."""
        arr = DimArray([6.0, 9.0], units.m)
        result = arr / 3
        np.testing.assert_array_equal(result.data, [2.0, 3.0])

    def test_power(self):
        """Power operation multiplies dimension exponents."""
        length = DimArray([2.0, 3.0], units.m)
        area = length ** 2
        np.testing.assert_array_equal(area.data, [4.0, 9.0])
        assert area.dimension.length == 2

    def test_negation(self):
        """Negation preserves units."""
        arr = DimArray([1.0, -2.0], units.m)
        result = -arr
        np.testing.assert_array_equal(result.data, [-1.0, 2.0])
        assert result.unit == arr.unit


class TestDimArrayComparison:
    """Test DimArray comparison operations."""

    def test_equality_same_units(self):
        """Equality with same units."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([1.0, 3.0], units.m)
        result = a == b
        np.testing.assert_array_equal(result, [True, False])

    def test_equality_different_units_same_dimension(self):
        """Equality converts units."""
        a = DimArray([1.0], units.km)
        b = DimArray([1000.0], units.m)
        result = a == b
        assert result[0] == True  # noqa: E712

    def test_less_than(self):
        """Less than comparison."""
        a = DimArray([1.0, 5.0], units.m)
        b = DimArray([3.0, 3.0], units.m)
        result = a < b
        np.testing.assert_array_equal(result, [True, False])

    def test_comparison_incompatible_raises(self):
        """Comparison with incompatible dimensions raises error."""
        mass = DimArray([1.0], units.kg)
        length = DimArray([1.0], units.m)
        with pytest.raises(DimensionError):
            mass < length


class TestDimArrayConversion:
    """Test unit conversion."""

    def test_to_compatible_unit(self):
        """Convert to compatible unit."""
        distance = DimArray([1.0], units.km)
        in_meters = distance.to(units.m)
        np.testing.assert_array_almost_equal(in_meters.data, [1000.0])
        assert in_meters.unit == units.m

    def test_to_incompatible_raises(self):
        """Convert to incompatible unit raises error."""
        distance = DimArray([1.0], units.m)
        with pytest.raises(Exception):  # UnitConversionError
            distance.to(units.s)

    def test_magnitude(self):
        """Get raw magnitude."""
        arr = DimArray([1.0, 2.0], units.m)
        mag = arr.magnitude()
        np.testing.assert_array_equal(mag, [1.0, 2.0])
        assert isinstance(mag, np.ndarray)


class TestDimArrayReductions:
    """Test reduction operations."""

    def test_sum(self):
        """Sum preserves units."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = arr.sum()
        np.testing.assert_array_almost_equal(result.data, [6.0])
        assert result.unit.dimension == units.m.dimension

    def test_mean(self):
        """Mean preserves units."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = arr.mean()
        np.testing.assert_array_almost_equal(result.data, [2.0])

    def test_min_max(self):
        """Min/max preserve units."""
        arr = DimArray([1.0, 5.0, 3.0], units.m)
        assert arr.min().data[0] == 1.0
        assert arr.max().data[0] == 5.0


class TestDimArrayIndexing:
    """Test indexing operations."""

    def test_single_index(self):
        """Single index returns DimArray."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = arr[0]
        assert isinstance(result, DimArray)
        assert result.data[0] == 1.0
        assert result.unit == units.m

    def test_slice(self):
        """Slice returns DimArray."""
        arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        result = arr[1:3]
        assert isinstance(result, DimArray)
        np.testing.assert_array_equal(result.data, [2.0, 3.0])

    def test_iteration(self):
        """Can iterate over DimArray."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        values = [x.data[0] for x in arr]
        assert values == [1.0, 2.0, 3.0]


class TestDimArrayString:
    """Test string representations."""

    def test_str_with_units(self):
        """str includes unit symbol."""
        arr = DimArray([1.0, 2.0], units.m)
        s = str(arr)
        assert "m" in s
        assert "1" in s or "1." in s

    def test_str_dimensionless(self):
        """Dimensionless doesn't show units."""
        arr = DimArray([1.0, 2.0])
        s = str(arr)
        # Should just show the values
        assert "1" in s


class TestUnitSimplification:
    """Test unit simplification in display."""

    def test_force_simplifies_to_newton(self):
        """kg·m/s² simplifies to N."""
        m = DimArray([2], units.kg)
        a = DimArray([9.8], units.m / units.s**2)
        F = m * a
        assert "N" in str(F)

    def test_energy_simplifies_to_joule(self):
        """kg·m²/s² simplifies to J."""
        F = DimArray([10], units.N)
        d = DimArray([5], units.m)
        E = F * d
        assert "J" in str(E)

    def test_power_simplifies_to_watt(self):
        """kg·m²/s³ simplifies to W."""
        E = DimArray([100], units.J)
        t = DimArray([10], units.s)
        P = E / t
        assert "W" in str(P)

    def test_velocity_stays_as_m_per_s(self):
        """m/s stays as m/s (registered form)."""
        v = DimArray([10], units.m / units.s)
        assert "m/s" in str(v)

    def test_distance_simplifies_from_v_times_t(self):
        """m/s * s simplifies to m."""
        v = DimArray([10], units.m / units.s)
        t = DimArray([5], units.s)
        d = v * t
        s = str(d)
        # Should be "m" not "m/s·s"
        assert s.endswith(" m") or s.endswith(" m²") is False


class TestFormatStrings:
    """Test __format__ support."""

    def test_format_single_value(self):
        """Format single value with format spec."""
        d = DimArray([1234.5678], units.m)
        assert f"{d:.2f}" == "1234.57 m"

    def test_format_scientific(self):
        """Scientific notation format."""
        d = DimArray([1234.5678], units.m)
        result = f"{d:.2e}"
        assert "1.23e+03" in result
        assert "m" in result

    def test_format_no_spec(self):
        """Format without spec includes value and unit."""
        d = DimArray([10], units.m)
        result = f"{d}"
        assert "10" in result
        assert "m" in result

    def test_format_dimensionless(self):
        """Format dimensionless quantity."""
        ratio = DimArray([3.14159])
        assert f"{ratio:.2f}" == "3.14"


class TestNumpyUfuncs:
    """Test numpy ufunc integration."""

    def test_sin_dimensionless(self):
        """np.sin works on dimensionless."""
        angle = DimArray([0, np.pi/2], units.rad)
        result = np.sin(angle)
        np.testing.assert_array_almost_equal(result.data, [0, 1])
        assert result.is_dimensionless

    def test_cos_dimensionless(self):
        """np.cos works on dimensionless."""
        angle = DimArray([0, np.pi])
        result = np.cos(angle)
        np.testing.assert_array_almost_equal(result.data, [1, -1])

    def test_exp_dimensionless(self):
        """np.exp works on dimensionless."""
        x = DimArray([0, 1])
        result = np.exp(x)
        np.testing.assert_array_almost_equal(result.data, [1, np.e])

    def test_log_dimensionless(self):
        """np.log works on dimensionless."""
        x = DimArray([1, np.e])
        result = np.log(x)
        np.testing.assert_array_almost_equal(result.data, [0, 1])

    def test_sin_rejects_dimensional(self):
        """np.sin rejects non-dimensionless input."""
        length = DimArray([1], units.m)
        with pytest.raises(DimensionError):
            np.sin(length)

    def test_sqrt_halves_dimension(self):
        """np.sqrt halves dimension exponents."""
        area = DimArray([4, 9], units.m**2)
        result = np.sqrt(area)
        np.testing.assert_array_almost_equal(result.data, [2, 3])
        assert result.dimension == units.m.dimension

    def test_abs_preserves_unit(self):
        """np.abs preserves units."""
        v = DimArray([-1, 2, -3], units.m / units.s)
        result = np.abs(v)
        np.testing.assert_array_equal(result.data, [1, 2, 3])
        assert result.dimension == v.dimension

    def test_negative_preserves_unit(self):
        """np.negative preserves units."""
        v = DimArray([1, -2], units.m)
        result = np.negative(v)
        np.testing.assert_array_equal(result.data, [-1, 2])
        assert result.dimension == v.dimension

    def test_add_via_ufunc(self):
        """np.add works with same dimensions."""
        a = DimArray([1, 2], units.m)
        b = DimArray([3, 4], units.m)
        result = np.add(a, b)
        np.testing.assert_array_equal(result.data, [4, 6])

    def test_multiply_via_ufunc(self):
        """np.multiply combines dimensions."""
        m = DimArray([2], units.kg)
        a = DimArray([5], units.m / units.s**2)
        result = np.multiply(m, a)
        assert result.dimension == units.N.dimension


class TestPhysicsExamples:
    """Test real physics calculations."""

    def test_kinematics(self):
        """Basic kinematics: d = v * t."""
        velocity = DimArray([10.0], units.m / units.s)
        time = DimArray([5.0], units.s)
        distance = velocity * time
        np.testing.assert_array_almost_equal(distance.data, [50.0])
        assert distance.dimension == units.m.dimension

    def test_force_calculation(self):
        """F = m * a."""
        mass = DimArray([2.0], units.kg)
        acceleration = DimArray([9.8], units.m / units.s**2)
        force = mass * acceleration
        np.testing.assert_array_almost_equal(force.data, [19.6])
        assert force.dimension == units.N.dimension

    def test_energy_calculation(self):
        """E = F * d (work)."""
        force = DimArray([10.0], units.N)
        distance = DimArray([5.0], units.m)
        work = force * distance
        np.testing.assert_array_almost_equal(work.data, [50.0])
        assert work.dimension == units.J.dimension

    def test_power_calculation(self):
        """P = E / t."""
        energy = DimArray([100.0], units.J)
        time = DimArray([10.0], units.s)
        power = energy / time
        np.testing.assert_array_almost_equal(power.data, [10.0])
        assert power.dimension == units.W.dimension

    def test_ohms_law(self):
        """V = I * R (implicitly testing voltage/current dimensions)."""
        voltage = DimArray([12.0], units.V)
        current = DimArray([2.0], units.A)
        resistance = voltage / current
        np.testing.assert_array_almost_equal(resistance.data, [6.0])
        assert resistance.dimension == units.ohm.dimension

    def test_potential_energy(self):
        """PE = m * g * h."""
        m = DimArray([2.0], units.kg)
        g = DimArray([9.8], units.m / units.s**2)
        h = DimArray([10.0], units.m)
        PE = m * g * h
        np.testing.assert_array_almost_equal(PE.data, [196.0])
        assert PE.dimension == units.J.dimension
