"""Tests for SymPy integration."""

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, units

pytest.importorskip("sympy")
import sympy as sp
from sympy.physics.units import meter, second, kilogram, newton, joule

from dimtensor.sympy import (
    to_sympy,
    from_sympy,
    sympy_unit_for,
    symbolic_diff,
    symbolic_integrate,
)
from dimtensor.sympy.calculus import simplify_units, substitute


class TestSympyUnitFor:
    """Tests for sympy_unit_for conversion."""

    def test_base_units(self):
        """Test conversion of base SI units."""
        # Use simplify to handle 1.0*meter vs meter
        assert sp.simplify(sympy_unit_for(units.m) - meter) == 0
        assert sp.simplify(sympy_unit_for(units.kg) - kilogram) == 0
        assert sp.simplify(sympy_unit_for(units.s) - second) == 0

    def test_derived_units(self):
        """Test conversion of derived units."""
        n_unit = sympy_unit_for(units.N)
        # Newton = kg*m/s^2
        assert sp.simplify(n_unit - newton) == 0

    def test_compound_units(self):
        """Test conversion of compound units."""
        velocity_unit = sympy_unit_for(units.m / units.s)
        # Should be meter/second
        assert sp.simplify(velocity_unit - meter / second) == 0


class TestToSympy:
    """Tests for to_sympy conversion."""

    def test_scalar_numerical(self):
        """Test conversion of scalar DimArray to numerical SymPy."""
        arr = DimArray([10], units.m)
        expr = to_sympy(arr)

        assert sp.simplify(expr - 10 * meter) == 0

    def test_scalar_symbolic(self):
        """Test conversion to symbolic SymPy expression."""
        arr = DimArray([10], units.m)
        expr = to_sympy(arr, symbol="x")

        x = sp.Symbol("x", real=True)
        assert sp.simplify(expr - x * meter) == 0

    def test_with_derived_unit(self):
        """Test conversion with derived unit."""
        force = DimArray([100], units.N)
        expr = to_sympy(force)

        assert sp.simplify(expr - 100 * newton) == 0


class TestFromSympy:
    """Tests for from_sympy conversion."""

    def test_simple_quantity(self):
        """Test conversion of simple quantity."""
        expr = 10 * meter
        arr = from_sympy(expr)

        assert isinstance(arr, DimArray)
        np.testing.assert_almost_equal(arr.data[0], 10.0)
        assert arr.unit.dimension == Dimension(length=1)

    def test_compound_quantity(self):
        """Test conversion of compound quantity."""
        expr = 100 * meter / second
        arr = from_sympy(expr)

        assert isinstance(arr, DimArray)
        np.testing.assert_almost_equal(arr.data[0], 100.0)
        assert arr.unit.dimension == Dimension(length=1, time=-1)

    def test_derived_unit(self):
        """Test conversion of derived unit."""
        expr = 50 * newton
        arr = from_sympy(expr)

        assert isinstance(arr, DimArray)
        # Newton = kg*m/s^2
        assert arr.unit.dimension == Dimension(length=1, mass=1, time=-2)


class TestSymbolicDiff:
    """Tests for symbolic differentiation."""

    def test_position_to_velocity(self):
        """Test differentiating position to get velocity."""
        t = sp.Symbol("t")
        # Position x = 5*t^2 meters
        x = 5 * t**2 * meter

        # Velocity = dx/dt
        v = symbolic_diff(x, t, var_unit=units.s)

        # v = 10*t meters/second
        expected = 10 * t * meter / second
        assert sp.simplify(v - expected) == 0

    def test_velocity_to_acceleration(self):
        """Test differentiating velocity to get acceleration."""
        t = sp.Symbol("t")
        # Velocity v = 10*t m/s
        v = 10 * t * meter / second

        # Acceleration = dv/dt
        a = symbolic_diff(v, t, var_unit=units.s)

        # a = 10 m/s^2
        expected = 10 * meter / second**2
        assert sp.simplify(a - expected) == 0


class TestSymbolicIntegrate:
    """Tests for symbolic integration."""

    def test_velocity_to_position(self):
        """Test integrating velocity to get position."""
        t = sp.Symbol("t")
        # Constant velocity v = 10 m/s
        v = 10 * meter / second

        # Position = integral of v dt
        x = symbolic_integrate(v, t, var_unit=units.s)

        # x = 10*t meters
        expected = 10 * t * meter
        assert sp.simplify(x - expected) == 0

    def test_acceleration_to_velocity(self):
        """Test integrating acceleration to get velocity."""
        t = sp.Symbol("t")
        # Constant acceleration a = 9.8 m/s^2
        a = sp.Rational(98, 10) * meter / second**2

        # Velocity = integral of a dt
        v = symbolic_integrate(a, t, var_unit=units.s)

        # v = 9.8*t m/s
        expected = sp.Rational(98, 10) * t * meter / second
        assert sp.simplify(v - expected) == 0


class TestSimplifyUnits:
    """Tests for unit simplification."""

    def test_simplify_fraction(self):
        """Test simplifying unit fractions."""
        expr = meter / meter
        result = simplify_units(expr)

        assert result == 1


class TestSubstitute:
    """Tests for substitution."""

    def test_substitute_numbers(self):
        """Test substituting numerical values."""
        t = sp.Symbol("t")
        v = sp.Symbol("v")
        x = v * t * meter

        result = substitute(x, {"v": 10, "t": 5})

        assert sp.simplify(result - 50 * meter) == 0


class TestRoundtrip:
    """Tests for roundtrip conversion."""

    def test_dimarray_roundtrip(self):
        """Test DimArray -> SymPy -> DimArray roundtrip."""
        original = DimArray([25.5], units.m)

        # Convert to sympy and back
        expr = to_sympy(original)
        restored = from_sympy(expr)

        np.testing.assert_almost_equal(restored.data[0], original.data[0])
        assert restored.unit.dimension == original.unit.dimension

    def test_velocity_roundtrip(self):
        """Test velocity roundtrip."""
        original = DimArray([100], units.m / units.s)

        expr = to_sympy(original)
        restored = from_sympy(expr)

        np.testing.assert_almost_equal(restored.data[0], 100.0)
        assert restored.unit.dimension == Dimension(length=1, time=-1)
