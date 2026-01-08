"""Tests for physical constants module.

Tests cover:
- Constant class behavior (creation, properties, string representations)
- CODATA 2022 values for universal, electromagnetic, atomic, and physico-chemical constants
- Arithmetic operations between constants and with DimArrays
- Integration with the DimArray calculation system
"""

import math

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.constants import (
    Constant,
    # Universal
    c,
    G,
    h,
    hbar,
    speed_of_light,
    gravitational_constant,
    planck_constant,
    # Electromagnetic
    e,
    mu_0,
    epsilon_0,
    alpha,
    elementary_charge,
    # Atomic
    m_e,
    m_p,
    m_n,
    a_0,
    R_inf,
    electron_mass,
    # Physico-chemical
    N_A,
    k_B,
    R,
    F,
    sigma,
    avogadro_constant,
    # Derived
    l_P,
    m_P,
    t_P,
    T_P,
    E_h,
)


class TestConstantClass:
    """Tests for the Constant class."""

    def test_constant_creation(self):
        """Create a constant with all attributes."""
        const = Constant(
            symbol="test",
            name="test constant",
            value=1.234,
            unit=units.m,
            uncertainty=0.001,
        )
        assert const.symbol == "test"
        assert const.name == "test constant"
        assert const.value == 1.234
        assert const.unit == units.m
        assert const.uncertainty == 0.001

    def test_exact_constant(self):
        """Exact constants have is_exact=True and zero uncertainty."""
        assert c.is_exact is True
        assert c.uncertainty == 0.0

    def test_inexact_constant(self):
        """Inexact constants have is_exact=False and non-zero uncertainty."""
        assert G.is_exact is False
        assert G.uncertainty > 0

    def test_relative_uncertainty(self):
        """Relative uncertainty computed correctly."""
        # G has known relative uncertainty ~2.2e-5
        rel_unc = G.relative_uncertainty
        assert rel_unc > 0
        assert abs(rel_unc - G.uncertainty / G.value) < 1e-15

    def test_relative_uncertainty_exact(self):
        """Exact constants have zero relative uncertainty."""
        assert c.relative_uncertainty == 0.0

    def test_dimension_property(self):
        """Dimension property returns unit's dimension."""
        assert c.dimension == c.unit.dimension
        assert c.dimension.length == 1
        assert c.dimension.time == -1

    def test_to_dimarray(self):
        """Convert constant to DimArray."""
        arr = c.to_dimarray()
        assert isinstance(arr, DimArray)
        assert arr.data[0] == c.value
        assert arr.unit == c.unit


class TestConstantStrings:
    """Tests for string representations."""

    def test_str_exact(self):
        """String representation for exact constants shows (exact)."""
        s = str(c)
        assert "299792458" in s
        assert "(exact)" in s

    def test_str_inexact(self):
        """String representation for inexact constants shows uncertainty."""
        s = str(G)
        assert "+/-" in s or "±" in s or "e-" in s

    def test_repr(self):
        """Repr shows Constant constructor form."""
        r = repr(c)
        assert "Constant" in r
        assert "c" in r

    def test_format_spec(self):
        """Format specification applies to value."""
        formatted = f"{c:.2e}"
        assert "3.00e+08" in formatted
        assert "m/s" in formatted


class TestUniversalConstants:
    """Tests for universal constants values (CODATA 2022)."""

    def test_speed_of_light_value(self):
        """Speed of light has correct value."""
        assert c.value == 299792458.0
        assert speed_of_light.value == c.value

    def test_speed_of_light_exact(self):
        """Speed of light is exact by definition."""
        assert c.is_exact

    def test_speed_of_light_unit(self):
        """Speed of light has velocity dimension."""
        assert c.dimension.length == 1
        assert c.dimension.time == -1

    def test_gravitational_constant_value(self):
        """Gravitational constant has approximately correct value."""
        assert abs(G.value - 6.67430e-11) < 1e-15
        assert gravitational_constant.value == G.value

    def test_gravitational_constant_uncertainty(self):
        """G has non-zero uncertainty."""
        assert G.uncertainty > 0
        assert not G.is_exact

    def test_planck_constant_value(self):
        """Planck constant has correct value."""
        assert h.value == 6.62607015e-34
        assert planck_constant.value == h.value

    def test_planck_constant_exact(self):
        """Planck constant is exact by definition."""
        assert h.is_exact

    def test_hbar_derived_from_h(self):
        """hbar = h / (2*pi)."""
        expected = h.value / (2 * math.pi)
        assert abs(hbar.value - expected) < 1e-50


class TestElectromagneticConstants:
    """Tests for electromagnetic constants."""

    def test_elementary_charge_value(self):
        """Elementary charge has correct value."""
        assert e.value == 1.602176634e-19
        assert elementary_charge.value == e.value

    def test_elementary_charge_exact(self):
        """Elementary charge is exact by definition."""
        assert e.is_exact

    def test_fine_structure_dimensionless(self):
        """Fine structure constant is dimensionless."""
        assert alpha.dimension.is_dimensionless

    def test_fine_structure_value(self):
        """Fine structure constant ~1/137."""
        assert abs(alpha.value - 1 / 137) < 1e-4


class TestAtomicConstants:
    """Tests for atomic constants."""

    def test_electron_mass_value(self):
        """Electron mass has approximately correct value."""
        assert abs(m_e.value - 9.1093837139e-31) < 1e-40
        assert electron_mass.value == m_e.value

    def test_proton_electron_mass_ratio(self):
        """Proton is ~1836 times heavier than electron."""
        ratio = m_p.value / m_e.value
        assert abs(ratio - 1836) < 1

    def test_bohr_radius_value(self):
        """Bohr radius has approximately correct value."""
        assert abs(a_0.value - 5.29e-11) < 1e-13


class TestPhysicoChemicalConstants:
    """Tests for physico-chemical constants."""

    def test_avogadro_value(self):
        """Avogadro constant has correct value."""
        assert N_A.value == 6.02214076e23
        assert avogadro_constant.value == N_A.value

    def test_avogadro_exact(self):
        """Avogadro constant is exact by definition."""
        assert N_A.is_exact

    def test_boltzmann_value(self):
        """Boltzmann constant has correct value."""
        assert k_B.value == 1.380649e-23

    def test_boltzmann_exact(self):
        """Boltzmann constant is exact by definition."""
        assert k_B.is_exact

    def test_gas_constant_derived(self):
        """R = N_A * k_B."""
        expected = N_A.value * k_B.value
        assert abs(R.value - expected) < 1e-10

    def test_faraday_derived(self):
        """F = N_A * e."""
        expected = N_A.value * e.value
        assert abs(F.value - expected) < 1e-20


class TestDerivedConstants:
    """Tests for derived Planck units."""

    def test_planck_length_value(self):
        """Planck length has approximately correct value."""
        assert abs(l_P.value - 1.616e-35) < 1e-37

    def test_planck_mass_value(self):
        """Planck mass has approximately correct value."""
        assert abs(m_P.value - 2.176e-8) < 1e-10

    def test_planck_time_value(self):
        """Planck time has approximately correct value."""
        assert abs(t_P.value - 5.39e-44) < 1e-46

    def test_planck_temperature_value(self):
        """Planck temperature has approximately correct value."""
        assert abs(T_P.value - 1.42e32) < 1e30

    def test_hartree_energy_value(self):
        """Hartree energy has approximately correct value."""
        assert abs(E_h.value - 4.36e-18) < 1e-19


class TestConstantArithmetic:
    """Tests for arithmetic operations with constants."""

    def test_constant_times_constant(self):
        """Multiply two constants."""
        result = c * h
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], c.value * h.value)

    def test_constant_times_scalar(self):
        """Multiply constant by scalar."""
        result = c * 2
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], 2 * c.value)

    def test_scalar_times_constant(self):
        """Multiply scalar by constant (rmul)."""
        result = 2 * c
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], 2 * c.value)

    def test_constant_divide_constant(self):
        """Divide two constants."""
        result = c / h
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], c.value / h.value)

    def test_constant_divide_scalar(self):
        """Divide constant by scalar."""
        result = c / 2
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], c.value / 2)

    def test_scalar_divide_constant(self):
        """Divide scalar by constant (rtruediv)."""
        result = 1.0 / c
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], 1.0 / c.value)

    def test_constant_squared(self):
        """Square a constant."""
        result = c**2
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], c.value**2)
        # Check dimension: velocity squared
        assert result.dimension.length == 2
        assert result.dimension.time == -2

    def test_constant_negative(self):
        """Negate a constant."""
        result = -c
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data[0], -c.value)


class TestConstantDimArrayArithmetic:
    """Tests for arithmetic between constants and DimArrays."""

    def test_constant_times_dimarray(self):
        """Constant * DimArray."""
        mass = DimArray([1.0, 2.0], units.kg)
        result = c**2 * mass  # E = mc^2 style
        assert isinstance(result, DimArray)
        assert result.dimension == units.J.dimension
        np.testing.assert_allclose(result.data, [c.value**2, 2 * c.value**2])

    def test_dimarray_times_constant(self):
        """DimArray * Constant."""
        mass = DimArray([1.0, 2.0], units.kg)
        result = mass * c**2
        assert isinstance(result, DimArray)
        np.testing.assert_allclose(result.data, [c.value**2, 2 * c.value**2])

    def test_dimarray_divide_constant(self):
        """DimArray / Constant."""
        energy = DimArray([8.98755e16], units.J)
        result = energy / c**2
        assert isinstance(result, DimArray)
        # Should be approximately 1 kg
        np.testing.assert_allclose(result.data[0], 1.0, rtol=1e-5)

    def test_constant_divide_dimarray(self):
        """Constant / DimArray."""
        time = DimArray([1.0], units.s)
        result = c / time
        assert isinstance(result, DimArray)
        # Acceleration dimension
        assert result.dimension.length == 1
        assert result.dimension.time == -2


class TestPhysicsCalculations:
    """Integration tests with real physics calculations."""

    def test_emc2(self):
        """E = mc^2 calculation."""
        mass = DimArray([1.0], units.kg)
        energy = mass * c**2
        expected_joules = 299792458.0**2
        np.testing.assert_allclose(energy.data[0], expected_joules)
        assert energy.dimension == units.J.dimension

    def test_gravitational_force(self):
        """F = G * m1 * m2 / r^2 calculation."""
        m1 = DimArray([5.972e24], units.kg)  # Earth mass
        m2 = DimArray([7.342e22], units.kg)  # Moon mass
        r = DimArray([3.844e8], units.m)  # Earth-Moon distance

        F = G * m1 * m2 / r**2
        assert F.dimension == units.N.dimension
        # Force should be around 1.98e20 N
        np.testing.assert_allclose(F.data[0], 1.98e20, rtol=0.01)

    def test_photon_energy(self):
        """E = h * f for a photon."""
        frequency = DimArray([5e14], units.Hz)  # Green light
        energy = h * frequency
        assert energy.dimension == units.J.dimension
        # Around 3.3e-19 J
        np.testing.assert_allclose(energy.data[0], 3.3e-19, rtol=0.01)

    def test_thermal_energy(self):
        """E = k_B * T."""
        temperature = DimArray([300.0], units.K)  # Room temperature
        energy = k_B * temperature
        assert energy.dimension == units.J.dimension
        # Around 4.14e-21 J
        np.testing.assert_allclose(energy.data[0], 4.14e-21, rtol=0.01)

    def test_ideal_gas_law(self):
        """PV = nRT."""
        n = DimArray([1.0], units.mol)
        T = DimArray([273.15], units.K)  # 0 Celsius

        # R * n * T should give energy (J)
        result = R * n * T
        # At STP, 1 mol at 273.15 K: R * T ~ 2271 J/mol
        expected = 8.314462618 * 273.15
        np.testing.assert_allclose(result.data[0], expected, rtol=1e-6)


class TestConstantImports:
    """Tests that all constants are importable from the right places."""

    def test_import_from_package_level(self):
        """Constants importable from dimtensor.constants."""
        from dimtensor.constants import c, G, h, k_B, N_A, e, m_e

        assert c.symbol == "c"
        assert G.symbol == "G"

    def test_import_from_submodule(self):
        """Constants importable from domain submodules."""
        from dimtensor.constants.universal import c, G
        from dimtensor.constants.electromagnetic import e, alpha
        from dimtensor.constants.atomic import m_e, m_p
        from dimtensor.constants.physico_chemical import N_A, k_B
        from dimtensor.constants.derived import l_P, m_P

        assert c.symbol == "c"
        assert e.symbol == "e"
        assert m_e.symbol == "m_e"
        assert N_A.symbol == "N_A"
        assert l_P.symbol == "l_P"

    def test_import_constant_class(self):
        """Constant class is importable."""
        from dimtensor.constants import Constant

        assert Constant is not None

    def test_constants_module_in_dimtensor(self):
        """Constants module accessible from main dimtensor."""
        import dimtensor

        assert hasattr(dimtensor, "constants")
        assert dimtensor.constants.c.value == 299792458.0
