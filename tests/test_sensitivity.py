"""Tests for sensitivity analysis tools."""

import pytest
import numpy as np

from dimtensor import DimArray, units
from dimtensor.analysis import (
    local_sensitivity,
    sensitivity_matrix,
    rank_parameters,
    normalized_sensitivity,
    tornado_diagram_data,
    SensitivityResult,
)


class TestLocalSensitivity:
    """Tests for local_sensitivity function."""

    def test_simple_linear_function(self):
        """Test sensitivity of linear function f(x) = 2*x."""
        def linear(x):
            return 2.0 * x

        x = DimArray(5.0, units.m)
        sens = local_sensitivity(linear, x)

        # df/dx = 2.0, dimensionless sensitivity (m/m)
        assert sens.unit.dimension == units.dimensionless.dimension
        np.testing.assert_allclose(sens._data, 2.0, rtol=1e-4)

    def test_quadratic_function(self):
        """Test sensitivity of quadratic function f(x) = 0.5 * m * x^2."""
        def kinetic_energy(x, m):
            return 0.5 * m * x**2

        x = DimArray(10.0, units.m / units.s)
        m = DimArray(2.0, units.kg)

        sens = local_sensitivity(kinetic_energy, x, args=(m,))

        # df/dx = m * x = 2.0 * 10.0 = 20.0 kg·m/s
        expected_value = 20.0
        expected_unit = units.kg * units.m / units.s  # J·s/m

        assert sens.unit.dimension == expected_unit.dimension
        np.testing.assert_allclose(sens._data, expected_value, rtol=1e-3)

    def test_sensitivity_with_respect_to_mass(self):
        """Test sensitivity of kinetic energy with respect to mass."""
        def kinetic_energy(m, v):
            return 0.5 * m * v**2

        m = DimArray(2.0, units.kg)
        v = DimArray(10.0, units.m / units.s)

        sens = local_sensitivity(kinetic_energy, m, args=(v,))

        # df/dm = 0.5 * v^2 = 0.5 * 100 = 50.0 m^2/s^2 = 50.0 J/kg
        expected_value = 50.0
        expected_unit = (units.m / units.s) ** 2  # J/kg

        assert sens.unit.dimension == expected_unit.dimension
        np.testing.assert_allclose(sens._data, expected_value, rtol=1e-3)

    def test_forward_difference_method(self):
        """Test forward difference method."""
        def square(x):
            return x**2

        x = DimArray(3.0, units.m)
        sens = local_sensitivity(square, x, method="forward")

        # df/dx = 2*x = 6.0 m
        np.testing.assert_allclose(sens._data, 6.0, rtol=1e-3)

    def test_backward_difference_method(self):
        """Test backward difference method."""
        def square(x):
            return x**2

        x = DimArray(3.0, units.m)
        sens = local_sensitivity(square, x, method="backward")

        # df/dx = 2*x = 6.0 m
        np.testing.assert_allclose(sens._data, 6.0, rtol=1e-3)

    def test_central_difference_is_most_accurate(self):
        """Test that central difference is more accurate than forward/backward."""
        def cube(x):
            return x**3

        x = DimArray(2.0, units.m)

        sens_central = local_sensitivity(cube, x, method="central")
        sens_forward = local_sensitivity(cube, x, method="forward")
        sens_backward = local_sensitivity(cube, x, method="backward")

        # Analytical: df/dx = 3*x^2 = 12.0 m^2
        analytical = 12.0

        error_central = abs(sens_central._data - analytical)
        error_forward = abs(sens_forward._data - analytical)
        error_backward = abs(sens_backward._data - analytical)

        # Central should have smallest error
        assert error_central < error_forward
        assert error_central < error_backward

    def test_vector_input(self):
        """Test sensitivity with vector input."""
        def sum_squares(x):
            return (x**2).sum()

        x = DimArray([1.0, 2.0, 3.0], units.m)
        sens = local_sensitivity(sum_squares, x)

        # df/dx = 2*x = [2, 4, 6] m
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(sens._data, expected, rtol=1e-3)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        def linear(x):
            return 2.0 * x

        x = DimArray(5.0, units.m)

        with pytest.raises(ValueError, match="Unknown method"):
            local_sensitivity(linear, x, method="invalid")

    def test_zero_parameter(self):
        """Test sensitivity when parameter is zero."""
        def linear(x):
            return 2.0 * x

        x = DimArray(0.0, units.m)
        sens = local_sensitivity(linear, x)

        # Should still work, using absolute step
        np.testing.assert_allclose(sens._data, 2.0, rtol=1e-2)


class TestSensitivityMatrix:
    """Tests for sensitivity_matrix function."""

    def test_two_parameter_function(self):
        """Test sensitivity matrix for function with two parameters."""
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        params = {
            "mass": DimArray(2.0, units.kg),
            "velocity": DimArray(10.0, units.m / units.s),
        }

        sens_matrix = sensitivity_matrix(kinetic_energy, params)

        # Check that we got sensitivities for both parameters
        assert "mass" in sens_matrix
        assert "velocity" in sens_matrix

        # df/dm = 0.5 * v^2 = 50.0 m^2/s^2
        np.testing.assert_allclose(sens_matrix["mass"]._data, 50.0, rtol=1e-3)

        # df/dv = m * v = 20.0 kg·m/s
        np.testing.assert_allclose(sens_matrix["velocity"]._data, 20.0, rtol=1e-3)

    def test_three_parameter_function(self):
        """Test sensitivity matrix for function with three parameters."""
        def gravitational_force(m1, m2, r):
            G = DimArray(6.674e-11, units.m**3 / (units.kg * units.s**2))
            return G * m1 * m2 / r**2

        params = {
            "m1": DimArray(1.0e24, units.kg),
            "m2": DimArray(1.0e24, units.kg),
            "r": DimArray(1.0e6, units.m),
        }

        sens_matrix = sensitivity_matrix(gravitational_force, params)

        assert len(sens_matrix) == 3
        assert "m1" in sens_matrix
        assert "m2" in sens_matrix
        assert "r" in sens_matrix


class TestRankParameters:
    """Tests for rank_parameters function."""

    def test_kinetic_energy_relative_ranking(self):
        """Test parameter ranking for kinetic energy with relative normalization."""
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        params = {
            "mass": DimArray(2.0, units.kg),
            "velocity": DimArray(10.0, units.m / units.s),
        }

        result = rank_parameters(kinetic_energy, params, normalization="relative")

        # Check result structure
        assert isinstance(result, SensitivityResult)
        assert len(result.ranking) == 2

        # For E = 0.5 * m * v^2:
        # Relative sensitivity for m: (∂E/∂m) * (m/E) = (0.5*v^2) * (m/(0.5*m*v^2)) = 1
        # Relative sensitivity for v: (∂E/∂v) * (v/E) = (m*v) * (v/(0.5*m*v^2)) = 2
        # So velocity should be ranked higher

        assert result.ranking[0][0] == "velocity"
        assert result.ranking[1][0] == "mass"

        # Check importance scores
        assert result.normalized["velocity"] > result.normalized["mass"]
        np.testing.assert_allclose(result.normalized["velocity"], 2.0, rtol=1e-3)
        np.testing.assert_allclose(result.normalized["mass"], 1.0, rtol=1e-3)

    def test_ranking_with_absolute_normalization(self):
        """Test parameter ranking with absolute normalization."""
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        params = {
            "mass": DimArray(2.0, units.kg),
            "velocity": DimArray(10.0, units.m / units.s),
        }

        result = rank_parameters(kinetic_energy, params, normalization="absolute")

        # Absolute sensitivities:
        # |∂E/∂m| = 50.0
        # |∂E/∂v| = 20.0
        # So mass should be ranked higher

        assert result.ranking[0][0] == "mass"
        assert result.ranking[1][0] == "velocity"

    def test_ranking_with_scaled_normalization(self):
        """Test parameter ranking with scaled normalization."""
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        params = {
            "mass": DimArray(2.0, units.kg),
            "velocity": DimArray(10.0, units.m / units.s),
        }

        result = rank_parameters(kinetic_energy, params, normalization="scaled")

        # Scaled sensitivities:
        # |∂E/∂m| * |m| = 50.0 * 2.0 = 100.0 J
        # |∂E/∂v| * |v| = 20.0 * 10.0 = 200.0 J
        # So velocity should be ranked higher

        assert result.ranking[0][0] == "velocity"
        assert result.ranking[1][0] == "mass"

    def test_invalid_normalization_raises_error(self):
        """Test that invalid normalization raises ValueError."""
        def linear(x, y):
            return x + y

        params = {
            "x": DimArray(1.0, units.m),
            "y": DimArray(2.0, units.m),
        }

        with pytest.raises(ValueError, match="Unknown normalization"):
            rank_parameters(linear, params, normalization="invalid")


class TestNormalizedSensitivity:
    """Tests for normalized_sensitivity function."""

    def test_normalized_sensitivity_calculation(self):
        """Test normalized sensitivity calculation."""
        # E = 100 J, v = 10 m/s, dE/dv = 20 kg·m/s
        dE_dv = DimArray(20.0, units.kg * units.m / units.s)
        v = DimArray(10.0, units.m / units.s)
        E = DimArray(100.0, units.J)

        norm_sens = normalized_sensitivity(dE_dv, v, E)

        # (20 * 10) / 100 = 2.0
        np.testing.assert_allclose(norm_sens, 2.0, rtol=1e-6)

    def test_normalized_sensitivity_is_dimensionless(self):
        """Test that normalized sensitivity is dimensionless."""
        dE_dv = DimArray(20.0, units.kg * units.m / units.s)
        v = DimArray(10.0, units.m / units.s)
        E = DimArray(100.0, units.J)

        norm_sens = normalized_sensitivity(dE_dv, v, E)

        # Should return a float, not a DimArray
        assert isinstance(norm_sens, float)

    def test_zero_output_raises_error(self):
        """Test that zero output raises ZeroDivisionError."""
        dE_dv = DimArray(20.0, units.kg * units.m / units.s)
        v = DimArray(10.0, units.m / units.s)
        E = DimArray(0.0, units.J)

        with pytest.raises(ZeroDivisionError):
            normalized_sensitivity(dE_dv, v, E)


class TestTornadoDiagramData:
    """Tests for tornado_diagram_data function."""

    def test_tornado_diagram_basic(self):
        """Test tornado diagram data generation."""
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        params = {
            "mass": DimArray(2.0, units.kg),
            "velocity": DimArray(10.0, units.m / units.s),
        }

        data = tornado_diagram_data(kinetic_energy, params, relative_variation=0.1)

        # Check structure
        assert "mass" in data
        assert "velocity" in data

        for param_name, param_data in data.items():
            assert "low" in param_data
            assert "high" in param_data
            assert "range" in param_data
            assert "baseline" in param_data

        # Baseline should be E = 0.5 * 2 * 100 = 100 J
        np.testing.assert_allclose(data["mass"]["baseline"], 100.0, rtol=1e-6)

        # Velocity variation should have larger impact (quadratic dependence)
        assert data["velocity"]["range"] > data["mass"]["range"]

    def test_tornado_diagram_with_custom_variations(self):
        """Test tornado diagram with custom variation ranges."""
        def linear(x, y):
            return 2.0 * x + 3.0 * y

        params = {
            "x": DimArray(5.0, units.m),
            "y": DimArray(10.0, units.m),
        }

        variations = {
            "x": (DimArray(4.0, units.m), DimArray(6.0, units.m)),
            "y": (DimArray(8.0, units.m), DimArray(12.0, units.m)),
        }

        data = tornado_diagram_data(linear, params, variations=variations)

        # For x: low = 2*4 + 3*10 = 38, high = 2*6 + 3*10 = 42, range = 4
        # For y: low = 2*5 + 3*8 = 34, high = 2*5 + 3*12 = 46, range = 12

        np.testing.assert_allclose(data["x"]["range"], 4.0, rtol=1e-6)
        np.testing.assert_allclose(data["y"]["range"], 12.0, rtol=1e-6)


class TestSensitivityResultDataClass:
    """Tests for SensitivityResult dataclass."""

    def test_sensitivity_result_structure(self):
        """Test that SensitivityResult has expected structure."""
        def linear(x, y):
            return x + y

        params = {
            "x": DimArray(1.0, units.m),
            "y": DimArray(2.0, units.m),
        }

        result = rank_parameters(linear, params, normalization="relative")

        # Check all fields exist
        assert hasattr(result, "sensitivities")
        assert hasattr(result, "normalized")
        assert hasattr(result, "ranking")
        assert hasattr(result, "output")

        # Check types
        assert isinstance(result.sensitivities, dict)
        assert isinstance(result.normalized, dict)
        assert isinstance(result.ranking, list)
        assert isinstance(result.output, DimArray)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dimensionless_parameter(self):
        """Test sensitivity with dimensionless parameter."""
        def exponential(x):
            # exp requires dimensionless input
            return DimArray(np.exp(x._data), units.dimensionless)

        x = DimArray(2.0, units.dimensionless)
        sens = local_sensitivity(exponential, x)

        # df/dx = exp(x) ≈ 7.389
        expected = np.exp(2.0)
        np.testing.assert_allclose(sens._data, expected, rtol=1e-3)

    def test_complex_unit_combinations(self):
        """Test sensitivity with complex unit combinations."""
        def pressure_from_ideal_gas(n, T):
            # PV = nRT, so P = nRT/V
            R = DimArray(8.314, units.J / (units.mol * units.K))
            V = DimArray(1.0, units.m**3)
            return n * R * T / V

        n = DimArray(10.0, units.mol)
        T = DimArray(300.0, units.K)

        sens_n = local_sensitivity(pressure_from_ideal_gas, n, args=(T,))
        sens_T = local_sensitivity(pressure_from_ideal_gas, T, args=(n,))

        # df/dn = RT/V = 8.314 * 300 / 1 = 2494.2 J/(m^3·mol) = Pa/mol
        expected_sens_n = 8.314 * 300.0
        np.testing.assert_allclose(sens_n._data, expected_sens_n, rtol=1e-2)

        # df/dT = nR/V = 10 * 8.314 / 1 = 83.14 J/(m^3·K) = Pa/K
        expected_sens_T = 10.0 * 8.314
        np.testing.assert_allclose(sens_T._data, expected_sens_T, rtol=1e-2)
