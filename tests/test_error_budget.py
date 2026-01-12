"""Tests for error budget calculator."""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.uncertainty import ErrorBudget, compute_error_budget


class TestErrorBudgetBasic:
    """Basic tests for ErrorBudget class."""

    def test_error_budget_creation(self):
        """Test ErrorBudget can be created."""
        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.8, 'y': 0.6}
        sensitivities = {'x': 2.0, 'y': 1.5}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        assert budget.result._data == 10.0
        assert budget.total_uncertainty == 1.0
        assert 'x' in budget.contributions
        assert 'y' in budget.contributions

    def test_percent_contributions_sum_to_100(self):
        """Test that percentage contributions sum to ~100%."""
        result = DimArray(10.0, units.m)
        # u_c = √(0.6² + 0.8²) = 1.0
        contribs = {'x': 0.6, 'y': 0.8}
        sensitivities = {'x': 1.0, 'y': 1.0}
        total_unc = 1.0

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        total_percent = sum(budget.percent_contributions.values())
        assert np.isclose(total_percent, 100.0, rtol=1e-10)

    def test_to_dict(self):
        """Test serialization to dict."""
        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.8, 'y': 0.6}
        sensitivities = {'x': 2.0, 'y': 1.5}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)
        data = budget.to_dict()

        assert 'result' in data
        assert 'total_uncertainty' in data
        assert 'contributions' in data
        assert 'sensitivities' in data
        assert 'percent_contributions' in data

    def test_repr(self):
        """Test string representation."""
        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.8, 'y': 0.6}
        sensitivities = {'x': 2.0, 'y': 1.5}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)
        repr_str = repr(budget)

        assert 'ErrorBudget' in repr_str
        assert 'n_inputs=2' in repr_str


class TestComputeErrorBudget:
    """Tests for compute_error_budget function."""

    def test_linear_function(self):
        """Test error budget for linear function: f(x) = a*x + b."""
        # f(x) = 2*x + 3*m
        # ∂f/∂x = 2
        def func(inputs):
            return 2 * inputs['x'] + DimArray(3.0, units.m)

        inputs = {
            'x': DimArray(5.0, units.m, uncertainty=0.1),
        }

        budget = compute_error_budget(func, inputs)

        # Expected: sensitivity = 2, contribution = 2 * 0.1 = 0.2
        assert np.isclose(budget.sensitivities['x']._data, 2.0, rtol=1e-2)
        assert np.isclose(budget.contributions['x'], 0.2, rtol=1e-2)
        assert np.isclose(budget.total_uncertainty, 0.2, rtol=1e-2)

    def test_power_function(self):
        """Test error budget for power function: f(x) = x^2."""
        # f(x) = x²
        # ∂f/∂x = 2x
        def func(inputs):
            return inputs['x'] ** 2

        x_val = 3.0
        x_unc = 0.1
        inputs = {
            'x': DimArray(x_val, units.m, uncertainty=x_unc),
        }

        budget = compute_error_budget(func, inputs)

        # Expected: sensitivity ≈ 2*3 = 6
        expected_sensitivity = 2 * x_val
        assert np.isclose(
            budget.sensitivities['x']._data,
            expected_sensitivity,
            rtol=1e-2
        )

        # Expected contribution: |6| * 0.1 = 0.6
        expected_contribution = abs(expected_sensitivity) * x_unc
        assert np.isclose(
            budget.contributions['x'],
            expected_contribution,
            rtol=1e-2
        )

    def test_product_function(self):
        """Test error budget for product: f(x,y) = x*y."""
        # f(x,y) = x*y
        # ∂f/∂x = y, ∂f/∂y = x
        def func(inputs):
            return inputs['x'] * inputs['y']

        x_val, y_val = 3.0, 4.0
        x_unc, y_unc = 0.1, 0.2
        inputs = {
            'x': DimArray(x_val, units.m, uncertainty=x_unc),
            'y': DimArray(y_val, units.s, uncertainty=y_unc),
        }

        budget = compute_error_budget(func, inputs)

        # Expected sensitivities: ∂f/∂x = y = 4, ∂f/∂y = x = 3
        assert np.isclose(budget.sensitivities['x']._data, y_val, rtol=1e-2)
        assert np.isclose(budget.sensitivities['y']._data, x_val, rtol=1e-2)

        # Expected contributions
        contrib_x = y_val * x_unc  # 4 * 0.1 = 0.4
        contrib_y = x_val * y_unc  # 3 * 0.2 = 0.6
        assert np.isclose(budget.contributions['x'], contrib_x, rtol=1e-2)
        assert np.isclose(budget.contributions['y'], contrib_y, rtol=1e-2)

        # Total uncertainty: √(0.4² + 0.6²) = √(0.16 + 0.36) = √0.52
        expected_total = np.sqrt(contrib_x**2 + contrib_y**2)
        assert np.isclose(budget.total_uncertainty, expected_total, rtol=1e-2)

    def test_pendulum_period(self):
        """Test error budget for pendulum period: T = 2π√(L/g)."""
        def pendulum_period(inputs):
            L = inputs['length']
            g = inputs['gravity']
            return 2 * np.pi * (L / g) ** 0.5

        L_val, g_val = 1.0, 9.8
        L_unc, g_unc = 0.01, 0.1
        inputs = {
            'length': DimArray(L_val, units.m, uncertainty=L_unc),
            'gravity': DimArray(g_val, units.m/units.s**2, uncertainty=g_unc),
        }

        budget = compute_error_budget(func=pendulum_period, inputs=inputs)

        # T = 2π√(L/g)
        # ∂T/∂L = π/√(gL)
        # ∂T/∂g = -π√(L)/g^(3/2) = -πL/(g^(3/2))
        expected_dT_dL = np.pi / np.sqrt(g_val * L_val)
        expected_dT_dg = -np.pi * np.sqrt(L_val) / g_val**(3/2)

        assert np.isclose(
            budget.sensitivities['length']._data,
            expected_dT_dL,
            rtol=1e-2
        )
        assert np.isclose(
            budget.sensitivities['gravity']._data,
            expected_dT_dg,
            rtol=1e-2
        )

        # Contributions
        contrib_L = abs(expected_dT_dL) * L_unc
        contrib_g = abs(expected_dT_dg) * g_unc
        assert np.isclose(budget.contributions['length'], contrib_L, rtol=1e-2)
        assert np.isclose(budget.contributions['gravity'], contrib_g, rtol=1e-2)

    def test_three_variable_function(self):
        """Test error budget with three inputs: f(x,y,z) = x*y*z."""
        def func(inputs):
            return inputs['x'] * inputs['y'] * inputs['z']

        inputs = {
            'x': DimArray(2.0, units.m, uncertainty=0.1),
            'y': DimArray(3.0, units.s, uncertainty=0.2),
            'z': DimArray(4.0, units.kg, uncertainty=0.3),
        }

        budget = compute_error_budget(func, inputs)

        # ∂f/∂x = y*z = 12, ∂f/∂y = x*z = 8, ∂f/∂z = x*y = 6
        assert np.isclose(budget.sensitivities['x']._data, 12.0, rtol=1e-2)
        assert np.isclose(budget.sensitivities['y']._data, 8.0, rtol=1e-2)
        assert np.isclose(budget.sensitivities['z']._data, 6.0, rtol=1e-2)

        # Verify percentage contributions sum to 100%
        total_percent = sum(budget.percent_contributions.values())
        assert np.isclose(total_percent, 100.0, rtol=1e-6)


class TestErrorBudgetEdgeCases:
    """Test edge cases and error handling."""

    def test_no_uncertainty_raises_error(self):
        """Test that inputs without uncertainty raise ValueError."""
        def func(inputs):
            return inputs['x'] + inputs['y']

        inputs = {
            'x': DimArray(5.0, units.m),  # No uncertainty!
            'y': DimArray(3.0, units.m, uncertainty=0.1),
        }

        with pytest.raises(ValueError, match="has no uncertainty"):
            compute_error_budget(func, inputs)

    def test_non_dimarray_input_raises_error(self):
        """Test that non-DimArray inputs raise ValueError."""
        def func(inputs):
            return inputs['x'] + inputs['y']

        inputs = {
            'x': 5.0,  # Not a DimArray!
            'y': DimArray(3.0, units.m, uncertainty=0.1),
        }

        with pytest.raises(ValueError, match="not a DimArray"):
            compute_error_budget(func, inputs)

    def test_invalid_step_factor(self):
        """Test that negative step_factor raises ValueError."""
        def func(inputs):
            return inputs['x']

        inputs = {
            'x': DimArray(5.0, units.m, uncertainty=0.1),
        }

        with pytest.raises(ValueError, match="step_factor must be positive"):
            compute_error_budget(func, inputs, step_factor=-0.01)

    def test_zero_total_uncertainty_handled(self):
        """Test that zero total uncertainty is handled gracefully."""
        # This could happen if all inputs have zero uncertainty (edge case)
        result = DimArray(10.0, units.m)
        total_unc = 0.0
        contribs = {'x': 0.0}
        sensitivities = {'x': 2.0}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        # Should not crash, percent should be 0
        assert budget.percent_contributions['x'] == 0.0

    def test_one_dominant_contribution(self):
        """Test when one input dominates the error budget."""
        def func(inputs):
            return inputs['x'] + inputs['y']

        inputs = {
            'x': DimArray(10.0, units.m, uncertainty=1.0),  # Large uncertainty
            'y': DimArray(5.0, units.m, uncertainty=0.01),  # Small uncertainty
        }

        budget = compute_error_budget(func, inputs)

        # x should contribute ~99.99% of uncertainty
        assert budget.percent_contributions['x'] > 99.0
        assert budget.percent_contributions['y'] < 1.0


class TestErrorBudgetVisualization:
    """Test visualization methods (smoke tests only)."""

    def test_plot_pie_no_error(self):
        """Test that plot_pie runs without error."""
        pytest.importorskip('matplotlib')

        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.6, 'y': 0.8}
        sensitivities = {'x': 1.0, 'y': 1.0}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        # Should not raise
        ax = budget.plot_pie()
        assert ax is not None

    def test_plot_bar_no_error(self):
        """Test that plot_bar runs without error."""
        pytest.importorskip('matplotlib')

        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.6, 'y': 0.8}
        sensitivities = {'x': 1.0, 'y': 1.0}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        # Should not raise
        ax = budget.plot_bar()
        assert ax is not None

    def test_plot_pareto_no_error(self):
        """Test that plot_pareto runs without error."""
        pytest.importorskip('matplotlib')

        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'a': 0.5, 'b': 0.3, 'c': 0.2, 'd': 0.1}
        sensitivities = {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        # Should not raise
        ax = budget.plot_pareto()
        assert ax is not None

    def test_to_dataframe_no_error(self):
        """Test that to_dataframe runs without error."""
        pd = pytest.importorskip('pandas')

        result = DimArray(10.0, units.m)
        total_unc = 1.0
        contribs = {'x': 0.6, 'y': 0.8}
        sensitivities = {'x': 1.0, 'y': 1.0}

        budget = ErrorBudget(result, total_unc, contribs, sensitivities)

        df = budget.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 inputs + 1 total row
        assert 'input' in df.columns
        assert 'sensitivity' in df.columns
        assert 'contribution' in df.columns
        assert 'percent' in df.columns


class TestErrorBudgetRealWorld:
    """Real-world physics examples."""

    def test_ohms_law_power(self):
        """Test error budget for Ohm's law: P = V²/R."""
        def power(inputs):
            V = inputs['voltage']
            R = inputs['resistance']
            return V**2 / R

        inputs = {
            'voltage': DimArray(12.0, units.V, uncertainty=0.5),
            'resistance': DimArray(10.0, units.ohm, uncertainty=0.2),
        }

        budget = compute_error_budget(power, inputs)

        # P = V²/R
        # ∂P/∂V = 2V/R, ∂P/∂R = -V²/R²
        V, R = 12.0, 10.0
        expected_dP_dV = 2 * V / R  # 2.4
        expected_dP_dR = -V**2 / R**2  # -14.4

        assert np.isclose(
            budget.sensitivities['voltage']._data,
            expected_dP_dV,
            rtol=1e-2
        )
        assert np.isclose(
            budget.sensitivities['resistance']._data,
            expected_dP_dR,
            rtol=1e-2
        )

        # Voltage should dominate (appears squared in formula)
        contrib_V_pct = budget.percent_contributions['voltage']
        contrib_R_pct = budget.percent_contributions['resistance']
        assert contrib_V_pct > contrib_R_pct

    def test_kinetic_energy(self):
        """Test error budget for kinetic energy: KE = 0.5*m*v²."""
        def kinetic_energy(inputs):
            m = inputs['mass']
            v = inputs['velocity']
            return 0.5 * m * v**2

        inputs = {
            'mass': DimArray(2.0, units.kg, uncertainty=0.01),
            'velocity': DimArray(10.0, units.m/units.s, uncertainty=0.5),
        }

        budget = compute_error_budget(kinetic_energy, inputs)

        # KE = 0.5*m*v²
        # ∂KE/∂m = 0.5*v², ∂KE/∂v = m*v
        m, v = 2.0, 10.0
        expected_dKE_dm = 0.5 * v**2  # 50
        expected_dKE_dv = m * v  # 20

        assert np.isclose(
            budget.sensitivities['mass']._data,
            expected_dKE_dm,
            rtol=1e-2
        )
        assert np.isclose(
            budget.sensitivities['velocity']._data,
            expected_dKE_dv,
            rtol=1e-2
        )

        # Percentage contributions should sum to 100%
        total_percent = sum(budget.percent_contributions.values())
        assert np.isclose(total_percent, 100.0, rtol=1e-6)
