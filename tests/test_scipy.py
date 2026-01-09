"""Tests for SciPy integration."""

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, units
from dimtensor.core.units import Unit

pytest.importorskip("scipy")

from dimtensor.scipy import (
    minimize,
    curve_fit,
    least_squares,
    solve_ivp,
    quad,
    interp1d,
)
from dimtensor.scipy.interpolate import DimUnivariateSpline


class TestMinimize:
    """Tests for dimension-aware minimization."""

    def test_basic_minimize(self):
        """Test basic minimization with units."""
        def objective(x):
            # Sum of squared deviations from [5, 5]
            return float(((x.data - 5.0)**2).sum())

        x0 = DimArray([1.0, 2.0], units.m)
        result = minimize(objective, x0, method='BFGS')

        assert result.success
        assert isinstance(result.x, DimArray)
        assert result.x.unit.dimension == Dimension(length=1)
        np.testing.assert_array_almost_equal(result.x.data, [5.0, 5.0], decimal=5)

    def test_minimize_with_bounds(self):
        """Test bounded minimization."""
        def objective(x):
            return float(((x.data - 10.0)**2).sum())

        x0 = DimArray([1.0], units.m)
        lower = DimArray([0.0], units.m)
        upper = DimArray([5.0], units.m)

        result = minimize(objective, x0, method='L-BFGS-B',
                         bounds=(lower, upper))

        assert result.success
        # Should hit upper bound
        np.testing.assert_array_almost_equal(result.x.data, [5.0], decimal=5)

    def test_minimize_requires_dimarray(self):
        """Test that x0 must be DimArray."""
        def objective(x):
            return float(x.sum())

        with pytest.raises(TypeError, match="must be a DimArray"):
            minimize(objective, np.array([1.0, 2.0]))


class TestCurveFit:
    """Tests for dimension-aware curve fitting."""

    def test_linear_fit(self):
        """Test fitting linear model to data."""
        def linear_model(x, m, b):
            return m * x.data + b

        # Generate noisy linear data
        np.random.seed(42)
        t = DimArray(np.linspace(0, 10, 20), units.s)
        y_true = 2.0 * t.data + 3.0
        y = DimArray(y_true + np.random.normal(0, 0.5, len(t.data)), units.m)

        # Need to provide p0 for scipy to determine parameter count
        popt, pcov = curve_fit(linear_model, t, y, p0=[1.0, 1.0])

        # Check fitted parameters (m ≈ 2, b ≈ 3 with some tolerance for noise)
        np.testing.assert_almost_equal(popt[0], 2.0, decimal=1)
        assert 2.0 < popt[1] < 4.5  # b is approximately 3 but has variance

    def test_curve_fit_requires_dimarray(self):
        """Test that inputs must be DimArrays."""
        def model(x, a):
            return a * x

        with pytest.raises(TypeError, match="must be a DimArray"):
            curve_fit(model, np.array([1, 2, 3]), DimArray([1, 2, 3], units.m))


class TestLeastSquares:
    """Tests for dimension-aware least squares."""

    def test_basic_least_squares(self):
        """Test basic least squares optimization."""
        def residuals(x):
            # Target: [5, 5]
            return DimArray(x.data - 5.0, x.unit)

        x0 = DimArray([1.0, 2.0], units.m)
        result = least_squares(residuals, x0)

        assert result.success
        assert isinstance(result.x, DimArray)
        np.testing.assert_array_almost_equal(result.x.data, [5.0, 5.0], decimal=5)


class TestSolveIvp:
    """Tests for dimension-aware ODE solving."""

    def test_simple_ode(self):
        """Test solving simple ODE dy/dt = -y."""
        def decay(t, y):
            # dy/dt = -y (exponential decay)
            return DimArray(-y.data, y.unit / t.unit)

        y0 = DimArray([1.0], units.m)
        t_span = (0.0, 5.0)
        t_eval = np.linspace(0, 5, 10)

        sol = solve_ivp(decay, t_span, y0, t_eval=t_eval)

        assert sol.success
        assert isinstance(sol.t, DimArray)
        assert isinstance(sol.y, DimArray)
        assert sol.y.unit.dimension == Dimension(length=1)

        # Check exponential decay
        expected = np.exp(-sol.t.data)
        np.testing.assert_array_almost_equal(
            sol.y.data[0], expected, decimal=3
        )

    def test_ode_with_dimensional_time(self):
        """Test ODE with dimensional time span."""
        def dynamics(t, y):
            # Return rate with proper dimensions
            # dy/dt has dimension m/s
            rate = -0.5 * y.data
            return DimArray(rate, units.m / units.s)

        y0 = DimArray([10.0], units.m)
        t0 = DimArray(0.0, units.s)
        tf = DimArray(2.0, units.s)

        sol = solve_ivp(dynamics, (t0, tf), y0)

        assert sol.success
        assert sol.t.unit.dimension == Dimension(time=1)

    def test_ode_requires_dimarray(self):
        """Test that y0 must be DimArray."""
        def dynamics(t, y):
            return -y

        with pytest.raises(TypeError, match="must be a DimArray"):
            solve_ivp(dynamics, (0, 1), np.array([1.0]))


class TestQuad:
    """Tests for dimension-aware numerical integration."""

    def test_constant_velocity(self):
        """Test integrating constant velocity to get distance."""
        def velocity(t):
            # Constant 10 m/s
            return DimArray(10.0, units.m / units.s)

        # Use dimensional limits
        t0 = DimArray(0.0, units.s)
        t1 = DimArray(5.0, units.s)
        distance, err = quad(velocity, t0, t1)

        # d = v * t = 10 m/s * 5 s = 50 m
        assert isinstance(distance, DimArray)
        # Result has dimension L (m/s * s = m)
        assert distance.unit.dimension == Dimension(length=1)
        np.testing.assert_almost_equal(distance.data, 50.0, decimal=3)

    def test_dimensional_limits(self):
        """Test integration with dimensional limits."""
        def velocity(t):
            return DimArray(2.0 * t.data, units.m / units.s)

        t0 = DimArray(0.0, units.s)
        t1 = DimArray(3.0, units.s)

        distance, err = quad(velocity, t0, t1)

        # d = ∫2t dt from 0 to 3 = t² |₀³ = 9 m
        assert distance.unit.dimension == Dimension(length=1)
        np.testing.assert_almost_equal(distance.data, 9.0, decimal=3)


class TestInterp1d:
    """Tests for dimension-aware 1D interpolation."""

    def test_linear_interpolation(self):
        """Test linear interpolation preserves units."""
        t = DimArray([0, 1, 2, 3], units.s)
        y = DimArray([0, 2, 4, 6], units.m)

        f = interp1d(t, y)

        # Interpolate at t=1.5
        t_new = DimArray([1.5], units.s)
        y_new = f(t_new)

        assert isinstance(y_new, DimArray)
        assert y_new.unit.dimension == Dimension(length=1)
        np.testing.assert_almost_equal(y_new.data, [3.0])

    def test_cubic_interpolation(self):
        """Test cubic interpolation."""
        t = DimArray(np.linspace(0, 2*np.pi, 10), units.s)
        y = DimArray(np.sin(t.data), units.m)

        f = interp1d(t, y, kind='cubic')

        t_new = DimArray([np.pi/2], units.s)
        y_new = f(t_new)

        np.testing.assert_almost_equal(y_new.data, [1.0], decimal=2)

    def test_dimension_mismatch_error(self):
        """Test error on dimension mismatch."""
        t = DimArray([0, 1, 2], units.s)
        y = DimArray([0, 1, 4], units.m)

        f = interp1d(t, y)

        # Try to interpolate with wrong units
        t_wrong = DimArray([0.5], units.kg)

        with pytest.raises(ValueError, match="doesn't match"):
            f(t_wrong)


class TestUnivariateSpline:
    """Tests for dimension-aware spline."""

    def test_spline_interpolation(self):
        """Test spline interpolation."""
        x = DimArray(np.linspace(0, 10, 20), units.s)
        y = DimArray(np.sin(x.data), units.m)

        spline = DimUnivariateSpline(x, y, s=0)

        x_new = DimArray([np.pi/2], units.s)
        y_new = spline(x_new)

        assert isinstance(y_new, DimArray)
        assert y_new.unit.dimension == Dimension(length=1)
        np.testing.assert_almost_equal(y_new.data, [1.0], decimal=2)

    def test_spline_derivative(self):
        """Test spline derivative with correct dimensions."""
        # y = t² => dy/dt = 2t
        t = DimArray(np.linspace(0, 5, 20), units.s)
        y = DimArray(t.data**2, units.m)

        spline = DimUnivariateSpline(t, y, k=3, s=0)
        deriv = spline.derivative(n=1)

        t_test = DimArray([2.0], units.s)
        dy_dt = deriv(t_test)

        # dy/dt should have dimension m/s
        assert dy_dt.unit.dimension == Dimension(length=1, time=-1)
        # At t=2, dy/dt = 2*2 = 4
        np.testing.assert_almost_equal(dy_dt.data, [4.0], decimal=1)

    def test_spline_integral(self):
        """Test spline integral with correct dimensions."""
        # y = 1 m (constant) => ∫y dt = t * y
        t = DimArray(np.linspace(0, 5, 20), units.s)
        y = DimArray(np.ones_like(t.data), units.m)

        spline = DimUnivariateSpline(t, y, s=0)

        a = DimArray(0.0, units.s)
        b = DimArray(3.0, units.s)
        integral = spline.integral(a, b)

        # ∫1 m dt from 0 to 3 s = 3 m·s
        # The integral value should be 3.0 (1 * 3)
        np.testing.assert_almost_equal(integral.data, 3.0, decimal=2)
        # Dimension should be length*time
        assert integral.unit.dimension == Dimension(length=1, time=1)
