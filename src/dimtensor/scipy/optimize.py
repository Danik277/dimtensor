"""Dimension-aware SciPy optimization wrappers.

Provides wrappers that preserve physical units through optimization.
"""
# mypy: disable-error-code="type-arg,assignment"

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit

try:
    from scipy import optimize as sp_optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _check_scipy() -> None:
    """Check that scipy is available."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for optimization functions. "
            "Install with: pip install scipy"
        )


def minimize(
    fun: Callable[[DimArray], float],
    x0: DimArray,
    method: str | None = None,
    bounds: tuple[DimArray, DimArray] | None = None,
    constraints: Any = None,
    **kwargs: Any,
) -> "MinimizeResult":
    """Dimension-aware minimization.

    Wraps scipy.optimize.minimize to preserve units through optimization.

    Args:
        fun: Objective function taking DimArray, returning scalar.
        x0: Initial guess with physical units.
        method: Optimization method (e.g., 'BFGS', 'L-BFGS-B').
        bounds: (lower, upper) DimArrays with same units as x0.
        constraints: Optimization constraints (passed to scipy).
        **kwargs: Additional arguments passed to scipy.optimize.minimize.

    Returns:
        MinimizeResult with optimal DimArray x.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.scipy import minimize
        >>>
        >>> def objective(x):
        ...     # x is DimArray with meters
        ...     return float((x.data - 5.0).sum()**2)
        >>>
        >>> x0 = DimArray([1.0, 2.0], units.m)
        >>> result = minimize(objective, x0)
        >>> print(result.x)  # DimArray in meters
    """
    _check_scipy()

    if not isinstance(x0, DimArray):
        raise TypeError("x0 must be a DimArray with units")

    x0_unit = x0.unit
    x0_data = np.asarray(x0.data)

    # Wrapper that adds units before calling user function
    def wrapped_fun(x_raw: np.ndarray) -> float:
        x_dim = DimArray._from_data_and_unit(x_raw, x0_unit)
        return float(fun(x_dim))

    # Convert bounds if provided
    scipy_bounds = None
    if bounds is not None:
        lower, upper = bounds
        if isinstance(lower, DimArray):
            if lower.unit.dimension != x0_unit.dimension:
                raise ValueError(
                    f"Lower bound dimension {lower.unit.dimension} "
                    f"doesn't match x0 dimension {x0_unit.dimension}"
                )
            lower_data = lower.to(x0_unit).data
        else:
            lower_data = lower

        if isinstance(upper, DimArray):
            if upper.unit.dimension != x0_unit.dimension:
                raise ValueError(
                    f"Upper bound dimension {upper.unit.dimension} "
                    f"doesn't match x0 dimension {x0_unit.dimension}"
                )
            upper_data = upper.to(x0_unit).data
        else:
            upper_data = upper

        scipy_bounds = list(zip(lower_data, upper_data))

    # Run optimization
    result = sp_optimize.minimize(
        wrapped_fun,
        x0_data,
        method=method,
        bounds=scipy_bounds,
        constraints=constraints,
        **kwargs,
    )

    # Convert result back to DimArray
    return MinimizeResult(
        x=DimArray._from_data_and_unit(result.x, x0_unit),
        fun=result.fun,
        success=result.success,
        message=result.message,
        nfev=result.nfev,
        nit=result.nit if hasattr(result, 'nit') else 0,
        _scipy_result=result,
    )


class MinimizeResult:
    """Result of dimension-aware minimization.

    Attributes:
        x: Optimal DimArray with units.
        fun: Objective value at optimum.
        success: Whether optimization succeeded.
        message: Status message.
        nfev: Number of function evaluations.
        nit: Number of iterations.
    """

    def __init__(
        self,
        x: DimArray,
        fun: float,
        success: bool,
        message: str,
        nfev: int,
        nit: int,
        _scipy_result: Any = None,
    ):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nit = nit
        self._scipy_result = _scipy_result

    def __repr__(self) -> str:
        return (
            f"MinimizeResult(x={self.x}, fun={self.fun:.6g}, "
            f"success={self.success})"
        )


def curve_fit(
    f: Callable[..., DimArray],
    xdata: DimArray,
    ydata: DimArray,
    p0: list[float] | None = None,
    sigma: DimArray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Dimension-aware curve fitting.

    Wraps scipy.optimize.curve_fit to work with dimensional data.

    Args:
        f: Model function f(x, *params) -> y. Must handle DimArray input.
        xdata: Independent variable data with units.
        ydata: Dependent variable data with units.
        p0: Initial parameter guesses (dimensionless).
        sigma: Standard deviation of ydata (same units as ydata).
        **kwargs: Additional arguments to scipy.optimize.curve_fit.

    Returns:
        (popt, pcov): Optimal parameters and covariance matrix.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.scipy import curve_fit
        >>>
        >>> def linear(x, m, b):
        ...     return m * x.data + b  # returns raw values
        >>>
        >>> x = DimArray([1, 2, 3, 4], units.s)
        >>> y = DimArray([2.1, 3.9, 6.2, 7.8], units.m)
        >>> popt, pcov = curve_fit(linear, x, y)
    """
    _check_scipy()

    if not isinstance(xdata, DimArray):
        raise TypeError("xdata must be a DimArray")
    if not isinstance(ydata, DimArray):
        raise TypeError("ydata must be a DimArray")

    x_raw = np.asarray(xdata.data)
    y_raw = np.asarray(ydata.data)

    # Wrapper that passes DimArray to user function
    def wrapped_f(x_raw_inner: np.ndarray, *params: float) -> np.ndarray:
        x_dim = DimArray._from_data_and_unit(x_raw_inner, xdata.unit)
        result = f(x_dim, *params)
        if isinstance(result, DimArray):
            return np.asarray(result.data)
        return np.asarray(result)

    sigma_raw = None
    if sigma is not None:
        if isinstance(sigma, DimArray):
            if sigma.unit.dimension != ydata.unit.dimension:
                raise ValueError(
                    f"sigma dimension {sigma.unit.dimension} "
                    f"doesn't match ydata dimension {ydata.unit.dimension}"
                )
            sigma_raw = sigma.to(ydata.unit).data
        else:
            sigma_raw = sigma

    popt, pcov = sp_optimize.curve_fit(
        wrapped_f,
        x_raw,
        y_raw,
        p0=p0,
        sigma=sigma_raw,
        **kwargs,
    )

    return popt, pcov


def least_squares(
    fun: Callable[[DimArray], np.ndarray],
    x0: DimArray,
    bounds: tuple[DimArray | float, DimArray | float] = (-np.inf, np.inf),
    **kwargs: Any,
) -> "LeastSquaresResult":
    """Dimension-aware least squares optimization.

    Wraps scipy.optimize.least_squares.

    Args:
        fun: Residual function taking DimArray, returning residuals.
        x0: Initial guess with physical units.
        bounds: (lower, upper) bounds.
        **kwargs: Additional arguments to scipy.optimize.least_squares.

    Returns:
        LeastSquaresResult with optimal DimArray x.
    """
    _check_scipy()

    if not isinstance(x0, DimArray):
        raise TypeError("x0 must be a DimArray")

    x0_unit = x0.unit
    x0_data = np.asarray(x0.data)

    def wrapped_fun(x_raw: np.ndarray) -> np.ndarray:
        x_dim = DimArray._from_data_and_unit(x_raw, x0_unit)
        result = fun(x_dim)
        if isinstance(result, DimArray):
            return np.asarray(result.data)
        return np.asarray(result)

    # Process bounds
    lb, ub = bounds
    if isinstance(lb, DimArray):
        lb = lb.to(x0_unit).data
    if isinstance(ub, DimArray):
        ub = ub.to(x0_unit).data

    result = sp_optimize.least_squares(
        wrapped_fun,
        x0_data,
        bounds=(lb, ub),
        **kwargs,
    )

    return LeastSquaresResult(
        x=DimArray._from_data_and_unit(result.x, x0_unit),
        cost=result.cost,
        fun=result.fun,
        optimality=result.optimality,
        success=result.success,
        _scipy_result=result,
    )


class LeastSquaresResult:
    """Result of dimension-aware least squares.

    Attributes:
        x: Optimal DimArray with units.
        cost: Value of cost function.
        fun: Residuals at solution.
        optimality: Optimality measure.
        success: Whether optimization succeeded.
    """

    def __init__(
        self,
        x: DimArray,
        cost: float,
        fun: np.ndarray,
        optimality: float,
        success: bool,
        _scipy_result: Any = None,
    ):
        self.x = x
        self.cost = cost
        self.fun = fun
        self.optimality = optimality
        self.success = success
        self._scipy_result = _scipy_result

    def __repr__(self) -> str:
        return (
            f"LeastSquaresResult(x={self.x}, cost={self.cost:.6g}, "
            f"success={self.success})"
        )
