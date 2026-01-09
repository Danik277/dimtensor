"""Dimension-aware SciPy integration wrappers.

Provides wrappers for numerical integration and ODE solving with units.
"""
# mypy: disable-error-code="type-arg"

from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit

try:
    from scipy import integrate as sp_integrate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _check_scipy() -> None:
    """Check that scipy is available."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for integration functions. "
            "Install with: pip install scipy"
        )


def solve_ivp(
    fun: Callable[[DimArray, DimArray], DimArray],
    t_span: tuple[DimArray, DimArray] | tuple[float, float],
    y0: DimArray,
    method: str = "RK45",
    t_eval: DimArray | np.ndarray | None = None,
    **kwargs: Any,
) -> "OdeResult":
    """Dimension-aware initial value problem solver.

    Wraps scipy.integrate.solve_ivp to preserve units through ODE solving.
    For dy/dt = f(t, y), the result y has the same dimension as y0.

    Args:
        fun: Function computing dy/dt. Takes (t, y) DimArrays, returns DimArray.
        t_span: (t0, tf) time interval. Can be DimArrays or floats.
        y0: Initial state with physical units.
        method: Integration method ('RK45', 'RK23', 'DOP853', etc.).
        t_eval: Times at which to store solution.
        **kwargs: Additional arguments to scipy.integrate.solve_ivp.

    Returns:
        OdeResult with dimensional t and y.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.scipy import solve_ivp
        >>>
        >>> def harmonic(t, y):
        ...     # y = [position, velocity]
        ...     # dy/dt = [velocity, -omega^2 * position]
        ...     omega = 2.0  # rad/s
        ...     return DimArray([y.data[1], -omega**2 * y.data[0]], y.unit)
        >>>
        >>> t_span = (0.0, 10.0)  # seconds
        >>> y0 = DimArray([1.0, 0.0], units.m)  # [position, velocity]
        >>> sol = solve_ivp(harmonic, t_span, y0)
    """
    _check_scipy()

    if not isinstance(y0, DimArray):
        raise TypeError("y0 must be a DimArray with units")

    y0_unit = y0.unit
    y0_data = np.asarray(y0.data)

    # Extract time unit if provided
    t_unit = None
    if isinstance(t_span[0], DimArray):
        t_unit = t_span[0].unit
        t0 = float(np.asarray(t_span[0]._data).flatten()[0])
        t1_conv = t_span[1].to(t_unit)
        tf = float(np.asarray(t1_conv._data).flatten()[0])
        t_span_raw = (t0, tf)
    else:
        t_span_raw = t_span

    # Handle t_eval
    t_eval_raw = None
    if t_eval is not None:
        if isinstance(t_eval, DimArray):
            if t_unit is not None:
                t_eval_raw = t_eval.to(t_unit).data
            else:
                t_eval_raw = t_eval.data
                t_unit = t_eval.unit
        else:
            t_eval_raw = t_eval

    # Wrapper that adds units before calling user function
    def wrapped_fun(t_raw: float, y_raw: np.ndarray) -> np.ndarray:
        if t_unit is not None:
            t_dim = DimArray._from_data_and_unit(np.atleast_1d(t_raw), t_unit)
        else:
            t_dim = DimArray._from_data_and_unit(
                np.atleast_1d(t_raw), Unit("s", Dimension(time=1), 1.0)
            )
        y_dim = DimArray._from_data_and_unit(np.atleast_1d(y_raw), y0_unit)
        result = fun(t_dim, y_dim)
        if isinstance(result, DimArray):
            return np.asarray(result.data).flatten()
        return np.asarray(result).flatten()

    result = sp_integrate.solve_ivp(
        wrapped_fun,
        t_span_raw,
        y0_data,
        method=method,
        t_eval=t_eval_raw,
        **kwargs,
    )

    # Convert results back to DimArrays
    if t_unit is not None:
        t_dim = DimArray._from_data_and_unit(result.t, t_unit)
    else:
        t_dim = DimArray._from_data_and_unit(
            result.t, Unit("s", Dimension(time=1), 1.0)
        )
    y_dim = DimArray._from_data_and_unit(result.y, y0_unit)

    return OdeResult(
        t=t_dim,
        y=y_dim,
        success=result.success,
        message=result.message,
        nfev=result.nfev,
        _scipy_result=result,
    )


class OdeResult:
    """Result of dimension-aware ODE solving.

    Attributes:
        t: Time points (DimArray).
        y: Solution values at t (DimArray, shape (n, len(t))).
        success: Whether integration succeeded.
        message: Status message.
        nfev: Number of function evaluations.
    """

    def __init__(
        self,
        t: DimArray,
        y: DimArray,
        success: bool,
        message: str,
        nfev: int,
        _scipy_result: Any = None,
    ):
        self.t = t
        self.y = y
        self.success = success
        self.message = message
        self.nfev = nfev
        self._scipy_result = _scipy_result

    def __repr__(self) -> str:
        return (
            f"OdeResult(t=DimArray(shape={self.t.data.shape}), "
            f"y=DimArray(shape={self.y.data.shape}), success={self.success})"
        )


def quad(
    func: Callable[[DimArray], DimArray],
    a: DimArray | float,
    b: DimArray | float,
    **kwargs: Any,
) -> tuple[DimArray, float]:
    """Dimension-aware numerical integration.

    Computes ∫f(x)dx from a to b. The result has dimension of f(x) * x.

    Args:
        func: Function to integrate. Takes DimArray, returns DimArray.
        a: Lower integration limit.
        b: Upper integration limit.
        **kwargs: Additional arguments to scipy.integrate.quad.

    Returns:
        (result, error): Integral result and estimated error.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.scipy import quad
        >>>
        >>> def velocity(t):
        ...     # v(t) = 10 m/s (constant)
        ...     return DimArray(10.0, units.m / units.s)
        >>>
        >>> distance, err = quad(velocity, 0.0, 5.0)  # seconds
        >>> print(distance)  # Should be 50 m
    """
    _check_scipy()

    # Determine x unit
    if isinstance(a, DimArray):
        x_unit = a.unit
        a_raw = float(np.asarray(a._data).flatten()[0])
    else:
        x_unit = Unit("1", Dimension(), 1.0)  # dimensionless
        a_raw = float(a)

    if isinstance(b, DimArray):
        if isinstance(a, DimArray):
            b_conv = b.to(x_unit)
            b_raw = float(np.asarray(b_conv._data).flatten()[0])
        else:
            x_unit = b.unit
            b_raw = float(np.asarray(b._data).flatten()[0])
    else:
        b_raw = float(b)

    # Evaluate func once to get output dimension
    mid = (a_raw + b_raw) / 2
    x_mid = DimArray._from_data_and_unit(np.atleast_1d(mid), x_unit)
    f_mid = func(x_mid)
    if isinstance(f_mid, DimArray):
        f_unit = f_mid.unit
    else:
        f_unit = Unit("1", Dimension(), 1.0)

    # Result dimension = f dimension * x dimension
    result_dim = f_unit.dimension * x_unit.dimension
    result_scale = f_unit.scale * x_unit.scale
    result_unit = Unit(str(result_dim), result_dim, result_scale)

    # Wrapper
    def wrapped_func(x_raw: float) -> float:
        x_dim = DimArray._from_data_and_unit(np.atleast_1d(x_raw), x_unit)
        result = func(x_dim)
        if isinstance(result, DimArray):
            return float(np.asarray(result.data).flatten()[0])
        return float(result)

    integral, error = sp_integrate.quad(wrapped_func, a_raw, b_raw, **kwargs)

    return DimArray._from_data_and_unit(np.atleast_1d(integral), result_unit), error
