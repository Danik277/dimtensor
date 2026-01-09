"""SciPy integration for dimension-aware scientific computing.

Provides wrappers around SciPy functions that preserve units through
optimization, integration, and interpolation operations.

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.scipy import minimize, solve_ivp
    >>>
    >>> # Dimensional optimization
    >>> x0 = DimArray([1.0, 2.0], units.m)
    >>> result = minimize(objective, x0)
    >>>
    >>> # Dimensional ODE solving
    >>> y0 = DimArray([0.0], units.m)
    >>> sol = solve_ivp(dynamics, t_span, y0)
"""

from .optimize import minimize, curve_fit, least_squares
from .integrate import solve_ivp, quad
from .interpolate import interp1d

__all__ = [
    # Optimization
    "minimize",
    "curve_fit",
    "least_squares",
    # Integration
    "solve_ivp",
    "quad",
    # Interpolation
    "interp1d",
]
