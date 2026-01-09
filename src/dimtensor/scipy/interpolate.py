"""Dimension-aware SciPy interpolation wrappers.

Provides wrappers for interpolation that preserve physical units.
"""
# mypy: disable-error-code="type-arg,assignment"

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from ..core.dimarray import DimArray
from ..core.units import Unit

try:
    from scipy import interpolate as sp_interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _check_scipy() -> None:
    """Check that scipy is available."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for interpolation functions. "
            "Install with: pip install scipy"
        )


class DimInterp1d:
    """Dimension-aware 1D interpolation.

    Wraps scipy.interpolate.interp1d to preserve units.

    Attributes:
        x_unit: Unit of x data.
        y_unit: Unit of y data.
    """

    def __init__(
        self,
        x: DimArray,
        y: DimArray,
        kind: str = "linear",
        fill_value: float | str = np.nan,
        bounds_error: bool = True,
        **kwargs: Any,
    ):
        """Create dimension-aware interpolator.

        Args:
            x: Independent variable data with units.
            y: Dependent variable data with units.
            kind: Interpolation kind ('linear', 'cubic', 'quadratic', etc.).
            fill_value: Value for extrapolation (or 'extrapolate').
            bounds_error: Whether to raise error for out-of-bounds.
            **kwargs: Additional arguments to scipy.interpolate.interp1d.

        Example:
            >>> from dimtensor import DimArray, units
            >>> from dimtensor.scipy.interpolate import DimInterp1d
            >>>
            >>> t = DimArray([0, 1, 2, 3], units.s)
            >>> y = DimArray([0, 1, 4, 9], units.m)
            >>> interp = DimInterp1d(t, y, kind='cubic')
            >>> y_new = interp(DimArray([0.5, 1.5, 2.5], units.s))
        """
        _check_scipy()

        if not isinstance(x, DimArray):
            raise TypeError("x must be a DimArray")
        if not isinstance(y, DimArray):
            raise TypeError("y must be a DimArray")

        self.x_unit = x.unit
        self.y_unit = y.unit

        self._interp = sp_interpolate.interp1d(
            np.asarray(x.data),
            np.asarray(y.data),
            kind=kind,
            fill_value=fill_value,
            bounds_error=bounds_error,
            **kwargs,
        )

    def __call__(self, x_new: DimArray | np.ndarray | float) -> DimArray:
        """Evaluate interpolation at new x values.

        Args:
            x_new: New x values (DimArray, array, or scalar).

        Returns:
            Interpolated y values as DimArray.
        """
        if isinstance(x_new, DimArray):
            if x_new.unit.dimension != self.x_unit.dimension:
                raise ValueError(
                    f"x_new dimension {x_new.unit.dimension} "
                    f"doesn't match interpolator x dimension {self.x_unit.dimension}"
                )
            x_raw = x_new.to(self.x_unit).data
        else:
            x_raw = x_new

        y_raw = self._interp(x_raw)
        return DimArray._from_data_and_unit(y_raw, self.y_unit)

    def __repr__(self) -> str:
        return f"DimInterp1d(x_unit={self.x_unit}, y_unit={self.y_unit})"


def interp1d(
    x: DimArray,
    y: DimArray,
    kind: str = "linear",
    fill_value: float | str = np.nan,
    bounds_error: bool = True,
    **kwargs: Any,
) -> DimInterp1d:
    """Create dimension-aware 1D interpolator.

    Convenience function that creates a DimInterp1d object.

    Args:
        x: Independent variable data with units.
        y: Dependent variable data with units.
        kind: Interpolation kind ('linear', 'cubic', 'quadratic', etc.).
        fill_value: Value for extrapolation (or 'extrapolate').
        bounds_error: Whether to raise error for out-of-bounds.
        **kwargs: Additional arguments.

    Returns:
        DimInterp1d interpolator object.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.scipy import interp1d
        >>>
        >>> t = DimArray([0, 1, 2, 3], units.s)
        >>> pos = DimArray([0, 2, 8, 18], units.m)
        >>> f = interp1d(t, pos, kind='cubic')
        >>> pos_interp = f(DimArray([0.5, 1.5], units.s))
    """
    return DimInterp1d(x, y, kind=kind, fill_value=fill_value,
                       bounds_error=bounds_error, **kwargs)


class DimUnivariateSpline:
    """Dimension-aware univariate spline.

    Wraps scipy.interpolate.UnivariateSpline to preserve units.
    """

    def __init__(
        self,
        x: DimArray,
        y: DimArray,
        w: np.ndarray | None = None,
        k: int = 3,
        s: float | None = None,
        **kwargs: Any,
    ):
        """Create dimension-aware spline.

        Args:
            x: Independent variable data with units.
            y: Dependent variable data with units.
            w: Weights for spline fitting.
            k: Degree of spline (1-5).
            s: Smoothing factor.
            **kwargs: Additional arguments.
        """
        _check_scipy()

        if not isinstance(x, DimArray):
            raise TypeError("x must be a DimArray")
        if not isinstance(y, DimArray):
            raise TypeError("y must be a DimArray")

        self.x_unit = x.unit
        self.y_unit = y.unit

        self._spline = sp_interpolate.UnivariateSpline(
            np.asarray(x.data),
            np.asarray(y.data),
            w=w,
            k=k,
            s=s,
            **kwargs,
        )

    def __call__(self, x_new: DimArray | np.ndarray | float) -> DimArray:
        """Evaluate spline at new x values."""
        if isinstance(x_new, DimArray):
            if x_new.unit.dimension != self.x_unit.dimension:
                raise ValueError(
                    f"x_new dimension {x_new.unit.dimension} "
                    f"doesn't match spline x dimension {self.x_unit.dimension}"
                )
            x_raw = x_new.to(self.x_unit).data
        else:
            x_raw = x_new

        y_raw = self._spline(x_raw)
        return DimArray._from_data_and_unit(y_raw, self.y_unit)

    def derivative(self, n: int = 1) -> "DimUnivariateSplineDerivative":
        """Get derivative of spline.

        The derivative has dimension y_dim / x_dim^n.

        Args:
            n: Order of derivative.

        Returns:
            Callable that evaluates the derivative.
        """
        return DimUnivariateSplineDerivative(
            self._spline.derivative(n),
            self.x_unit,
            self.y_unit,
            n,
        )

    def integral(
        self,
        a: DimArray | float,
        b: DimArray | float,
    ) -> DimArray:
        """Compute definite integral of spline.

        Args:
            a: Lower limit.
            b: Upper limit.

        Returns:
            Integral value with dimension y_dim * x_dim.
        """
        if isinstance(a, DimArray):
            a_conv = a.to(self.x_unit)
            a_raw = float(np.asarray(a_conv._data).flatten()[0])
        else:
            a_raw = float(a)

        if isinstance(b, DimArray):
            b_conv = b.to(self.x_unit)
            b_raw = float(np.asarray(b_conv._data).flatten()[0])
        else:
            b_raw = float(b)

        result_raw = self._spline.integral(a_raw, b_raw)

        # Result dimension = y_dim * x_dim
        result_dim = self.y_unit.dimension * self.x_unit.dimension
        result_scale = self.y_unit.scale * self.x_unit.scale
        result_unit = Unit(str(result_dim), result_dim, result_scale)

        return DimArray._from_data_and_unit(np.atleast_1d(result_raw), result_unit)


class DimUnivariateSplineDerivative:
    """Derivative of a dimension-aware spline."""

    def __init__(
        self,
        derivative: Any,
        x_unit: Unit,
        y_unit: Unit,
        order: int,
    ):
        self._derivative = derivative
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.order = order

        # Derivative dimension = y_dim / x_dim^order
        self._result_dim = y_unit.dimension
        for _ in range(order):
            self._result_dim = self._result_dim / x_unit.dimension
        self._result_scale = y_unit.scale / (x_unit.scale ** order)
        self._result_unit = Unit(str(self._result_dim), self._result_dim, self._result_scale)

    def __call__(self, x_new: DimArray | np.ndarray | float) -> DimArray:
        """Evaluate derivative at new x values."""
        if isinstance(x_new, DimArray):
            x_raw = x_new.to(self.x_unit).data
        else:
            x_raw = x_new

        y_raw = self._derivative(x_raw)
        return DimArray._from_data_and_unit(y_raw, self._result_unit)
