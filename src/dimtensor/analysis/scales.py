"""Characteristic scale identification for non-dimensionalization.

This module provides tools for automatically identifying characteristic scales
from problem parameters, which is the first step in non-dimensionalization.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.dimensions import Dimension
from ..core.dimarray import DimArray
from ..core.units import Unit


class CharacteristicScalesFinder:
    """Identifies characteristic scales from problem parameters.

    Given a set of dimensional parameters (e.g., velocity, length, density),
    this class identifies a representative scale for each physical dimension.
    These scales are used for non-dimensionalization.

    Strategy:
    - For each dimension, find all parameters with that dimension
    - Take the maximum absolute value as the characteristic scale
    - If no parameter has a dimension, that dimension has no characteristic scale

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.analysis import CharacteristicScalesFinder
        >>>
        >>> # Pipe flow problem
        >>> params = {
        ...     'velocity': DimArray([1.0, 2.0, 3.0], units.m / units.s),
        ...     'diameter': DimArray(0.1, units.m),
        ...     'density': DimArray(1000.0, units.kg / units.m**3),
        ...     'viscosity': DimArray(0.001, units.Pa * units.s),
        ... }
        >>>
        >>> finder = CharacteristicScalesFinder()
        >>> scales = finder.find_scales(params)
        >>> print(scales)  # {dimension: scale_value}
    """

    def __init__(
        self,
        strategy: str = "max",
        preferred_scales: Optional[Dict[Dimension, float]] = None,
    ):
        """Initialize the characteristic scales finder.

        Args:
            strategy: Strategy for choosing scales when multiple parameters
                     have the same dimension. Options:
                     - 'max': Use maximum absolute value (default)
                     - 'mean': Use arithmetic mean
                     - 'median': Use median value
            preferred_scales: Optional dict mapping dimensions to user-specified
                            characteristic scales. These override automatic detection.
        """
        if strategy not in ("max", "mean", "median"):
            raise ValueError(f"Unknown strategy: {strategy}. Choose 'max', 'mean', or 'median'")

        self.strategy = strategy
        self.preferred_scales = preferred_scales or {}

    def find_scales(
        self,
        parameters: Dict[str, DimArray],
    ) -> Dict[Dimension, float]:
        """Find characteristic scales for each dimension in the problem.

        Args:
            parameters: Dictionary mapping parameter names to DimArrays.
                       For example: {'velocity': v, 'length': L, 'time': t}

        Returns:
            Dictionary mapping each Dimension to its characteristic scale value.
            Only dimensions present in the parameters are included.
            Dimensionless parameters are excluded.

        Examples:
            >>> from dimtensor import DimArray, units
            >>> params = {
            ...     'v': DimArray(10.0, units.m / units.s),
            ...     'L': DimArray(2.0, units.m),
            ... }
            >>> finder = CharacteristicScalesFinder()
            >>> scales = finder.find_scales(params)
            >>> # Returns scales for length and velocity dimensions
        """
        # Group parameters by dimension
        dimension_groups: Dict[Dimension, list[float]] = {}

        for name, param in parameters.items():
            if not isinstance(param, DimArray):
                raise TypeError(f"Parameter '{name}' must be a DimArray, got {type(param)}")

            dim = param.dimension

            # Skip dimensionless parameters
            if dim.is_dimensionless:
                continue

            # Extract scalar value(s) from the parameter
            values = np.abs(param._data).flatten()

            # Only use finite, non-zero values
            values = values[np.isfinite(values) & (values > 0)]

            if len(values) > 0:
                if dim not in dimension_groups:
                    dimension_groups[dim] = []
                dimension_groups[dim].extend(values.tolist())

        # Compute characteristic scale for each dimension
        scales: Dict[Dimension, float] = {}

        for dim, values in dimension_groups.items():
            # Check if user provided a preferred scale
            if dim in self.preferred_scales:
                scales[dim] = self.preferred_scales[dim]
                continue

            # Apply strategy to choose scale
            if self.strategy == "max":
                scale = float(np.max(values))
            elif self.strategy == "mean":
                scale = float(np.mean(values))
            elif self.strategy == "median":
                scale = float(np.median(values))
            else:
                # Should never reach here due to __init__ validation
                scale = float(np.max(values))

            # Avoid zero or near-zero scales (numerical stability)
            if scale < 1e-100:
                scale = 1.0

            scales[dim] = scale

        return scales

    def find_scales_by_base_dimension(
        self,
        parameters: Dict[str, DimArray],
    ) -> Dict[str, float]:
        """Find characteristic scales for each base SI dimension.

        This method decomposes composite dimensions into base dimensions
        and finds scales for length, mass, time, etc. separately.

        Args:
            parameters: Dictionary mapping parameter names to DimArrays.

        Returns:
            Dictionary mapping base dimension names to characteristic scales.
            Keys are: 'length', 'mass', 'time', 'current', 'temperature',
            'amount', 'luminosity'.

        Examples:
            >>> from dimtensor import DimArray, units
            >>> params = {
            ...     'velocity': DimArray(10.0, units.m / units.s),
            ...     'force': DimArray(100.0, units.N),
            ... }
            >>> finder = CharacteristicScalesFinder()
            >>> scales = finder.find_scales_by_base_dimension(params)
            >>> # Returns separate scales for length, mass, time
        """
        from ..core.dimensions import (
            LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOSITY,
            _DIMENSION_NAMES
        )

        # Collect values for each base dimension
        base_dimension_values: Dict[int, list[float]] = {
            i: [] for i in range(7)
        }

        for name, param in parameters.items():
            if not isinstance(param, DimArray):
                raise TypeError(f"Parameter '{name}' must be a DimArray, got {type(param)}")

            dim = param.dimension

            # Skip dimensionless
            if dim.is_dimensionless:
                continue

            # Extract values
            values = np.abs(param._data).flatten()
            values = values[np.isfinite(values) & (values > 0)]

            if len(values) == 0:
                continue

            # Get the scale in SI base units
            scale_si = param.unit.scale
            values_si = values * scale_si

            # For each base dimension with non-zero exponent
            exponents = dim._exponents
            for i, exp in enumerate(exponents):
                if exp != 0:
                    # Raise values to 1/exp power to get contribution to base dimension
                    # Example: if velocity has L¹T⁻¹ and v=10 m/s
                    # Then length scale ≈ 10^1 = 10 m, time scale ≈ 10^(-1/(-1)) = 10 s
                    contribution = np.power(values_si, 1.0 / float(exp))
                    base_dimension_values[i].extend(contribution.tolist())

        # Compute scale for each base dimension
        result: Dict[str, float] = {}

        for i, name in enumerate(_DIMENSION_NAMES):
            values = base_dimension_values[i]

            if len(values) == 0:
                continue

            # Apply strategy
            if self.strategy == "max":
                scale = float(np.max(values))
            elif self.strategy == "mean":
                scale = float(np.mean(values))
            elif self.strategy == "median":
                scale = float(np.median(values))
            else:
                scale = float(np.max(values))

            # Numerical stability
            if scale < 1e-100:
                scale = 1.0

            result[name] = scale

        return result

    def create_scaling_units(
        self,
        parameters: Dict[str, DimArray],
    ) -> Dict[Dimension, Unit]:
        """Create units representing characteristic scales.

        This is a convenience method that converts the scale dictionary
        into Unit objects, which can be used directly for scaling.

        Args:
            parameters: Dictionary mapping parameter names to DimArrays.

        Returns:
            Dictionary mapping each Dimension to a Unit with that dimension
            and scale equal to the characteristic scale.

        Examples:
            >>> from dimtensor import DimArray, units
            >>> params = {'v': DimArray(10.0, units.m / units.s)}
            >>> finder = CharacteristicScalesFinder()
            >>> scale_units = finder.create_scaling_units(params)
            >>> # scale_units contains Unit with velocity dimension, scale=10.0
        """
        scales = self.find_scales(parameters)

        result: Dict[Dimension, Unit] = {}
        for dim, scale in scales.items():
            # Create a unit with this dimension and scale
            symbol = f"scale_{len(result)}"
            result[dim] = Unit(symbol=symbol, dimension=dim, scale=scale)

        return result
