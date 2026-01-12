"""Dimensional analysis tools for dimtensor.

This module provides tools for automatic non-dimensionalization, characteristic
scale identification, computation of dimensionless numbers, scaling law discovery,
and sensitivity analysis.

Key components:
- CharacteristicScalesFinder: Identifies characteristic scales from problem parameters
- Dimensionless numbers database: Compute common dimensionless numbers (Re, Ma, Fr, etc.)
- Regime inference: Determine physical regimes from dimensionless number values
- PowerLawFitter: Discover power law relationships with dimensional constraints
- MultiPowerLawFitter: Fit multi-variable power laws with dimensional consistency
- Sensitivity analysis: Local sensitivity analysis and parameter importance ranking

Examples:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.analysis import CharacteristicScalesFinder, dimensionless_numbers as dn
    >>> from dimtensor.analysis import PowerLawFitter
    >>>
    >>> # Define problem parameters
    >>> params = {
    ...     'density': DimArray(1000, units.kg / units.m**3),
    ...     'velocity': DimArray(1.0, units.m / units.s),
    ...     'length': DimArray(0.1, units.m),
    ...     'viscosity': DimArray(0.001, units.Pa * units.s),
    ... }
    >>>
    >>> # Find characteristic scales
    >>> finder = CharacteristicScalesFinder()
    >>> scales = finder.find_scales(params)
    >>>
    >>> # Compute Reynolds number
    >>> Re = dn.compute('Re', **params)
    >>> regime = dn.infer_regime('Re', Re)
    >>> print(f"Re = {Re:.0f} ({regime})")
    >>>
    >>> # Discover scaling laws
    >>> x = DimArray([1, 2, 3, 4], units.m)
    >>> y = DimArray([1, 4, 9, 16], units.m**2)
    >>> fitter = PowerLawFitter()
    >>> result = fitter.fit(x, y)
    >>> print(f"Exponent: {result.exponents[0]:.2f}")
    >>>
    >>> # Sensitivity analysis
    >>> from dimtensor.analysis import local_sensitivity, rank_parameters
    >>> def kinetic_energy(mass, velocity):
    ...     return 0.5 * mass * velocity**2
    >>> params = {
    ...     'mass': DimArray(2.0, units.kg),
    ...     'velocity': DimArray(10.0, units.m / units.s),
    ... }
    >>> result = rank_parameters(kinetic_energy, params)
    >>> print(result.ranking)  # [('velocity', 2.0), ('mass', 1.0)]
"""

from . import dimensionless_numbers
from .scales import CharacteristicScalesFinder
from .scaling import PowerLawFitter, MultiPowerLawFitter, FitResult
from .sensitivity import (
    local_sensitivity,
    sensitivity_matrix,
    rank_parameters,
    normalized_sensitivity,
    tornado_diagram_data,
    SensitivityResult,
)
from .buckingham import (
    buckingham_pi,
    PiGroup,
    KNOWN_PI_GROUPS,
)

__all__ = [
    "CharacteristicScalesFinder",
    "dimensionless_numbers",
    "PowerLawFitter",
    "MultiPowerLawFitter",
    "FitResult",
    "local_sensitivity",
    "sensitivity_matrix",
    "rank_parameters",
    "normalized_sensitivity",
    "tornado_diagram_data",
    "SensitivityResult",
    "buckingham_pi",
    "PiGroup",
    "KNOWN_PI_GROUPS",
]
