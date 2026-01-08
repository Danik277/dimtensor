"""Universal physical constants from CODATA 2022.

This module provides the fundamental universal constants that appear
throughout physics: the speed of light, gravitational constant, and
Planck constant.

Examples:
    >>> from dimtensor.constants import c, G, h, hbar
    >>> print(c)
    c = 299792458 m/s (exact)
    >>> print(G)
    G = 6.6743e-11 +/- 1.5e-15 m^3/(kg s^2)

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
"""

import math

from ..core.dimensions import Dimension
from ..core.units import Unit
from ._base import Constant

# =============================================================================
# Speed of light in vacuum
# =============================================================================
# Exact by definition since 2019 SI redefinition

speed_of_light = Constant(
    symbol="c",
    name="speed of light in vacuum",
    value=299792458.0,
    unit=Unit("m/s", Dimension(length=1, time=-1), 1.0),
    uncertainty=0.0,  # exact
)
c = speed_of_light

# =============================================================================
# Newtonian constant of gravitation
# =============================================================================
# CODATA 2022: 6.67430(15) x 10^-11 m^3 kg^-1 s^-2
# Relative uncertainty: 2.2 x 10^-5

gravitational_constant = Constant(
    symbol="G",
    name="Newtonian constant of gravitation",
    value=6.67430e-11,
    unit=Unit("m^3/(kg s^2)", Dimension(length=3, mass=-1, time=-2), 1.0),
    uncertainty=1.5e-15,
)
G = gravitational_constant

# =============================================================================
# Planck constant
# =============================================================================
# Exact by definition since 2019 SI redefinition

planck_constant = Constant(
    symbol="h",
    name="Planck constant",
    value=6.62607015e-34,
    unit=Unit("J s", Dimension(mass=1, length=2, time=-1), 1.0),
    uncertainty=0.0,  # exact
)
h = planck_constant

# =============================================================================
# Reduced Planck constant (h-bar)
# =============================================================================
# hbar = h / (2 * pi), exact since h is exact

reduced_planck_constant = Constant(
    symbol="hbar",
    name="reduced Planck constant",
    value=6.62607015e-34 / (2.0 * math.pi),  # 1.054571817646156e-34
    unit=Unit("J s", Dimension(mass=1, length=2, time=-1), 1.0),
    uncertainty=0.0,  # exact (derived from exact h)
)
hbar = reduced_planck_constant

__all__ = [
    "speed_of_light",
    "c",
    "gravitational_constant",
    "G",
    "planck_constant",
    "h",
    "reduced_planck_constant",
    "hbar",
]
