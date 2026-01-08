"""Domain-specific unit collections.

This module provides specialized units for different scientific domains:

- astronomy: parsec, AU, solar_mass, light_year, etc.
- chemistry: molar, dalton, ppm, angstrom, etc.
- engineering: MPa, ksi, BTU, horsepower, etc.

Example:
    >>> from dimtensor.domains.astronomy import parsec, AU
    >>> from dimtensor.domains.chemistry import molar, dalton
    >>> from dimtensor.domains.engineering import MPa, hp
"""

from . import astronomy
from . import chemistry
from . import engineering

__all__ = [
    "astronomy",
    "chemistry",
    "engineering",
]
