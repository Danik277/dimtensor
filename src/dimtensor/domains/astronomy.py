"""Astronomy units for astrophysics and space science.

This module provides units commonly used in astronomy and astrophysics,
including distance units (parsec, light-year, AU), mass units (solar masses,
Earth masses), and related quantities.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.astronomy import parsec, solar_mass
    >>> distance = DimArray([4.2], parsec)  # Distance to Proxima Centauri
    >>> mass = DimArray([1.0], solar_mass)  # Mass of the Sun

Reference values from IAU 2015 Resolution B3 and CODATA 2022.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Distance Units
# =============================================================================

# Astronomical Unit (IAU 2012 definition: exactly 149597870700 m)
astronomical_unit = Unit("AU", Dimension(length=1), 1.495978707e11)
AU = astronomical_unit

# Parsec (IAU 2015: 648000/pi AU)
# 1 pc = 3.0856775814913673e16 m
parsec = Unit("pc", Dimension(length=1), 3.0856775814913673e16)
pc = parsec

# Kiloparsec and Megaparsec
kiloparsec = Unit("kpc", Dimension(length=1), 3.0856775814913673e19)
megaparsec = Unit("Mpc", Dimension(length=1), 3.0856775814913673e22)
kpc = kiloparsec
Mpc = megaparsec

# Light-year (IAU: exactly 9460730472580800 m based on Julian year)
light_year = Unit("ly", Dimension(length=1), 9.460730472580800e15)
ly = light_year


# =============================================================================
# Mass Units
# =============================================================================

# Solar mass (IAU 2015 nominal value: GM_sun / G)
# M_sun = 1.98841e30 kg (using IAU nominal solar mass parameter)
solar_mass = Unit("M_sun", Dimension(mass=1), 1.98841e30)
M_sun = solar_mass

# Earth mass (IAU nominal value)
earth_mass = Unit("M_earth", Dimension(mass=1), 5.9722e24)
M_earth = earth_mass

# Jupiter mass (IAU nominal value)
jupiter_mass = Unit("M_jup", Dimension(mass=1), 1.8982e27)
M_jup = jupiter_mass


# =============================================================================
# Length/Radius Units
# =============================================================================

# Solar radius (IAU 2015 nominal value)
solar_radius = Unit("R_sun", Dimension(length=1), 6.957e8)
R_sun = solar_radius

# Earth radius (IAU nominal equatorial radius)
earth_radius = Unit("R_earth", Dimension(length=1), 6.3781e6)
R_earth = earth_radius

# Jupiter radius (IAU nominal equatorial radius)
jupiter_radius = Unit("R_jup", Dimension(length=1), 7.1492e7)
R_jup = jupiter_radius


# =============================================================================
# Luminosity/Power Units
# =============================================================================

# Solar luminosity (IAU 2015 nominal value)
solar_luminosity = Unit("L_sun", Dimension(mass=1, length=2, time=-3), 3.828e26)
L_sun = solar_luminosity


# =============================================================================
# Angular Units (used in astrometry)
# =============================================================================

# Arcsecond (1/3600 degree = pi/648000 radians)
arcsecond = Unit("arcsec", DIMENSIONLESS, 4.84813681109536e-6)
arcsec = arcsecond

# Milliarcsecond
milliarcsecond = Unit("mas", DIMENSIONLESS, 4.84813681109536e-9)
mas = milliarcsecond

# Microarcsecond
microarcsecond = Unit("uas", DIMENSIONLESS, 4.84813681109536e-12)
uas = microarcsecond


# =============================================================================
# Time Units
# =============================================================================

# Julian year (exactly 365.25 days, used in light-year definition)
julian_year = Unit("yr", Dimension(time=1), 3.15576e7)
yr = julian_year


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Distance
    "astronomical_unit", "AU",
    "parsec", "pc",
    "kiloparsec", "kpc",
    "megaparsec", "Mpc",
    "light_year", "ly",
    # Mass
    "solar_mass", "M_sun",
    "earth_mass", "M_earth",
    "jupiter_mass", "M_jup",
    # Radius
    "solar_radius", "R_sun",
    "earth_radius", "R_earth",
    "jupiter_radius", "R_jup",
    # Luminosity
    "solar_luminosity", "L_sun",
    # Angular
    "arcsecond", "arcsec",
    "milliarcsecond", "mas",
    "microarcsecond", "uas",
    # Time
    "julian_year", "yr",
]
