"""Chemistry units for analytical and physical chemistry.

This module provides units commonly used in chemistry and biochemistry,
including concentration units (molar, molal), atomic mass units (dalton),
and dimensionless concentration ratios (ppm, ppb).

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.chemistry import molar, dalton
    >>> concentration = DimArray([0.1], molar)  # 0.1 M solution
    >>> mass = DimArray([12.011], dalton)  # Mass of carbon atom

Reference values from CODATA 2022 and IUPAC recommendations.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Atomic Mass Units
# =============================================================================

# Dalton (unified atomic mass unit)
# 1 Da = 1 u = 1.66053906892e-27 kg (CODATA 2022)
dalton = Unit("Da", Dimension(mass=1), 1.66053906892e-27)
Da = dalton
atomic_mass_unit = dalton
u = dalton


# =============================================================================
# Length Units (common in chemistry/crystallography)
# =============================================================================

# Angstrom (10^-10 m, common in crystallography and molecular dimensions)
angstrom = Unit("A", Dimension(length=1), 1e-10)

# Bohr radius (atomic unit of length)
# a_0 = 5.29177210544e-11 m (CODATA 2022)
bohr_radius = Unit("a0", Dimension(length=1), 5.29177210544e-11)
a0 = bohr_radius


# =============================================================================
# Concentration Units
# =============================================================================

# Molar concentration (mol/L = mol/dm^3)
# Dimension: amount / length^3 = N * L^-3
molar = Unit("M", Dimension(amount=1, length=-3), 1000.0)  # mol/m^3 = 1000 mol/L
M = molar

# Millimolar
millimolar = Unit("mM", Dimension(amount=1, length=-3), 1.0)  # 1 mM = 1 mol/m^3
mM = millimolar

# Micromolar
micromolar = Unit("uM", Dimension(amount=1, length=-3), 1e-3)  # 1 uM = 0.001 mol/m^3
uM = micromolar

# Nanomolar
nanomolar = Unit("nM", Dimension(amount=1, length=-3), 1e-6)
nM = nanomolar

# Picomolar
picomolar = Unit("pM", Dimension(amount=1, length=-3), 1e-9)
pM = picomolar


# =============================================================================
# Molality Units
# =============================================================================

# Molality (mol/kg solvent)
# Dimension: amount / mass = N * M^-1
molal = Unit("m", Dimension(amount=1, mass=-1), 1.0)


# =============================================================================
# Dimensionless Concentration Ratios
# =============================================================================

# Parts per million (10^-6)
# Note: ppm is context-dependent (mass/mass, volume/volume, etc.)
# We define it as a generic dimensionless ratio
ppm = Unit("ppm", DIMENSIONLESS, 1e-6)

# Parts per billion (10^-9)
ppb = Unit("ppb", DIMENSIONLESS, 1e-9)

# Parts per trillion (10^-12)
ppt = Unit("ppt", DIMENSIONLESS, 1e-12)

# Percent
percent = Unit("%", DIMENSIONLESS, 0.01)


# =============================================================================
# Energy Units (common in chemistry)
# =============================================================================

# Hartree (atomic unit of energy)
# E_h = 4.3597447222060e-18 J (CODATA 2022)
hartree = Unit("E_h", Dimension(mass=1, length=2, time=-2), 4.3597447222060e-18)
E_h = hartree

# Kilocalorie per mole (common in biochemistry)
# 1 kcal/mol = 4184 J/mol = 6.9477e-21 J per molecule
kcal_per_mol = Unit("kcal/mol", Dimension(mass=1, length=2, time=-2, amount=-1), 4184.0)

# Kilojoule per mole
kJ_per_mol = Unit("kJ/mol", Dimension(mass=1, length=2, time=-2, amount=-1), 1000.0)


# =============================================================================
# Electric Dipole Moment
# =============================================================================

# Debye (unit of electric dipole moment)
# 1 D = 3.33564e-30 C*m
debye = Unit("D", Dimension(current=1, time=1, length=1), 3.33564e-30)
D = debye


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Atomic mass
    "dalton", "Da", "atomic_mass_unit", "u",
    # Length
    "angstrom",
    "bohr_radius", "a0",
    # Concentration
    "molar", "M",
    "millimolar", "mM",
    "micromolar", "uM",
    "nanomolar", "nM",
    "picomolar", "pM",
    # Molality
    "molal",
    # Dimensionless ratios
    "ppm", "ppb", "ppt", "percent",
    # Energy
    "hartree", "E_h",
    "kcal_per_mol",
    "kJ_per_mol",
    # Dipole moment
    "debye", "D",
]
