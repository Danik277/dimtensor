"""Electromagnetic constants from CODATA 2022.

This module provides electromagnetic constants including the elementary
charge, vacuum permeability, vacuum permittivity, and fine-structure constant.

Examples:
    >>> from dimtensor.constants.electromagnetic import e, mu_0, epsilon_0, alpha
    >>> print(e)
    e = 1.602176634e-19 C (exact)
    >>> print(alpha)
    alpha = 0.0072973525643 +/- 1.1e-12 1

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
"""

from ..core.dimensions import DIMENSIONLESS, Dimension
from ..core.units import Unit
from ._base import Constant

# =============================================================================
# Elementary charge
# =============================================================================
# Exact by definition since 2019 SI redefinition

elementary_charge = Constant(
    symbol="e",
    name="elementary charge",
    value=1.602176634e-19,
    unit=Unit("C", Dimension(current=1, time=1), 1.0),
    uncertainty=0.0,  # exact
)
e = elementary_charge

# =============================================================================
# Vacuum magnetic permeability (mu_0)
# =============================================================================
# CODATA 2022: 1.25663706127(20) x 10^-6 N/A^2
# No longer exact after 2019 SI redefinition

vacuum_permeability = Constant(
    symbol="mu_0",
    name="vacuum magnetic permeability",
    value=1.25663706127e-6,
    unit=Unit("N/A^2", Dimension(mass=1, length=1, time=-2, current=-2), 1.0),
    uncertainty=2.0e-16,
)
mu_0 = vacuum_permeability

# =============================================================================
# Vacuum electric permittivity (epsilon_0)
# =============================================================================
# CODATA 2022: 8.8541878188(14) x 10^-12 F/m
# epsilon_0 = 1 / (mu_0 * c^2)

vacuum_permittivity = Constant(
    symbol="epsilon_0",
    name="vacuum electric permittivity",
    value=8.8541878188e-12,
    unit=Unit("F/m", Dimension(mass=-1, length=-3, time=4, current=2), 1.0),
    uncertainty=1.4e-21,
)
epsilon_0 = vacuum_permittivity

# =============================================================================
# Fine-structure constant (alpha)
# =============================================================================
# CODATA 2022: 7.2973525643(11) x 10^-3
# alpha = e^2 / (4 * pi * epsilon_0 * hbar * c)
# Dimensionless

fine_structure_constant = Constant(
    symbol="alpha",
    name="fine-structure constant",
    value=7.2973525643e-3,
    unit=Unit("1", DIMENSIONLESS, 1.0),
    uncertainty=1.1e-12,
)
alpha = fine_structure_constant

# =============================================================================
# Magnetic flux quantum (Phi_0)
# =============================================================================
# Phi_0 = h / (2e), exact since h and e are exact
# CODATA 2022: 2.067833848... x 10^-15 Wb

magnetic_flux_quantum = Constant(
    symbol="Phi_0",
    name="magnetic flux quantum",
    value=2.067833848461929e-15,
    unit=Unit("Wb", Dimension(mass=1, length=2, time=-2, current=-1), 1.0),
    uncertainty=0.0,  # exact
)
Phi_0 = magnetic_flux_quantum

# =============================================================================
# Conductance quantum (G_0)
# =============================================================================
# G_0 = 2e^2 / h, exact since e and h are exact
# CODATA 2022: 7.748091729... x 10^-5 S

conductance_quantum = Constant(
    symbol="G_0",
    name="conductance quantum",
    value=7.748091729863649e-5,
    unit=Unit("S", Dimension(mass=-1, length=-2, time=3, current=2), 1.0),
    uncertainty=0.0,  # exact
)
G_0 = conductance_quantum

# =============================================================================
# Impedance of vacuum (Z_0)
# =============================================================================
# Z_0 = mu_0 * c = sqrt(mu_0 / epsilon_0)
# CODATA 2022: 376.730313412(59) ohm

impedance_of_vacuum = Constant(
    symbol="Z_0",
    name="characteristic impedance of vacuum",
    value=376.730313412,
    unit=Unit("ohm", Dimension(mass=1, length=2, time=-3, current=-2), 1.0),
    uncertainty=5.9e-8,
)
Z_0 = impedance_of_vacuum

__all__ = [
    "elementary_charge",
    "e",
    "vacuum_permeability",
    "mu_0",
    "vacuum_permittivity",
    "epsilon_0",
    "fine_structure_constant",
    "alpha",
    "magnetic_flux_quantum",
    "Phi_0",
    "conductance_quantum",
    "G_0",
    "impedance_of_vacuum",
    "Z_0",
]
