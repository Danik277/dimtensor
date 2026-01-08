"""Atomic and nuclear constants from CODATA 2022.

This module provides atomic and nuclear constants including particle masses,
the Bohr radius, Rydberg constant, and related quantities.

Examples:
    >>> from dimtensor.constants.atomic import m_e, m_p, a_0, R_inf
    >>> print(m_e)
    m_e = 9.1093837139e-31 +/- 2.8e-40 kg
    >>> print(a_0)
    a_0 = 5.29177210544e-11 +/- 8.2e-21 m

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
"""

from ..core.dimensions import Dimension
from ..core.units import Unit
from ._base import Constant

# =============================================================================
# Electron mass
# =============================================================================
# CODATA 2022: 9.1093837139(28) x 10^-31 kg
# Relative uncertainty: 3.1 x 10^-10

electron_mass = Constant(
    symbol="m_e",
    name="electron mass",
    value=9.1093837139e-31,
    unit=Unit("kg", Dimension(mass=1), 1.0),
    uncertainty=2.8e-40,
)
m_e = electron_mass

# =============================================================================
# Proton mass
# =============================================================================
# CODATA 2022: 1.67262192595(52) x 10^-27 kg
# Relative uncertainty: 3.1 x 10^-10

proton_mass = Constant(
    symbol="m_p",
    name="proton mass",
    value=1.67262192595e-27,
    unit=Unit("kg", Dimension(mass=1), 1.0),
    uncertainty=5.2e-37,
)
m_p = proton_mass

# =============================================================================
# Neutron mass
# =============================================================================
# CODATA 2022: 1.67492750056(85) x 10^-27 kg
# Relative uncertainty: 5.1 x 10^-10

neutron_mass = Constant(
    symbol="m_n",
    name="neutron mass",
    value=1.67492750056e-27,
    unit=Unit("kg", Dimension(mass=1), 1.0),
    uncertainty=8.5e-37,
)
m_n = neutron_mass

# =============================================================================
# Atomic mass constant (1/12 of C-12 mass)
# =============================================================================
# CODATA 2022: 1.66053906892(52) x 10^-27 kg
# Relative uncertainty: 3.1 x 10^-10

atomic_mass_constant = Constant(
    symbol="m_u",
    name="atomic mass constant",
    value=1.66053906892e-27,
    unit=Unit("kg", Dimension(mass=1), 1.0),
    uncertainty=5.2e-37,
)
m_u = atomic_mass_constant

# =============================================================================
# Bohr radius
# =============================================================================
# CODATA 2022: 5.29177210544(82) x 10^-11 m
# a_0 = hbar / (m_e * c * alpha)
# Relative uncertainty: 1.6 x 10^-10

bohr_radius = Constant(
    symbol="a_0",
    name="Bohr radius",
    value=5.29177210544e-11,
    unit=Unit("m", Dimension(length=1), 1.0),
    uncertainty=8.2e-21,
)
a_0 = bohr_radius

# =============================================================================
# Classical electron radius
# =============================================================================
# CODATA 2022: 2.8179403205(13) x 10^-15 m
# r_e = alpha^2 * a_0

classical_electron_radius = Constant(
    symbol="r_e",
    name="classical electron radius",
    value=2.8179403205e-15,
    unit=Unit("m", Dimension(length=1), 1.0),
    uncertainty=1.3e-24,
)
r_e = classical_electron_radius

# =============================================================================
# Rydberg constant
# =============================================================================
# CODATA 2022: 10973731.568157(12) m^-1
# R_inf = m_e * c * alpha^2 / (2 * h)
# Relative uncertainty: 1.1 x 10^-12

rydberg_constant = Constant(
    symbol="R_inf",
    name="Rydberg constant",
    value=10973731.568157,
    unit=Unit("1/m", Dimension(length=-1), 1.0),
    uncertainty=1.2e-5,
)
R_inf = rydberg_constant

# =============================================================================
# Compton wavelength
# =============================================================================
# CODATA 2022: 2.42631023538(76) x 10^-12 m
# lambda_C = h / (m_e * c)

compton_wavelength = Constant(
    symbol="lambda_C",
    name="Compton wavelength",
    value=2.42631023538e-12,
    unit=Unit("m", Dimension(length=1), 1.0),
    uncertainty=7.6e-22,
)
lambda_C = compton_wavelength

# =============================================================================
# Bohr magneton
# =============================================================================
# CODATA 2022: 9.2740100657(29) x 10^-24 J/T
# mu_B = e * hbar / (2 * m_e)

bohr_magneton = Constant(
    symbol="mu_B",
    name="Bohr magneton",
    value=9.2740100657e-24,
    unit=Unit("J/T", Dimension(mass=1, length=2, time=-2, current=-1), 1.0),
    uncertainty=2.9e-33,
)
mu_B = bohr_magneton

# =============================================================================
# Nuclear magneton
# =============================================================================
# CODATA 2022: 5.0507837393(16) x 10^-27 J/T
# mu_N = e * hbar / (2 * m_p)

nuclear_magneton = Constant(
    symbol="mu_N",
    name="nuclear magneton",
    value=5.0507837393e-27,
    unit=Unit("J/T", Dimension(mass=1, length=2, time=-2, current=-1), 1.0),
    uncertainty=1.6e-36,
)
mu_N = nuclear_magneton

__all__ = [
    "electron_mass",
    "m_e",
    "proton_mass",
    "m_p",
    "neutron_mass",
    "m_n",
    "atomic_mass_constant",
    "m_u",
    "bohr_radius",
    "a_0",
    "classical_electron_radius",
    "r_e",
    "rydberg_constant",
    "R_inf",
    "compton_wavelength",
    "lambda_C",
    "bohr_magneton",
    "mu_B",
    "nuclear_magneton",
    "mu_N",
]
