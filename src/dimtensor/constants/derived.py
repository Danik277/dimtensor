"""Derived physical constants from CODATA 2022.

This module provides derived constants including the Planck units
(length, mass, time, temperature) and the Hartree energy.

Examples:
    >>> from dimtensor.constants.derived import l_P, m_P, t_P, T_P, E_h
    >>> print(l_P)
    l_P = 1.616255e-35 +/- 1.8e-40 m
    >>> print(E_h)
    E_h = 4.3597447222e-18 +/- 8.5e-28 J

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
"""

from ..core.dimensions import Dimension
from ..core.units import Unit
from ._base import Constant

# =============================================================================
# Planck length
# =============================================================================
# l_P = sqrt(hbar * G / c^3)
# CODATA 2022: 1.616255(18) x 10^-35 m
# Relative uncertainty: 1.1 x 10^-5 (dominated by G uncertainty)

planck_length = Constant(
    symbol="l_P",
    name="Planck length",
    value=1.616255e-35,
    unit=Unit("m", Dimension(length=1), 1.0),
    uncertainty=1.8e-40,
)
l_P = planck_length

# =============================================================================
# Planck mass
# =============================================================================
# m_P = sqrt(hbar * c / G)
# CODATA 2022: 2.176434(24) x 10^-8 kg
# Relative uncertainty: 1.1 x 10^-5

planck_mass = Constant(
    symbol="m_P",
    name="Planck mass",
    value=2.176434e-8,
    unit=Unit("kg", Dimension(mass=1), 1.0),
    uncertainty=2.4e-13,
)
m_P = planck_mass

# =============================================================================
# Planck time
# =============================================================================
# t_P = sqrt(hbar * G / c^5)
# CODATA 2022: 5.391247(60) x 10^-44 s
# Relative uncertainty: 1.1 x 10^-5

planck_time = Constant(
    symbol="t_P",
    name="Planck time",
    value=5.391247e-44,
    unit=Unit("s", Dimension(time=1), 1.0),
    uncertainty=6.0e-49,
)
t_P = planck_time

# =============================================================================
# Planck temperature
# =============================================================================
# T_P = sqrt(hbar * c^5 / (G * k_B^2))
# CODATA 2022: 1.416784(16) x 10^32 K
# Relative uncertainty: 1.1 x 10^-5

planck_temperature = Constant(
    symbol="T_P",
    name="Planck temperature",
    value=1.416784e32,
    unit=Unit("K", Dimension(temperature=1), 1.0),
    uncertainty=1.6e27,
)
T_P = planck_temperature

# =============================================================================
# Planck charge
# =============================================================================
# q_P = sqrt(4 * pi * epsilon_0 * hbar * c)
# CODATA 2022: 1.875546... x 10^-18 C
# Has uncertainty from epsilon_0

planck_charge = Constant(
    symbol="q_P",
    name="Planck charge",
    value=1.8755460653237244e-18,
    unit=Unit("C", Dimension(current=1, time=1), 1.0),
    uncertainty=1.5e-28,  # from epsilon_0 uncertainty
)
q_P = planck_charge

# =============================================================================
# Hartree energy
# =============================================================================
# E_h = m_e * c^2 * alpha^2 = 2 * R_inf * h * c
# CODATA 2022: 4.3597447222(85) x 10^-18 J
# Relative uncertainty: 2.0 x 10^-10

hartree_energy = Constant(
    symbol="E_h",
    name="Hartree energy",
    value=4.3597447222e-18,
    unit=Unit("J", Dimension(mass=1, length=2, time=-2), 1.0),
    uncertainty=8.5e-28,
)
E_h = hartree_energy

__all__ = [
    "planck_length",
    "l_P",
    "planck_mass",
    "m_P",
    "planck_time",
    "t_P",
    "planck_temperature",
    "T_P",
    "planck_charge",
    "q_P",
    "hartree_energy",
    "E_h",
]
