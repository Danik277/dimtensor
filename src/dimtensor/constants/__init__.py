"""Physical constants from CODATA 2022.

This module provides fundamental physical constants with proper units
and uncertainty values based on the 2022 CODATA recommended values.

Basic Usage:
    >>> from dimtensor.constants import c, G, h, k_B, N_A
    >>> print(c)
    c = 299792458 m/s (exact)
    >>> print(G.uncertainty)
    1.5e-15

    >>> from dimtensor import DimArray, units
    >>> m = DimArray([1.0], units.kg)
    >>> E = m * c**2  # E = mc^2
    >>> print(E)
    [8.98755e+16] J

Domain-specific imports for additional constants:
    >>> from dimtensor.constants.electromagnetic import mu_0, epsilon_0, Phi_0
    >>> from dimtensor.constants.atomic import m_e, m_p, a_0, R_inf
    >>> from dimtensor.constants.physico_chemical import R, F, sigma
    >>> from dimtensor.constants.derived import l_P, m_P, t_P, T_P

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
    arXiv:2409.03787 (CODATA 2022 Technical Paper)
"""

from ._base import Constant

# Atomic constants (most commonly used)
from .atomic import (
    R_inf,
    a_0,
    bohr_radius,
    electron_mass,
    m_e,
    m_n,
    m_p,
    neutron_mass,
    proton_mass,
    rydberg_constant,
)

# Derived constants (Planck units)
from .derived import (
    T_P,
    E_h,
    hartree_energy,
    l_P,
    m_P,
    planck_length,
    planck_mass,
    planck_temperature,
    planck_time,
    t_P,
)

# Electromagnetic constants (most commonly used)
from .electromagnetic import (
    alpha,
    e,
    elementary_charge,
    epsilon_0,
    fine_structure_constant,
    mu_0,
    vacuum_permeability,
    vacuum_permittivity,
)

# Physico-chemical constants (most commonly used)
from .physico_chemical import (
    N_A,
    F,
    R,
    avogadro_constant,
    boltzmann_constant,
    faraday_constant,
    k_B,
    molar_gas_constant,
    sigma,
    stefan_boltzmann_constant,
)

# Universal constants
from .universal import (
    G,
    c,
    gravitational_constant,
    h,
    hbar,
    planck_constant,
    reduced_planck_constant,
    speed_of_light,
)

__all__ = [
    # Class
    "Constant",
    # Universal
    "speed_of_light",
    "c",
    "gravitational_constant",
    "G",
    "planck_constant",
    "h",
    "reduced_planck_constant",
    "hbar",
    # Electromagnetic
    "elementary_charge",
    "e",
    "vacuum_permeability",
    "mu_0",
    "vacuum_permittivity",
    "epsilon_0",
    "fine_structure_constant",
    "alpha",
    # Atomic
    "electron_mass",
    "m_e",
    "proton_mass",
    "m_p",
    "neutron_mass",
    "m_n",
    "bohr_radius",
    "a_0",
    "rydberg_constant",
    "R_inf",
    # Physico-chemical
    "avogadro_constant",
    "N_A",
    "boltzmann_constant",
    "k_B",
    "molar_gas_constant",
    "R",
    "faraday_constant",
    "F",
    "stefan_boltzmann_constant",
    "sigma",
    # Derived (Planck units)
    "planck_length",
    "l_P",
    "planck_mass",
    "m_P",
    "planck_time",
    "t_P",
    "planck_temperature",
    "T_P",
    "hartree_energy",
    "E_h",
]
