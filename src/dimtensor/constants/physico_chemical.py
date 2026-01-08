"""Physico-chemical constants from CODATA 2022.

This module provides physico-chemical constants including Avogadro's number,
Boltzmann constant, molar gas constant, Faraday constant, and Stefan-Boltzmann
constant.

Examples:
    >>> from dimtensor.constants.physico_chemical import N_A, k_B, R, F
    >>> print(N_A)
    N_A = 6.02214076e+23 1/mol (exact)
    >>> print(k_B)
    k_B = 1.380649e-23 J/K (exact)

References:
    CODATA 2022: https://physics.nist.gov/cuu/Constants/
"""

from ..core.dimensions import Dimension
from ..core.units import Unit
from ._base import Constant

# =============================================================================
# Avogadro constant
# =============================================================================
# Exact by definition since 2019 SI redefinition

avogadro_constant = Constant(
    symbol="N_A",
    name="Avogadro constant",
    value=6.02214076e23,
    unit=Unit("1/mol", Dimension(amount=-1), 1.0),
    uncertainty=0.0,  # exact
)
N_A = avogadro_constant

# =============================================================================
# Boltzmann constant
# =============================================================================
# Exact by definition since 2019 SI redefinition

boltzmann_constant = Constant(
    symbol="k_B",
    name="Boltzmann constant",
    value=1.380649e-23,
    unit=Unit("J/K", Dimension(mass=1, length=2, time=-2, temperature=-1), 1.0),
    uncertainty=0.0,  # exact
)
k_B = boltzmann_constant

# =============================================================================
# Molar gas constant
# =============================================================================
# R = N_A * k_B, exact since both are exact
# CODATA 2022: 8.314462618... J/(mol K)

molar_gas_constant = Constant(
    symbol="R",
    name="molar gas constant",
    value=8.314462618153024,  # N_A * k_B
    unit=Unit(
        "J/(mol K)",
        Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
        1.0,
    ),
    uncertainty=0.0,  # exact (derived from exact constants)
)
R = molar_gas_constant

# =============================================================================
# Faraday constant
# =============================================================================
# F = N_A * e, exact since both are exact
# CODATA 2022: 96485.33212... C/mol

faraday_constant = Constant(
    symbol="F",
    name="Faraday constant",
    value=96485.33212331001,  # N_A * e
    unit=Unit("C/mol", Dimension(current=1, time=1, amount=-1), 1.0),
    uncertainty=0.0,  # exact (derived from exact constants)
)
F = faraday_constant

# =============================================================================
# Stefan-Boltzmann constant
# =============================================================================
# sigma = (pi^2 / 60) * k_B^4 / (hbar^3 * c^2), exact since all inputs are exact
# CODATA 2022: 5.670374419... x 10^-8 W/(m^2 K^4)

stefan_boltzmann_constant = Constant(
    symbol="sigma",
    name="Stefan-Boltzmann constant",
    value=5.670374419184429e-8,
    unit=Unit(
        "W/(m^2 K^4)",
        Dimension(mass=1, length=0, time=-3, temperature=-4),
        1.0,
    ),
    uncertainty=0.0,  # exact (derived from exact constants)
)
sigma = stefan_boltzmann_constant

# =============================================================================
# First radiation constant
# =============================================================================
# c_1 = 2 * pi * h * c^2, exact since h and c are exact
# CODATA 2022: 3.741771852... x 10^-16 W m^2

first_radiation_constant = Constant(
    symbol="c_1",
    name="first radiation constant",
    value=3.7417718521927573e-16,
    unit=Unit("W m^2", Dimension(mass=1, length=4, time=-3), 1.0),
    uncertainty=0.0,  # exact
)
c_1 = first_radiation_constant

# =============================================================================
# Second radiation constant
# =============================================================================
# c_2 = h * c / k_B, exact since all are exact
# CODATA 2022: 1.438776877... x 10^-2 m K

second_radiation_constant = Constant(
    symbol="c_2",
    name="second radiation constant",
    value=1.4387768775039337e-2,
    unit=Unit("m K", Dimension(length=1, temperature=1), 1.0),
    uncertainty=0.0,  # exact
)
c_2 = second_radiation_constant

# =============================================================================
# Wien displacement law constant
# =============================================================================
# b = c_2 / 4.965114231744276 (from Wien's law)
# CODATA 2022: 2.897771955... x 10^-3 m K

wien_displacement_constant = Constant(
    symbol="b",
    name="Wien displacement law constant",
    value=2.897771955e-3,
    unit=Unit("m K", Dimension(length=1, temperature=1), 1.0),
    uncertainty=0.0,  # exact (derived from exact c_2)
)
b = wien_displacement_constant

__all__ = [
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
    "first_radiation_constant",
    "c_1",
    "second_radiation_constant",
    "c_2",
    "wien_displacement_constant",
    "b",
]
