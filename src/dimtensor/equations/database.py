"""Equation database for physics and engineering.

Provides a registry of physics equations with dimensional metadata
for use in dimensional analysis, validation, and physics-informed ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..core.dimensions import Dimension


@dataclass
class Equation:
    """A physics equation with dimensional metadata.

    Attributes:
        name: Human-readable name (e.g., "Newton's Second Law").
        formula: Symbolic formula (e.g., "F = ma").
        variables: Dict mapping variable names to their dimensions.
        domain: Physics domain (e.g., "mechanics").
        tags: List of tags for categorization.
        description: Longer description of the equation.
        assumptions: List of assumptions/conditions.
        latex: LaTeX representation.
        related: List of related equation names.
    """

    name: str
    formula: str
    variables: dict[str, Dimension]
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    assumptions: list[str] = field(default_factory=list)
    latex: str = ""
    related: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "variables": {k: str(v) for k, v in self.variables.items()},
            "domain": self.domain,
            "tags": self.tags,
            "description": self.description,
            "assumptions": self.assumptions,
            "latex": self.latex,
            "related": self.related,
        }


# Global equation registry
_EQUATIONS: dict[str, Equation] = {}


def register_equation(equation: Equation) -> None:
    """Register an equation in the database."""
    _EQUATIONS[equation.name] = equation


def get_equation(name: str) -> Equation:
    """Get an equation by name.

    Args:
        name: Equation name.

    Returns:
        Equation object.

    Raises:
        KeyError: If equation not found.
    """
    if name not in _EQUATIONS:
        raise KeyError(f"Equation '{name}' not found")
    return _EQUATIONS[name]


def get_equations(
    domain: str | None = None,
    tags: list[str] | None = None,
) -> list[Equation]:
    """Get equations, optionally filtered.

    Args:
        domain: Filter by physics domain.
        tags: Filter by tags (must have ALL tags).

    Returns:
        List of matching equations.
    """
    results = list(_EQUATIONS.values())

    if domain is not None:
        results = [eq for eq in results if eq.domain == domain]

    if tags is not None:
        results = [eq for eq in results if all(t in eq.tags for t in tags)]

    return results


def search_equations(query: str) -> list[Equation]:
    """Search equations by name, description, or variable names.

    Args:
        query: Search query (case-insensitive).

    Returns:
        List of matching equations.
    """
    query_lower = query.lower()
    results = []

    for eq in _EQUATIONS.values():
        if (
            query_lower in eq.name.lower()
            or query_lower in eq.description.lower()
            or query_lower in eq.formula.lower()
            or any(query_lower in var.lower() for var in eq.variables.keys())
            or any(query_lower in tag.lower() for tag in eq.tags)
        ):
            results.append(eq)

    return results


def list_domains() -> list[str]:
    """List all registered physics domains.

    Returns:
        Sorted list of unique domain names.
    """
    domains = set(eq.domain for eq in _EQUATIONS.values())
    return sorted(domains)


def clear_equations() -> None:
    """Clear all registered equations (for testing)."""
    _EQUATIONS.clear()


# ============================================================================
# MECHANICS EQUATIONS
# ============================================================================

# Dimensions
_L = Dimension(length=1)
_M = Dimension(mass=1)
_T = Dimension(time=1)
_V = Dimension(length=1, time=-1)
_A = Dimension(length=1, time=-2)
_F = Dimension(mass=1, length=1, time=-2)
_E = Dimension(mass=1, length=2, time=-2)
_P = Dimension(mass=1, length=1, time=-1)  # momentum
_W = Dimension(mass=1, length=2, time=-3)  # power
_DIMLESS = Dimension()

register_equation(Equation(
    name="Newton's Second Law",
    formula="F = ma",
    variables={"F": _F, "m": _M, "a": _A},
    domain="mechanics",
    tags=["newton", "force", "acceleration", "fundamental"],
    description="Force equals mass times acceleration",
    latex=r"F = ma",
    related=["Newton's First Law", "Newton's Third Law"],
))

register_equation(Equation(
    name="Kinetic Energy",
    formula="KE = (1/2)mv^2",
    variables={"KE": _E, "m": _M, "v": _V},
    domain="mechanics",
    tags=["energy", "kinetic", "motion"],
    description="Energy of motion",
    latex=r"KE = \frac{1}{2}mv^2",
    related=["Potential Energy", "Work-Energy Theorem"],
))

register_equation(Equation(
    name="Potential Energy (Gravitational)",
    formula="PE = mgh",
    variables={"PE": _E, "m": _M, "g": _A, "h": _L},
    domain="mechanics",
    tags=["energy", "potential", "gravity"],
    description="Gravitational potential energy near Earth's surface",
    latex=r"PE = mgh",
    assumptions=["Uniform gravitational field", "Near Earth's surface"],
    related=["Kinetic Energy", "Work-Energy Theorem"],
))

register_equation(Equation(
    name="Work",
    formula="W = F * d",
    variables={"W": _E, "F": _F, "d": _L},
    domain="mechanics",
    tags=["work", "energy", "force"],
    description="Work done by a constant force over a distance",
    latex=r"W = \vec{F} \cdot \vec{d}",
    assumptions=["Constant force", "Parallel to displacement"],
    related=["Power", "Kinetic Energy"],
))

register_equation(Equation(
    name="Power",
    formula="P = W/t",
    variables={"P": _W, "W": _E, "t": _T},
    domain="mechanics",
    tags=["power", "energy", "time"],
    description="Rate of doing work",
    latex=r"P = \frac{W}{t}",
    related=["Work", "Energy"],
))

register_equation(Equation(
    name="Momentum",
    formula="p = mv",
    variables={"p": _P, "m": _M, "v": _V},
    domain="mechanics",
    tags=["momentum", "motion", "mass"],
    description="Linear momentum",
    latex=r"\vec{p} = m\vec{v}",
    related=["Impulse", "Conservation of Momentum"],
))

register_equation(Equation(
    name="Impulse",
    formula="J = F * t",
    variables={"J": _P, "F": _F, "t": _T},
    domain="mechanics",
    tags=["impulse", "momentum", "force"],
    description="Change in momentum due to force over time",
    latex=r"\vec{J} = \vec{F} \Delta t",
    related=["Momentum", "Newton's Second Law"],
))

register_equation(Equation(
    name="Hooke's Law",
    formula="F = -kx",
    variables={"F": _F, "k": Dimension(mass=1, time=-2), "x": _L},
    domain="mechanics",
    tags=["spring", "elastic", "oscillation"],
    description="Force exerted by a spring",
    latex=r"F = -kx",
    assumptions=["Linear elastic region"],
    related=["Simple Harmonic Motion"],
))

register_equation(Equation(
    name="Gravitational Force",
    formula="F = G*m1*m2/r^2",
    variables={
        "F": _F,
        "G": Dimension(length=3, mass=-1, time=-2),
        "m1": _M, "m2": _M, "r": _L
    },
    domain="mechanics",
    tags=["gravity", "newton", "universal"],
    description="Newton's law of universal gravitation",
    latex=r"F = G\frac{m_1 m_2}{r^2}",
    related=["Gravitational Potential Energy"],
))

register_equation(Equation(
    name="Centripetal Acceleration",
    formula="a_c = v^2/r",
    variables={"a_c": _A, "v": _V, "r": _L},
    domain="mechanics",
    tags=["circular", "motion", "acceleration"],
    description="Acceleration toward center in circular motion",
    latex=r"a_c = \frac{v^2}{r}",
    related=["Centripetal Force"],
))

# ============================================================================
# THERMODYNAMICS EQUATIONS
# ============================================================================

_TEMP = Dimension(temperature=1)
_PRESSURE = Dimension(mass=1, length=-1, time=-2)
_VOLUME = Dimension(length=3)
_N_MOL = Dimension(amount=1)
_ENTROPY = Dimension(mass=1, length=2, time=-2, temperature=-1)
_HEAT_CAPACITY = Dimension(mass=1, length=2, time=-2, temperature=-1)

register_equation(Equation(
    name="Ideal Gas Law",
    formula="PV = nRT",
    variables={
        "P": _PRESSURE,
        "V": _VOLUME,
        "n": _N_MOL,
        "R": Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
        "T": _TEMP
    },
    domain="thermodynamics",
    tags=["gas", "ideal", "fundamental", "state"],
    description="Relates pressure, volume, and temperature of ideal gas",
    latex=r"PV = nRT",
    assumptions=["Ideal gas behavior", "No intermolecular forces"],
))

register_equation(Equation(
    name="First Law of Thermodynamics",
    formula="dU = Q - W",
    variables={"dU": _E, "Q": _E, "W": _E},
    domain="thermodynamics",
    tags=["energy", "conservation", "heat", "work", "fundamental"],
    description="Conservation of energy for thermodynamic systems",
    latex=r"\Delta U = Q - W",
))

register_equation(Equation(
    name="Heat Capacity",
    formula="Q = mc*dT",
    variables={"Q": _E, "m": _M, "c": _HEAT_CAPACITY / _M, "dT": _TEMP},
    domain="thermodynamics",
    tags=["heat", "temperature", "capacity"],
    description="Heat required to change temperature",
    latex=r"Q = mc\Delta T",
))

register_equation(Equation(
    name="Stefan-Boltzmann Law",
    formula="P = sigma*A*T^4",
    variables={
        "P": _W,
        "sigma": Dimension(mass=1, time=-3, temperature=-4),
        "A": Dimension(length=2),
        "T": _TEMP
    },
    domain="thermodynamics",
    tags=["radiation", "blackbody", "heat"],
    description="Power radiated by a blackbody",
    latex=r"P = \sigma A T^4",
))

register_equation(Equation(
    name="Entropy Change",
    formula="dS = dQ/T",
    variables={"dS": _ENTROPY, "dQ": _E, "T": _TEMP},
    domain="thermodynamics",
    tags=["entropy", "heat", "reversible"],
    description="Entropy change for reversible process",
    latex=r"dS = \frac{dQ}{T}",
    assumptions=["Reversible process"],
))
# ============================================================================
# STATISTICAL MECHANICS EQUATIONS
# ============================================================================

# Additional dimensions for statistical mechanics
_BOLTZMANN_CONST = Dimension(mass=1, length=2, time=-2, temperature=-1)
_CHEMICAL_POTENTIAL = Dimension(mass=1, length=2, time=-2)  # energy
_NUMBER_DENSITY = Dimension(length=-3)
_PROBABILITY = _DIMLESS

register_equation(Equation(
    name="Boltzmann Distribution",
    formula="P_i = exp(-E_i/kT) / Z",
    variables={
        "P_i": _PROBABILITY,
        "E_i": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP,
        "Z": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["boltzmann", "canonical", "ensemble", "probability", "fundamental"],
    description="Probability of finding system in state i in canonical ensemble",
    latex=r"P_i = \frac{e^{-E_i/k_B T}}{Z}",
    assumptions=["Canonical ensemble (N,V,T fixed)", "Thermal equilibrium"],
    related=["Canonical Partition Function", "Helmholtz Free Energy"],
))

register_equation(Equation(
    name="Canonical Partition Function",
    formula="Z = sum(exp(-E_i/kT))",
    variables={
        "Z": _DIMLESS,
        "E_i": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["partition function", "canonical", "ensemble", "fundamental"],
    description="Sum over all states weighted by Boltzmann factor",
    latex=r"Z = \sum_i e^{-E_i/k_B T}",
    assumptions=["Canonical ensemble (N,V,T fixed)", "Discrete energy levels"],
    related=["Boltzmann Distribution", "Helmholtz Free Energy from Partition Function"],
))

register_equation(Equation(
    name="Grand Canonical Partition Function",
    formula="Xi = sum(exp((mu*N - E)/kT))",
    variables={
        "Xi": _DIMLESS,
        "mu": _CHEMICAL_POTENTIAL,
        "N": _DIMLESS,  # particle number
        "E": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["partition function", "grand canonical", "ensemble"],
    description="Sum over all states and particle numbers",
    latex=r"\Xi = \sum_{N,i} e^{(\mu N - E_i)/k_B T}",
    assumptions=["Grand canonical ensemble (mu,V,T fixed)"],
    related=["Grand Potential", "Chemical Potential"],
))

register_equation(Equation(
    name="Microcanonical Density of States",
    formula="Omega(E,V,N) = count of states",
    variables={
        "Omega": _DIMLESS,
        "E": _E,
        "V": _VOLUME,
        "N": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["partition function", "microcanonical", "ensemble", "density of states"],
    description="Number of microstates with given energy, volume, and particle number",
    latex=r"\Omega(E,V,N)",
    assumptions=["Microcanonical ensemble (E,V,N fixed)", "Isolated system"],
    related=["Boltzmann Entropy"],
))

register_equation(Equation(
    name="Fermi-Dirac Distribution",
    formula="n(E) = 1/(exp((E-mu)/kT) + 1)",
    variables={
        "n": _DIMLESS,
        "E": _E,
        "mu": _CHEMICAL_POTENTIAL,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["fermi-dirac", "fermion", "quantum", "distribution", "occupation"],
    description="Average occupation number for fermions (particles with half-integer spin)",
    latex=r"n(E) = \frac{1}{e^{(E-\mu)/k_B T} + 1}",
    assumptions=["Quantum statistics", "Non-interacting fermions", "Grand canonical ensemble"],
    related=["Bose-Einstein Distribution", "Chemical Potential"],
))

register_equation(Equation(
    name="Fermi Function at T=0",
    formula="n(E) = 1 if E < mu else 0",
    variables={
        "n": _DIMLESS,
        "E": _E,
        "mu": _CHEMICAL_POTENTIAL
    },
    domain="statistical_mechanics",
    tags=["fermi-dirac", "fermion", "quantum", "zero temperature", "step function"],
    description="Fermi-Dirac distribution at zero temperature (step function)",
    latex=r"n(E) = \begin{cases} 1 & E < \mu \\ 0 & E > \mu \end{cases}",
    assumptions=["T = 0", "Non-interacting fermions"],
    related=["Fermi-Dirac Distribution"],
))

register_equation(Equation(
    name="Bose-Einstein Distribution",
    formula="n(E) = 1/(exp((E-mu)/kT) - 1)",
    variables={
        "n": _DIMLESS,
        "E": _E,
        "mu": _CHEMICAL_POTENTIAL,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["bose-einstein", "boson", "quantum", "distribution", "occupation"],
    description="Average occupation number for bosons (particles with integer spin)",
    latex=r"n(E) = \frac{1}{e^{(E-\mu)/k_B T} - 1}",
    assumptions=["Quantum statistics", "Non-interacting bosons", "mu < E_min"],
    related=["Fermi-Dirac Distribution", "Planck Distribution"],
))

register_equation(Equation(
    name="Planck Distribution",
    formula="n(E) = 1/(exp(E/kT) - 1)",
    variables={
        "n": _DIMLESS,
        "E": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["planck", "photon", "boson", "quantum", "distribution", "blackbody"],
    description="Occupation number for photons (bosons with mu = 0)",
    latex=r"n(E) = \frac{1}{e^{E/k_B T} - 1}",
    assumptions=["Photons (massless bosons)", "Chemical potential mu = 0"],
    related=["Bose-Einstein Distribution"],
))

register_equation(Equation(
    name="Helmholtz Free Energy",
    formula="A = U - TS",
    variables={
        "A": _E,
        "U": _E,
        "T": _TEMP,
        "S": _ENTROPY
    },
    domain="statistical_mechanics",
    tags=["free energy", "helmholtz", "canonical", "thermodynamic potential"],
    description="Free energy for canonical ensemble (natural variables: T, V, N)",
    latex=r"A = U - TS",
    assumptions=["Equilibrium thermodynamics"],
    related=["Helmholtz Free Energy from Partition Function", "Gibbs Free Energy", "First Law of Thermodynamics"],
))

register_equation(Equation(
    name="Helmholtz Free Energy from Partition Function",
    formula="A = -kT*ln(Z)",
    variables={
        "A": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP,
        "Z": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["free energy", "helmholtz", "canonical", "partition function"],
    description="Relation between Helmholtz free energy and canonical partition function",
    latex=r"A = -k_B T \ln Z",
    assumptions=["Canonical ensemble"],
    related=["Canonical Partition Function", "Helmholtz Free Energy"],
))

register_equation(Equation(
    name="Gibbs Free Energy",
    formula="G = H - TS",
    variables={
        "G": _E,
        "H": _E,  # enthalpy
        "T": _TEMP,
        "S": _ENTROPY
    },
    domain="statistical_mechanics",
    tags=["free energy", "gibbs", "thermodynamic potential"],
    description="Free energy for systems at constant pressure (natural variables: T, P, N)",
    latex=r"G = H - TS",
    assumptions=["Equilibrium thermodynamics"],
    related=["Helmholtz Free Energy", "Chemical Potential"],
))

register_equation(Equation(
    name="Grand Potential",
    formula="Omega = U - TS - mu*N",
    variables={
        "Omega": _E,
        "U": _E,
        "T": _TEMP,
        "S": _ENTROPY,
        "mu": _CHEMICAL_POTENTIAL,
        "N": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["free energy", "grand potential", "grand canonical"],
    description="Thermodynamic potential for grand canonical ensemble (natural variables: mu, V, T)",
    latex=r"\Omega = U - TS - \mu N",
    assumptions=["Grand canonical ensemble"],
    related=["Grand Canonical Partition Function"],
))

register_equation(Equation(
    name="Grand Potential from Partition Function",
    formula="Omega = -kT*ln(Xi)",
    variables={
        "Omega": _E,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP,
        "Xi": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["free energy", "grand potential", "grand canonical", "partition function"],
    description="Relation between grand potential and grand canonical partition function",
    latex=r"\Omega = -k_B T \ln \Xi",
    assumptions=["Grand canonical ensemble"],
    related=["Grand Canonical Partition Function", "Grand Potential"],
))

register_equation(Equation(
    name="Boltzmann Entropy",
    formula="S = k*ln(Omega)",
    variables={
        "S": _ENTROPY,
        "k": _BOLTZMANN_CONST,
        "Omega": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["entropy", "boltzmann", "fundamental", "microcanonical"],
    description="Entropy as measure of number of accessible microstates",
    latex=r"S = k_B \ln \Omega",
    assumptions=["All microstates equally probable", "Microcanonical ensemble"],
    related=["Gibbs Entropy", "Entropy Change"],
))

register_equation(Equation(
    name="Gibbs Entropy",
    formula="S = -k*sum(P_i*ln(P_i))",
    variables={
        "S": _ENTROPY,
        "k": _BOLTZMANN_CONST,
        "P_i": _PROBABILITY
    },
    domain="statistical_mechanics",
    tags=["entropy", "gibbs", "canonical", "information"],
    description="Entropy for non-uniform probability distribution over microstates",
    latex=r"S = -k_B \sum_i P_i \ln P_i",
    assumptions=["Canonical or grand canonical ensemble"],
    related=["Boltzmann Entropy", "Von Neumann Entropy"],
))

register_equation(Equation(
    name="Von Neumann Entropy",
    formula="S = -k*Tr(rho*ln(rho))",
    variables={
        "S": _ENTROPY,
        "k": _BOLTZMANN_CONST,
        "rho": _DIMLESS  # density matrix
    },
    domain="statistical_mechanics",
    tags=["entropy", "von neumann", "quantum", "density matrix"],
    description="Quantum mechanical entropy for mixed states",
    latex=r"S = -k_B \text{Tr}(\rho \ln \rho)",
    assumptions=["Quantum system", "Density matrix formalism"],
    related=["Gibbs Entropy"],
))

register_equation(Equation(
    name="Maxwell-Boltzmann Speed Distribution",
    formula="f(v) = 4*pi*n*(m/(2*pi*kT))^(3/2)*v^2*exp(-m*v^2/(2*kT))",
    variables={
        "f": Dimension(time=1, length=-1),  # probability per unit speed
        "v": _V,
        "n": _NUMBER_DENSITY,
        "m": _M,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["maxwell-boltzmann", "classical", "distribution", "gas", "kinetic theory"],
    description="Speed distribution of particles in classical ideal gas",
    latex=r"f(v) = 4\pi n \left(\frac{m}{2\pi k_B T}\right)^{3/2} v^2 e^{-mv^2/2k_B T}",
    assumptions=["Classical limit", "Ideal gas", "Thermal equilibrium"],
    related=["Maxwell-Boltzmann Energy Distribution"],
))

register_equation(Equation(
    name="Maxwell-Boltzmann Energy Distribution",
    formula="f(E) = 2*pi*n*(pi*kT)^(-3/2)*E^(1/2)*exp(-E/kT)",
    variables={
        "f": Dimension(mass=-1, length=-2, time=2),  # probability per unit energy
        "E": _E,
        "n": _NUMBER_DENSITY,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["maxwell-boltzmann", "classical", "distribution", "gas", "energy"],
    description="Energy distribution of particles in classical ideal gas",
    latex=r"f(E) = 2\pi n (\pi k_B T)^{-3/2} E^{1/2} e^{-E/k_B T}",
    assumptions=["Classical limit", "Ideal gas", "Thermal equilibrium"],
    related=["Maxwell-Boltzmann Speed Distribution"],
))

register_equation(Equation(
    name="Chemical Potential (Thermodynamic)",
    formula="mu = dG/dN",
    variables={
        "mu": _CHEMICAL_POTENTIAL,
        "G": _E,
        "N": _DIMLESS
    },
    domain="statistical_mechanics",
    tags=["chemical potential", "thermodynamic", "gibbs"],
    description="Change in Gibbs free energy per particle added at constant T and P",
    latex=r"\mu = \left(\frac{\partial G}{\partial N}\right)_{T,P}",
    assumptions=["Constant temperature and pressure"],
    related=["Gibbs Free Energy", "Fermi-Dirac Distribution", "Bose-Einstein Distribution"],
))

register_equation(Equation(
    name="Chemical Potential (Ideal Gas)",
    formula="mu = kT*ln(n/n_Q)",
    variables={
        "mu": _CHEMICAL_POTENTIAL,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP,
        "n": _NUMBER_DENSITY,
        "n_Q": _NUMBER_DENSITY  # quantum concentration
    },
    domain="statistical_mechanics",
    tags=["chemical potential", "ideal gas", "classical"],
    description="Chemical potential for classical ideal gas",
    latex=r"\mu = k_B T \ln\left(\frac{n}{n_Q}\right)",
    assumptions=["Classical ideal gas", "n_Q = (mkT/2pi*hbar^2)^(3/2)"],
    related=["Chemical Potential (Thermodynamic)"],
))

register_equation(Equation(
    name="Internal Energy from Partition Function",
    formula="U = -d(ln(Z))/d(beta)",
    variables={
        "U": _E,
        "Z": _DIMLESS,
        "beta": Dimension(mass=-1, length=-2, time=2, temperature=1)  # 1/kT
    },
    domain="statistical_mechanics",
    tags=["internal energy", "partition function", "canonical"],
    description="Internal energy from canonical partition function derivative",
    latex=r"U = -\frac{\partial \ln Z}{\partial \beta}",
    assumptions=["Canonical ensemble", "beta = 1/k_B T"],
    related=["Canonical Partition Function"],
))

register_equation(Equation(
    name="Pressure from Partition Function",
    formula="P = kT*d(ln(Z))/dV",
    variables={
        "P": _PRESSURE,
        "k": _BOLTZMANN_CONST,
        "T": _TEMP,
        "Z": _DIMLESS,
        "V": _VOLUME
    },
    domain="statistical_mechanics",
    tags=["pressure", "partition function", "canonical"],
    description="Pressure from canonical partition function derivative",
    latex=r"P = k_B T \frac{\partial \ln Z}{\partial V}",
    assumptions=["Canonical ensemble"],
    related=["Canonical Partition Function"],
))

register_equation(Equation(
    name="Equipartition Theorem",
    formula="E_avg = (f/2)*kT",
    variables={
        "E_avg": _E,
        "f": _DIMLESS,  # degrees of freedom
        "k": _BOLTZMANN_CONST,
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["equipartition", "classical", "energy", "fundamental"],
    description="Average energy per quadratic degree of freedom in classical system",
    latex=r"\langle E \rangle = \frac{f}{2} k_B T",
    assumptions=["Classical limit", "Quadratic degrees of freedom", "Thermal equilibrium"],
    related=["Maxwell-Boltzmann Speed Distribution"],
))


# ============================================================================
# ELECTROMAGNETISM EQUATIONS
# ============================================================================

_CHARGE = Dimension(current=1, time=1)
_VOLTAGE = Dimension(mass=1, length=2, time=-3, current=-1)
_CURRENT = Dimension(current=1)
_RESISTANCE = Dimension(mass=1, length=2, time=-3, current=-2)
_CAPACITANCE = Dimension(mass=-1, length=-2, time=4, current=2)
_INDUCTANCE = Dimension(mass=1, length=2, time=-2, current=-2)
_E_FIELD = Dimension(mass=1, length=1, time=-3, current=-1)
_B_FIELD = Dimension(mass=1, time=-2, current=-1)

register_equation(Equation(
    name="Ohm's Law",
    formula="V = IR",
    variables={"V": _VOLTAGE, "I": _CURRENT, "R": _RESISTANCE},
    domain="electromagnetism",
    tags=["circuits", "resistance", "current", "fundamental"],
    description="Voltage equals current times resistance",
    latex=r"V = IR",
    related=["Electric Power", "Kirchhoff's Laws"],
))

register_equation(Equation(
    name="Electric Power",
    formula="P = IV",
    variables={"P": _W, "I": _CURRENT, "V": _VOLTAGE},
    domain="electromagnetism",
    tags=["power", "circuits", "current"],
    description="Electrical power dissipation",
    latex=r"P = IV",
    related=["Ohm's Law"],
))

register_equation(Equation(
    name="Coulomb's Law",
    formula="F = k*q1*q2/r^2",
    variables={
        "F": _F,
        "k": Dimension(mass=1, length=3, time=-4, current=-2),
        "q1": _CHARGE, "q2": _CHARGE, "r": _L
    },
    domain="electromagnetism",
    tags=["electrostatic", "force", "charge"],
    description="Force between two point charges",
    latex=r"F = k\frac{q_1 q_2}{r^2}",
))

register_equation(Equation(
    name="Capacitor Energy",
    formula="E = (1/2)CV^2",
    variables={"E": _E, "C": _CAPACITANCE, "V": _VOLTAGE},
    domain="electromagnetism",
    tags=["capacitor", "energy", "circuits"],
    description="Energy stored in a capacitor",
    latex=r"E = \frac{1}{2}CV^2",
))

register_equation(Equation(
    name="Inductor Energy",
    formula="E = (1/2)LI^2",
    variables={"E": _E, "L": _INDUCTANCE, "I": _CURRENT},
    domain="electromagnetism",
    tags=["inductor", "energy", "circuits"],
    description="Energy stored in an inductor",
    latex=r"E = \frac{1}{2}LI^2",
))

register_equation(Equation(
    name="Lorentz Force",
    formula="F = q(E + v x B)",
    variables={"F": _F, "q": _CHARGE, "E": _E_FIELD, "v": _V, "B": _B_FIELD},
    domain="electromagnetism",
    tags=["force", "magnetic", "electric", "charged particle"],
    description="Force on a charged particle in electromagnetic field",
    latex=r"\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})",
))

# ============================================================================
# FLUID DYNAMICS EQUATIONS
# ============================================================================

_DENSITY = Dimension(mass=1, length=-3)
_VISCOSITY = Dimension(mass=1, length=-1, time=-1)

register_equation(Equation(
    name="Bernoulli's Equation",
    formula="P + (1/2)rho*v^2 + rho*g*h = const",
    variables={
        "P": _PRESSURE,
        "rho": _DENSITY,
        "v": _V,
        "g": _A,
        "h": _L
    },
    domain="fluid_dynamics",
    tags=["fluid", "pressure", "energy", "flow"],
    description="Conservation of energy in fluid flow",
    latex=r"P + \frac{1}{2}\rho v^2 + \rho gh = \text{const}",
    assumptions=["Incompressible", "Inviscid", "Steady flow", "Along streamline"],
))

register_equation(Equation(
    name="Continuity Equation",
    formula="A1*v1 = A2*v2",
    variables={
        "A1": Dimension(length=2), "v1": _V,
        "A2": Dimension(length=2), "v2": _V
    },
    domain="fluid_dynamics",
    tags=["fluid", "mass", "conservation", "flow"],
    description="Mass conservation in fluid flow",
    latex=r"A_1 v_1 = A_2 v_2",
    assumptions=["Incompressible", "Steady flow"],
))

register_equation(Equation(
    name="Reynolds Number",
    formula="Re = rho*v*L/mu",
    variables={
        "Re": _DIMLESS,
        "rho": _DENSITY,
        "v": _V,
        "L": _L,
        "mu": _VISCOSITY
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "turbulence", "laminar", "flow"],
    description="Ratio of inertial to viscous forces",
    latex=r"Re = \frac{\rho v L}{\mu}",
))

register_equation(Equation(
    name="Navier-Stokes (incompressible)",
    formula="rho(dv/dt + v.grad(v)) = -grad(P) + mu*laplacian(v) + f",
    variables={
        "rho": _DENSITY,
        "v": _V,
        "P": _PRESSURE,
        "mu": _VISCOSITY,
        "f": Dimension(mass=1, length=-2, time=-2)  # force per volume
    },
    domain="fluid_dynamics",
    tags=["pde", "momentum", "viscous", "fundamental"],
    description="Momentum equation for incompressible viscous flow",
    latex=r"\rho\left(\frac{\partial \vec{v}}{\partial t} + \vec{v} \cdot \nabla\vec{v}\right) = -\nabla P + \mu \nabla^2 \vec{v} + \vec{f}",
    assumptions=["Incompressible", "Newtonian fluid"],
))

# ============================================================================
# SPECIAL RELATIVITY
# ============================================================================

register_equation(Equation(
    name="Mass-Energy Equivalence",
    formula="E = mc^2",
    variables={
        "E": _E,
        "m": _M,
        "c": _V  # speed of light
    },
    domain="relativity",
    tags=["einstein", "energy", "mass", "fundamental"],
    description="Rest energy of a massive object",
    latex=r"E = mc^2",
))

register_equation(Equation(
    name="Lorentz Factor",
    formula="gamma = 1/sqrt(1 - v^2/c^2)",
    variables={"gamma": _DIMLESS, "v": _V, "c": _V},
    domain="relativity",
    tags=["lorentz", "time dilation", "length contraction"],
    description="Relativistic factor for time dilation and length contraction",
    latex=r"\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}",
))

# ============================================================================
# QUANTUM MECHANICS
# ============================================================================

_ANGULAR_FREQ = Dimension(time=-1)
_WAVE_NUMBER = Dimension(length=-1)

register_equation(Equation(
    name="Planck-Einstein Relation",
    formula="E = hbar * omega",
    variables={
        "E": _E,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "omega": _ANGULAR_FREQ
    },
    domain="quantum",
    tags=["photon", "energy", "frequency", "fundamental"],
    description="Energy of a photon",
    latex=r"E = \hbar\omega",
))

register_equation(Equation(
    name="de Broglie Wavelength",
    formula="lambda = h/p",
    variables={
        "lambda": _L,
        "h": Dimension(mass=1, length=2, time=-1),
        "p": _P
    },
    domain="quantum",
    tags=["wave-particle", "wavelength", "momentum"],
    description="Wavelength associated with a particle",
    latex=r"\lambda = \frac{h}{p}",
))

register_equation(Equation(
    name="Heisenberg Uncertainty (position-momentum)",
    formula="delta_x * delta_p >= hbar/2",
    variables={
        "delta_x": _L,
        "delta_p": _P,
        "hbar": Dimension(mass=1, length=2, time=-1)
    },
    domain="quantum",
    tags=["uncertainty", "fundamental", "measurement"],
    description="Fundamental limit on simultaneous measurement precision",
    latex=r"\Delta x \Delta p \geq \frac{\hbar}{2}",
))

register_equation(Equation(
    name="Heisenberg Uncertainty (energy-time)",
    formula="delta_E * delta_t >= hbar/2",
    variables={
        "delta_E": _E,
        "delta_t": _T,
        "hbar": Dimension(mass=1, length=2, time=-1)
    },
    domain="quantum",
    tags=["uncertainty", "fundamental", "measurement", "energy"],
    description="Uncertainty relation between energy and time",
    latex=r"\Delta E \Delta t \geq \frac{\hbar}{2}",
))

register_equation(Equation(
    name="Schrödinger Equation (time-independent)",
    formula="H*psi = E*psi",
    variables={
        "H": Dimension(mass=1, length=2, time=-2),  # Hamiltonian (energy)
        "psi": _DIMLESS,  # wavefunction (normalized)
        "E": _E
    },
    domain="quantum",
    tags=["schrodinger", "eigenvalue", "fundamental"],
    description="Time-independent Schrödinger equation for stationary states",
    latex=r"\hat{H}\psi = E\psi",
))

register_equation(Equation(
    name="Compton Scattering",
    formula="delta_lambda = (h/(m_e*c))*(1 - cos(theta))",
    variables={
        "delta_lambda": _L,
        "h": Dimension(mass=1, length=2, time=-1),
        "m_e": _M,
        "c": _V,
        "theta": _DIMLESS
    },
    domain="quantum",
    tags=["photon", "scattering", "wavelength"],
    description="Wavelength shift in photon-electron scattering",
    latex=r"\Delta\lambda = \frac{h}{m_e c}(1 - \cos\theta)",
))

register_equation(Equation(
    name="Bohr Radius",
    formula="a_0 = hbar^2/(m_e*e^2*k_e)",
    variables={
        "a_0": _L,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "m_e": _M,
        "e": _CHARGE,
        "k_e": Dimension(mass=1, length=3, time=-4, current=-2)
    },
    domain="quantum",
    tags=["atomic", "hydrogen", "fundamental"],
    description="Radius of lowest-energy electron orbit in hydrogen",
    latex=r"a_0 = \frac{\hbar^2}{m_e e^2 k_e}",
))

register_equation(Equation(
    name="Rydberg Formula",
    formula="1/lambda = R*(1/n1^2 - 1/n2^2)",
    variables={
        "lambda": _L,
        "R": _WAVE_NUMBER,
        "n1": _DIMLESS,
        "n2": _DIMLESS
    },
    domain="quantum",
    tags=["atomic", "spectroscopy", "hydrogen"],
    description="Wavelengths of spectral lines in hydrogen",
    latex=r"\frac{1}{\lambda} = R\left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right)",
))

register_equation(Equation(
    name="Quantum Harmonic Oscillator Energy",
    formula="E_n = hbar*omega*(n + 1/2)",
    variables={
        "E_n": _E,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "omega": _ANGULAR_FREQ,
        "n": _DIMLESS
    },
    domain="quantum",
    tags=["oscillator", "energy", "quantization"],
    description="Energy levels of quantum harmonic oscillator",
    latex=r"E_n = \hbar\omega\left(n + \frac{1}{2}\right)",
))

# ============================================================================
# RELATIVITY (EXPANDED)
# ============================================================================

register_equation(Equation(
    name="Time Dilation",
    formula="delta_t = gamma * delta_t0",
    variables={
        "delta_t": _T,
        "gamma": _DIMLESS,
        "delta_t0": _T
    },
    domain="relativity",
    tags=["time", "lorentz", "special relativity"],
    description="Time interval in moving frame appears longer",
    latex=r"\Delta t = \gamma \Delta t_0",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Length Contraction",
    formula="L = L0/gamma",
    variables={
        "L": _L,
        "L0": _L,
        "gamma": _DIMLESS
    },
    domain="relativity",
    tags=["length", "lorentz", "special relativity"],
    description="Length in direction of motion appears contracted",
    latex=r"L = \frac{L_0}{\gamma}",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Relativistic Momentum",
    formula="p = gamma*m*v",
    variables={
        "p": _P,
        "gamma": _DIMLESS,
        "m": _M,
        "v": _V
    },
    domain="relativity",
    tags=["momentum", "special relativity"],
    description="Momentum of a relativistic particle",
    latex=r"\vec{p} = \gamma m\vec{v}",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Relativistic Energy",
    formula="E = gamma*m*c^2",
    variables={
        "E": _E,
        "gamma": _DIMLESS,
        "m": _M,
        "c": _V
    },
    domain="relativity",
    tags=["energy", "special relativity"],
    description="Total energy of a relativistic particle",
    latex=r"E = \gamma mc^2",
    related=["Mass-Energy Equivalence", "Lorentz Factor"],
))

register_equation(Equation(
    name="Energy-Momentum Relation",
    formula="E^2 = (pc)^2 + (mc^2)^2",
    variables={
        "E": _E,
        "p": _P,
        "m": _M,
        "c": _V
    },
    domain="relativity",
    tags=["energy", "momentum", "special relativity"],
    description="Relation between energy and momentum in special relativity",
    latex=r"E^2 = (pc)^2 + (mc^2)^2",
    related=["Mass-Energy Equivalence"],
))

register_equation(Equation(
    name="Doppler Effect (relativistic)",
    formula="f = f0*sqrt((1-beta)/(1+beta))",
    variables={
        "f": Dimension(time=-1),
        "f0": Dimension(time=-1),
        "beta": _DIMLESS  # v/c
    },
    domain="relativity",
    tags=["doppler", "frequency", "special relativity"],
    description="Frequency shift for source moving radially",
    latex=r"f = f_0\sqrt{\frac{1-\beta}{1+\beta}}",
))

register_equation(Equation(
    name="Schwarzschild Radius",
    formula="r_s = 2*G*M/c^2",
    variables={
        "r_s": _L,
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "c": _V
    },
    domain="relativity",
    tags=["black hole", "gravity", "general relativity"],
    description="Radius of the event horizon of a non-rotating black hole",
    latex=r"r_s = \frac{2GM}{c^2}",
))

# ============================================================================
# GENERAL RELATIVITY
# ============================================================================

# Dimensions for GR quantities
_CURVATURE = Dimension(time=-2)  # Ricci scalar
_ENERGY_DENSITY = Dimension(mass=1, length=-3)
_HUBBLE = Dimension(time=-1)
_COSMOLOGICAL_CONST = Dimension(length=-2)
_STRAIN = _DIMLESS  # gravitational wave strain

# Schwarzschild Solution
register_equation(Equation(
    name="Schwarzschild Metric (time component)",
    formula="g_tt = -(1 - r_s/r)",
    variables={
        "g_tt": _DIMLESS,
        "r_s": _L,
        "r": _L
    },
    domain="general_relativity",
    tags=["schwarzschild", "metric", "black hole", "GR"],
    description="Time component of Schwarzschild metric in standard coordinates",
    latex=r"g_{tt} = -\left(1 - \frac{r_s}{r}\right)",
    assumptions=["Metric signature (-+++)", "Schwarzschild coordinates", "r > r_s"],
    related=["Schwarzschild Radius", "Schwarzschild Metric (radial component)"],
))

register_equation(Equation(
    name="Schwarzschild Metric (radial component)",
    formula="g_rr = 1/(1 - r_s/r)",
    variables={
        "g_rr": _DIMLESS,
        "r_s": _L,
        "r": _L
    },
    domain="general_relativity",
    tags=["schwarzschild", "metric", "black hole", "GR"],
    description="Radial component of Schwarzschild metric in standard coordinates",
    latex=r"g_{rr} = \frac{1}{1 - \frac{r_s}{r}}",
    assumptions=["Metric signature (-+++)", "Schwarzschild coordinates", "r > r_s"],
    related=["Schwarzschild Radius", "Schwarzschild Metric (time component)"],
))

register_equation(Equation(
    name="Gravitational Time Dilation",
    formula="tau = t*sqrt(1 - r_s/r)",
    variables={
        "tau": _T,  # proper time
        "t": _T,    # coordinate time
        "r_s": _L,
        "r": _L
    },
    domain="general_relativity",
    tags=["time dilation", "schwarzschild", "black hole", "GR"],
    description="Proper time relative to coordinate time near massive object",
    latex=r"\tau = t\sqrt{1 - \frac{r_s}{r}}",
    assumptions=["Schwarzschild geometry", "r > r_s"],
    related=["Schwarzschild Radius", "Gravitational Redshift"],
))

register_equation(Equation(
    name="Gravitational Redshift",
    formula="z = 1/sqrt(1 - r_s/r) - 1",
    variables={
        "z": _DIMLESS,
        "r_s": _L,
        "r": _L
    },
    domain="general_relativity",
    tags=["redshift", "schwarzschild", "black hole", "GR"],
    description="Gravitational redshift of light escaping from radius r",
    latex=r"z = \frac{1}{\sqrt{1 - \frac{r_s}{r}}} - 1",
    assumptions=["Schwarzschild geometry", "r > r_s"],
    related=["Schwarzschild Radius", "Gravitational Time Dilation"],
))

# Friedmann Equations (Cosmology)
register_equation(Equation(
    name="First Friedmann Equation",
    formula="H^2 = (8*pi*G/3)*rho - k*c^2/a^2 + Lambda*c^2/3",
    variables={
        "H": _HUBBLE,
        "G": Dimension(length=3, mass=-1, time=-2),
        "rho": _ENERGY_DENSITY,
        "k": Dimension(length=-2),  # curvature
        "c": _V,
        "a": _DIMLESS,  # scale factor (relative)
        "Lambda": _COSMOLOGICAL_CONST
    },
    domain="general_relativity",
    tags=["cosmology", "friedmann", "flrw", "universe", "GR"],
    description="Expansion rate of the universe in FLRW cosmology",
    latex=r"H^2 = \frac{8\pi G}{3}\rho - \frac{kc^2}{a^2} + \frac{\Lambda c^2}{3}",
    assumptions=["Homogeneous and isotropic universe", "FLRW metric"],
    related=["Second Friedmann Equation", "Critical Density"],
))

register_equation(Equation(
    name="Second Friedmann Equation",
    formula="a_dot_dot/a = -(4*pi*G/3)*(rho + 3*p/c^2) + Lambda*c^2/3",
    variables={
        "a_dot_dot": Dimension(time=-2),  # second derivative of scale factor
        "a": _DIMLESS,
        "G": Dimension(length=3, mass=-1, time=-2),
        "rho": _ENERGY_DENSITY,
        "p": _PRESSURE,
        "c": _V,
        "Lambda": _COSMOLOGICAL_CONST
    },
    domain="general_relativity",
    tags=["cosmology", "friedmann", "acceleration", "flrw", "GR"],
    description="Acceleration equation for scale factor in FLRW cosmology",
    latex=r"\frac{\ddot{a}}{a} = -\frac{4\pi G}{3}\left(\rho + \frac{3p}{c^2}\right) + \frac{\Lambda c^2}{3}",
    assumptions=["Homogeneous and isotropic universe", "FLRW metric"],
    related=["First Friedmann Equation"],
))

register_equation(Equation(
    name="Fluid Equation (Cosmology)",
    formula="rho_dot + 3*H*(rho + p/c^2) = 0",
    variables={
        "rho_dot": Dimension(mass=1, length=-3, time=-1),  # time derivative
        "H": _HUBBLE,
        "rho": _ENERGY_DENSITY,
        "p": _PRESSURE,
        "c": _V
    },
    domain="general_relativity",
    tags=["cosmology", "conservation", "flrw", "GR"],
    description="Energy conservation in expanding universe",
    latex=r"\dot{\rho} + 3H\left(\rho + \frac{p}{c^2}\right) = 0",
    assumptions=["Homogeneous and isotropic universe", "FLRW metric"],
    related=["First Friedmann Equation"],
))

register_equation(Equation(
    name="Critical Density",
    formula="rho_c = 3*H^2/(8*pi*G)",
    variables={
        "rho_c": _ENERGY_DENSITY,
        "H": _HUBBLE,
        "G": Dimension(length=3, mass=-1, time=-2)
    },
    domain="general_relativity",
    tags=["cosmology", "density", "critical", "GR"],
    description="Density required for flat universe (k=0)",
    latex=r"\rho_c = \frac{3H^2}{8\pi G}",
    related=["First Friedmann Equation", "Density Parameter"],
))

register_equation(Equation(
    name="Density Parameter",
    formula="Omega = rho/rho_c",
    variables={
        "Omega": _DIMLESS,
        "rho": _ENERGY_DENSITY,
        "rho_c": _ENERGY_DENSITY
    },
    domain="general_relativity",
    tags=["cosmology", "density", "parameter", "GR"],
    description="Ratio of density to critical density",
    latex=r"\Omega = \frac{\rho}{\rho_c}",
    related=["Critical Density"],
))

register_equation(Equation(
    name="Scale Factor (Matter-dominated)",
    formula="a = a_0*(t/t_0)^(2/3)",
    variables={
        "a": _DIMLESS,
        "a_0": _DIMLESS,
        "t": _T,
        "t_0": _T
    },
    domain="general_relativity",
    tags=["cosmology", "scale factor", "matter", "GR"],
    description="Scale factor evolution in matter-dominated era",
    latex=r"a = a_0\left(\frac{t}{t_0}\right)^{2/3}",
    assumptions=["Matter-dominated universe", "No cosmological constant", "k=0"],
    related=["First Friedmann Equation"],
))

register_equation(Equation(
    name="Scale Factor (Radiation-dominated)",
    formula="a = a_0*(t/t_0)^(1/2)",
    variables={
        "a": _DIMLESS,
        "a_0": _DIMLESS,
        "t": _T,
        "t_0": _T
    },
    domain="general_relativity",
    tags=["cosmology", "scale factor", "radiation", "GR"],
    description="Scale factor evolution in radiation-dominated era",
    latex=r"a = a_0\left(\frac{t}{t_0}\right)^{1/2}",
    assumptions=["Radiation-dominated universe", "No cosmological constant", "k=0"],
    related=["First Friedmann Equation"],
))

register_equation(Equation(
    name="Hubble Time",
    formula="t_H = 1/H",
    variables={
        "t_H": _T,
        "H": _HUBBLE
    },
    domain="general_relativity",
    tags=["cosmology", "hubble", "time", "GR"],
    description="Characteristic time scale of universe expansion",
    latex=r"t_H = \frac{1}{H}",
    related=["First Friedmann Equation"],
))

# Kerr Metric (Rotating Black Hole)
register_equation(Equation(
    name="Kerr Angular Momentum Parameter",
    formula="a_kerr = J/(M*c)",
    variables={
        "a_kerr": _L,
        "J": Dimension(mass=1, length=2, time=-1),  # angular momentum
        "M": _M,
        "c": _V
    },
    domain="general_relativity",
    tags=["kerr", "black hole", "rotation", "GR"],
    description="Specific angular momentum parameter for rotating black hole",
    latex=r"a = \frac{J}{Mc}",
    related=["Kerr Outer Horizon"],
))

register_equation(Equation(
    name="Kerr Outer Horizon",
    formula="r_plus = G*M/c^2 + sqrt((G*M/c^2)^2 - a_kerr^2)",
    variables={
        "r_plus": _L,
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "c": _V,
        "a_kerr": _L
    },
    domain="general_relativity",
    tags=["kerr", "black hole", "horizon", "rotation", "GR"],
    description="Outer event horizon radius for rotating black hole",
    latex=r"r_+ = \frac{GM}{c^2} + \sqrt{\left(\frac{GM}{c^2}\right)^2 - a^2}",
    assumptions=["a <= GM/c^2 (no naked singularity)"],
    related=["Kerr Angular Momentum Parameter", "Ergosphere Radius"],
))

register_equation(Equation(
    name="Ergosphere Radius",
    formula="r_ergo = G*M/c^2 + sqrt((G*M/c^2)^2 - a_kerr^2*cos^2(theta))",
    variables={
        "r_ergo": _L,
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "c": _V,
        "a_kerr": _L,
        "theta": _DIMLESS
    },
    domain="general_relativity",
    tags=["kerr", "black hole", "ergosphere", "rotation", "GR"],
    description="Ergosphere boundary for rotating black hole (depends on polar angle)",
    latex=r"r_{\text{ergo}} = \frac{GM}{c^2} + \sqrt{\left(\frac{GM}{c^2}\right)^2 - a^2\cos^2\theta}",
    related=["Kerr Angular Momentum Parameter", "Kerr Outer Horizon"],
))

# Gravitational Waves
register_equation(Equation(
    name="Gravitational Wave Strain",
    formula="h = (G/(c^4*r))*E",
    variables={
        "h": _STRAIN,
        "G": Dimension(length=3, mass=-1, time=-2),
        "c": _V,
        "r": _L,
        "E": _E  # characteristic energy
    },
    domain="general_relativity",
    tags=["gravitational waves", "strain", "GR"],
    description="Order-of-magnitude gravitational wave strain amplitude",
    latex=r"h \sim \frac{G}{c^4 r}E",
    assumptions=["Far-field approximation", "Weak field"],
    related=["Gravitational Wave Energy Flux"],
))

register_equation(Equation(
    name="Chirp Mass",
    formula="M_chirp = (m1*m2)^(3/5)/(m1+m2)^(1/5)",
    variables={
        "M_chirp": _M,
        "m1": _M,
        "m2": _M
    },
    domain="general_relativity",
    tags=["gravitational waves", "binary", "chirp mass", "GR"],
    description="Combination of masses determining gravitational wave amplitude",
    latex=r"\mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}}",
    related=["Gravitational Wave Strain"],
))

register_equation(Equation(
    name="Gravitational Wave Energy Flux",
    formula="dE/dt = (32/5)*(G^4/c^5)*(m1*m2)^2*(m1+m2)/r^5",
    variables={
        "dE/dt": _W,
        "G": Dimension(length=3, mass=-1, time=-2),
        "c": _V,
        "m1": _M,
        "m2": _M,
        "r": _L  # orbital separation
    },
    domain="general_relativity",
    tags=["gravitational waves", "energy", "binary", "GR"],
    description="Energy radiated by gravitational waves from binary system",
    latex=r"\frac{dE}{dt} = \frac{32}{5}\frac{G^4}{c^5}\frac{(m_1 m_2)^2(m_1+m_2)}{r^5}",
    assumptions=["Circular orbit", "Newtonian limit", "Quadrupole approximation"],
    related=["Chirp Mass"],
))

# Curvature Scalars
register_equation(Equation(
    name="Kretschmann Scalar",
    formula="K = R_abcd*R^abcd",
    variables={
        "K": Dimension(time=-4),  # curvature^2
        "R_abcd": Dimension(time=-2)  # Riemann tensor component
    },
    domain="general_relativity",
    tags=["curvature", "scalar", "invariant", "GR"],
    description="Curvature invariant (contraction of Riemann tensor)",
    latex=r"K = R_{\alpha\beta\gamma\delta}R^{\alpha\beta\gamma\delta}",
    related=["Kretschmann Scalar (Schwarzschild)"],
))

register_equation(Equation(
    name="Kretschmann Scalar (Schwarzschild)",
    formula="K = 48*(G*M)^2/(c^4*r^6)",
    variables={
        "K": Dimension(time=-4),
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "c": _V,
        "r": _L
    },
    domain="general_relativity",
    tags=["curvature", "schwarzschild", "black hole", "GR"],
    description="Kretschmann scalar for Schwarzschild spacetime",
    latex=r"K = \frac{48(GM)^2}{c^4 r^6}",
    related=["Schwarzschild Radius", "Kretschmann Scalar"],
))

register_equation(Equation(
    name="Ricci Scalar",
    formula="R = g^ab*R_ab",
    variables={
        "R": _CURVATURE,
        "g^ab": _DIMLESS,  # inverse metric
        "R_ab": _CURVATURE  # Ricci tensor
    },
    domain="general_relativity",
    tags=["curvature", "ricci", "scalar", "GR"],
    description="Trace of Ricci curvature tensor",
    latex=r"R = g^{\alpha\beta}R_{\alpha\beta}",
    related=["Einstein Field Equations"],
))

# Einstein Field Equations
register_equation(Equation(
    name="Einstein Field Equations",
    formula="R_ab - (1/2)*R*g_ab + Lambda*g_ab = (8*pi*G/c^4)*T_ab",
    variables={
        "R_ab": _CURVATURE,
        "R": _CURVATURE,
        "g_ab": _DIMLESS,
        "Lambda": _COSMOLOGICAL_CONST,
        "G": Dimension(length=3, mass=-1, time=-2),
        "c": _V,
        "T_ab": Dimension(mass=1, length=-1, time=-2)  # energy-momentum tensor
    },
    domain="general_relativity",
    tags=["einstein", "field equations", "fundamental", "GR"],
    description="Fundamental equations relating spacetime curvature to energy-momentum",
    latex=r"R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}",
    assumptions=["Metric signature (-+++)"],
    related=["Ricci Scalar", "Energy-Momentum Tensor (Perfect Fluid)"],
))

register_equation(Equation(
    name="Energy-Momentum Tensor (Perfect Fluid)",
    formula="T_ab = (rho + p/c^2)*u_a*u_b + p*g_ab",
    variables={
        "T_ab": Dimension(mass=1, length=-1, time=-2),
        "rho": _ENERGY_DENSITY,
        "p": _PRESSURE,
        "c": _V,
        "u_a": _DIMLESS,  # 4-velocity (normalized)
        "g_ab": _DIMLESS
    },
    domain="general_relativity",
    tags=["energy-momentum", "perfect fluid", "GR"],
    description="Energy-momentum tensor for perfect fluid",
    latex=r"T_{\mu\nu} = \left(\rho + \frac{p}{c^2}\right)u_\mu u_\nu + p g_{\mu\nu}",
    related=["Einstein Field Equations", "Fluid Equation (Cosmology)"],
))

# Compact Objects
register_equation(Equation(
    name="Compactness Parameter",
    formula="C = G*M/(R*c^2)",
    variables={
        "C": _DIMLESS,
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "R": _L,  # object radius
        "c": _V
    },
    domain="general_relativity",
    tags=["compact object", "neutron star", "black hole", "GR"],
    description="Compactness of a spherical mass distribution",
    latex=r"C = \frac{GM}{Rc^2}",
    assumptions=["C < 1/2 for stable configuration"],
    related=["Schwarzschild Radius"],
))

register_equation(Equation(
    name="Buchdahl Limit",
    formula="R_min = (9/4)*r_s",
    variables={
        "R_min": _L,
        "r_s": _L
    },
    domain="general_relativity",
    tags=["compact object", "neutron star", "buchdahl", "GR"],
    description="Minimum radius for static, spherical perfect fluid with uniform density",
    latex=r"R_{\text{min}} = \frac{9}{4}r_s",
    assumptions=["Static", "Spherically symmetric", "Perfect fluid"],
    related=["Schwarzschild Radius", "Compactness Parameter"],
))



# ============================================================================
# OPTICS
# ============================================================================

_FREQUENCY = Dimension(time=-1)
_REFRACTIVE_INDEX = _DIMLESS
_FOCAL_LENGTH = _L
_ANGLE = _DIMLESS

register_equation(Equation(
    name="Snell's Law",
    formula="n1*sin(theta1) = n2*sin(theta2)",
    variables={
        "n1": _REFRACTIVE_INDEX,
        "theta1": _ANGLE,
        "n2": _REFRACTIVE_INDEX,
        "theta2": _ANGLE
    },
    domain="optics",
    tags=["refraction", "fundamental", "wave"],
    description="Law of refraction at interface between media",
    latex=r"n_1\sin\theta_1 = n_2\sin\theta_2",
))

register_equation(Equation(
    name="Thin Lens Equation",
    formula="1/f = 1/d_o + 1/d_i",
    variables={
        "f": _FOCAL_LENGTH,
        "d_o": _L,  # object distance
        "d_i": _L   # image distance
    },
    domain="optics",
    tags=["lens", "imaging", "geometric optics"],
    description="Relates object and image distances to focal length",
    latex=r"\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}",
))

register_equation(Equation(
    name="Lens Maker's Equation",
    formula="1/f = (n-1)*(1/R1 - 1/R2)",
    variables={
        "f": _FOCAL_LENGTH,
        "n": _REFRACTIVE_INDEX,
        "R1": _L,  # radius of curvature 1
        "R2": _L   # radius of curvature 2
    },
    domain="optics",
    tags=["lens", "fabrication", "geometric optics"],
    description="Focal length from lens geometry and material",
    latex=r"\frac{1}{f} = (n-1)\left(\frac{1}{R_1} - \frac{1}{R_2}\right)",
))

register_equation(Equation(
    name="Magnification",
    formula="M = -d_i/d_o",
    variables={
        "M": _DIMLESS,
        "d_i": _L,
        "d_o": _L
    },
    domain="optics",
    tags=["lens", "imaging", "magnification"],
    description="Linear magnification of optical system",
    latex=r"M = -\frac{d_i}{d_o}",
))

register_equation(Equation(
    name="Diffraction Grating",
    formula="d*sin(theta) = m*lambda",
    variables={
        "d": _L,  # grating spacing
        "theta": _ANGLE,
        "m": _DIMLESS,  # order
        "lambda": _L
    },
    domain="optics",
    tags=["diffraction", "interference", "wave"],
    description="Condition for constructive interference in diffraction grating",
    latex=r"d\sin\theta = m\lambda",
))

register_equation(Equation(
    name="Rayleigh Criterion",
    formula="theta_min = 1.22*lambda/D",
    variables={
        "theta_min": _ANGLE,
        "lambda": _L,
        "D": _L  # aperture diameter
    },
    domain="optics",
    tags=["resolution", "diffraction", "imaging"],
    description="Minimum resolvable angle for circular aperture",
    latex=r"\theta_{\text{min}} = 1.22\frac{\lambda}{D}",
))

register_equation(Equation(
    name="Brewster's Angle",
    formula="tan(theta_B) = n2/n1",
    variables={
        "theta_B": _ANGLE,
        "n1": _REFRACTIVE_INDEX,
        "n2": _REFRACTIVE_INDEX
    },
    domain="optics",
    tags=["polarization", "reflection", "refraction"],
    description="Angle of incidence for complete p-polarization",
    latex=r"\tan\theta_B = \frac{n_2}{n_1}",
))

register_equation(Equation(
    name="Critical Angle",
    formula="sin(theta_c) = n2/n1",
    variables={
        "theta_c": _ANGLE,
        "n1": _REFRACTIVE_INDEX,
        "n2": _REFRACTIVE_INDEX
    },
    domain="optics",
    tags=["total internal reflection", "refraction"],
    description="Angle for total internal reflection (n1 > n2)",
    latex=r"\sin\theta_c = \frac{n_2}{n_1}",
    assumptions=["n1 > n2"],
))

register_equation(Equation(
    name="Malus's Law",
    formula="I = I0*cos^2(theta)",
    variables={
        "I": Dimension(mass=1, time=-3),  # intensity
        "I0": Dimension(mass=1, time=-3),
        "theta": _ANGLE
    },
    domain="optics",
    tags=["polarization", "intensity"],
    description="Intensity of light through polarizer",
    latex=r"I = I_0\cos^2\theta",
))

register_equation(Equation(
    name="Wave Equation (electromagnetic)",
    formula="c = lambda*f",
    variables={
        "c": _V,
        "lambda": _L,
        "f": _FREQUENCY
    },
    domain="optics",
    tags=["wave", "fundamental", "electromagnetic"],
    description="Relates wavelength, frequency, and speed of light",
    latex=r"c = \lambda f",
))

# ============================================================================
# ACOUSTICS
# ============================================================================

_SOUND_INTENSITY = Dimension(mass=1, time=-3)  # W/m²
_SOUND_PRESSURE = Dimension(mass=1, length=-1, time=-2)  # Pa
_BULK_MODULUS = Dimension(mass=1, length=-1, time=-2)

register_equation(Equation(
    name="Speed of Sound in Fluid",
    formula="v = sqrt(B/rho)",
    variables={
        "v": _V,
        "B": _BULK_MODULUS,
        "rho": _DENSITY
    },
    domain="acoustics",
    tags=["wave", "speed", "fluid"],
    description="Speed of sound in terms of bulk modulus and density",
    latex=r"v = \sqrt{\frac{B}{\rho}}",
))

register_equation(Equation(
    name="Speed of Sound in Ideal Gas",
    formula="v = sqrt(gamma*R*T/M)",
    variables={
        "v": _V,
        "gamma": _DIMLESS,  # heat capacity ratio
        "R": Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
        "T": _TEMP,
        "M": Dimension(mass=1, amount=-1)  # molar mass
    },
    domain="acoustics",
    tags=["wave", "speed", "gas"],
    description="Speed of sound in ideal gas",
    latex=r"v = \sqrt{\frac{\gamma RT}{M}}",
))

register_equation(Equation(
    name="Sound Intensity",
    formula="I = P^2/(2*rho*v)",
    variables={
        "I": _SOUND_INTENSITY,
        "P": _SOUND_PRESSURE,
        "rho": _DENSITY,
        "v": _V
    },
    domain="acoustics",
    tags=["intensity", "pressure", "wave"],
    description="Sound intensity from pressure amplitude",
    latex=r"I = \frac{P^2}{2\rho v}",
))

register_equation(Equation(
    name="Sound Intensity Level",
    formula="L = 10*log10(I/I0)",
    variables={
        "L": _DIMLESS,  # decibels
        "I": _SOUND_INTENSITY,
        "I0": _SOUND_INTENSITY  # reference intensity
    },
    domain="acoustics",
    tags=["intensity", "decibel", "logarithmic"],
    description="Sound level in decibels",
    latex=r"L = 10\log_{10}\left(\frac{I}{I_0}\right)",
))

register_equation(Equation(
    name="Doppler Effect (sound)",
    formula="f = f0*(v + v_r)/(v - v_s)",
    variables={
        "f": _FREQUENCY,
        "f0": _FREQUENCY,
        "v": _V,  # speed of sound
        "v_r": _V,  # receiver velocity
        "v_s": _V   # source velocity
    },
    domain="acoustics",
    tags=["doppler", "frequency", "wave"],
    description="Frequency shift due to relative motion in sound",
    latex=r"f = f_0\frac{v + v_r}{v - v_s}",
))

register_equation(Equation(
    name="Acoustic Impedance",
    formula="Z = rho*v",
    variables={
        "Z": Dimension(mass=1, length=-2, time=-1),
        "rho": _DENSITY,
        "v": _V
    },
    domain="acoustics",
    tags=["impedance", "wave", "material"],
    description="Characteristic acoustic impedance of medium",
    latex=r"Z = \rho v",
))

register_equation(Equation(
    name="Wave Equation (1D)",
    formula="d^2y/dt^2 = v^2*d^2y/dx^2",
    variables={
        "y": _L,  # displacement
        "t": _T,
        "v": _V,
        "x": _L
    },
    domain="acoustics",
    tags=["wave", "pde", "fundamental"],
    description="One-dimensional wave equation",
    latex=r"\frac{\partial^2 y}{\partial t^2} = v^2\frac{\partial^2 y}{\partial x^2}",
))

register_equation(Equation(
    name="Standing Wave Frequency (string)",
    formula="f_n = n*v/(2*L)",
    variables={
        "f_n": _FREQUENCY,
        "n": _DIMLESS,  # harmonic number
        "v": _V,
        "L": _L  # string length
    },
    domain="acoustics",
    tags=["standing wave", "resonance", "string"],
    description="Natural frequencies of vibrating string",
    latex=r"f_n = \frac{nv}{2L}",
    assumptions=["Fixed ends"],
))

register_equation(Equation(
    name="Acoustic Power",
    formula="P = I*A",
    variables={
        "P": _W,
        "I": _SOUND_INTENSITY,
        "A": Dimension(length=2)
    },
    domain="acoustics",
    tags=["power", "intensity"],
    description="Total acoustic power through area",
    latex=r"P = IA",
))

# ============================================================================
# FLUID DYNAMICS (EXPANDED)
# ============================================================================

register_equation(Equation(
    name="Stokes' Law",
    formula="F_d = 6*pi*mu*r*v",
    variables={
        "F_d": _F,  # drag force
        "mu": _VISCOSITY,
        "r": _L,  # sphere radius
        "v": _V
    },
    domain="fluid_dynamics",
    tags=["drag", "viscous", "sphere"],
    description="Drag force on a sphere in viscous fluid (low Re)",
    latex=r"F_d = 6\pi\mu rv",
    assumptions=["Low Reynolds number", "Spherical object"],
))

register_equation(Equation(
    name="Poiseuille's Law",
    formula="Q = pi*r^4*delta_P/(8*mu*L)",
    variables={
        "Q": Dimension(length=3, time=-1),  # volume flow rate
        "r": _L,  # pipe radius
        "delta_P": _PRESSURE,
        "mu": _VISCOSITY,
        "L": _L  # pipe length
    },
    domain="fluid_dynamics",
    tags=["flow", "viscous", "pipe"],
    description="Volume flow rate in cylindrical pipe",
    latex=r"Q = \frac{\pi r^4 \Delta P}{8\mu L}",
    assumptions=["Laminar flow", "Newtonian fluid"],
))

register_equation(Equation(
    name="Drag Equation",
    formula="F_d = (1/2)*rho*v^2*C_d*A",
    variables={
        "F_d": _F,
        "rho": _DENSITY,
        "v": _V,
        "C_d": _DIMLESS,  # drag coefficient
        "A": Dimension(length=2)  # reference area
    },
    domain="fluid_dynamics",
    tags=["drag", "turbulent", "aerodynamics"],
    description="Drag force on object in fluid flow",
    latex=r"F_d = \frac{1}{2}\rho v^2 C_d A",
))

register_equation(Equation(
    name="Froude Number",
    formula="Fr = v/sqrt(g*L)",
    variables={
        "Fr": _DIMLESS,
        "v": _V,
        "g": _A,
        "L": _L  # characteristic length
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "wave", "gravity"],
    description="Ratio of inertial to gravitational forces",
    latex=r"Fr = \frac{v}{\sqrt{gL}}",
))

register_equation(Equation(
    name="Mach Number",
    formula="M = v/c",
    variables={
        "M": _DIMLESS,
        "v": _V,  # flow velocity
        "c": _V   # speed of sound
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "compressible", "supersonic"],
    description="Ratio of flow velocity to speed of sound",
    latex=r"M = \frac{v}{c}",
))

# ============================================================================
# CHEMICAL KINETICS
# ============================================================================

# Dimensions for chemical kinetics
_CONCENTRATION = Dimension(amount=1, length=-3)  # mol/m³
_RATE_ZERO = Dimension(amount=1, length=-3, time=-1)  # mol/(m³·s)
_RATE_FIRST = Dimension(time=-1)  # 1/s
_RATE_SECOND = Dimension(amount=-1, length=3, time=-1)  # m³/(mol·s)
_ACTIVATION_ENERGY = Dimension(mass=1, length=2, time=-2, amount=-1)  # J/mol
_GAS_CONSTANT = Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1)  # J/(mol·K)
_EQUILIBRIUM_CONSTANT = _DIMLESS  # dimensionless (activity-based)

# Rate Laws - Integrated Forms

register_equation(Equation(
    name="Rate Law (Zero Order, Integrated)",
    formula="[A] = [A]0 - k*t",
    variables={
        "[A]": _CONCENTRATION,
        "[A]0": _CONCENTRATION,
        "k": _RATE_ZERO,
        "t": _T
    },
    domain="kinetics",
    tags=["rate law", "zero order", "integrated", "concentration"],
    description="Integrated rate law for zero-order reaction",
    latex=r"[A] = [A]_0 - kt",
    assumptions=["Zero-order kinetics", "Constant rate"],
    related=["Rate Law (Zero Order, Differential)"],
))

register_equation(Equation(
    name="Rate Law (First Order, Integrated)",
    formula="[A] = [A]0 * exp(-k*t)",
    variables={
        "[A]": _CONCENTRATION,
        "[A]0": _CONCENTRATION,
        "k": _RATE_FIRST,
        "t": _T
    },
    domain="kinetics",
    tags=["rate law", "first order", "integrated", "exponential"],
    description="Integrated rate law for first-order reaction",
    latex=r"[A] = [A]_0 e^{-kt}",
    assumptions=["First-order kinetics"],
    related=["Rate Law (First Order, Differential)", "Half-life (First Order)"],
))

register_equation(Equation(
    name="Rate Law (Second Order, Integrated)",
    formula="1/[A] = 1/[A]0 + k*t",
    variables={
        "1/[A]": Dimension(amount=-1, length=3),
        "1/[A]0": Dimension(amount=-1, length=3),
        "k": _RATE_SECOND,
        "t": _T
    },
    domain="kinetics",
    tags=["rate law", "second order", "integrated"],
    description="Integrated rate law for second-order reaction",
    latex=r"\frac{1}{[A]} = \frac{1}{[A]_0} + kt",
    assumptions=["Second-order kinetics", "Single reactant"],
    related=["Rate Law (Second Order, Differential)", "Half-life (Second Order)"],
))

# Rate Laws - Differential Forms

register_equation(Equation(
    name="Rate Law (Zero Order, Differential)",
    formula="d[A]/dt = -k",
    variables={
        "d[A]/dt": _RATE_ZERO,
        "k": _RATE_ZERO
    },
    domain="kinetics",
    tags=["rate law", "zero order", "differential"],
    description="Differential rate law for zero-order reaction",
    latex=r"\frac{d[A]}{dt} = -k",
    assumptions=["Zero-order kinetics"],
    related=["Rate Law (Zero Order, Integrated)"],
))

register_equation(Equation(
    name="Rate Law (First Order, Differential)",
    formula="d[A]/dt = -k*[A]",
    variables={
        "d[A]/dt": _RATE_ZERO,
        "k": _RATE_FIRST,
        "[A]": _CONCENTRATION
    },
    domain="kinetics",
    tags=["rate law", "first order", "differential"],
    description="Differential rate law for first-order reaction",
    latex=r"\frac{d[A]}{dt} = -k[A]",
    assumptions=["First-order kinetics"],
    related=["Rate Law (First Order, Integrated)"],
))

register_equation(Equation(
    name="Rate Law (Second Order, Differential)",
    formula="d[A]/dt = -k*[A]^2",
    variables={
        "d[A]/dt": _RATE_ZERO,
        "k": _RATE_SECOND,
        "[A]": _CONCENTRATION
    },
    domain="kinetics",
    tags=["rate law", "second order", "differential"],
    description="Differential rate law for second-order reaction",
    latex=r"\frac{d[A]}{dt} = -k[A]^2",
    assumptions=["Second-order kinetics", "Single reactant"],
    related=["Rate Law (Second Order, Integrated)"],
))

# Temperature Dependence

register_equation(Equation(
    name="Arrhenius Equation",
    formula="k = A * exp(-Ea/(R*T))",
    variables={
        "k": _RATE_FIRST,  # varies by order, using first-order as example
        "A": _RATE_FIRST,  # pre-exponential factor (same dimension as k)
        "Ea": _ACTIVATION_ENERGY,
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="kinetics",
    tags=["arrhenius", "temperature", "activation energy", "fundamental"],
    description="Temperature dependence of reaction rate constant",
    latex=r"k = A e^{-E_a/(RT)}",
    assumptions=["Arrhenius behavior"],
    related=["Eyring Equation", "Collision Theory"],
))

register_equation(Equation(
    name="Eyring Equation",
    formula="k = (kB*T/h) * exp(-dG_act/(R*T))",
    variables={
        "k": _RATE_FIRST,
        "kB": Dimension(mass=1, length=2, time=-2, temperature=-1),  # Boltzmann constant
        "T": _TEMP,
        "h": Dimension(mass=1, length=2, time=-1),  # Planck constant
        "dG_act": _ACTIVATION_ENERGY,  # activation Gibbs energy
        "R": _GAS_CONSTANT
    },
    domain="kinetics",
    tags=["eyring", "transition state", "temperature", "activation energy"],
    description="Transition state theory equation for rate constant",
    latex=r"k = \frac{k_B T}{h} e^{-\Delta G^\ddagger/(RT)}",
    assumptions=["Transition state theory", "Equilibrium between reactants and transition state"],
    related=["Arrhenius Equation"],
))

register_equation(Equation(
    name="Van't Hoff Equation",
    formula="d(ln K)/dT = dH/(R*T^2)",
    variables={
        "d(ln K)/dT": Dimension(temperature=-1),
        "dH": _ACTIVATION_ENERGY,  # standard enthalpy change
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="kinetics",
    tags=["van't hoff", "equilibrium", "temperature", "thermodynamics"],
    description="Temperature dependence of equilibrium constant",
    latex=r"\frac{d(\ln K)}{dT} = \frac{\Delta H^\circ}{RT^2}",
    assumptions=["dH is temperature-independent"],
    related=["Equilibrium Constant (Thermodynamic)"],
))

register_equation(Equation(
    name="Collision Theory",
    formula="k = Z * p * exp(-Ea/(R*T))",
    variables={
        "k": _RATE_SECOND,  # bimolecular rate constant
        "Z": _RATE_SECOND,  # collision frequency factor
        "p": _DIMLESS,  # steric factor
        "Ea": _ACTIVATION_ENERGY,
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="kinetics",
    tags=["collision theory", "bimolecular", "activation energy"],
    description="Rate constant from collision theory",
    latex=r"k = Zp e^{-E_a/(RT)}",
    assumptions=["Hard sphere collision model", "Bimolecular reaction"],
    related=["Arrhenius Equation"],
))

# Equilibrium Constants

register_equation(Equation(
    name="Equilibrium Constant (Kinetic Definition)",
    formula="K = k_forward / k_reverse",
    variables={
        "K": _EQUILIBRIUM_CONSTANT,
        "k_forward": _RATE_FIRST,  # varies by reaction
        "k_reverse": _RATE_FIRST
    },
    domain="kinetics",
    tags=["equilibrium", "rate constant", "fundamental"],
    description="Equilibrium constant from forward and reverse rate constants",
    latex=r"K = \frac{k_{\text{forward}}}{k_{\text{reverse}}}",
    assumptions=["Elementary reaction", "Equilibrium"],
    related=["Equilibrium Constant (Thermodynamic)"],
))

register_equation(Equation(
    name="Equilibrium Constant (Thermodynamic)",
    formula="K = exp(-dG/(R*T))",
    variables={
        "K": _EQUILIBRIUM_CONSTANT,
        "dG": _ACTIVATION_ENERGY,  # standard Gibbs energy change
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="kinetics",
    tags=["equilibrium", "gibbs energy", "thermodynamics"],
    description="Equilibrium constant from Gibbs energy",
    latex=r"K = e^{-\Delta G^\circ/(RT)}",
    assumptions=["Standard state", "Activity-based definition"],
    related=["Equilibrium Constant (Kinetic Definition)", "Van't Hoff Equation"],
))

# Half-lives

register_equation(Equation(
    name="Half-life (First Order)",
    formula="t_half = ln(2)/k",
    variables={
        "t_half": _T,
        "k": _RATE_FIRST
    },
    domain="kinetics",
    tags=["half-life", "first order", "decay"],
    description="Half-life for first-order reaction (independent of concentration)",
    latex=r"t_{1/2} = \frac{\ln(2)}{k}",
    assumptions=["First-order kinetics"],
    related=["Rate Law (First Order, Integrated)"],
))

register_equation(Equation(
    name="Half-life (Second Order)",
    formula="t_half = 1/(k*[A]0)",
    variables={
        "t_half": _T,
        "k": _RATE_SECOND,
        "[A]0": _CONCENTRATION
    },
    domain="kinetics",
    tags=["half-life", "second order"],
    description="Half-life for second-order reaction (depends on initial concentration)",
    latex=r"t_{1/2} = \frac{1}{k[A]_0}",
    assumptions=["Second-order kinetics", "Single reactant"],
    related=["Rate Law (Second Order, Integrated)"],
))

# Enzyme Kinetics

register_equation(Equation(
    name="Michaelis-Menten Equation",
    formula="v = Vmax*[S] / (Km + [S])",
    variables={
        "v": _RATE_ZERO,  # reaction velocity
        "Vmax": _RATE_ZERO,  # maximum velocity
        "[S]": _CONCENTRATION,  # substrate concentration
        "Km": _CONCENTRATION  # Michaelis constant
    },
    domain="kinetics",
    tags=["enzyme", "biochemistry", "catalysis", "michaelis-menten"],
    description="Rate of enzyme-catalyzed reaction",
    latex=r"v = \frac{V_{\max}[S]}{K_m + [S]}",
    assumptions=["Steady-state approximation", "Single substrate", "No product inhibition"],
    related=["Lineweaver-Burk Equation", "Hill Equation"],
))

# ============================================================================
# SOLID STATE PHYSICS
# ============================================================================

# Additional dimensions for solid state
_K = Dimension(length=-1)  # wave vector
_HBAR = Dimension(mass=1, length=2, time=-1)  # reduced Planck's constant
_NUMBER_DENSITY = Dimension(length=-3)  # carrier/particle density
_DOS_3D = Dimension(mass=-1, length=-2, time=2)  # density of states (3D)
_CONDUCTIVITY = Dimension(mass=-1, length=-3, time=3, current=2)  # electrical conductivity
_HALL_COEFF = Dimension(length=3, current=-1, time=-1)  # Hall coefficient
_MOBILITY = Dimension(current=1, time=2, mass=-1)  # carrier mobility

# Band Structure & Electronic Properties

register_equation(Equation(
    name="Effective Mass",
    formula="1/m* = (1/hbar^2)*d^2E/dk^2",
    variables={
        "m*": _M,
        "hbar": _HBAR,
        "E": _E,
        "k": _K
    },
    domain="solid_state",
    tags=["effective mass", "band structure", "semiconductor"],
    description="Effective mass tensor component from band curvature",
    latex=r"\frac{1}{m^*} = \frac{1}{\hbar^2}\frac{d^2E}{dk^2}",
    assumptions=["Parabolic bands near extrema"],
    related=["Fermi Energy (3D)"],
))

register_equation(Equation(
    name="Fermi Energy (3D)",
    formula="E_F = (hbar^2/2m)*(3*pi^2*n)^(2/3)",
    variables={
        "E_F": _E,
        "hbar": _HBAR,
        "m": _M,
        "n": _NUMBER_DENSITY
    },
    domain="solid_state",
    tags=["fermi", "energy", "free electrons", "3D"],
    description="Fermi energy for 3D free electron gas at zero temperature",
    latex=r"E_F = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}",
    assumptions=["Free electron gas", "Zero temperature", "3D system"],
    related=["Fermi Wavevector", "DOS 3D Free Electrons"],
))

register_equation(Equation(
    name="Fermi Wavevector",
    formula="k_F = (3*pi^2*n)^(1/3)",
    variables={
        "k_F": _K,
        "n": _NUMBER_DENSITY
    },
    domain="solid_state",
    tags=["fermi", "wavevector", "free electrons"],
    description="Fermi wavevector for 3D free electron gas",
    latex=r"k_F = (3\pi^2 n)^{1/3}",
    assumptions=["Free electron gas", "3D system"],
    related=["Fermi Energy (3D)"],
))

# Density of States

register_equation(Equation(
    name="DOS 3D Free Electrons",
    formula="g(E) = (V/2pi^2)*(2m/hbar^2)^(3/2)*sqrt(E)",
    variables={
        "g(E)": _DOS_3D,
        "V": _VOLUME,
        "m": _M,
        "hbar": _HBAR,
        "E": _E
    },
    domain="solid_state",
    tags=["dos", "density of states", "3D", "free electrons"],
    description="Density of states for 3D free electron gas",
    latex=r"g(E) = \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}",
    assumptions=["Free electron approximation", "Parabolic bands", "3D system"],
    related=["Fermi Energy (3D)", "DOS 2D Free Electrons"],
))

register_equation(Equation(
    name="DOS 2D Free Electrons",
    formula="g(E) = A*m/(pi*hbar^2)",
    variables={
        "g(E)": Dimension(mass=-1, length=-2, time=2),  # 2D DOS
        "A": Dimension(length=2),  # area
        "m": _M,
        "hbar": _HBAR
    },
    domain="solid_state",
    tags=["dos", "density of states", "2D", "free electrons"],
    description="Density of states for 2D free electron gas (constant)",
    latex=r"g(E) = \frac{Am}{\pi\hbar^2}",
    assumptions=["Free electron approximation", "Parabolic bands", "2D system"],
    related=["DOS 3D Free Electrons"],
))

# Phonons & Lattice Dynamics

register_equation(Equation(
    name="Phonon Dispersion (Debye)",
    formula="omega = v_s*|k|",
    variables={
        "omega": _ANGULAR_FREQ,
        "v_s": _V,  # sound velocity
        "k": _K
    },
    domain="solid_state",
    tags=["phonon", "dispersion", "debye", "acoustic"],
    description="Linear phonon dispersion relation (Debye model)",
    latex=r"\omega = v_s |k|",
    assumptions=["Linear dispersion", "Acoustic branch only", "Debye approximation"],
    related=["Debye Temperature", "Debye Heat Capacity (low T)"],
))

register_equation(Equation(
    name="Debye Temperature",
    formula="Theta_D = hbar*omega_D/k_B",
    variables={
        "Theta_D": _TEMP,
        "hbar": _HBAR,
        "omega_D": _ANGULAR_FREQ,  # Debye frequency
        "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1)  # Boltzmann constant
    },
    domain="solid_state",
    tags=["debye", "temperature", "phonon"],
    description="Characteristic temperature scale for lattice vibrations",
    latex=r"\Theta_D = \frac{\hbar\omega_D}{k_B}",
    related=["Phonon Dispersion (Debye)", "Debye Heat Capacity (low T)"],
))

register_equation(Equation(
    name="Debye Heat Capacity (low T)",
    formula="C_V = (12*pi^4/5)*N*k_B*(T/Theta_D)^3",
    variables={
        "C_V": _HEAT_CAPACITY,
        "N": _DIMLESS,  # number of atoms
        "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1),
        "T": _TEMP,
        "Theta_D": _TEMP
    },
    domain="solid_state",
    tags=["debye", "heat capacity", "low temperature"],
    description="Low-temperature heat capacity (T^3 law)",
    latex=r"C_V = \frac{12\pi^4}{5}Nk_B\left(\frac{T}{\Theta_D}\right)^3",
    assumptions=["T << Theta_D", "Debye model"],
    related=["Debye Temperature"],
))

# Transport Properties

register_equation(Equation(
    name="Drude Conductivity",
    formula="sigma = n*e^2*tau/m",
    variables={
        "sigma": _CONDUCTIVITY,
        "n": _NUMBER_DENSITY,
        "e": _CHARGE,
        "tau": _T,  # scattering time
        "m": _M
    },
    domain="solid_state",
    tags=["drude", "conductivity", "transport"],
    description="Electrical conductivity in Drude model",
    latex=r"\sigma = \frac{ne^2\tau}{m}",
    assumptions=["Classical free electrons", "Constant scattering time", "Drude model"],
    related=["Drude Mobility", "Ohm's Law"],
))

register_equation(Equation(
    name="Drude Mobility",
    formula="mu = e*tau/m",
    variables={
        "mu": _MOBILITY,
        "e": _CHARGE,
        "tau": _T,
        "m": _M
    },
    domain="solid_state",
    tags=["drude", "mobility", "transport"],
    description="Carrier mobility in Drude model",
    latex=r"\mu = \frac{e\tau}{m}",
    assumptions=["Classical free electrons", "Drude model"],
    related=["Drude Conductivity"],
))

register_equation(Equation(
    name="Hall Coefficient (classical)",
    formula="R_H = 1/(n*e)",
    variables={
        "R_H": _HALL_COEFF,
        "n": _NUMBER_DENSITY,
        "e": _CHARGE
    },
    domain="solid_state",
    tags=["hall effect", "transport", "classical"],
    description="Hall coefficient for single carrier type",
    latex=r"R_H = \frac{1}{ne}",
    assumptions=["Single carrier type", "Classical limit", "Isotropic scattering"],
    related=["Hall Voltage"],
))

register_equation(Equation(
    name="Hall Voltage",
    formula="V_H = I*B*R_H/d",
    variables={
        "V_H": _VOLTAGE,
        "I": _CURRENT,
        "B": _B_FIELD,
        "R_H": _HALL_COEFF,
        "d": _L  # sample thickness
    },
    domain="solid_state",
    tags=["hall effect", "voltage", "transport"],
    description="Hall voltage across sample in magnetic field",
    latex=r"V_H = \frac{IBR_H}{d}",
    related=["Hall Coefficient (classical)"],
))

register_equation(Equation(
    name="Mean Free Path",
    formula="ell = v_F*tau",
    variables={
        "ell": _L,
        "v_F": _V,  # Fermi velocity
        "tau": _T
    },
    domain="solid_state",
    tags=["transport", "scattering", "mean free path"],
    description="Average distance between scattering events",
    latex=r"\ell = v_F\tau",
    related=["Drude Conductivity", "Fermi Energy (3D)"],
))

# Superconductivity (BCS Theory)

register_equation(Equation(
    name="BCS Energy Gap (zero T)",
    formula="Delta(0) = 1.76*k_B*T_c",
    variables={
        "Delta(0)": _E,  # energy gap at T=0
        "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1),
        "T_c": _TEMP  # critical temperature
    },
    domain="solid_state",
    tags=["bcs", "superconductivity", "energy gap"],
    description="BCS energy gap at zero temperature",
    latex=r"\Delta(0) = 1.76 k_B T_c",
    assumptions=["Weak-coupling limit", "BCS theory"],
    related=["Cooper Pair Binding Energy"],
))

register_equation(Equation(
    name="Cooper Pair Binding Energy",
    formula="E_b = 2*Delta",
    variables={
        "E_b": _E,
        "Delta": _E  # energy gap
    },
    domain="solid_state",
    tags=["bcs", "cooper pairs", "superconductivity"],
    description="Energy required to break a Cooper pair",
    latex=r"E_b = 2\Delta",
    related=["BCS Energy Gap (zero T)"],
))

register_equation(Equation(
    name="BCS Coherence Length",
    formula="xi_0 = hbar*v_F/(pi*Delta)",
    variables={
        "xi_0": _L,
        "hbar": _HBAR,
        "v_F": _V,
        "Delta": _E
    },
    domain="solid_state",
    tags=["bcs", "coherence length", "superconductivity"],
    description="BCS coherence length (size of Cooper pair)",
    latex=r"\xi_0 = \frac{\hbar v_F}{\pi\Delta}",
    assumptions=["Clean limit", "BCS theory"],
    related=["BCS Energy Gap (zero T)", "London Penetration Depth"],
))

register_equation(Equation(
    name="London Penetration Depth",
    formula="lambda_L = sqrt(m/(mu_0*n_s*e^2))",
    variables={
        "lambda_L": _L,
        "m": _M,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),  # permeability
        "n_s": _NUMBER_DENSITY,  # superfluid density
        "e": _CHARGE
    },
    domain="solid_state",
    tags=["superconductivity", "london", "penetration depth"],
    description="Magnetic field penetration depth in superconductor",
    latex=r"\lambda_L = \sqrt{\frac{m}{\mu_0 n_s e^2}}",
    assumptions=["London theory", "Local approximation"],
    related=["BCS Coherence Length"],
))

# ============================================================================
# BIOPHYSICS EQUATIONS
# ============================================================================

# Additional biophysics dimensions
_PERMEABILITY = Dimension(length=1, time=-1)  # m/s
_DIFFUSIVITY = Dimension(length=2, time=-1)  # m²/s
_FLUX = Dimension(amount=1, length=-2, time=-1)  # mol/(m²·s)
_CONDUCTANCE_DENSITY = Dimension(mass=-1, length=-2, time=3, current=2)  # S/m²

# Update enzyme kinetics domain in existing Michaelis-Menten equation
# and add more biophysics equations

register_equation(Equation(
    name="Lineweaver-Burk Equation",
    formula="1/v = (Km/Vmax)*(1/[S]) + 1/Vmax",
    variables={
        "v": _RATE_ZERO,
        "Vmax": _RATE_ZERO,
        "[S]": _CONCENTRATION,
        "Km": _CONCENTRATION
    },
    domain="biophysics",
    tags=["enzyme", "kinetics", "linearization", "biochemistry"],
    description="Linearized form of Michaelis-Menten equation for parameter estimation",
    latex=r"\frac{1}{v} = \frac{K_m}{V_{\max}}\frac{1}{[S]} + \frac{1}{V_{\max}}",
    assumptions=["Same as Michaelis-Menten"],
    related=["Michaelis-Menten Equation"],
))

register_equation(Equation(
    name="Hill Equation",
    formula="v = Vmax * [S]^n / (K^n + [S]^n)",
    variables={
        "v": _RATE_ZERO,
        "Vmax": _RATE_ZERO,
        "[S]": _CONCENTRATION,
        "K": _CONCENTRATION,
        "n": _DIMLESS
    },
    domain="biophysics",
    tags=["enzyme", "cooperativity", "allosteric", "hill", "biochemistry"],
    description="Enzyme kinetics with cooperative substrate binding",
    latex=r"v = \frac{V_{\max}[S]^n}{K^n + [S]^n}",
    assumptions=[
        "Cooperative binding",
        "n > 1: positive cooperativity",
        "n = 1: reduces to Michaelis-Menten",
        "n < 1: negative cooperativity"
    ],
    related=["Michaelis-Menten Equation"],
))

# Membrane Biophysics
register_equation(Equation(
    name="Nernst Equation",
    formula="E = (R*T / (z*F)) * ln([ion]_out / [ion]_in)",
    variables={
        "E": _VOLTAGE,
        "R": _GAS_CONSTANT,
        "T": _TEMP,
        "z": _DIMLESS,
        "F": Dimension(current=1, time=1, amount=-1),
        "ion_out": _CONCENTRATION,
        "ion_in": _CONCENTRATION
    },
    domain="biophysics",
    tags=["membrane", "potential", "ion", "equilibrium", "nernst", "fundamental"],
    description="Equilibrium potential across a membrane for a single ion species",
    latex=r"E = \frac{RT}{zF}\ln\frac{[\text{ion}]_{out}}{[\text{ion}]_{in}}",
    assumptions=[
        "Equilibrium conditions",
        "Single ion species",
        "Ideal behavior"
    ],
    related=["Goldman-Hodgkin-Katz Equation"],
))

register_equation(Equation(
    name="Goldman-Hodgkin-Katz Equation",
    formula="E = (RT/F) * ln((P_K[K+]_out + P_Na[Na+]_out + P_Cl[Cl-]_in)/(P_K[K+]_in + P_Na[Na+]_in + P_Cl[Cl-]_out))",
    variables={
        "E": _VOLTAGE,
        "R": _GAS_CONSTANT,
        "T": _TEMP,
        "F": Dimension(current=1, time=1, amount=-1),
        "P_K": _PERMEABILITY,
        "P_Na": _PERMEABILITY,
        "P_Cl": _PERMEABILITY,
        "K_out": _CONCENTRATION,
        "K_in": _CONCENTRATION,
        "Na_out": _CONCENTRATION,
        "Na_in": _CONCENTRATION,
        "Cl_out": _CONCENTRATION,
        "Cl_in": _CONCENTRATION
    },
    domain="biophysics",
    tags=["membrane", "potential", "ion", "permeability", "ghk"],
    description="Membrane potential for multiple permeable ion species",
    latex=r"E = \frac{RT}{F}\ln\frac{P_K[K^+]_{out} + P_{Na}[Na^+]_{out} + P_{Cl}[Cl^-]_{in}}{P_K[K^+]_{in} + P_{Na}[Na^+]_{in} + P_{Cl}[Cl^-]_{out}}",
    assumptions=[
        "Constant field assumption",
        "Multiple permeable ions",
        "Steady state"
    ],
    related=["Nernst Equation"],
))

register_equation(Equation(
    name="Hodgkin-Huxley Current Equation",
    formula="I = g_Na*m^3*h*(V - E_Na) + g_K*n^4*(V - E_K) + g_L*(V - E_L)",
    variables={
        "I": Dimension(current=1, length=-2),  # current density
        "g_Na": _CONDUCTANCE_DENSITY,
        "g_K": _CONDUCTANCE_DENSITY,
        "g_L": _CONDUCTANCE_DENSITY,
        "m": _DIMLESS,
        "h": _DIMLESS,
        "n": _DIMLESS,
        "V": _VOLTAGE,
        "E_Na": _VOLTAGE,
        "E_K": _VOLTAGE,
        "E_L": _VOLTAGE
    },
    domain="biophysics",
    tags=["hodgkin-huxley", "action-potential", "neuron", "ion-channel", "electrophysiology"],
    description="Ionic current in Hodgkin-Huxley model (component of full HH system)",
    latex=r"I = g_{Na}m^3h(V - E_{Na}) + g_Kn^4(V - E_K) + g_L(V - E_L)",
    assumptions=[
        "Voltage-gated channels",
        "Gating variables satisfy auxiliary ODEs",
        "Full model includes membrane capacitance equation"
    ],
    related=["Cable Equation", "Nernst Equation"],
))

# Diffusion & Transport
register_equation(Equation(
    name="Fick's First Law",
    formula="J = -D * dC/dx",
    variables={
        "J": _FLUX,
        "D": _DIFFUSIVITY,
        "C": _CONCENTRATION,
        "x": _L
    },
    domain="biophysics",
    tags=["diffusion", "transport", "fick", "fundamental"],
    description="Diffusive flux proportional to concentration gradient",
    latex=r"J = -D\frac{dC}{dx}",
    assumptions=["Steady state", "Isotropic medium"],
    related=["Fick's Second Law", "Einstein-Stokes Relation"],
))

register_equation(Equation(
    name="Fick's Second Law",
    formula="dC/dt = D * d²C/dx²",
    variables={
        "C": _CONCENTRATION,
        "t": _T,
        "D": _DIFFUSIVITY,
        "x": _L
    },
    domain="biophysics",
    tags=["diffusion", "pde", "fick"],
    description="Time evolution of concentration by diffusion (requires numerical solver)",
    latex=r"\frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2}",
    assumptions=["Constant diffusion coefficient", "One-dimensional"],
    related=["Fick's First Law"],
))

register_equation(Equation(
    name="Cable Equation",
    formula="lambda²*d²V/dx² = tau*dV/dt + V - V_rest",
    variables={
        "lambda": _L,  # length constant
        "V": _VOLTAGE,
        "x": _L,
        "tau": _T,  # time constant
        "t": _T,
        "V_rest": _VOLTAGE
    },
    domain="biophysics",
    tags=["neuron", "cable", "electrophysiology", "pde"],
    description="Voltage distribution in passive dendrite/axon (PDE, requires numerical solver)",
    latex=r"\lambda^2\frac{\partial^2 V}{\partial x^2} = \tau\frac{\partial V}{\partial t} + V - V_{rest}",
    assumptions=[
        "Passive membrane (no active channels)",
        "Cylindrical geometry",
        "Uniform cable properties"
    ],
    related=["Hodgkin-Huxley Current Equation"],
))

register_equation(Equation(
    name="Einstein-Stokes Relation",
    formula="D = kB*T/(6*pi*eta*r)",
    variables={
        "D": _DIFFUSIVITY,
        "kB": Dimension(mass=1, length=2, time=-2, temperature=-1),
        "T": _TEMP,
        "eta": _VISCOSITY,
        "r": _L
    },
    domain="biophysics",
    tags=["diffusion", "stokes", "brownian"],
    description="Diffusion coefficient from Stokes drag on spherical particle",
    latex=r"D = \frac{k_BT}{6\pi\eta r}",
    assumptions=["Spherical particle", "Low Reynolds number"],
    related=["Fick's First Law", "Stokes' Law"],
))

# Population Dynamics
register_equation(Equation(
    name="Logistic Growth",
    formula="dN/dt = r*N*(1 - N/K)",
    variables={
        "N": _DIMLESS,  # population count
        "t": _T,
        "r": _RATE_FIRST,  # growth rate
        "K": _DIMLESS  # carrying capacity
    },
    domain="biophysics",
    tags=["population", "ecology", "growth", "logistic", "ode"],
    description="Population growth with density-dependent limiting",
    latex=r"\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right)",
    assumptions=[
        "Density-dependent growth",
        "Well-mixed population",
        "No age structure"
    ],
    related=["Lotka-Volterra Predator-Prey"],
))

register_equation(Equation(
    name="Lotka-Volterra Predator-Prey",
    formula="dx/dt = alpha*x - beta*x*y; dy/dt = delta*x*y - gamma*y",
    variables={
        "x": _DIMLESS,  # prey population
        "y": _DIMLESS,  # predator population
        "t": _T,
        "alpha": _RATE_FIRST,
        "beta": Dimension(time=-1),  # 1/(pop*time) treated as 1/time for dimensionless pop
        "delta": Dimension(time=-1),
        "gamma": _RATE_FIRST
    },
    domain="biophysics",
    tags=["population", "ecology", "predator-prey", "ode", "lotka-volterra"],
    description="Coupled ODEs for predator-prey population dynamics",
    latex=r"\frac{dx}{dt} = \alpha x - \beta xy; \quad \frac{dy}{dt} = \delta xy - \gamma y",
    assumptions=[
        "Well-mixed populations",
        "No environmental limits",
        "Instantaneous predation effect"
    ],
    related=["Logistic Growth"],
))

# Temperature Effects (Q10 coefficient)
register_equation(Equation(
    name="Q10 Temperature Coefficient",
    formula="k2 = k1 * Q10^((T2-T1)/10K)",
    variables={
        "k2": _RATE_FIRST,
        "k1": _RATE_FIRST,
        "Q10": _DIMLESS,
        "T2": _TEMP,
        "T1": _TEMP
    },
    domain="biophysics",
    tags=["kinetics", "temperature", "biophysics", "q10"],
    description="Empirical temperature dependence of biological rates",
    latex=r"k_2 = k_1 Q_{10}^{(T_2-T_1)/10K}",
    assumptions=["Typically Q10 = 2-3 for biological processes"],
    related=["Arrhenius Equation"],
))
# ============================================================================
# PLASMA PHYSICS EQUATIONS
# ============================================================================

# Plasma-specific dimensions
_N_DENSITY = Dimension(length=-3)  # number density [m⁻³]
_CONDUCTIVITY = Dimension(mass=-1, length=-3, time=3, current=2)  # [S/m]
_RESISTIVITY = Dimension(mass=1, length=3, time=-3, current=-2)  # [Ω·m]
_DIST_FUNC = Dimension(length=-6, time=3)  # phase space density [m⁻⁶·s³]
_CURRENT_DENSITY = Dimension(current=1, length=-2)  # [A/m²]

# MHD Equations (Ideal)
register_equation(Equation(
    name="Ideal Ohm's Law",
    formula="E + v x B = 0",
    variables={
        "E": _E_FIELD,
        "v": _V,
        "B": _B_FIELD
    },
    domain="plasma",
    tags=["mhd", "ideal", "electromagnetic"],
    description="Electric field in perfectly conducting plasma (ideal MHD)",
    latex=r"\vec{E} + \vec{v} \times \vec{B} = 0",
    assumptions=["Perfect conductivity", "Single fluid MHD", "SI units: B in Tesla"],
    related=["Resistive Ohm's Law", "Lorentz Force"],
))

register_equation(Equation(
    name="Magnetic Flux Freezing",
    formula="dB/dt = curl(v x B)",
    variables={
        "B": _B_FIELD,
        "t": _T,
        "v": _V
    },
    domain="plasma",
    tags=["mhd", "ideal", "induction"],
    description="Magnetic field evolution in ideal MHD (flux freezing)",
    latex=r"\frac{\partial \vec{B}}{\partial t} = \nabla \times (\vec{v} \times \vec{B})",
    assumptions=["Ideal MHD", "Infinite conductivity"],
    related=["Ideal Ohm's Law", "Alfvén Velocity"],
))

register_equation(Equation(
    name="Ideal MHD Equilibrium",
    formula="J x B = grad(P)",
    variables={
        "J": _CURRENT_DENSITY,
        "B": _B_FIELD,
        "P": _PRESSURE
    },
    domain="plasma",
    tags=["mhd", "ideal", "equilibrium"],
    description="Force balance in magnetohydrodynamic equilibrium",
    latex=r"\vec{J} \times \vec{B} = \nabla P",
    assumptions=["Static equilibrium", "Single fluid MHD"],
    related=["Ideal Ohm's Law", "Lorentz Force"],
))

# MHD Equations (Resistive)
register_equation(Equation(
    name="Resistive Ohm's Law",
    formula="E + v x B = eta*J",
    variables={
        "E": _E_FIELD,
        "v": _V,
        "B": _B_FIELD,
        "eta": _RESISTIVITY,
        "J": _CURRENT_DENSITY
    },
    domain="plasma",
    tags=["mhd", "resistive", "electromagnetic"],
    description="Generalized Ohm's law including resistivity",
    latex=r"\vec{E} + \vec{v} \times \vec{B} = \eta\vec{J}",
    assumptions=["Single fluid MHD", "SI units: B in Tesla, η in Ω·m"],
    related=["Ideal Ohm's Law", "Magnetic Diffusion"],
))

register_equation(Equation(
    name="Magnetic Diffusion",
    formula="dB/dt = curl(v x B) + (eta/mu_0)*laplacian(B)",
    variables={
        "B": _B_FIELD,
        "t": _T,
        "v": _V,
        "eta": _RESISTIVITY,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2)  # permeability
    },
    domain="plasma",
    tags=["mhd", "resistive", "diffusion"],
    description="Magnetic field evolution with resistivity (resistive MHD)",
    latex=r"\frac{\partial \vec{B}}{\partial t} = \nabla \times (\vec{v} \times \vec{B}) + \frac{\eta}{\mu_0}\nabla^2\vec{B}",
    assumptions=["Resistive MHD", "Finite conductivity"],
    related=["Resistive Ohm's Law", "Magnetic Flux Freezing", "Lundquist Number"],
))

# Kinetic Theory
register_equation(Equation(
    name="Vlasov Equation",
    formula="df/dt + v*grad(f) + (q/m)*(E + v x B)*grad_v(f) = 0",
    variables={
        "f": _DIST_FUNC,
        "t": _T,
        "v": _V,
        "q": _CHARGE,
        "m": _M,
        "E": _E_FIELD,
        "B": _B_FIELD
    },
    domain="plasma",
    tags=["kinetic", "collisionless", "distribution"],
    description="Evolution of particle distribution in phase space (collisionless)",
    latex=r"\frac{\partial f}{\partial t} + \vec{v} \cdot \nabla f + \frac{q}{m}(\vec{E} + \vec{v} \times \vec{B}) \cdot \nabla_v f = 0",
    assumptions=["Collisionless plasma", "Phase space density f(x,v,t)", "SI units"],
    related=["Lorentz Force"],
))

# Characteristic Scales
register_equation(Equation(
    name="Debye Length",
    formula="lambda_D = sqrt(epsilon_0*k_B*T_e/(n_e*e^2))",
    variables={
        "lambda_D": _L,
        "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),  # permittivity
        "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1),  # Boltzmann constant
        "T_e": _TEMP,
        "n_e": _N_DENSITY,
        "e": _CHARGE  # elementary charge
    },
    domain="plasma",
    tags=["scale", "shielding", "electron"],
    description="Characteristic shielding length in plasma",
    latex=r"\lambda_D = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}",
    assumptions=["Quasi-neutral", "Non-relativistic electrons", "SI units"],
    related=["Plasma Frequency (electron)"],
))

register_equation(Equation(
    name="Plasma Frequency (electron)",
    formula="omega_pe = sqrt(n_e*e^2/(epsilon_0*m_e))",
    variables={
        "omega_pe": _ANGULAR_FREQ,
        "n_e": _N_DENSITY,
        "e": _CHARGE,
        "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),
        "m_e": _M
    },
    domain="plasma",
    tags=["scale", "frequency", "electron", "oscillation"],
    description="Characteristic electron oscillation frequency (angular frequency ω, not f)",
    latex=r"\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}",
    assumptions=["Quasi-neutral", "Angular frequency in rad/s", "Ordinary frequency f = ω/(2π)"],
    related=["Debye Length", "Inertial Length"],
))

register_equation(Equation(
    name="Plasma Frequency (ion)",
    formula="omega_pi = sqrt(n_i*Z^2*e^2/(epsilon_0*m_i))",
    variables={
        "omega_pi": _ANGULAR_FREQ,
        "n_i": _N_DENSITY,
        "Z": _DIMLESS,  # ion charge number
        "e": _CHARGE,
        "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),
        "m_i": _M
    },
    domain="plasma",
    tags=["scale", "frequency", "ion", "oscillation"],
    description="Characteristic ion oscillation frequency (angular frequency ω, not f)",
    latex=r"\omega_{pi} = \sqrt{\frac{n_i Z^2 e^2}{\epsilon_0 m_i}}",
    assumptions=["Quasi-neutral", "Angular frequency in rad/s"],
    related=["Plasma Frequency (electron)"],
))

register_equation(Equation(
    name="Electron Gyrofrequency",
    formula="omega_ce = e*B/m_e",
    variables={
        "omega_ce": _ANGULAR_FREQ,
        "e": _CHARGE,
        "B": _B_FIELD,
        "m_e": _M
    },
    domain="plasma",
    tags=["scale", "frequency", "electron", "magnetic"],
    description="Electron cyclotron frequency in magnetic field",
    latex=r"\omega_{ce} = \frac{eB}{m_e}",
    assumptions=["Non-relativistic", "Uniform magnetic field", "SI units: B in Tesla"],
    related=["Electron Gyroradius", "Ion Gyrofrequency"],
))

register_equation(Equation(
    name="Ion Gyrofrequency",
    formula="omega_ci = Z*e*B/m_i",
    variables={
        "omega_ci": _ANGULAR_FREQ,
        "Z": _DIMLESS,
        "e": _CHARGE,
        "B": _B_FIELD,
        "m_i": _M
    },
    domain="plasma",
    tags=["scale", "frequency", "ion", "magnetic"],
    description="Ion cyclotron frequency in magnetic field",
    latex=r"\omega_{ci} = \frac{ZeB}{m_i}",
    assumptions=["Non-relativistic", "Uniform magnetic field", "SI units: B in Tesla"],
    related=["Ion Gyroradius", "Electron Gyrofrequency"],
))

register_equation(Equation(
    name="Electron Gyroradius",
    formula="r_Le = m_e*v_th_e/(e*B)",
    variables={
        "r_Le": _L,
        "m_e": _M,
        "v_th_e": _V,  # electron thermal velocity
        "e": _CHARGE,
        "B": _B_FIELD
    },
    domain="plasma",
    tags=["scale", "length", "electron", "magnetic"],
    description="Electron Larmor radius (thermal velocity v_th = sqrt(k_B*T/m))",
    latex=r"r_{Le} = \frac{m_e v_{th,e}}{eB}",
    assumptions=["Thermal velocity = sqrt(k_B*T/m)", "Non-relativistic", "SI units: B in Tesla"],
    related=["Electron Gyrofrequency", "Ion Gyroradius"],
))

register_equation(Equation(
    name="Ion Gyroradius",
    formula="r_Li = m_i*v_th_i/(Z*e*B)",
    variables={
        "r_Li": _L,
        "m_i": _M,
        "v_th_i": _V,  # ion thermal velocity
        "Z": _DIMLESS,
        "e": _CHARGE,
        "B": _B_FIELD
    },
    domain="plasma",
    tags=["scale", "length", "ion", "magnetic"],
    description="Ion Larmor radius (thermal velocity v_th = sqrt(k_B*T/m))",
    latex=r"r_{Li} = \frac{m_i v_{th,i}}{ZeB}",
    assumptions=["Thermal velocity = sqrt(k_B*T/m)", "Non-relativistic", "SI units: B in Tesla"],
    related=["Ion Gyrofrequency", "Electron Gyroradius"],
))

register_equation(Equation(
    name="Inertial Length",
    formula="delta_e = c/omega_pe",
    variables={
        "delta_e": _L,
        "c": _V,  # speed of light
        "omega_pe": _ANGULAR_FREQ
    },
    domain="plasma",
    tags=["scale", "length", "electron", "skin depth"],
    description="Electron inertial length (skin depth), characteristic scale for electron dynamics",
    latex=r"\delta_e = \frac{c}{\omega_{pe}}",
    assumptions=["Collisionless plasma", "Also called skin depth"],
    related=["Plasma Frequency (electron)"],
))

# Wave Physics
register_equation(Equation(
    name="Alfvén Velocity",
    formula="v_A = B/sqrt(mu_0*rho)",
    variables={
        "v_A": _V,
        "B": _B_FIELD,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
        "rho": _DENSITY
    },
    domain="plasma",
    tags=["wave", "alfven", "mhd"],
    description="Characteristic velocity of Alfvén waves in magnetized plasma",
    latex=r"v_A = \frac{B}{\sqrt{\mu_0\rho}}",
    assumptions=["Single fluid MHD", "SI units: B in Tesla"],
    related=["Alfvén Transit Time", "Lundquist Number"],
))

register_equation(Equation(
    name="Alfvén Transit Time",
    formula="tau_A = L/v_A",
    variables={
        "tau_A": _T,
        "L": _L,  # characteristic length
        "v_A": _V
    },
    domain="plasma",
    tags=["wave", "alfven", "mhd", "time"],
    description="Characteristic time for Alfvén wave propagation over length L",
    latex=r"\tau_A = \frac{L}{v_A}",
    assumptions=["Single fluid MHD"],
    related=["Alfvén Velocity", "Lundquist Number"],
))

# Dimensionless Parameters
register_equation(Equation(
    name="Lundquist Number",
    formula="S = mu_0*L*v_A/eta",
    variables={
        "S": _DIMLESS,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
        "L": _L,
        "v_A": _V,
        "eta": _RESISTIVITY
    },
    domain="plasma",
    tags=["dimensionless", "mhd", "resistive"],
    description="Ratio of resistive diffusion time to Alfvén transit time (S = τ_R/τ_A)",
    latex=r"S = \frac{\mu_0 L v_A}{\eta}",
    assumptions=["Resistive MHD", "Also S = μ₀σLv_A where σ = 1/η"],
    related=["Alfvén Velocity", "Magnetic Reynolds Number", "Magnetic Diffusion"],
))

register_equation(Equation(
    name="Magnetic Reynolds Number",
    formula="R_m = mu_0*sigma*L*v",
    variables={
        "R_m": _DIMLESS,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
        "sigma": _CONDUCTIVITY,
        "L": _L,
        "v": _V
    },
    domain="plasma",
    tags=["dimensionless", "mhd", "resistive"],
    description="Ratio of magnetic advection to diffusion (R_m = μ₀σLv)",
    latex=r"R_m = \mu_0\sigma L v",
    assumptions=["Resistive MHD", "Also R_m = μ₀Lv/η where η = 1/σ"],
    related=["Lundquist Number", "Magnetic Diffusion"],
))

register_equation(Equation(
    name="Plasma Beta",
    formula="beta = 2*mu_0*P/B^2",
    variables={
        "beta": _DIMLESS,
        "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
        "P": _PRESSURE,
        "B": _B_FIELD
    },
    domain="plasma",
    tags=["dimensionless", "mhd", "pressure"],
    description="Ratio of plasma pressure to magnetic pressure",
    latex=r"\beta = \frac{2\mu_0 P}{B^2}",
    assumptions=["β = P/(B²/(2μ₀))", "SI units: B in Tesla, P in Pascal"],
    related=["Ideal MHD Equilibrium"],
))

register_equation(Equation(
    name="Magnetic Mirror Ratio",
    formula="R_mirror = B_max/B_min",
    variables={
        "R_mirror": _DIMLESS,
        "B_max": _B_FIELD,
        "B_min": _B_FIELD
    },
    domain="plasma",
    tags=["dimensionless", "mirror", "confinement"],
    description="Ratio of maximum to minimum magnetic field in mirror configuration",
    latex=r"R_{mirror} = \frac{B_{max}}{B_{min}}",
    assumptions=["Magnetic mirror geometry", "Loss cone angle θ_lc = arcsin(sqrt(1/R))"],
    related=["Electron Gyrofrequency", "Ion Gyrofrequency"],
))

# ============================================================================
# MATERIALS SCIENCE EQUATIONS
# ============================================================================

# Dimensions for materials science
_STRESS = Dimension(mass=1, length=-1, time=-2)  # Pa
_STRAIN = _DIMLESS
_STRAIN_RATE = Dimension(time=-1)
_YOUNGS_MODULUS = Dimension(mass=1, length=-1, time=-2)  # Pa
_FRACTURE_ENERGY = Dimension(mass=1, time=-2)  # J/m²
_STRESS_INTENSITY = Dimension(mass=1, length=Fraction(-1, 2), time=-2)  # Pa·√m
_DIFFUSIVITY = Dimension(length=2, time=-1)  # m²/s
_FLUX = Dimension(amount=1, length=-2, time=-1)  # mol/(m²·s)
_GRAIN_SIZE = _L
_DISLOCATION_DENSITY = Dimension(length=-2)  # 1/m²
_BURGERS_VECTOR = _L
_SHEAR_MODULUS = Dimension(mass=1, length=-1, time=-2)  # Pa

# ============================================================================
# Stress-Strain Relationships
# ============================================================================

register_equation(Equation(
    name="Hooke's Law (materials)",
    formula="sigma = E*epsilon",
    variables={
        "sigma": _STRESS,
        "E": _YOUNGS_MODULUS,
        "epsilon": _STRAIN
    },
    domain="materials_science",
    tags=["stress-strain", "elastic", "fundamental"],
    description="Linear elastic stress-strain relationship",
    latex=r"\sigma = E\varepsilon",
    assumptions=["Linear elastic region", "Uniaxial stress", "Isotropic material"],
    related=["Poisson's Ratio", "Shear Modulus"],
))

register_equation(Equation(
    name="Von Mises Yield Criterion",
    formula="sigma_VM = sqrt((sigma_1 - sigma_2)^2 + (sigma_2 - sigma_3)^2 + (sigma_3 - sigma_1)^2)/sqrt(2)",
    variables={
        "sigma_VM": _STRESS,
        "sigma_1": _STRESS,
        "sigma_2": _STRESS,
        "sigma_3": _STRESS
    },
    domain="materials_science",
    tags=["stress-strain", "yield", "plasticity"],
    description="Effective stress for yield in ductile materials",
    latex=r"\sigma_{VM} = \sqrt{\frac{(\sigma_1 - \sigma_2)^2 + (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2}{2}}",
    assumptions=["Principal stresses known", "Ductile material"],
    related=["Tresca Yield Criterion"],
))

register_equation(Equation(
    name="True Stress-Strain",
    formula="sigma_true = sigma_eng*(1 + epsilon_eng)",
    variables={
        "sigma_true": _STRESS,
        "sigma_eng": _STRESS,
        "epsilon_eng": _STRAIN
    },
    domain="materials_science",
    tags=["stress-strain", "plasticity"],
    description="Conversion from engineering to true stress",
    latex=r"\sigma_{\text{true}} = \sigma_{\text{eng}}(1 + \varepsilon_{\text{eng}})",
    assumptions=["Uniaxial tension", "Constant volume (incompressible)"],
))

register_equation(Equation(
    name="Poisson's Ratio",
    formula="epsilon_trans = -nu*epsilon_axial",
    variables={
        "epsilon_trans": _STRAIN,
        "nu": _DIMLESS,
        "epsilon_axial": _STRAIN
    },
    domain="materials_science",
    tags=["stress-strain", "elastic"],
    description="Transverse strain from axial strain",
    latex=r"\varepsilon_{\text{trans}} = -\nu\varepsilon_{\text{axial}}",
    assumptions=["Linear elastic", "Isotropic material"],
    related=["Hooke's Law (materials)"],
))

# ============================================================================
# Fracture Mechanics
# ============================================================================

register_equation(Equation(
    name="Griffith Criterion",
    formula="sigma_f = sqrt(2*E*gamma/(pi*a))",
    variables={
        "sigma_f": _STRESS,
        "E": _YOUNGS_MODULUS,
        "gamma": _FRACTURE_ENERGY,
        "a": _L  # crack length
    },
    domain="materials_science",
    tags=["fracture", "brittle", "failure"],
    description="Critical stress for brittle fracture",
    latex=r"\sigma_f = \sqrt{\frac{2E\gamma}{\pi a}}",
    assumptions=["Brittle material", "Plane stress", "Through-thickness crack"],
    related=["Stress Intensity Factor"],
))

register_equation(Equation(
    name="Stress Intensity Factor",
    formula="K_I = Y*sigma*sqrt(pi*a)",
    variables={
        "K_I": _STRESS_INTENSITY,
        "Y": _DIMLESS,  # geometry factor
        "sigma": _STRESS,
        "a": _L  # crack length
    },
    domain="materials_science",
    tags=["fracture", "crack", "failure"],
    description="Mode I stress intensity at crack tip",
    latex=r"K_I = Y\sigma\sqrt{\pi a}",
    assumptions=["Linear elastic fracture mechanics"],
    related=["Paris Law", "Fracture Toughness"],
))

register_equation(Equation(
    name="Paris Law",
    formula="da/dN = C*(delta_K)^m",
    variables={
        "a": _L,  # crack length
        "N": _DIMLESS,  # number of cycles
        "C": Dimension(length=1, time=1),  # simplified dimension (material-dependent)
        "delta_K": _STRESS_INTENSITY,
        "m": _DIMLESS  # Paris exponent
    },
    domain="materials_science",
    tags=["fatigue", "crack-growth", "failure"],
    description="Fatigue crack growth rate",
    latex=r"\frac{da}{dN} = C(\Delta K)^m",
    assumptions=["Region II (stable crack growth)", "Constant amplitude loading"],
    related=["Stress Intensity Factor"],
))

register_equation(Equation(
    name="Fracture Toughness",
    formula="K_IC = sigma_f*sqrt(pi*a_c)",
    variables={
        "K_IC": _STRESS_INTENSITY,
        "sigma_f": _STRESS,
        "a_c": _L  # critical crack length
    },
    domain="materials_science",
    tags=["fracture", "toughness", "material-property"],
    description="Critical stress intensity for unstable crack propagation",
    latex=r"K_{IC} = \sigma_f\sqrt{\pi a_c}",
    assumptions=["Mode I loading", "Linear elastic fracture mechanics"],
    related=["Stress Intensity Factor", "Griffith Criterion"],
))

# ============================================================================
# Fatigue
# ============================================================================

register_equation(Equation(
    name="Basquin's Law",
    formula="delta_sigma/2 = sigma_f_prime*(2*N_f)^b",
    variables={
        "delta_sigma": _STRESS,
        "sigma_f_prime": _STRESS,  # fatigue strength coefficient
        "N_f": _DIMLESS,  # cycles to failure
        "b": _DIMLESS  # fatigue strength exponent
    },
    domain="materials_science",
    tags=["fatigue", "high-cycle", "S-N-curve"],
    description="High-cycle fatigue life prediction",
    latex=r"\frac{\Delta\sigma}{2} = \sigma'_f(2N_f)^b",
    assumptions=["Elastic strain dominates", "High cycle fatigue (N > 10^4)"],
    related=["Coffin-Manson"],
))

register_equation(Equation(
    name="Coffin-Manson",
    formula="delta_epsilon_p/2 = epsilon_f_prime*(2*N_f)^c",
    variables={
        "delta_epsilon_p": _STRAIN,
        "epsilon_f_prime": _STRAIN,  # fatigue ductility coefficient
        "N_f": _DIMLESS,
        "c": _DIMLESS  # fatigue ductility exponent
    },
    domain="materials_science",
    tags=["fatigue", "low-cycle", "plasticity"],
    description="Low-cycle fatigue life prediction",
    latex=r"\frac{\Delta\varepsilon_p}{2} = \varepsilon'_f(2N_f)^c",
    assumptions=["Plastic strain dominates", "Low cycle fatigue (N < 10^4)"],
    related=["Basquin's Law"],
))

register_equation(Equation(
    name="Goodman Relation",
    formula="sigma_a/sigma_e + sigma_m/sigma_u = 1",
    variables={
        "sigma_a": _STRESS,  # alternating stress amplitude
        "sigma_e": _STRESS,  # endurance limit
        "sigma_m": _STRESS,  # mean stress
        "sigma_u": _STRESS   # ultimate tensile strength
    },
    domain="materials_science",
    tags=["fatigue", "mean-stress", "design"],
    description="Effect of mean stress on fatigue life",
    latex=r"\frac{\sigma_a}{\sigma_e} + \frac{\sigma_m}{\sigma_u} = 1",
    assumptions=["Linear damage accumulation", "Ductile material"],
))

# ============================================================================
# Creep
# ============================================================================

register_equation(Equation(
    name="Norton Power Law",
    formula="epsilon_dot = A*sigma^n*exp(-Q/(R*T))",
    variables={
        "epsilon_dot": _STRAIN_RATE,
        "A": Dimension(mass=-1, length=1, time=-1),  # material constant (simplified)
        "sigma": _STRESS,
        "n": _DIMLESS,  # stress exponent
        "Q": _ACTIVATION_ENERGY,
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="materials_science",
    tags=["creep", "high-temperature", "deformation"],
    description="Steady-state creep rate",
    latex=r"\dot{\varepsilon} = A\sigma^n \exp\left(-\frac{Q}{RT}\right)",
    assumptions=["Steady-state creep", "Constant stress and temperature"],
    related=["Larson-Miller Parameter"],
))

register_equation(Equation(
    name="Larson-Miller Parameter",
    formula="LMP = T*(C + log10(t_r))",
    variables={
        "LMP": Dimension(temperature=1),  # K (with implicit log(hours))
        "T": _TEMP,
        "C": _DIMLESS,  # material constant (~20 for many alloys)
        "t_r": _T  # rupture time
    },
    domain="materials_science",
    tags=["creep", "rupture", "life-prediction"],
    description="Time-temperature parameter for creep rupture",
    latex=r"LMP = T(C + \log_{10}(t_r))",
    assumptions=["Single creep mechanism", "Constant stress"],
))

register_equation(Equation(
    name="Monkman-Grant",
    formula="epsilon_dot_min*t_r = C",
    variables={
        "epsilon_dot_min": _STRAIN_RATE,
        "t_r": _T,
        "C": _STRAIN  # Monkman-Grant constant
    },
    domain="materials_science",
    tags=["creep", "rupture", "failure"],
    description="Minimum creep rate and rupture time relationship",
    latex=r"\dot{\varepsilon}_{\text{min}} \cdot t_r = C",
    assumptions=["Material-specific constant"],
))

# ============================================================================
# Diffusion
# ============================================================================

register_equation(Equation(
    name="Fick's First Law",
    formula="J = -D*dC/dx",
    variables={
        "J": _FLUX,
        "D": _DIFFUSIVITY,
        "C": _CONCENTRATION,
        "x": _L
    },
    domain="materials_science",
    tags=["diffusion", "mass-transport", "steady-state"],
    description="Steady-state diffusion flux",
    latex=r"J = -D\frac{dC}{dx}",
    assumptions=["Steady state", "One-dimensional", "Constant D"],
    related=["Fick's Second Law"],
))

register_equation(Equation(
    name="Fick's Second Law",
    formula="dC/dt = D*d^2C/dx^2",
    variables={
        "C": _CONCENTRATION,
        "t": _T,
        "D": _DIFFUSIVITY,
        "x": _L
    },
    domain="materials_science",
    tags=["diffusion", "mass-transport", "transient", "pde"],
    description="Transient diffusion equation",
    latex=r"\frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2}",
    assumptions=["One-dimensional", "Constant D"],
    related=["Fick's First Law", "Arrhenius (Diffusivity)"],
))

register_equation(Equation(
    name="Arrhenius (Diffusivity)",
    formula="D = D0*exp(-Q/(R*T))",
    variables={
        "D": _DIFFUSIVITY,
        "D0": _DIFFUSIVITY,  # pre-exponential factor
        "Q": _ACTIVATION_ENERGY,
        "R": _GAS_CONSTANT,
        "T": _TEMP
    },
    domain="materials_science",
    tags=["diffusion", "temperature", "activation-energy"],
    description="Temperature dependence of diffusivity",
    latex=r"D = D_0 \exp\left(-\frac{Q}{RT}\right)",
    related=["Fick's First Law", "Fick's Second Law"],
))

# ============================================================================
# Phase Transformations
# ============================================================================

register_equation(Equation(
    name="Avrami Equation",
    formula="f = 1 - exp(-k*t^n)",
    variables={
        "f": _DIMLESS,  # transformed fraction
        "k": Dimension(time=-1),  # rate constant (depends on n)
        "t": _T,
        "n": _DIMLESS  # Avrami exponent
    },
    domain="materials_science",
    tags=["phase-transformation", "kinetics", "JMAK"],
    description="Fraction transformed during phase transformation (JMAK model)",
    latex=r"f = 1 - \exp(-kt^n)",
    assumptions=["Random nucleation", "Isothermal transformation"],
))

register_equation(Equation(
    name="Lever Rule",
    formula="f_alpha = (C_beta - C_0)/(C_beta - C_alpha)",
    variables={
        "f_alpha": _DIMLESS,  # fraction of alpha phase
        "C_beta": _DIMLESS,  # composition of beta phase
        "C_0": _DIMLESS,    # overall composition
        "C_alpha": _DIMLESS  # composition of alpha phase
    },
    domain="materials_science",
    tags=["phase-diagram", "equilibrium", "composition"],
    description="Phase fractions at equilibrium from phase diagram",
    latex=r"f_\alpha = \frac{C_\beta - C_0}{C_\beta - C_\alpha}",
    assumptions=["Two-phase equilibrium", "Binary system"],
))

# ============================================================================
# Hardening Mechanisms
# ============================================================================

register_equation(Equation(
    name="Hall-Petch Relation",
    formula="sigma_y = sigma_0 + k_y/sqrt(d)",
    variables={
        "sigma_y": _STRESS,
        "sigma_0": _STRESS,  # lattice friction stress
        "k_y": Dimension(mass=1, length=Fraction(-1, 2), time=-2),  # Hall-Petch constant
        "d": _GRAIN_SIZE
    },
    domain="materials_science",
    tags=["strengthening", "grain-size", "yield"],
    description="Yield strength dependence on grain size",
    latex=r"\sigma_y = \sigma_0 + \frac{k_y}{\sqrt{d}}",
    assumptions=["Polycrystalline material", "Room temperature"],
))

register_equation(Equation(
    name="Taylor Hardening",
    formula="sigma_y = sigma_0 + alpha*G*b*sqrt(rho)",
    variables={
        "sigma_y": _STRESS,
        "sigma_0": _STRESS,
        "alpha": _DIMLESS,  # material constant (~0.3-0.5)
        "G": _SHEAR_MODULUS,
        "b": _BURGERS_VECTOR,
        "rho": _DISLOCATION_DENSITY
    },
    domain="materials_science",
    tags=["strengthening", "dislocation", "yield"],
    description="Yield strength from dislocation density",
    latex=r"\sigma_y = \sigma_0 + \alpha G b \sqrt{\rho}",
    assumptions=["Dislocation-dislocation interactions"],
    related=["Hall-Petch Relation"],
))
# ============================================================================
# NUCLEAR PHYSICS EQUATIONS
# ============================================================================

# Additional dimensions for nuclear physics
_DECAY_CONST = Dimension(time=-1)  # decay constant/activity (T^-1)
_COUNT = _DIMLESS  # particle count (dimensionless)

# ----------------------------------------------------------------------------
# Binding Energy Equations
# ----------------------------------------------------------------------------

register_equation(Equation(
    name="Mass Defect",
    formula="delta_m = Z*m_p + N*m_n - M_nucleus",
    variables={
        "delta_m": _M,
        "Z": _COUNT,  # proton number (dimensionless)
        "m_p": _M,  # proton mass
        "N": _COUNT,  # neutron number (dimensionless)
        "m_n": _M,  # neutron mass
        "M_nucleus": _M  # nuclear mass
    },
    domain="nuclear",
    tags=["binding", "mass", "nuclear structure"],
    description="Mass difference between separated nucleons and bound nucleus",
    latex=r"\Delta m = Zm_p + Nm_n - M_{\text{nucleus}}",
    related=["Binding Energy", "Mass-Energy Equivalence"],
))

register_equation(Equation(
    name="Binding Energy",
    formula="BE = delta_m * c^2",
    variables={
        "BE": _E,  # binding energy
        "delta_m": _M,  # mass defect
        "c": _V  # speed of light
    },
    domain="nuclear",
    tags=["binding", "energy", "nuclear structure"],
    description="Total binding energy from mass defect",
    latex=r"BE = \Delta m c^2",
    related=["Mass Defect", "Mass-Energy Equivalence", "Binding Energy per Nucleon"],
))

register_equation(Equation(
    name="Binding Energy per Nucleon",
    formula="BE_A = BE/A",
    variables={
        "BE_A": _E,  # binding energy per nucleon
        "BE": _E,  # total binding energy
        "A": _COUNT  # mass number (dimensionless)
    },
    domain="nuclear",
    tags=["binding", "energy", "nuclear structure", "stability"],
    description="Average binding energy per nucleon (stability indicator)",
    latex=r"BE/A = \frac{BE}{A}",
    related=["Binding Energy"],
))

register_equation(Equation(
    name="Semi-Empirical Mass Formula (Volume Term)",
    formula="E_V = a_V * A",
    variables={
        "E_V": _E,  # volume term energy
        "a_V": _E,  # volume coefficient (~15.75 MeV)
        "A": _COUNT  # mass number
    },
    domain="nuclear",
    tags=["binding", "SEMF", "nuclear structure"],
    description="Volume term in semi-empirical mass formula (proportional to A)",
    latex=r"E_V = a_V A",
    related=["Binding Energy"],
))

register_equation(Equation(
    name="Semi-Empirical Mass Formula (Surface Term)",
    formula="E_S = -a_S * A^(2/3)",
    variables={
        "E_S": _E,  # surface term energy
        "a_S": _E,  # surface coefficient (~17.8 MeV)
        "A": _COUNT  # mass number
    },
    domain="nuclear",
    tags=["binding", "SEMF", "nuclear structure"],
    description="Surface term in SEMF (correction for surface nucleons)",
    latex=r"E_S = -a_S A^{2/3}",
    related=["Binding Energy"],
))

register_equation(Equation(
    name="Semi-Empirical Mass Formula (Coulomb Term)",
    formula="E_C = -a_C * Z*(Z-1)/A^(1/3)",
    variables={
        "E_C": _E,  # Coulomb term energy
        "a_C": _E,  # Coulomb coefficient (~0.711 MeV)
        "Z": _COUNT,  # proton number
        "A": _COUNT  # mass number
    },
    domain="nuclear",
    tags=["binding", "SEMF", "nuclear structure", "coulomb"],
    description="Coulomb repulsion term in SEMF",
    latex=r"E_C = -a_C \frac{Z(Z-1)}{A^{1/3}}",
    related=["Binding Energy"],
))

register_equation(Equation(
    name="Semi-Empirical Mass Formula (Asymmetry Term)",
    formula="E_A = -a_A * (A - 2*Z)^2/A",
    variables={
        "E_A": _E,  # asymmetry term energy
        "a_A": _E,  # asymmetry coefficient (~23.7 MeV)
        "A": _COUNT,  # mass number
        "Z": _COUNT  # proton number
    },
    domain="nuclear",
    tags=["binding", "SEMF", "nuclear structure"],
    description="Asymmetry term in SEMF (favors N≈Z)",
    latex=r"E_A = -a_A \frac{(A - 2Z)^2}{A}",
    related=["Binding Energy"],
))

register_equation(Equation(
    name="Semi-Empirical Mass Formula (Pairing Term)",
    formula="E_P = delta * A^(-1/2)",
    variables={
        "E_P": _E,  # pairing term energy
        "delta": _E,  # pairing coefficient (~12 MeV, sign depends on parity)
        "A": _COUNT  # mass number
    },
    domain="nuclear",
    tags=["binding", "SEMF", "nuclear structure", "pairing"],
    description="Pairing term in SEMF (even-even/odd-odd correction)",
    latex=r"E_P = \delta A^{-1/2}",
    assumptions=["delta = +12 MeV (even-even)", "delta = 0 (odd-A)", "delta = -12 MeV (odd-odd)"],
    related=["Binding Energy"],
))

# ----------------------------------------------------------------------------
# Radioactive Decay Equations
# ----------------------------------------------------------------------------

register_equation(Equation(
    name="Exponential Decay Law",
    formula="N(t) = N_0 * exp(-lambda*t)",
    variables={
        "N": _COUNT,  # number of nuclei at time t
        "N_0": _COUNT,  # initial number of nuclei
        "lambda": _DECAY_CONST,  # decay constant
        "t": _T  # time
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "exponential", "fundamental"],
    description="Number of undecayed nuclei as function of time",
    latex=r"N(t) = N_0 e^{-\lambda t}",
    related=["Activity", "Half-life"],
))

register_equation(Equation(
    name="Activity",
    formula="A(t) = lambda * N(t)",
    variables={
        "A": _DECAY_CONST,  # activity (decays per unit time)
        "lambda": _DECAY_CONST,  # decay constant
        "N": _COUNT  # number of nuclei
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "activity"],
    description="Decay rate (activity) of radioactive sample",
    latex=r"A(t) = \lambda N(t)",
    related=["Exponential Decay Law", "Half-life"],
))

register_equation(Equation(
    name="Activity Decay",
    formula="A(t) = A_0 * exp(-lambda*t)",
    variables={
        "A": _DECAY_CONST,  # activity at time t
        "A_0": _DECAY_CONST,  # initial activity
        "lambda": _DECAY_CONST,  # decay constant
        "t": _T  # time
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "activity", "exponential"],
    description="Activity as exponential function of time",
    latex=r"A(t) = A_0 e^{-\lambda t}",
    related=["Activity", "Exponential Decay Law"],
))

register_equation(Equation(
    name="Half-life (Nuclear)",
    formula="t_half = ln(2)/lambda",
    variables={
        "t_half": _T,  # half-life
        "lambda": _DECAY_CONST  # decay constant
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "half-life"],
    description="Time for half of nuclei to decay",
    latex=r"t_{1/2} = \frac{\ln 2}{\lambda} \approx \frac{0.693}{\lambda}",
    related=["Exponential Decay Law", "Mean Lifetime (Nuclear)"],
))

register_equation(Equation(
    name="Mean Lifetime (Nuclear)",
    formula="tau = 1/lambda",
    variables={
        "tau": _T,  # mean lifetime
        "lambda": _DECAY_CONST  # decay constant
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "lifetime"],
    description="Average lifetime of a nucleus before decay",
    latex=r"\tau = \frac{1}{\lambda}",
    related=["Half-life (Nuclear)", "Exponential Decay Law"],
))

register_equation(Equation(
    name="Half-life to Mean Lifetime (Nuclear)",
    formula="tau = t_half/ln(2)",
    variables={
        "tau": _T,  # mean lifetime
        "t_half": _T  # half-life
    },
    domain="nuclear",
    tags=["decay", "radioactivity", "lifetime", "half-life"],
    description="Conversion between half-life and mean lifetime",
    latex=r"\tau = \frac{t_{1/2}}{\ln 2} \approx 1.44 \, t_{1/2}",
    related=["Half-life (Nuclear)", "Mean Lifetime (Nuclear)"],
))

register_equation(Equation(
    name="Bateman Equation (Two-Step Decay)",
    formula="N_2(t) = N_10 * lambda_1/(lambda_2 - lambda_1) * (exp(-lambda_1*t) - exp(-lambda_2*t))",
    variables={
        "N_2": _COUNT,  # daughter nuclei count
        "N_10": _COUNT,  # initial parent nuclei count
        "lambda_1": _DECAY_CONST,  # parent decay constant
        "lambda_2": _DECAY_CONST,  # daughter decay constant
        "t": _T  # time
    },
    domain="nuclear",
    tags=["decay", "chain", "bateman", "radioactivity"],
    description="Daughter nucleus population in two-step decay chain",
    latex=r"N_2(t) = \frac{N_{1,0}\lambda_1}{\lambda_2 - \lambda_1}\left(e^{-\lambda_1 t} - e^{-\lambda_2 t}\right)",
    assumptions=["lambda_1 ≠ lambda_2", "Initially pure parent (N_2(0) = 0)"],
    related=["Exponential Decay Law"],
))

# ----------------------------------------------------------------------------
# Q-value and Nuclear Reactions
# ----------------------------------------------------------------------------

register_equation(Equation(
    name="Q-value (Mass-Energy)",
    formula="Q = (sum(m_reactants) - sum(m_products)) * c^2",
    variables={
        "Q": _E,  # reaction Q-value (energy released)
        "m_reactants": _M,  # total mass of reactants
        "m_products": _M,  # total mass of products
        "c": _V  # speed of light
    },
    domain="nuclear",
    tags=["Q-value", "reaction", "energy"],
    description="Energy released (Q>0) or required (Q<0) in nuclear reaction",
    latex=r"Q = \left(\sum m_{\text{reactants}} - \sum m_{\text{products}}\right)c^2",
    related=["Mass-Energy Equivalence"],
))

register_equation(Equation(
    name="Q-value (Kinetic Energy)",
    formula="Q = sum(KE_products) - sum(KE_reactants)",
    variables={
        "Q": _E,  # reaction Q-value
        "KE_products": _E,  # kinetic energy of products
        "KE_reactants": _E  # kinetic energy of reactants
    },
    domain="nuclear",
    tags=["Q-value", "reaction", "energy", "kinetic"],
    description="Q-value from kinetic energy balance",
    latex=r"Q = \sum KE_{\text{products}} - \sum KE_{\text{reactants}}",
    related=["Q-value (Mass-Energy)"],
))

register_equation(Equation(
    name="Threshold Energy (Endothermic Reaction)",
    formula="K_th = -Q * (1 + m_products/m_target + Q/(2*m_target*c^2))",
    variables={
        "K_th": _E,  # threshold kinetic energy
        "Q": _E,  # Q-value (negative for endothermic)
        "m_products": _M,  # total mass of products
        "m_target": _M,  # target mass
        "c": _V  # speed of light
    },
    domain="nuclear",
    tags=["threshold", "reaction", "energy", "endothermic"],
    description="Minimum projectile energy for endothermic reaction",
    latex=r"K_{\text{th}} = -Q\left(1 + \frac{m_{\text{products}}}{m_{\text{target}}} + \frac{Q}{2m_{\text{target}}c^2}\right)",
    assumptions=["Q < 0 (endothermic)", "Target at rest"],
    related=["Q-value (Mass-Energy)"],
))

# ----------------------------------------------------------------------------
# Cross-sections and Reaction Rates
# ----------------------------------------------------------------------------

register_equation(Equation(
    name="Breit-Wigner Cross-section",
    formula="sigma(E) = pi*lambda^2 * g * Gamma_n*Gamma_gamma/((E-E_R)^2 + (Gamma/2)^2)",
    variables={
        "sigma": Dimension(length=2),  # cross-section
        "lambda": _L,  # reduced wavelength (h/(2*pi*p))
        "g": _DIMLESS,  # spin statistical factor
        "Gamma_n": _E,  # neutron width
        "Gamma_gamma": _E,  # gamma width
        "E": _E,  # energy
        "E_R": _E,  # resonance energy
        "Gamma": _E  # total width
    },
    domain="nuclear",
    tags=["cross-section", "resonance", "breit-wigner"],
    description="Resonance cross-section (Breit-Wigner formula)",
    latex=r"\sigma(E) = \pi\lambda^2 g \frac{\Gamma_n\Gamma_\gamma}{(E-E_R)^2 + (\Gamma/2)^2}",
    assumptions=["Single isolated resonance", "Non-relativistic"],
    related=["Reaction Rate (Nuclear)"],
))

register_equation(Equation(
    name="Reaction Rate (Nuclear)",
    formula="R = n_1 * n_2 * sigma * v",
    variables={
        "R": Dimension(length=-3, time=-1),  # reaction rate per volume
        "n_1": _NUMBER_DENSITY,  # number density species 1
        "n_2": _NUMBER_DENSITY,  # number density species 2
        "sigma": Dimension(length=2),  # cross-section
        "v": _V  # relative velocity
    },
    domain="nuclear",
    tags=["cross-section", "reaction", "rate"],
    description="Nuclear reaction rate for two interacting species",
    latex=r"R = n_1 n_2 \langle\sigma v\rangle",
    related=["Breit-Wigner Cross-section", "Mean Free Path (Nuclear)"],
))

register_equation(Equation(
    name="Mean Free Path (Nuclear)",
    formula="lambda_mfp = 1/(n*sigma)",
    variables={
        "lambda_mfp": _L,  # mean free path
        "n": _NUMBER_DENSITY,  # number density
        "sigma": Dimension(length=2)  # cross-section
    },
    domain="nuclear",
    tags=["cross-section", "transport", "mean free path"],
    description="Average distance traveled between interactions",
    latex=r"\lambda_{\text{mfp}} = \frac{1}{n\sigma}",
    related=["Reaction Rate (Nuclear)"],
))

# ----------------------------------------------------------------------------
# Fusion and Fission
# ----------------------------------------------------------------------------

register_equation(Equation(
    name="Coulomb Barrier Energy",
    formula="E_C = Z_1*Z_2*e^2/(4*pi*epsilon_0*r)",
    variables={
        "E_C": _E,  # Coulomb barrier energy
        "Z_1": _COUNT,  # charge number of nucleus 1
        "Z_2": _COUNT,  # charge number of nucleus 2
        "e": _CHARGE,  # elementary charge
        "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),  # permittivity
        "r": _L  # separation distance
    },
    domain="nuclear",
    tags=["fusion", "coulomb", "barrier"],
    description="Electrostatic potential energy barrier for fusion",
    latex=r"E_C = \frac{Z_1 Z_2 e^2}{4\pi\epsilon_0 r}",
    related=["Gamow Factor"],
))

register_equation(Equation(
    name="Nuclear Radius",
    formula="R = r_0 * A^(1/3)",
    variables={
        "R": _L,  # nuclear radius
        "r_0": _L,  # radius constant (~1.2 fm)
        "A": _COUNT  # mass number
    },
    domain="nuclear",
    tags=["nuclear structure", "radius", "size"],
    description="Empirical nuclear radius formula",
    latex=r"R = r_0 A^{1/3}",
    assumptions=["r_0 ≈ 1.2 fm"],
    related=["Coulomb Barrier Energy"],
))

register_equation(Equation(
    name="Gamow Factor",
    formula="P = exp(-2*pi*eta)",
    variables={
        "P": _DIMLESS,  # tunneling probability
        "eta": _DIMLESS  # Sommerfeld parameter: Z_1*Z_2*e^2/(4*pi*epsilon_0*hbar*v)
    },
    domain="nuclear",
    tags=["fusion", "tunneling", "quantum", "gamow"],
    description="Quantum tunneling probability through Coulomb barrier",
    latex=r"P \propto e^{-2\pi\eta}",
    assumptions=["eta = Z_1*Z_2*e^2/(4*pi*epsilon_0*hbar*v)"],
    related=["Coulomb Barrier Energy"],
))

register_equation(Equation(
    name="Fusion Energy Release",
    formula="E_fusion = Q",
    variables={
        "E_fusion": _E,  # energy released in fusion
        "Q": _E  # Q-value of fusion reaction
    },
    domain="nuclear",
    tags=["fusion", "energy", "reaction"],
    description="Energy released in fusion reaction (positive Q-value)",
    latex=r"E_{\text{fusion}} = Q > 0",
    assumptions=["Exothermic fusion reaction"],
    related=["Q-value (Mass-Energy)"],
))

register_equation(Equation(
    name="Fission Energy Release",
    formula="E_fission = Q",
    variables={
        "E_fission": _E,  # energy released in fission
        "Q": _E  # Q-value of fission reaction
    },
    domain="nuclear",
    tags=["fission", "energy", "reaction"],
    description="Energy released in fission reaction (positive Q-value)",
    latex=r"E_{\text{fission}} = Q > 0",
    assumptions=["Exothermic fission reaction"],
    related=["Q-value (Mass-Energy)"],
))
# ============================================================================
# QUANTUM FIELD THEORY
# ============================================================================
# In natural units (c = ℏ = 1):
# - Energy, mass, momentum: dimension [M]
# - Length and time: dimension [M]^(-1)
# - Lagrangian density: [M]^4
# - Scalar field φ: [M]
# - Spinor field ψ: [M]^(3/2)
# - Vector field A_μ: [M]
# - Cross-section: [M]^(-2) (area in natural units)
# - Decay rate Γ: [M] (inverse time in natural units)
# - Coupling constants: dimensionless (fine structure α) or various powers

from fractions import Fraction

# Natural unit dimensions from domains/natural.py
_NAT_ENERGY = _E  # In SI this is M L² T⁻², represents energy/mass in natural units
_NAT_LENGTH = Dimension(mass=-1, length=-2, time=2)  # [E]^(-1) in natural units
_NAT_TIME = Dimension(mass=-1, length=-2, time=2)  # [E]^(-1) in natural units

# QFT-specific dimensions (in natural units, expressed as SI equivalents)
_NAT_MASS = _E  # Same as energy in natural units
_NAT_MOMENTUM = _E  # Same as energy in natural units
_LAGRANGIAN_DENSITY = Dimension(mass=4, length=8, time=-8)  # [M]^4 → M^4 L^8 T^-8 in SI
_SCALAR_FIELD = _E  # φ has dimension [M]^1
_SPINOR_FIELD = Dimension(mass=Fraction(3,2), length=3, time=-3)  # ψ has [M]^(3/2)
_VECTOR_FIELD = _E  # A_μ has dimension [M]^1
_CROSS_SECTION = Dimension(mass=-2, length=-4, time=4)  # [M]^(-2) in natural units
_DECAY_RATE = _E  # Γ has dimension [M]^1 = [E]^1

# Field equations
register_equation(Equation(
    name="Dirac Equation (free particle)",
    formula="(i*gamma^mu*d_mu - m)*psi = 0",
    variables={
        "gamma": _DIMLESS,  # gamma matrices (dimensionless)
        "d_mu": _E,  # partial derivative d/dx^mu has dimension [M] in natural units
        "m": _NAT_MASS,  # mass
        "psi": _SPINOR_FIELD  # 4-component Dirac spinor
    },
    domain="quantum_field_theory",
    tags=["qft", "dirac", "spinor", "field equation", "relativistic", "fermion"],
    description="Relativistic equation of motion for spin-1/2 fermions",
    latex=r"(i\gamma^\mu \partial_\mu - m)\psi = 0",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "ψ is a 4-component Dirac spinor",
        "γ^μ are Dirac gamma matrices satisfying {γ^μ, γ^ν} = 2g^μν",
        "Free particle (no external fields)"
    ],
    related=["Klein-Gordon Equation", "Dirac Field Lagrangian"],
))

register_equation(Equation(
    name="Klein-Gordon Equation",
    formula="(d^mu*d_mu + m^2)*phi = 0",
    variables={
        "d": _E,  # d'Alembertian operator components
        "m": _NAT_MASS,  # mass
        "phi": _SCALAR_FIELD  # scalar field
    },
    domain="quantum_field_theory",
    tags=["qft", "klein-gordon", "scalar", "field equation", "relativistic"],
    description="Relativistic wave equation for spin-0 bosons",
    latex=r"(\partial^\mu \partial_\mu + m^2)\phi = 0",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "φ is a real or complex scalar field",
        "∂^μ ∂_μ = ∂²/∂t² - ∇² (d'Alembertian operator)"
    ],
    related=["Dirac Equation (free particle)", "Scalar Field Lagrangian"],
))

# Lagrangian densities
register_equation(Equation(
    name="Scalar Field Lagrangian",
    formula="L = (1/2)*d_mu(phi)*d^mu(phi) - (1/2)*m^2*phi^2 - V(phi)",
    variables={
        "L": _LAGRANGIAN_DENSITY,
        "d_mu": _E,  # derivative
        "phi": _SCALAR_FIELD,
        "m": _NAT_MASS,
        "V": _LAGRANGIAN_DENSITY  # potential (same dimension as L)
    },
    domain="quantum_field_theory",
    tags=["qft", "lagrangian", "scalar field", "action"],
    description="Lagrangian density for real scalar field theory",
    latex=r"\mathcal{L} = \frac{1}{2}\partial_\mu\phi\partial^\mu\phi - \frac{1}{2}m^2\phi^2 - V(\phi)",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Real scalar field φ",
        "V(φ) is interaction potential (e.g., λφ^4/4!)",
        "Lagrangian density has dimension [M]^4"
    ],
    related=["Klein-Gordon Equation", "QED Lagrangian"],
))

register_equation(Equation(
    name="Dirac Field Lagrangian",
    formula="L = psi_bar*(i*gamma^mu*d_mu - m)*psi",
    variables={
        "L": _LAGRANGIAN_DENSITY,
        "psi_bar": _SPINOR_FIELD,  # adjoint spinor
        "gamma": _DIMLESS,
        "d_mu": _E,
        "m": _NAT_MASS,
        "psi": _SPINOR_FIELD
    },
    domain="quantum_field_theory",
    tags=["qft", "lagrangian", "dirac", "fermion", "spinor"],
    description="Lagrangian density for free Dirac fermion field",
    latex=r"\mathcal{L} = \bar{\psi}(i\gamma^\mu \partial_\mu - m)\psi",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "ψ is a 4-component Dirac spinor",
        "ψ̄ = ψ†γ^0 is the Dirac adjoint",
        "Lagrangian density has dimension [M]^4"
    ],
    related=["Dirac Equation (free particle)", "QED Lagrangian"],
))

register_equation(Equation(
    name="QED Lagrangian",
    formula="L = psi_bar*(i*gamma^mu*D_mu - m)*psi - (1/4)*F^mu_nu*F_mu^nu",
    variables={
        "L": _LAGRANGIAN_DENSITY,
        "psi_bar": _SPINOR_FIELD,
        "gamma": _DIMLESS,
        "D_mu": _E,  # covariant derivative
        "m": _NAT_MASS,
        "psi": _SPINOR_FIELD,
        "F": Dimension(mass=2, length=4, time=-4)  # field strength [M]^2
    },
    domain="quantum_field_theory",
    tags=["qft", "lagrangian", "qed", "gauge theory", "electromagnetism"],
    description="Quantum electrodynamics Lagrangian density",
    latex=r"\mathcal{L} = \bar{\psi}(i\gamma^\mu D_\mu - m)\psi - \frac{1}{4}F^{\mu\nu}F_{\mu\nu}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "D_μ = ∂_μ + ieA_μ is the covariant derivative",
        "F^μν = ∂^μA^ν - ∂^νA^μ is the electromagnetic field strength tensor",
        "e is the electric charge (dimensionless in natural units)",
        "Gauge group U(1)"
    ],
    related=["Dirac Field Lagrangian", "QCD Lagrangian"],
))

register_equation(Equation(
    name="QCD Lagrangian",
    formula="L = sum_f[psi_bar_f*(i*gamma^mu*D_mu - m_f)*psi_f] - (1/4)*G^a_mu_nu*G^mu_nu_a",
    variables={
        "L": _LAGRANGIAN_DENSITY,
        "psi_f": _SPINOR_FIELD,  # quark field (flavor f)
        "D_mu": _E,  # covariant derivative
        "m_f": _NAT_MASS,  # quark mass
        "G": Dimension(mass=2, length=4, time=-4)  # gluon field strength [M]^2
    },
    domain="quantum_field_theory",
    tags=["qft", "lagrangian", "qcd", "gauge theory", "strong force", "quarks", "gluons"],
    description="Quantum chromodynamics Lagrangian density",
    latex=r"\mathcal{L} = \sum_f \bar{\psi}_f(i\gamma^\mu D_\mu - m_f)\psi_f - \frac{1}{4}G^a_{\mu\nu}G^{\mu\nu}_a",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "D_μ = ∂_μ - ig_s T^a A^a_μ is the covariant derivative",
        "G^a_μν is the gluon field strength tensor",
        "T^a are SU(3) color generators",
        "g_s is the strong coupling constant",
        "Sum over quark flavors f = u,d,s,c,b,t"
    ],
    related=["QED Lagrangian", "Dirac Field Lagrangian"],
))

# Energy-momentum relations
register_equation(Equation(
    name="Relativistic Energy-Momentum (natural units)",
    formula="E^2 = p^2 + m^2",
    variables={
        "E": _NAT_ENERGY,
        "p": _NAT_MOMENTUM,
        "m": _NAT_MASS
    },
    domain="quantum_field_theory",
    tags=["qft", "relativistic", "energy", "momentum", "natural units"],
    description="Energy-momentum relation in natural units (c = ℏ = 1)",
    latex=r"E^2 = p^2 + m^2",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "On-shell particle (satisfies mass-shell condition)"
    ],
    related=["Energy-Momentum Relation", "Klein-Gordon Equation"],
))

# Feynman propagators
register_equation(Equation(
    name="Scalar Feynman Propagator",
    formula="D_F(p) = 1/(p^2 - m^2 + i*epsilon)",
    variables={
        "D_F": Dimension(mass=-2, length=-4, time=4),  # [M]^(-2)
        "p": _NAT_MOMENTUM,  # 4-momentum squared
        "m": _NAT_MASS,
        "epsilon": _DIMLESS  # infinitesimal
    },
    domain="quantum_field_theory",
    tags=["qft", "propagator", "scalar", "feynman rules"],
    description="Momentum-space Feynman propagator for scalar bosons",
    latex=r"D_F(p) = \frac{1}{p^2 - m^2 + i\epsilon}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "p^2 = p_μ p^μ = E^2 - |p⃗|^2",
        "iε prescription for time-ordering",
        "Momentum space representation"
    ],
    related=["Fermion Feynman Propagator", "Photon Feynman Propagator", "Klein-Gordon Equation"],
))

register_equation(Equation(
    name="Fermion Feynman Propagator",
    formula="S_F(p) = (gamma^mu*p_mu + m)/(p^2 - m^2 + i*epsilon)",
    variables={
        "S_F": Dimension(mass=-1, length=-2, time=2),  # [M]^(-1) with spinor indices
        "gamma": _DIMLESS,
        "p_mu": _NAT_MOMENTUM,
        "m": _NAT_MASS,
        "epsilon": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "propagator", "fermion", "dirac", "feynman rules"],
    description="Momentum-space Feynman propagator for Dirac fermions",
    latex=r"S_F(p) = \frac{\gamma^\mu p_\mu + m}{p^2 - m^2 + i\epsilon}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "γ^μ p_μ is Feynman slash notation",
        "Returns a 4×4 matrix in spinor space",
        "iε prescription for time-ordering"
    ],
    related=["Scalar Feynman Propagator", "Dirac Equation (free particle)"],
))

register_equation(Equation(
    name="Photon Feynman Propagator",
    formula="D^mu_nu(k) = -g^mu_nu/(k^2 + i*epsilon)",
    variables={
        "D": Dimension(mass=-2, length=-4, time=4),  # [M]^(-2) with Lorentz indices
        "g": _DIMLESS,  # metric tensor
        "k": _NAT_MOMENTUM,  # photon 4-momentum
        "epsilon": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "propagator", "photon", "gauge boson", "feynman rules", "qed"],
    description="Momentum-space Feynman propagator for photons (Feynman gauge)",
    latex=r"D^{\mu\nu}(k) = \frac{-g^{\mu\nu}}{k^2 + i\epsilon}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Feynman gauge (ξ = 1)",
        "Massless photon (k^2 = 0 on-shell)",
        "g^μν is the Minkowski metric"
    ],
    related=["Scalar Feynman Propagator", "QED Lagrangian"],
))

# Yukawa potential
register_equation(Equation(
    name="Yukawa Potential",
    formula="V(r) = -g^2*exp(-m*r)/(4*pi*r)",
    variables={
        "V": _E,  # potential energy
        "g": _DIMLESS,  # coupling constant
        "m": _NAT_MASS,  # mediator mass
        "r": _NAT_LENGTH,  # distance
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "potential", "yukawa", "force", "meson"],
    description="Potential from exchange of massive scalar boson",
    latex=r"V(r) = -\frac{g^2 e^{-mr}}{4\pi r}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Static potential approximation",
        "Mediated by scalar boson of mass m",
        "Reduces to Coulomb potential when m → 0"
    ],
    related=["Scalar Feynman Propagator"],
))

# Cross-sections
register_equation(Equation(
    name="Compton Scattering Cross-section",
    formula="sigma = (pi*alpha^2/m_e^2)*[(1+x)/x^3]*[2*x*(1+x)/(1+2*x) - ln(1+2*x)] + ln(1+2*x)/(2*x) - (1+3*x)/(1+2*x)^2",
    variables={
        "sigma": _CROSS_SECTION,
        "alpha": _DIMLESS,  # fine structure constant
        "m_e": _NAT_MASS,  # electron mass
        "x": _DIMLESS,  # E_gamma/m_e (photon energy ratio)
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "cross-section", "compton", "scattering", "qed", "photon", "electron"],
    description="Total cross-section for Compton scattering (Klein-Nishina formula)",
    latex=r"\sigma = \frac{\pi\alpha^2}{m_e^2}\frac{1+x}{x^3}\left[2x\frac{1+x}{1+2x} - \ln(1+2x)\right] + \frac{\ln(1+2x)}{2x} - \frac{1+3x}{(1+2x)^2}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "x = E_γ/m_e (photon energy in units of electron mass)",
        "Klein-Nishina formula (exact QED result)",
        "Unpolarized cross-section"
    ],
    related=["Fine Structure Constant", "QED Lagrangian"],
))

register_equation(Equation(
    name="Bhabha Scattering Cross-section (center-of-mass)",
    formula="dsigma/dOmega = (alpha^2/4*s)*[(s^2+u^2)/t^2 + (s^2+t^2)/u^2 + 2*m_e^2*s*(1/t^2+1/u^2)]",
    variables={
        "dsigma": _CROSS_SECTION,  # differential cross-section
        "dOmega": _DIMLESS,  # solid angle element
        "alpha": _DIMLESS,
        "s": Dimension(mass=2, length=4, time=-4),  # Mandelstam variable [M]^2
        "t": Dimension(mass=2, length=4, time=-4),
        "u": Dimension(mass=2, length=4, time=-4),
        "m_e": _NAT_MASS
    },
    domain="quantum_field_theory",
    tags=["qft", "cross-section", "bhabha", "scattering", "qed", "electron", "positron"],
    description="Differential cross-section for electron-positron scattering (e+e- → e+e-)",
    latex=r"\frac{d\sigma}{d\Omega} = \frac{\alpha^2}{4s}\left[\frac{s^2+u^2}{t^2} + \frac{s^2+t^2}{u^2} + \frac{2m_e^2 s}{t^2} + \frac{2m_e^2 s}{u^2}\right]",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Center-of-mass frame",
        "s, t, u are Mandelstam variables: s + t + u = 4m_e^2",
        "Tree-level QED (one-photon exchange)"
    ],
    related=["Møller Scattering Cross-section", "QED Lagrangian"],
))

register_equation(Equation(
    name="Møller Scattering Cross-section (center-of-mass)",
    formula="dsigma/dOmega = (alpha^2/4*s)*[(s^2+u^2)/t^2 + (s^2+t^2)/u^2 + (t^2+u^2)/s^2]",
    variables={
        "dsigma": _CROSS_SECTION,
        "dOmega": _DIMLESS,
        "alpha": _DIMLESS,
        "s": Dimension(mass=2, length=4, time=-4),
        "t": Dimension(mass=2, length=4, time=-4),
        "u": Dimension(mass=2, length=4, time=-4)
    },
    domain="quantum_field_theory",
    tags=["qft", "cross-section", "moller", "scattering", "qed", "electron"],
    description="Differential cross-section for electron-electron scattering (e-e- → e-e-)",
    latex=r"\frac{d\sigma}{d\Omega} = \frac{\alpha^2}{4s}\left[\frac{s^2+u^2}{t^2} + \frac{s^2+t^2}{u^2} + \frac{t^2+u^2}{s^2}\right]",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Center-of-mass frame",
        "s, t, u are Mandelstam variables",
        "Tree-level QED with exchange and annihilation diagrams",
        "Identical fermions (includes symmetrization)"
    ],
    related=["Bhabha Scattering Cross-section", "QED Lagrangian"],
))

register_equation(Equation(
    name="Pair Production Cross-section (threshold)",
    formula="sigma = (4*pi*alpha^2/3*m_e^2)*beta*(3-beta^4)/(2*(2-beta^2))",
    variables={
        "sigma": _CROSS_SECTION,
        "alpha": _DIMLESS,
        "m_e": _NAT_MASS,
        "beta": _DIMLESS,  # v/c = sqrt(1 - 4m_e^2/s)
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "cross-section", "pair production", "qed", "electron", "positron"],
    description="Total cross-section for electron-positron pair production from photons near threshold",
    latex=r"\sigma = \frac{4\pi\alpha^2}{3m_e^2}\beta\frac{3-\beta^4}{2(2-\beta^2)}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "β = v/c = √(1 - 4m_e²/s)",
        "s is the center-of-mass energy squared",
        "Near threshold approximation",
        "Unpolarized cross-section"
    ],
    related=["Compton Scattering Cross-section", "QED Lagrangian"],
))

# Decay rates
register_equation(Equation(
    name="Muon Decay Rate",
    formula="Gamma = (G_F^2*m_mu^5)/(192*pi^3)",
    variables={
        "Gamma": _DECAY_RATE,
        "G_F": Dimension(mass=-2, length=-4, time=4),  # Fermi constant [M]^(-2)
        "m_mu": _NAT_MASS,  # muon mass
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "decay", "muon", "weak force", "fermi", "lifetime"],
    description="Decay rate for muon decay (μ- → e- ν̄_e ν_μ)",
    latex=r"\Gamma = \frac{G_F^2 m_\mu^5}{192\pi^3}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "G_F ≈ 1.166 × 10^-5 GeV^-2 (Fermi constant)",
        "Neglects electron mass (m_e << m_μ)",
        "Tree-level weak interaction",
        "Lifetime τ = 1/Γ"
    ],
    related=["Fermi's Golden Rule", "Pion Decay Constant"],
))

register_equation(Equation(
    name="Pion Decay Constant",
    formula="Gamma_pi = (G_F^2*f_pi^2*m_pi*m_mu^2/4*pi)*(1 - m_mu^2/m_pi^2)^2",
    variables={
        "Gamma_pi": _DECAY_RATE,
        "G_F": Dimension(mass=-2, length=-4, time=4),
        "f_pi": _E,  # pion decay constant [M]
        "m_pi": _NAT_MASS,  # pion mass
        "m_mu": _NAT_MASS,  # muon mass
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "decay", "pion", "meson", "weak force"],
    description="Decay rate for charged pion decay (π+ → μ+ ν_μ)",
    latex=r"\Gamma_\pi = \frac{G_F^2 f_\pi^2 m_\pi m_\mu^2}{4\pi}\left(1 - \frac{m_\mu^2}{m_\pi^2}\right)^2",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "f_π ≈ 92.4 MeV (pion decay constant)",
        "Helicity suppression factor (1 - m_μ²/m_π²)²",
        "Two-body decay"
    ],
    related=["Muon Decay Rate"],
))

register_equation(Equation(
    name="Fermi's Golden Rule (decay rate)",
    formula="Gamma = (2*pi)*|M_fi|^2*rho(E_f)",
    variables={
        "Gamma": _DECAY_RATE,
        "M_fi": _DIMLESS,  # matrix element (dimensionless in chosen normalization)
        "rho": Dimension(mass=-1, length=-2, time=2),  # density of states [M]^(-1)
        "E_f": _NAT_ENERGY,  # final state energy
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "decay", "fermi", "golden rule", "transition rate"],
    description="Transition rate from initial to final state",
    latex=r"\Gamma = 2\pi |\mathcal{M}_{fi}|^2 \rho(E_f)",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Time-independent perturbation theory",
        "ρ(E_f) is the density of final states",
        "M_fi is the matrix element of the interaction Hamiltonian"
    ],
    related=["Muon Decay Rate", "Breit-Wigner Resonance"],
))

# Coupling constants and resonances
register_equation(Equation(
    name="Fine Structure Constant",
    formula="alpha = e^2/(4*pi)",
    variables={
        "alpha": _DIMLESS,
        "e": _DIMLESS,  # electric charge (dimensionless in natural units)
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "coupling constant", "qed", "alpha", "fundamental"],
    description="Electromagnetic coupling constant (α ≈ 1/137)",
    latex=r"\alpha = \frac{e^2}{4\pi}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "α ≈ 1/137.036 at low energy",
        "Running coupling: α = α(Q²)",
        "Dimensionless coupling"
    ],
    related=["QED Lagrangian", "Running Coupling Constant"],
))

register_equation(Equation(
    name="Running Coupling Constant (QED)",
    formula="alpha(Q^2) = alpha(mu^2)/(1 - (alpha(mu^2)/(3*pi))*ln(Q^2/mu^2))",
    variables={
        "alpha": _DIMLESS,
        "Q": _NAT_MOMENTUM,  # energy scale
        "mu": _NAT_MOMENTUM,  # reference scale
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "coupling constant", "qed", "running", "renormalization"],
    description="Energy-dependent fine structure constant (one-loop QED)",
    latex=r"\alpha(Q^2) = \frac{\alpha(\mu^2)}{1 - \frac{\alpha(\mu^2)}{3\pi}\ln\frac{Q^2}{\mu^2}}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "One-loop beta function approximation",
        "Q² is the energy scale (momentum transfer squared)",
        "Vacuum polarization effects",
        "Valid for Q² >> m_e²"
    ],
    related=["Fine Structure Constant", "QCD Running Coupling"],
))

register_equation(Equation(
    name="QCD Running Coupling (asymptotic freedom)",
    formula="alpha_s(Q^2) = (4*pi)/((11 - 2*n_f/3)*ln(Q^2/Lambda_QCD^2))",
    variables={
        "alpha_s": _DIMLESS,
        "Q": _NAT_MOMENTUM,
        "n_f": _DIMLESS,  # number of active quark flavors
        "Lambda_QCD": _NAT_ENERGY,  # QCD scale parameter
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "coupling constant", "qcd", "running", "asymptotic freedom", "strong force"],
    description="Energy-dependent strong coupling constant (one-loop QCD)",
    latex=r"\alpha_s(Q^2) = \frac{4\pi}{(11 - \frac{2n_f}{3})\ln\frac{Q^2}{\Lambda_{QCD}^2}}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "One-loop beta function",
        "n_f = number of active quark flavors (typically 3-6)",
        "Λ_QCD ≈ 200 MeV (QCD scale parameter)",
        "Asymptotic freedom: α_s → 0 as Q² → ∞",
        "Breaks down at low Q² (confinement regime)"
    ],
    related=["QCD Lagrangian", "Running Coupling Constant (QED)"],
))

register_equation(Equation(
    name="Breit-Wigner Resonance",
    formula="sigma(E) = (12*pi/m_R^2)*((2*J+1)/((2*s_a+1)*(2*s_b+1)))*Gamma_a*Gamma_b/((E^2-m_R^2)^2 + m_R^2*Gamma_R^2)",
    variables={
        "sigma": _CROSS_SECTION,
        "E": _NAT_ENERGY,  # center-of-mass energy
        "m_R": _NAT_MASS,  # resonance mass
        "J": _DIMLESS,  # total angular momentum
        "s_a": _DIMLESS,  # spin of particle a
        "s_b": _DIMLESS,  # spin of particle b
        "Gamma_a": _DECAY_RATE,  # partial width to channel a
        "Gamma_b": _DECAY_RATE,  # partial width to channel b
        "Gamma_R": _DECAY_RATE,  # total width
        "pi": _DIMLESS
    },
    domain="quantum_field_theory",
    tags=["qft", "resonance", "breit-wigner", "cross-section", "scattering"],
    description="Cross-section for resonant scattering (Breit-Wigner formula)",
    latex=r"\sigma(E) = \frac{12\pi}{m_R^2}\frac{2J+1}{(2s_a+1)(2s_b+1)}\frac{\Gamma_a\Gamma_b}{(E^2-m_R^2)^2 + m_R^2\Gamma_R^2}",
    assumptions=[
        "Natural units (c = ℏ = 1)",
        "Narrow resonance (Γ_R << m_R)",
        "J is the spin of the resonance",
        "Γ_a, Γ_b are partial widths to initial and final channels",
        "Relativistic Breit-Wigner form"
    ],
    related=["Fermi's Golden Rule (decay rate)"],
))
