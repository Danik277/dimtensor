"""Equation database for physics and engineering.

Provides a registry of physics equations with dimensional metadata
for use in dimensional analysis, validation, and physics-informed ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
