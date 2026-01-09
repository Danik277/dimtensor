"""Variable name heuristics for dimensional inference.

This module provides heuristic-based inference of physical dimensions
from variable names. It uses patterns like "velocity" → L/T to suggest
dimensions for unnamed quantities.

Example:
    >>> from dimtensor.inference import infer_dimension
    >>> dim, confidence = infer_dimension("velocity")
    >>> print(dim)  # Dimension(length=1, time=-1)
    >>> print(confidence)  # 0.9
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from ..core.dimensions import Dimension


class InferenceResult(NamedTuple):
    """Result of dimensional inference.

    Attributes:
        dimension: Inferred dimension.
        confidence: Confidence level (0.0 to 1.0).
        pattern: The pattern that matched.
        source: Source of the inference ("exact", "prefix", "suffix", "partial").
    """

    dimension: Dimension
    confidence: float
    pattern: str
    source: str


# ==============================================================================
# Variable Name → Dimension Mappings
# ==============================================================================

# High-confidence exact matches (0.9)
VARIABLE_PATTERNS: dict[str, Dimension] = {
    # Mechanics - Motion
    "velocity": Dimension(length=1, time=-1),
    "speed": Dimension(length=1, time=-1),
    "v": Dimension(length=1, time=-1),  # Common abbreviation
    "acceleration": Dimension(length=1, time=-2),
    "accel": Dimension(length=1, time=-2),
    "a": Dimension(length=1, time=-2),  # Common abbreviation
    "jerk": Dimension(length=1, time=-3),
    "position": Dimension(length=1),
    "displacement": Dimension(length=1),
    "x": Dimension(length=1),  # Common position variable
    "y": Dimension(length=1),
    "z": Dimension(length=1),
    "r": Dimension(length=1),  # Radius/position
    # Mechanics - Forces & Energy
    "force": Dimension(length=1, mass=1, time=-2),
    "f": Dimension(length=1, mass=1, time=-2),
    "momentum": Dimension(length=1, mass=1, time=-1),
    "p": Dimension(length=1, mass=1, time=-1),  # Momentum
    "impulse": Dimension(length=1, mass=1, time=-1),
    "energy": Dimension(length=2, mass=1, time=-2),
    "e": Dimension(length=2, mass=1, time=-2),
    "kinetic_energy": Dimension(length=2, mass=1, time=-2),
    "potential_energy": Dimension(length=2, mass=1, time=-2),
    "work": Dimension(length=2, mass=1, time=-2),
    "w": Dimension(length=2, mass=1, time=-2),
    "power": Dimension(length=2, mass=1, time=-3),
    "torque": Dimension(length=2, mass=1, time=-2),
    "angular_momentum": Dimension(length=2, mass=1, time=-1),
    "moment_of_inertia": Dimension(length=2, mass=1),
    # Mechanics - Pressure & Stress
    "pressure": Dimension(length=-1, mass=1, time=-2),
    "stress": Dimension(length=-1, mass=1, time=-2),
    "strain": Dimension(),  # Dimensionless
    # Geometry
    "distance": Dimension(length=1),
    "length": Dimension(length=1),
    "width": Dimension(length=1),
    "height": Dimension(length=1),
    "depth": Dimension(length=1),
    "radius": Dimension(length=1),
    "diameter": Dimension(length=1),
    "circumference": Dimension(length=1),
    "wavelength": Dimension(length=1),
    "area": Dimension(length=2),
    "surface_area": Dimension(length=2),
    "cross_section": Dimension(length=2),
    "volume": Dimension(length=3),
    "angle": Dimension(),  # Dimensionless (radians)
    "theta": Dimension(),
    "phi": Dimension(),
    "omega": Dimension(time=-1),  # Angular velocity
    "angular_velocity": Dimension(time=-1),
    "angular_acceleration": Dimension(time=-2),
    # Time
    "time": Dimension(time=1),
    "t": Dimension(time=1),
    "duration": Dimension(time=1),
    "period": Dimension(time=1),
    "frequency": Dimension(time=-1),
    "freq": Dimension(time=-1),
    "f": Dimension(time=-1),  # Note: conflicts with force, lower priority
    "rate": Dimension(time=-1),
    # Mass
    "mass": Dimension(mass=1),
    "m": Dimension(mass=1),
    "weight": Dimension(length=1, mass=1, time=-2),  # Actually force!
    "density": Dimension(length=-3, mass=1),
    "rho": Dimension(length=-3, mass=1),
    "specific_gravity": Dimension(),  # Dimensionless
    # Temperature & Thermodynamics
    "temperature": Dimension(temperature=1),
    "temp": Dimension(temperature=1),
    "t": Dimension(temperature=1),  # Note: conflicts with time
    "heat": Dimension(length=2, mass=1, time=-2),  # Energy
    "q": Dimension(length=2, mass=1, time=-2),  # Heat
    "entropy": Dimension(length=2, mass=1, time=-2, temperature=-1),
    "specific_heat": Dimension(length=2, time=-2, temperature=-1),
    "thermal_conductivity": Dimension(length=1, mass=1, time=-3, temperature=-1),
    # Electromagnetics
    "current": Dimension(current=1),
    "i": Dimension(current=1),
    "voltage": Dimension(length=2, mass=1, time=-3, current=-1),
    "potential": Dimension(length=2, mass=1, time=-3, current=-1),
    "v": Dimension(length=2, mass=1, time=-3, current=-1),  # Conflicts with velocity
    "resistance": Dimension(length=2, mass=1, time=-3, current=-2),
    "r": Dimension(length=2, mass=1, time=-3, current=-2),  # Conflicts with radius
    "impedance": Dimension(length=2, mass=1, time=-3, current=-2),
    "capacitance": Dimension(length=-2, mass=-1, time=4, current=2),
    "inductance": Dimension(length=2, mass=1, time=-2, current=-2),
    "charge": Dimension(current=1, time=1),
    "q": Dimension(current=1, time=1),  # Conflicts with heat
    "electric_field": Dimension(length=1, mass=1, time=-3, current=-1),
    "magnetic_field": Dimension(mass=1, time=-2, current=-1),
    "magnetic_flux": Dimension(length=2, mass=1, time=-2, current=-1),
    "flux": Dimension(length=2, mass=1, time=-2, current=-1),
    # Fluids
    "flow_rate": Dimension(length=3, time=-1),
    "viscosity": Dimension(length=-1, mass=1, time=-1),
    "kinematic_viscosity": Dimension(length=2, time=-1),
    "surface_tension": Dimension(mass=1, time=-2),
    # Chemistry
    "concentration": Dimension(amount=1, length=-3),
    "molarity": Dimension(amount=1, length=-3),
    "molar_mass": Dimension(mass=1, amount=-1),
    # Optics
    "intensity": Dimension(mass=1, time=-3),  # Power per area
    "irradiance": Dimension(mass=1, time=-3),
    "luminosity": Dimension(luminosity=1),
    # Acoustics
    "sound_pressure": Dimension(length=-1, mass=1, time=-2),
    "sound_intensity": Dimension(mass=1, time=-3),
}

# Suffix patterns (medium confidence, 0.7)
SUFFIX_PATTERNS: dict[str, Dimension] = {
    # SI base units
    "_m": Dimension(length=1),
    "_meter": Dimension(length=1),
    "_meters": Dimension(length=1),
    "_kg": Dimension(mass=1),
    "_kilogram": Dimension(mass=1),
    "_s": Dimension(time=1),
    "_sec": Dimension(time=1),
    "_second": Dimension(time=1),
    "_seconds": Dimension(time=1),
    "_a": Dimension(current=1),
    "_amp": Dimension(current=1),
    "_ampere": Dimension(current=1),
    "_k": Dimension(temperature=1),
    "_kelvin": Dimension(temperature=1),
    "_mol": Dimension(amount=1),
    "_mole": Dimension(amount=1),
    "_cd": Dimension(luminosity=1),
    "_candela": Dimension(luminosity=1),
    # Derived units
    "_n": Dimension(length=1, mass=1, time=-2),  # Newton
    "_newton": Dimension(length=1, mass=1, time=-2),
    "_j": Dimension(length=2, mass=1, time=-2),  # Joule
    "_joule": Dimension(length=2, mass=1, time=-2),
    "_w": Dimension(length=2, mass=1, time=-3),  # Watt
    "_watt": Dimension(length=2, mass=1, time=-3),
    "_pa": Dimension(length=-1, mass=1, time=-2),  # Pascal
    "_pascal": Dimension(length=-1, mass=1, time=-2),
    "_hz": Dimension(time=-1),  # Hertz
    "_hertz": Dimension(time=-1),
    "_v": Dimension(length=2, mass=1, time=-3, current=-1),  # Volt
    "_volt": Dimension(length=2, mass=1, time=-3, current=-1),
    "_ohm": Dimension(length=2, mass=1, time=-3, current=-2),
    "_c": Dimension(current=1, time=1),  # Coulomb
    "_coulomb": Dimension(current=1, time=1),
    # Compound units
    "_m_per_s": Dimension(length=1, time=-1),
    "_m_s": Dimension(length=1, time=-1),  # Alternative
    "_mps": Dimension(length=1, time=-1),
    "_m_per_s2": Dimension(length=1, time=-2),
    "_m_s2": Dimension(length=1, time=-2),
    "_kg_per_m3": Dimension(mass=1, length=-3),
    "_kg_m3": Dimension(mass=1, length=-3),
    "_n_per_m2": Dimension(length=-1, mass=1, time=-2),  # Pa
}

# Prefix patterns (medium confidence, 0.7)
# None means "modifier only" - inherit dimension from rest of name
PREFIX_PATTERNS: dict[str, Dimension | None] = {
    "initial_": None,  # Modifier, inherit from rest
    "final_": None,
    "max_": None,
    "min_": None,
    "avg_": None,
    "average_": None,
    "mean_": None,
    "total_": None,
    "net_": None,
    "delta_": None,  # Change in quantity
    "d_": None,  # Differential
}

# Component suffixes (lower priority, 0.6)
COMPONENT_PATTERNS: dict[str, None] = {
    "_x": None,  # X-component
    "_y": None,  # Y-component
    "_z": None,  # Z-component
    "_r": None,  # Radial
    "_theta": None,  # Angular
    "_phi": None,
    "_component": None,
    "_magnitude": None,
}


def infer_dimension(
    name: str, min_confidence: float = 0.5
) -> InferenceResult | None:
    """Infer physical dimension from a variable name.

    Uses heuristics based on common physics naming conventions to suggest
    a dimension for the given variable name.

    Args:
        name: Variable name to analyze.
        min_confidence: Minimum confidence threshold (0.0 to 1.0).

    Returns:
        InferenceResult with dimension and confidence, or None if no match.

    Examples:
        >>> result = infer_dimension("velocity")
        >>> result.dimension
        Dimension(length=1, time=-1)
        >>> result.confidence
        0.9

        >>> result = infer_dimension("initial_velocity_x")
        >>> result.dimension
        Dimension(length=1, time=-1)
        >>> result.confidence
        0.7

        >>> infer_dimension("foo")  # No match
        None
    """
    results = get_matching_patterns(name)

    if not results:
        return None

    # Return highest confidence result above threshold
    best = results[0]
    if best.confidence >= min_confidence:
        return best
    return None


def get_matching_patterns(name: str) -> list[InferenceResult]:
    """Get all matching patterns for a variable name.

    Returns a list of possible dimension inferences, sorted by confidence.

    Args:
        name: Variable name to analyze.

    Returns:
        List of InferenceResult, sorted by confidence (highest first).
    """
    name_lower = name.lower()
    results: list[InferenceResult] = []

    # Check exact match (highest confidence)
    if name_lower in VARIABLE_PATTERNS:
        results.append(
            InferenceResult(
                dimension=VARIABLE_PATTERNS[name_lower],
                confidence=0.9,
                pattern=name_lower,
                source="exact",
            )
        )

    # Strip prefixes and try again
    stripped_name = name_lower
    for prefix in PREFIX_PATTERNS:
        if name_lower.startswith(prefix):
            stripped_name = name_lower[len(prefix) :]
            break

    # Strip component suffixes
    for suffix in COMPONENT_PATTERNS:
        if stripped_name.endswith(suffix):
            stripped_name = stripped_name[: -len(suffix)]
            break

    # Check after stripping prefixes/suffixes
    if stripped_name != name_lower and stripped_name in VARIABLE_PATTERNS:
        results.append(
            InferenceResult(
                dimension=VARIABLE_PATTERNS[stripped_name],
                confidence=0.7,
                pattern=stripped_name,
                source="prefix_stripped",
            )
        )

    # Check suffix patterns (longer suffixes first)
    sorted_suffixes = sorted(SUFFIX_PATTERNS.keys(), key=len, reverse=True)
    for suffix in sorted_suffixes:
        if name_lower.endswith(suffix):
            results.append(
                InferenceResult(
                    dimension=SUFFIX_PATTERNS[suffix],
                    confidence=0.7,
                    pattern=suffix,
                    source="suffix",
                )
            )
            break

    # Check partial matches (lowest confidence)
    for pattern, dim in VARIABLE_PATTERNS.items():
        if len(pattern) >= 4 and pattern in name_lower and pattern not in [
            r.pattern for r in results
        ]:
            results.append(
                InferenceResult(
                    dimension=dim,
                    confidence=0.5,
                    pattern=pattern,
                    source="partial",
                )
            )

    # Sort by confidence (highest first)
    results.sort(key=lambda r: r.confidence, reverse=True)

    return results


def suggest_dimension(name: str) -> str:
    """Get a human-readable suggestion for a variable's dimension.

    Args:
        name: Variable name to analyze.

    Returns:
        Human-readable suggestion string.

    Example:
        >>> suggest_dimension("velocity")
        "velocity: likely L·T⁻¹ (m/s) - high confidence"
    """
    result = infer_dimension(name)

    if result is None:
        return f"{name}: no dimension inference available"

    # Format dimension nicely
    dim_str = str(result.dimension)

    # Confidence level
    if result.confidence >= 0.8:
        conf_str = "high confidence"
    elif result.confidence >= 0.6:
        conf_str = "medium confidence"
    else:
        conf_str = "low confidence"

    return f"{name}: likely {dim_str} - {conf_str} (from {result.source})"
