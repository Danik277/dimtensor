"""Database of dimensionless numbers and regime inference.

This module provides a comprehensive database of common dimensionless numbers
used in physics and engineering, along with functions to compute them from
dimensional parameters and infer physical regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.dimarray import DimArray


def _extract_dimensionless(arr: DimArray) -> float:
    """Extract scalar value from a dimensionless DimArray.

    Args:
        arr: DimArray that should be dimensionless

    Returns:
        Float value

    Raises:
        ValueError: If array is not dimensionless or not scalar
    """
    if not arr.is_dimensionless:
        raise ValueError(
            f"Expected dimensionless result but got dimension {arr.dimension}"
        )

    # Handle scalar or single-element arrays
    data = arr.magnitude()
    if data.size == 1:
        return float(data.item())
    else:
        # Take mean for multi-element arrays (conservative)
        return float(np.mean(data))


@dataclass(frozen=True)
class DimensionlessNumber:
    """Represents a dimensionless number with its formula and interpretation.

    Attributes:
        name: Full name (e.g., "Reynolds number")
        symbol: Standard symbol (e.g., "Re")
        formula: LaTeX formula for display
        required_params: List of parameter names required to compute it
        compute_func: Function that takes parameters dict and returns dimensionless value
        description: Physical interpretation
        regime_map: Optional function mapping value to regime name
    """

    name: str
    symbol: str
    formula: str
    required_params: Tuple[str, ...]
    compute_func: Callable[[Dict[str, DimArray]], float]
    description: str
    regime_map: Optional[Callable[[float], str]] = None


# ============================================================================
# Regime mapping functions
# ============================================================================


def reynolds_regime(Re: float) -> str:
    """Map Reynolds number to flow regime."""
    if Re < 1:
        return "Stokes flow (creeping)"
    elif Re < 2300:
        return "laminar"
    elif Re < 4000:
        return "transitional"
    else:
        return "turbulent"


def mach_regime(Ma: float) -> str:
    """Map Mach number to compressibility regime."""
    if Ma < 0.3:
        return "incompressible"
    elif Ma < 0.8:
        return "subsonic"
    elif Ma < 1.2:
        return "transonic"
    elif Ma < 5.0:
        return "supersonic"
    else:
        return "hypersonic"


def froude_regime(Fr: float) -> str:
    """Map Froude number to flow regime."""
    if Fr < 1:
        return "subcritical (gravity-dominated)"
    elif Fr == 1:
        return "critical"
    else:
        return "supercritical (inertia-dominated)"


def grashof_regime(Gr: float) -> str:
    """Map Grashof number to convection regime."""
    if Gr < 1e8:
        return "laminar natural convection"
    else:
        return "turbulent natural convection"


def rayleigh_regime(Ra: float) -> str:
    """Map Rayleigh number to convection regime."""
    if Ra < 1708:
        return "conductive (no convection)"
    elif Ra < 1e8:
        return "laminar convection"
    else:
        return "turbulent convection"


def knudsen_regime(Kn: float) -> str:
    """Map Knudsen number to flow regime."""
    if Kn < 0.01:
        return "continuum"
    elif Kn < 0.1:
        return "slip flow"
    elif Kn < 10:
        return "transitional"
    else:
        return "free molecular"


# ============================================================================
# Dimensionless number definitions
# ============================================================================

# Fluid Mechanics
REYNOLDS = DimensionlessNumber(
    name="Reynolds number",
    symbol="Re",
    formula=r"Re = \frac{\rho v L}{\mu} = \frac{v L}{\nu}",
    required_params=("density", "velocity", "length", "viscosity"),
    compute_func=lambda p: _extract_dimensionless(
        p["density"] * p["velocity"] * p["length"] / p["viscosity"]
    ),
    description="Ratio of inertial to viscous forces. Characterizes laminar vs turbulent flow.",
    regime_map=reynolds_regime,
)

MACH = DimensionlessNumber(
    name="Mach number",
    symbol="Ma",
    formula=r"Ma = \frac{v}{c}",
    required_params=("velocity", "sound_speed"),
    compute_func=lambda p: _extract_dimensionless(
        p["velocity"] / p["sound_speed"]
    ),
    description="Ratio of flow velocity to speed of sound. Characterizes compressibility effects.",
    regime_map=mach_regime,
)

FROUDE = DimensionlessNumber(
    name="Froude number",
    symbol="Fr",
    formula=r"Fr = \frac{v}{\sqrt{g L}}",
    required_params=("velocity", "gravity", "length"),
    compute_func=lambda p: _extract_dimensionless(
        p["velocity"] / (p["gravity"] * p["length"]) ** 0.5
    ),
    description="Ratio of inertial to gravitational forces. Important in open-channel flow.",
    regime_map=froude_regime,
)

WEBER = DimensionlessNumber(
    name="Weber number",
    symbol="We",
    formula=r"We = \frac{\rho v^2 L}{\sigma}",
    required_params=("density", "velocity", "length", "surface_tension"),
    compute_func=lambda p: _extract_dimensionless(
        p["density"] * p["velocity"] ** 2 * p["length"] / p["surface_tension"]
    ),
    description="Ratio of inertial to surface tension forces. Important in multiphase flows.",
)

CAPILLARY = DimensionlessNumber(
    name="Capillary number",
    symbol="Ca",
    formula=r"Ca = \frac{\mu v}{\sigma}",
    required_params=("viscosity", "velocity", "surface_tension"),
    compute_func=lambda p: _extract_dimensionless(
        p["viscosity"] * p["velocity"] / p["surface_tension"]
    ),
    description="Ratio of viscous to surface tension forces. Important in microfluidics.",
)

BOND = DimensionlessNumber(
    name="Bond number",
    symbol="Bo",
    formula=r"Bo = \frac{\rho g L^2}{\sigma}",
    required_params=("density", "gravity", "length", "surface_tension"),
    compute_func=lambda p: _extract_dimensionless(
        p["density"] * p["gravity"] * p["length"] ** 2 / p["surface_tension"]
    ),
    description="Ratio of gravitational to surface tension forces. Important for drops and bubbles.",
)

STROUHAL = DimensionlessNumber(
    name="Strouhal number",
    symbol="St",
    formula=r"St = \frac{f L}{v}",
    required_params=("frequency", "length", "velocity"),
    compute_func=lambda p: _extract_dimensionless(
        p["frequency"] * p["length"] / p["velocity"]
    ),
    description="Ratio of oscillation frequency to flow velocity. Important in vortex shedding.",
)

KNUDSEN = DimensionlessNumber(
    name="Knudsen number",
    symbol="Kn",
    formula=r"Kn = \frac{\lambda}{L}",
    required_params=("mean_free_path", "length"),
    compute_func=lambda p: _extract_dimensionless(
        p["mean_free_path"] / p["length"]
    ),
    description="Ratio of molecular mean free path to characteristic length. Determines continuum validity.",
    regime_map=knudsen_regime,
)

# Heat Transfer
PRANDTL = DimensionlessNumber(
    name="Prandtl number",
    symbol="Pr",
    formula=r"Pr = \frac{\nu}{\alpha} = \frac{\mu c_p}{k}",
    required_params=("viscosity", "specific_heat", "thermal_conductivity"),
    compute_func=lambda p: _extract_dimensionless(
        p["viscosity"] * p["specific_heat"] / p["thermal_conductivity"]
    ),
    description="Ratio of momentum to thermal diffusivity. Relates velocity and temperature profiles.",
)

NUSSELT = DimensionlessNumber(
    name="Nusselt number",
    symbol="Nu",
    formula=r"Nu = \frac{h L}{k}",
    required_params=("heat_transfer_coefficient", "length", "thermal_conductivity"),
    compute_func=lambda p: _extract_dimensionless(
        p["heat_transfer_coefficient"] * p["length"] / p["thermal_conductivity"]
    ),
    description="Ratio of convective to conductive heat transfer. Dimensionless heat transfer coefficient.",
)

GRASHOF = DimensionlessNumber(
    name="Grashof number",
    symbol="Gr",
    formula=r"Gr = \frac{g \beta \Delta T L^3}{\nu^2}",
    required_params=("gravity", "thermal_expansion", "temperature_diff", "length", "kinematic_viscosity"),
    compute_func=lambda p: _extract_dimensionless(
        
            p["gravity"]
            * p["thermal_expansion"]
            * p["temperature_diff"]
            * p["length"] ** 3
            / p["kinematic_viscosity"] ** 2
        
    ),
    description="Ratio of buoyancy to viscous forces. Drives natural convection.",
    regime_map=grashof_regime,
)

RAYLEIGH = DimensionlessNumber(
    name="Rayleigh number",
    symbol="Ra",
    formula=r"Ra = Gr \cdot Pr = \frac{g \beta \Delta T L^3}{\nu \alpha}",
    required_params=("gravity", "thermal_expansion", "temperature_diff", "length", "kinematic_viscosity", "thermal_diffusivity"),
    compute_func=lambda p: _extract_dimensionless(
        p["gravity"]
        * p["thermal_expansion"]
        * p["temperature_diff"]
        * p["length"] ** 3
        / (p["kinematic_viscosity"] * p["thermal_diffusivity"])
    ),
    description="Product of Grashof and Prandtl. Determines onset of natural convection.",
    regime_map=rayleigh_regime,
)

PECLET = DimensionlessNumber(
    name="Peclet number",
    symbol="Pe",
    formula=r"Pe = Re \cdot Pr = \frac{v L}{\alpha}",
    required_params=("velocity", "length", "thermal_diffusivity"),
    compute_func=lambda p: _extract_dimensionless(
        p["velocity"] * p["length"] / p["thermal_diffusivity"]
    ),
    description="Ratio of advective to diffusive heat transport.",
)

BIOT = DimensionlessNumber(
    name="Biot number",
    symbol="Bi",
    formula=r"Bi = \frac{h L}{k}",
    required_params=("heat_transfer_coefficient", "length", "thermal_conductivity"),
    compute_func=lambda p: _extract_dimensionless(
        p["heat_transfer_coefficient"] * p["length"] / p["thermal_conductivity"]
    ),
    description="Ratio of internal to external thermal resistance. Determines temperature distribution.",
)

FOURIER = DimensionlessNumber(
    name="Fourier number",
    symbol="Fo",
    formula=r"Fo = \frac{\alpha t}{L^2}",
    required_params=("thermal_diffusivity", "time", "length"),
    compute_func=lambda p: _extract_dimensionless(
        p["thermal_diffusivity"] * p["time"] / p["length"] ** 2
    ),
    description="Dimensionless time for heat conduction. Measures penetration of thermal disturbance.",
)

ECKERT = DimensionlessNumber(
    name="Eckert number",
    symbol="Ec",
    formula=r"Ec = \frac{v^2}{c_p \Delta T}",
    required_params=("velocity", "specific_heat", "temperature_diff"),
    compute_func=lambda p: _extract_dimensionless(
        p["velocity"] ** 2 / (p["specific_heat"] * p["temperature_diff"])
    ),
    description="Ratio of kinetic energy to enthalpy. Measures viscous heating importance.",
)

# Electromagnetics
MAGNETIC_REYNOLDS = DimensionlessNumber(
    name="Magnetic Reynolds number",
    symbol="Rm",
    formula=r"Rm = \frac{\mu_0 \sigma v L}{1}",
    required_params=("permeability", "conductivity", "velocity", "length"),
    compute_func=lambda p: _extract_dimensionless(
        p["permeability"] * p["conductivity"] * p["velocity"] * p["length"]
    ),
    description="Ratio of magnetic advection to diffusion. Important in magnetohydrodynamics.",
)

# Database registry
_DIMENSIONLESS_NUMBERS: Dict[str, DimensionlessNumber] = {
    "Re": REYNOLDS,
    "Ma": MACH,
    "Fr": FROUDE,
    "We": WEBER,
    "Ca": CAPILLARY,
    "Bo": BOND,
    "St": STROUHAL,
    "Kn": KNUDSEN,
    "Pr": PRANDTL,
    "Nu": NUSSELT,
    "Gr": GRASHOF,
    "Ra": RAYLEIGH,
    "Pe": PECLET,
    "Bi": BIOT,
    "Fo": FOURIER,
    "Ec": ECKERT,
    "Rm": MAGNETIC_REYNOLDS,
}


# ============================================================================
# Public API
# ============================================================================


def list_dimensionless_numbers() -> List[str]:
    """List all available dimensionless numbers.

    Returns:
        List of symbol strings (e.g., ['Re', 'Ma', 'Fr', ...])
    """
    return sorted(_DIMENSIONLESS_NUMBERS.keys())


def get_dimensionless_number(symbol: str) -> DimensionlessNumber:
    """Get dimensionless number definition by symbol.

    Args:
        symbol: Symbol like 'Re', 'Ma', etc.

    Returns:
        DimensionlessNumber dataclass with formula and computation function.

    Raises:
        KeyError: If symbol not found in database.
    """
    if symbol not in _DIMENSIONLESS_NUMBERS:
        available = ", ".join(list_dimensionless_numbers())
        raise KeyError(
            f"Unknown dimensionless number: {symbol}. Available: {available}"
        )
    return _DIMENSIONLESS_NUMBERS[symbol]


def compute(symbol: str, **params: DimArray) -> float:
    """Compute a dimensionless number from parameters.

    Args:
        symbol: Symbol of the dimensionless number (e.g., 'Re')
        **params: Keyword arguments mapping parameter names to DimArrays.
                 Parameter names must match those in required_params.

    Returns:
        Dimensionless value as a float.

    Raises:
        KeyError: If symbol not found or required parameters missing.
        ValueError: If dimensional analysis fails (wrong dimensions).

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.analysis import dimensionless_numbers as dn
        >>>
        >>> Re = dn.compute(
        ...     'Re',
        ...     density=DimArray(1000, units.kg / units.m**3),
        ...     velocity=DimArray(1.0, units.m / units.s),
        ...     length=DimArray(0.1, units.m),
        ...     viscosity=DimArray(0.001, units.Pa * units.s),
        ... )
        >>> print(f"Reynolds number: {Re}")
    """
    number = get_dimensionless_number(symbol)

    # Check all required parameters are present
    missing = set(number.required_params) - set(params.keys())
    if missing:
        raise KeyError(
            f"Missing required parameters for {symbol}: {missing}. "
            f"Required: {number.required_params}"
        )

    # Compute the dimensionless number
    try:
        result = number.compute_func(params)
    except Exception as e:
        raise ValueError(
            f"Failed to compute {symbol} with given parameters. "
            f"Check that parameter dimensions are correct. Error: {e}"
        ) from e

    return result


def infer_regime(symbol: str, value: float) -> Optional[str]:
    """Infer the physical regime from a dimensionless number value.

    Args:
        symbol: Symbol of the dimensionless number (e.g., 'Re')
        value: The computed dimensionless value

    Returns:
        String describing the regime (e.g., 'laminar', 'turbulent'),
        or None if no regime map is available for this number.

    Examples:
        >>> from dimtensor.analysis import dimensionless_numbers as dn
        >>> regime = dn.infer_regime('Re', 1000)
        >>> print(regime)  # 'laminar'
        >>> regime = dn.infer_regime('Re', 10000)
        >>> print(regime)  # 'turbulent'
    """
    number = get_dimensionless_number(symbol)

    if number.regime_map is None:
        return None

    return number.regime_map(value)


def compute_all_applicable(
    parameters: Dict[str, DimArray]
) -> Dict[str, Tuple[float, Optional[str]]]:
    """Compute all dimensionless numbers applicable to the given parameters.

    Args:
        parameters: Dictionary mapping parameter names to DimArrays.

    Returns:
        Dictionary mapping symbols to (value, regime) tuples.
        Only includes dimensionless numbers for which all required
        parameters are present.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.analysis import dimensionless_numbers as dn
        >>>
        >>> params = {
        ...     'density': DimArray(1000, units.kg / units.m**3),
        ...     'velocity': DimArray(1.0, units.m / units.s),
        ...     'length': DimArray(0.1, units.m),
        ...     'viscosity': DimArray(0.001, units.Pa * units.s),
        ...     'sound_speed': DimArray(340, units.m / units.s),
        ... }
        >>>
        >>> results = dn.compute_all_applicable(params)
        >>> for symbol, (value, regime) in results.items():
        ...     print(f"{symbol} = {value:.2f} ({regime})")
    """
    results: Dict[str, Tuple[float, Optional[str]]] = {}

    for symbol in list_dimensionless_numbers():
        number = get_dimensionless_number(symbol)

        # Check if all required parameters are present
        if all(param in parameters for param in number.required_params):
            try:
                value = compute(symbol, **parameters)
                regime = infer_regime(symbol, value)
                results[symbol] = (value, regime)
            except Exception:
                # Skip if computation fails (e.g., wrong dimensions)
                pass

    return results
