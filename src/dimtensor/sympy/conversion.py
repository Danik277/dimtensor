"""Conversion between DimArray and SymPy expressions.

Provides to_sympy() and from_sympy() for bridging numerical and symbolic computation.
"""
# mypy: disable-error-code="import-untyped,type-arg,arg-type,no-any-return"

from __future__ import annotations

from typing import Any
from fractions import Fraction

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit
from ..core.dimensions import Dimension

try:
    import sympy as sp
    from sympy.physics.units import (
        meter, kilogram, second, ampere, kelvin, mole, candela,
        newton, joule, watt, pascal, hertz, coulomb, volt, ohm,
        siemens, farad, henry, weber, tesla,
    )
    from sympy.physics.units import convert_to
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def _check_sympy() -> None:
    """Check that sympy is available."""
    if not HAS_SYMPY:
        raise ImportError(
            "sympy is required for SymPy integration. "
            "Install with: pip install sympy"
        )


# Mapping from Dimension tuple to SymPy unit expression
_DIMENSION_TO_SYMPY = {
    # Base dimensions
    (1, 0, 0, 0, 0, 0, 0): "meter",         # length
    (0, 1, 0, 0, 0, 0, 0): "kilogram",      # mass
    (0, 0, 1, 0, 0, 0, 0): "second",        # time
    (0, 0, 0, 1, 0, 0, 0): "ampere",        # current
    (0, 0, 0, 0, 1, 0, 0): "kelvin",        # temperature
    (0, 0, 0, 0, 0, 1, 0): "mole",          # amount
    (0, 0, 0, 0, 0, 0, 1): "candela",       # luminosity
    # Common derived dimensions
    (1, 0, -1, 0, 0, 0, 0): "meter/second",  # velocity
    (1, 0, -2, 0, 0, 0, 0): "meter/second**2",  # acceleration
    (1, 1, -2, 0, 0, 0, 0): "newton",        # force
    (2, 1, -2, 0, 0, 0, 0): "joule",         # energy
    (2, 1, -3, 0, 0, 0, 0): "watt",          # power
    (-1, 1, -2, 0, 0, 0, 0): "pascal",       # pressure
    (0, 0, -1, 0, 0, 0, 0): "hertz",         # frequency
    (0, 0, 1, 1, 0, 0, 0): "coulomb",        # charge
    (2, 1, -3, -1, 0, 0, 0): "volt",         # voltage
    (2, 1, -3, -2, 0, 0, 0): "ohm",          # resistance
    (-2, -1, 3, 2, 0, 0, 0): "siemens",      # conductance
    (-2, -1, 4, 2, 0, 0, 0): "farad",        # capacitance
    (2, 1, -2, -2, 0, 0, 0): "henry",        # inductance
    (2, 1, -2, -1, 0, 0, 0): "weber",        # magnetic flux
    (0, 1, -2, -1, 0, 0, 0): "tesla",        # magnetic field
}


def sympy_unit_for(unit: Unit) -> Any:
    """Get the SymPy unit expression for a dimtensor Unit.

    Args:
        unit: A dimtensor Unit object.

    Returns:
        SymPy unit expression.

    Example:
        >>> from dimtensor import units
        >>> from dimtensor.sympy import sympy_unit_for
        >>> sympy_unit_for(units.m)
        meter
        >>> sympy_unit_for(units.N)
        newton
    """
    _check_sympy()

    # Get dimension tuple
    dim = unit.dimension
    dim_tuple = (
        int(dim.length), int(dim.mass), int(dim.time),
        int(dim.current), int(dim.temperature),
        int(dim.amount), int(dim.luminosity)
    )

    # Check for known dimensions first
    if dim_tuple in _DIMENSION_TO_SYMPY:
        unit_expr = eval(_DIMENSION_TO_SYMPY[dim_tuple])
        # Apply scale factor
        return unit.scale * unit_expr

    # Build from base units for unknown dimensions
    result: Any = unit.scale
    if dim.length != 0:
        result = result * meter ** int(dim.length)
    if dim.mass != 0:
        result = result * kilogram ** int(dim.mass)
    if dim.time != 0:
        result = result * second ** int(dim.time)
    if dim.current != 0:
        result = result * ampere ** int(dim.current)
    if dim.temperature != 0:
        result = result * kelvin ** int(dim.temperature)
    if dim.amount != 0:
        result = result * mole ** int(dim.amount)
    if dim.luminosity != 0:
        result = result * candela ** int(dim.luminosity)

    return result


def to_sympy(
    arr: DimArray,
    symbol: str | None = None,
) -> Any:
    """Convert DimArray to SymPy expression with units.

    Args:
        arr: DimArray to convert.
        symbol: Optional symbol name for the value. If None, uses numerical value.

    Returns:
        SymPy expression with units attached.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.sympy import to_sympy
        >>>
        >>> distance = DimArray([10], units.m)
        >>> expr = to_sympy(distance)
        >>> print(expr)
        10*meter
        >>>
        >>> # With symbolic value
        >>> x = to_sympy(distance, symbol="x")
        >>> print(x)
        x*meter
    """
    _check_sympy()

    sympy_unit = sympy_unit_for(arr.unit)

    if symbol is not None:
        # Create symbolic expression
        sym = sp.Symbol(symbol, real=True)
        return sym * sympy_unit
    else:
        # Use numerical value
        data = np.asarray(arr._data)
        if data.size == 1:
            value = float(data.flatten()[0])
            return value * sympy_unit
        else:
            # For arrays, return a Matrix with units
            values = data.flatten().tolist()
            matrix = sp.Matrix(values)
            return matrix * sympy_unit


def from_sympy(
    expr: Any,
    target_unit: Unit | None = None,
) -> DimArray:
    """Convert SymPy expression with units to DimArray.

    Args:
        expr: SymPy expression with units.
        target_unit: Optional target unit for the result.

    Returns:
        DimArray with appropriate units.

    Example:
        >>> from sympy.physics.units import meter, second
        >>> from dimtensor.sympy import from_sympy
        >>>
        >>> expr = 100 * meter / second
        >>> arr = from_sympy(expr)
        >>> print(arr)
        [100.] m/s
    """
    _check_sympy()

    # Try to convert to base SI units
    try:
        # Get the numerical coefficient and unit parts
        base_units = [meter, kilogram, second, ampere, kelvin, mole, candela]
        converted = convert_to(expr, base_units)

        # Extract coefficient and unit
        # SymPy expressions have .as_coeff_Mul() method
        coeff, unit_part = converted.as_coeff_Mul()

        # Parse the unit part to get dimension
        dim = _parse_sympy_unit(unit_part)

        # Create the Unit
        from ..core import units as dt_units
        result_unit = _dimension_to_unit(dim)

        value = float(coeff)
        result = DimArray._from_data_and_unit(np.array([value]), result_unit)

        if target_unit is not None:
            return result.to(target_unit)
        return result

    except Exception as e:
        raise ValueError(f"Cannot convert SymPy expression to DimArray: {e}") from e


def _parse_sympy_unit(unit_expr: Any) -> Dimension:
    """Parse SymPy unit expression to Dimension."""
    _check_sympy()

    # Initialize dimension exponents
    length = Fraction(0)
    mass = Fraction(0)
    time = Fraction(0)
    current = Fraction(0)
    temperature = Fraction(0)
    amount = Fraction(0)
    luminosity = Fraction(0)

    # Get the base form
    base = unit_expr.as_base_exp()

    def process_term(term: Any, power: int = 1) -> None:
        nonlocal length, mass, time, current, temperature, amount, luminosity

        if term == meter or str(term) == "meter":
            length += power
        elif term == kilogram or str(term) == "kilogram":
            mass += power
        elif term == second or str(term) == "second":
            time += power
        elif term == ampere or str(term) == "ampere":
            current += power
        elif term == kelvin or str(term) == "kelvin":
            temperature += power
        elif term == mole or str(term) == "mole":
            amount += power
        elif term == candela or str(term) == "candela":
            luminosity += power

    # Handle power expressions
    if hasattr(unit_expr, "is_Pow") and unit_expr.is_Pow:
        base, exp = unit_expr.as_base_exp()
        process_term(base, int(exp))
    elif hasattr(unit_expr, "is_Mul") and unit_expr.is_Mul:
        for arg in unit_expr.args:
            if hasattr(arg, "is_Pow") and arg.is_Pow:
                base, exp = arg.as_base_exp()
                process_term(base, int(exp))
            else:
                process_term(arg, 1)
    elif unit_expr == 1:
        pass  # Dimensionless
    else:
        process_term(unit_expr, 1)

    return Dimension(
        length=length,
        mass=mass,
        time=time,
        current=current,
        temperature=temperature,
        amount=amount,
        luminosity=luminosity,
    )


def _dimension_to_unit(dim: Dimension) -> Unit:
    """Convert Dimension to a base Unit."""
    from ..core import units as dt_units

    # Start with dimensionless
    result = dt_units.dimensionless

    if dim.length != 0:
        result = result * (dt_units.m ** int(dim.length))
    if dim.mass != 0:
        result = result * (dt_units.kg ** int(dim.mass))
    if dim.time != 0:
        result = result * (dt_units.s ** int(dim.time))
    if dim.current != 0:
        result = result * (dt_units.A ** int(dim.current))
    if dim.temperature != 0:
        result = result * (dt_units.K ** int(dim.temperature))
    if dim.amount != 0:
        result = result * (dt_units.mol ** int(dim.amount))
    if dim.luminosity != 0:
        result = result * (dt_units.cd ** int(dim.luminosity))

    return result
