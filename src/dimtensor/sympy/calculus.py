"""Symbolic calculus with dimensional tracking.

Provides symbolic differentiation and integration that preserve physical units.
"""
# mypy: disable-error-code="import-untyped,type-arg,arg-type,no-any-return"

from __future__ import annotations

from typing import Any

from ..core.units import Unit

try:
    import sympy as sp
    from sympy.physics.units import meter, kilogram, second, ampere, kelvin, mole, candela
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from .conversion import to_sympy, from_sympy, sympy_unit_for, _check_sympy


def symbolic_diff(
    expr: Any,
    var: Any,
    var_unit: Unit | None = None,
) -> Any:
    """Symbolically differentiate an expression with units.

    The derivative of a quantity with units u1 with respect to a variable
    with units u2 has units u1/u2.

    Args:
        expr: SymPy expression with units.
        var: Variable to differentiate with respect to (symbol or string).
        var_unit: Unit of the differentiation variable (required if var is string).

    Returns:
        Differentiated SymPy expression with correct units.

    Example:
        >>> import sympy as sp
        >>> from sympy.physics.units import meter, second
        >>> from dimtensor.sympy import symbolic_diff
        >>>
        >>> t = sp.Symbol("t")
        >>> x = 5 * t**2 * meter  # position as function of time
        >>> v = symbolic_diff(x, t, var_unit=units.s)  # velocity
        >>> print(v)  # 10*t*meter/second
    """
    _check_sympy()

    # Get the variable symbol
    if isinstance(var, str):
        var = sp.Symbol(var, real=True)

    # Perform symbolic differentiation
    result = sp.diff(expr, var)

    # If var_unit is provided, divide by the unit
    if var_unit is not None:
        sympy_var_unit = sympy_unit_for(var_unit)
        result = result / sympy_var_unit

    return result


def symbolic_integrate(
    expr: Any,
    var: Any,
    var_unit: Unit | None = None,
    limits: tuple[Any, Any] | None = None,
) -> Any:
    """Symbolically integrate an expression with units.

    The integral of a quantity with units u1 with respect to a variable
    with units u2 has units u1*u2.

    Args:
        expr: SymPy expression with units.
        var: Variable to integrate with respect to (symbol or string).
        var_unit: Unit of the integration variable (required if var is string).
        limits: Optional (lower, upper) bounds for definite integration.

    Returns:
        Integrated SymPy expression with correct units.

    Example:
        >>> import sympy as sp
        >>> from sympy.physics.units import meter, second
        >>> from dimtensor.sympy import symbolic_integrate
        >>>
        >>> t = sp.Symbol("t")
        >>> a = 10 * meter / second**2  # constant acceleration
        >>> v = symbolic_integrate(a, t, var_unit=units.s)  # velocity
        >>> print(v)  # 10*t*meter/second
    """
    _check_sympy()

    # Get the variable symbol
    if isinstance(var, str):
        var = sp.Symbol(var, real=True)

    # Perform symbolic integration
    if limits is not None:
        result = sp.integrate(expr, (var, limits[0], limits[1]))
    else:
        result = sp.integrate(expr, var)

    # If var_unit is provided, multiply by the unit
    if var_unit is not None:
        sympy_var_unit = sympy_unit_for(var_unit)
        result = result * sympy_var_unit

    return result


def simplify_units(expr: Any) -> Any:
    """Simplify an expression with units.

    Combines and cancels units where possible.

    Args:
        expr: SymPy expression with units.

    Returns:
        Simplified expression.
    """
    _check_sympy()

    from sympy.physics.units.util import quantity_simplify
    return quantity_simplify(expr)


def substitute(
    expr: Any,
    substitutions: dict[str, Any],
) -> Any:
    """Substitute values into a symbolic expression.

    Args:
        expr: SymPy expression with units.
        substitutions: Dict mapping symbol names to values (can be DimArray or numbers).

    Returns:
        Expression with substitutions applied.

    Example:
        >>> import sympy as sp
        >>> from sympy.physics.units import meter, second
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.sympy import substitute
        >>>
        >>> t = sp.Symbol("t")
        >>> v = sp.Symbol("v")
        >>> x = v * t * meter / second  # formula
        >>> result = substitute(x, {"v": 10, "t": 5})
        >>> print(result)  # 50*meter
    """
    _check_sympy()

    from ..core.dimarray import DimArray

    # Find all symbols in the expression and match by name
    subs = {}
    for sym in expr.free_symbols:
        name = str(sym)
        if name in substitutions:
            value = substitutions[name]
            if isinstance(value, DimArray):
                # Convert DimArray to sympy expression
                subs[sym] = to_sympy(value)
            else:
                subs[sym] = value

    return expr.subs(subs)
