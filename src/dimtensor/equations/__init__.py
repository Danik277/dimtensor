"""Physics equation database for dimensional analysis.

Provides a comprehensive database of physics equations with dimensional
information, enabling equation lookup, dimensional verification, and
physics-informed machine learning.

Example:
    >>> from dimtensor.equations import get_equations, search_equations
    >>>
    >>> # Get equations by domain
    >>> mechanics = get_equations(domain="mechanics")
    >>> for eq in mechanics[:3]:
    ...     print(f"{eq.name}: {eq.formula}")
    >>>
    >>> # Search for equations
    >>> energy_eqs = search_equations("energy")
"""

from .database import (
    Equation,
    get_equations,
    search_equations,
    get_equation,
    list_domains,
    register_equation,
)

__all__ = [
    "Equation",
    "get_equations",
    "search_equations",
    "get_equation",
    "list_domains",
    "register_equation",
]
