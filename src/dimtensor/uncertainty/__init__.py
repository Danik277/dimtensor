"""Uncertainty analysis tools for dimtensor.

This subpackage provides tools for uncertainty propagation and error budget analysis,
following the Guide to Uncertainty in Measurement (GUM) methodology.

Main Components:
    - compute_error_budget: Decompose uncertainty into input contributions
    - ErrorBudget: Container for uncertainty contribution analysis with reporting

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.uncertainty import compute_error_budget
    >>> import numpy as np
    >>>
    >>> # Define a computation (pendulum period)
    >>> def pendulum_period(inputs):
    ...     L = inputs['length']
    ...     g = inputs['gravity']
    ...     return 2 * np.pi * (L / g) ** 0.5
    >>>
    >>> # Inputs with uncertainty
    >>> inputs = {
    ...     'length': DimArray(1.0, units.m, uncertainty=0.01),
    ...     'gravity': DimArray(9.8, units.m/units.s**2, uncertainty=0.1),
    ... }
    >>>
    >>> # Compute error budget
    >>> budget = compute_error_budget(pendulum_period, inputs)
    >>>
    >>> # View as table
    >>> print(budget.to_dataframe())
    >>>
    >>> # Visualize contributions
    >>> budget.plot_pie()
    >>> budget.plot_bar()
"""

from .error_budget import ErrorBudget, compute_error_budget

__all__ = [
    "ErrorBudget",
    "compute_error_budget",
]

# Optional: Import monte_carlo if available (future implementation)
try:
    from .monte_carlo import (
        LHSSampler,
        MCResult,
        RandomSampler,
        SobolSampler,
        monte_carlo,
        monte_carlo_dimarray,
    )
    __all__.extend([
        "monte_carlo",
        "monte_carlo_dimarray",
        "MCResult",
        "RandomSampler",
        "LHSSampler",
        "SobolSampler",
    ])
except ImportError:
    pass  # monte_carlo module not yet implemented
