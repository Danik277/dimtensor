"""Error budget calculator for uncertainty contribution analysis.

This module provides tools to decompose total uncertainty into contributions
from each input variable, following the Guide to Uncertainty in Measurement (GUM).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ErrorBudget:
    """Container for uncertainty contribution analysis.

    Stores the decomposition of total uncertainty into contributions from
    each input variable, following GUM methodology.

    Attributes:
        result: The computed result value.
        total_uncertainty: The combined standard uncertainty.
        contributions: Dict mapping input names to their uncertainty contributions.
        sensitivities: Dict mapping input names to their sensitivity coefficients.
        percent_contributions: Dict mapping input names to percentage contributions.
    """

    def __init__(
        self,
        result: Any,
        total_uncertainty: float | NDArray[Any],
        contributions: dict[str, float | NDArray[Any]],
        sensitivities: dict[str, Any],
    ) -> None:
        """Initialize ErrorBudget.

        Args:
            result: The computed result (scalar or DimArray).
            total_uncertainty: Combined standard uncertainty.
            contributions: Uncertainty contribution from each input.
            sensitivities: Sensitivity coefficient (∂f/∂x_i) for each input.
        """
        self.result = result
        self.total_uncertainty = total_uncertainty
        self.contributions = contributions
        self.sensitivities = sensitivities

        # Compute percentage contributions
        # Percent contribution: 100% * (contribution^2) / (total_uncertainty^2)
        total_var = np.asarray(total_uncertainty) ** 2
        self.percent_contributions = {}
        for name, contrib in contributions.items():
            contrib_var = np.asarray(contrib) ** 2
            # Handle case where total_uncertainty is zero
            with np.errstate(divide='ignore', invalid='ignore'):
                percent = 100.0 * contrib_var / total_var
                percent = np.nan_to_num(percent, nan=0.0, posinf=0.0, neginf=0.0)
            self.percent_contributions[name] = percent

    def to_dict(self) -> dict[str, Any]:
        """Export error budget as dictionary for serialization.

        Returns:
            Dictionary with all error budget data.
        """
        return {
            'result': self.result,
            'total_uncertainty': self.total_uncertainty,
            'contributions': self.contributions,
            'sensitivities': self.sensitivities,
            'percent_contributions': self.percent_contributions,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export error budget as pandas DataFrame for tabular reporting.

        Returns:
            DataFrame with columns: input, sensitivity, contribution, percent.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        # For scalar results, create simple table
        rows = []
        for name in self.contributions.keys():
            sensitivity = self.sensitivities[name]
            contribution = self.contributions[name]
            percent = self.percent_contributions[name]

            # Convert to scalar if needed
            if hasattr(sensitivity, 'item'):
                sensitivity = sensitivity.item() if sensitivity.size == 1 else sensitivity
            if hasattr(contribution, 'item'):
                contribution = contribution.item() if contribution.size == 1 else contribution
            if hasattr(percent, 'item'):
                percent = percent.item() if percent.size == 1 else percent

            rows.append({
                'input': name,
                'sensitivity': sensitivity,
                'contribution': contribution,
                'percent': percent,
            })

        df = pd.DataFrame(rows)

        # Add total row
        total_percent = df['percent'].sum()
        total_row = pd.DataFrame([{
            'input': 'TOTAL',
            'sensitivity': None,
            'contribution': self.total_uncertainty,
            'percent': total_percent,
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        return df

    def plot_pie(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Create pie chart showing percentage contributions.

        Args:
            ax: Optional matplotlib Axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.pie().

        Returns:
            The matplotlib Axes object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Get percentage contributions (handle scalar/array)
        names = list(self.percent_contributions.keys())
        percents = []
        for name in names:
            p = self.percent_contributions[name]
            if hasattr(p, 'item'):
                p = p.item() if np.isscalar(p) or p.size == 1 else float(np.mean(p))
            percents.append(p)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            percents,
            labels=names,
            autopct='%1.1f%%',
            startangle=90,
            **kwargs
        )

        ax.set_title('Uncertainty Contribution Breakdown')

        return ax

    def plot_bar(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Create horizontal bar chart of contributions, sorted by magnitude.

        Args:
            ax: Optional matplotlib Axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.barh().

        Returns:
            The matplotlib Axes object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Get contributions and sort by magnitude
        names = list(self.contributions.keys())
        contribs = []
        for name in names:
            c = self.contributions[name]
            if hasattr(c, 'item'):
                c = c.item() if np.isscalar(c) or c.size == 1 else float(np.mean(np.abs(c)))
            contribs.append(c)

        # Sort by contribution magnitude
        sorted_indices = np.argsort(contribs)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_contribs = [contribs[i] for i in sorted_indices]

        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_contribs, **kwargs)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Uncertainty Contribution')
        ax.set_title('Uncertainty Contributions by Input')
        ax.grid(axis='x', alpha=0.3)

        return ax

    def plot_pareto(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Create Pareto chart showing cumulative percentage contributions.

        Args:
            ax: Optional matplotlib Axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to plotting functions.

        Returns:
            The matplotlib Axes object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get percentage contributions and sort
        names = list(self.percent_contributions.keys())
        percents = []
        for name in names:
            p = self.percent_contributions[name]
            if hasattr(p, 'item'):
                p = p.item() if np.isscalar(p) or p.size == 1 else float(np.mean(p))
            percents.append(p)

        # Sort descending by percentage
        sorted_indices = np.argsort(percents)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_percents = [percents[i] for i in sorted_indices]

        # Compute cumulative percentages
        cumulative = np.cumsum(sorted_percents)

        # Create bar chart
        x_pos = np.arange(len(sorted_names))
        ax.bar(x_pos, sorted_percents, alpha=0.7, label='Individual')

        # Create cumulative line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(x_pos, cumulative, 'r-o', linewidth=2, label='Cumulative')
        ax2.set_ylabel('Cumulative Contribution (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 105])
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% line')

        # Configure primary axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_ylabel('Contribution (%)')
        ax.set_title('Pareto Chart of Uncertainty Contributions')
        ax.grid(axis='y', alpha=0.3)

        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        return ax

    def __repr__(self) -> str:
        """String representation showing key information."""
        n_inputs = len(self.contributions)
        return (
            f"ErrorBudget(n_inputs={n_inputs}, "
            f"total_uncertainty={self.total_uncertainty})"
        )


def compute_error_budget(
    func: Callable[[dict[str, Any]], Any],
    inputs: dict[str, Any],
    step_factor: float = 0.01,
) -> ErrorBudget:
    """Compute error budget by analyzing input uncertainty contributions.

    Uses finite difference approximation to compute sensitivity coefficients
    (∂f/∂x_i) and determines how each input's uncertainty contributes to
    the total output uncertainty.

    Following GUM (Guide to Uncertainty in Measurement):
    - Sensitivity coefficient: c_i = ∂f/∂x_i
    - Uncertainty contribution: u_i(y) = |c_i| * u(x_i)
    - Combined uncertainty: u_c = √(Σ u_i²)
    - Percent contribution: 100% * u_i² / u_c²

    Args:
        func: Function that computes result from inputs.
              Should accept dict of inputs and return result.
        inputs: Dict mapping input names to values (DimArray with uncertainty).
        step_factor: Relative step size for finite differences (default 0.01 = 1%).

    Returns:
        ErrorBudget with contribution breakdown.

    Raises:
        ValueError: If inputs don't have uncertainty or step_factor invalid.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.uncertainty import compute_error_budget
        >>>
        >>> # Pendulum period: T = 2π√(L/g)
        >>> def pendulum_period(inputs):
        ...     L = inputs['length']
        ...     g = inputs['gravity']
        ...     return 2 * np.pi * (L / g) ** 0.5
        >>>
        >>> inputs = {
        ...     'length': DimArray(1.0, units.m, uncertainty=0.01),
        ...     'gravity': DimArray(9.8, units.m/units.s**2, uncertainty=0.1),
        ... }
        >>>
        >>> budget = compute_error_budget(pendulum_period, inputs)
        >>> df = budget.to_dataframe()
        >>> print(df)
    """
    # Import here to avoid circular dependency
    from ..core.dimarray import DimArray

    # Validate inputs
    if step_factor <= 0:
        raise ValueError(f"step_factor must be positive, got {step_factor}")

    # Check that inputs have uncertainty
    for name, value in inputs.items():
        if isinstance(value, DimArray):
            if not value.has_uncertainty:
                raise ValueError(
                    f"Input '{name}' has no uncertainty. "
                    "All inputs must have uncertainty for error budget analysis."
                )
        else:
            # For non-DimArray inputs, we can't compute error budget
            raise ValueError(
                f"Input '{name}' is not a DimArray. "
                "Error budget requires DimArray inputs with uncertainty."
            )

    # Compute nominal result
    result = func(inputs)

    # Extract nominal result value (handle DimArray)
    if isinstance(result, DimArray):
        nominal_value = result._data
        result_unit = result._unit
    else:
        nominal_value = np.asarray(result)
        result_unit = None

    # Compute sensitivity coefficients and contributions for each input
    sensitivities = {}
    contributions = {}

    for name, input_value in inputs.items():
        # Get nominal value and uncertainty
        if isinstance(input_value, DimArray):
            x_nominal = input_value._data
            x_uncertainty = input_value._uncertainty
            x_unit = input_value._unit
        else:
            x_nominal = np.asarray(input_value)
            x_uncertainty = None
            x_unit = None

        if x_uncertainty is None:
            continue

        # Compute step size based on uncertainty and step_factor
        # Use max of relative step and absolute minimum
        step = np.maximum(
            step_factor * np.abs(x_nominal),
            step_factor * x_uncertainty
        )
        # Ensure non-zero step
        step = np.maximum(step, 1e-10 * np.maximum(np.abs(x_nominal), 1.0))

        # Perturb input and compute function
        inputs_perturbed = inputs.copy()
        if isinstance(input_value, DimArray):
            x_perturbed = x_nominal + step
            inputs_perturbed[name] = DimArray._from_data_and_unit(
                x_perturbed, x_unit, x_uncertainty
            )
        else:
            x_perturbed = x_nominal + step
            inputs_perturbed[name] = x_perturbed

        # Compute perturbed result
        result_perturbed = func(inputs_perturbed)

        # Extract perturbed value
        if isinstance(result_perturbed, DimArray):
            perturbed_value = result_perturbed._data
        else:
            perturbed_value = np.asarray(result_perturbed)

        # Compute sensitivity coefficient via finite difference
        # c_i = ∂f/∂x_i ≈ (f(x+h) - f(x)) / h
        sensitivity = (perturbed_value - nominal_value) / step

        # Store sensitivity (with units if available)
        if isinstance(result, DimArray) and isinstance(input_value, DimArray):
            # Create DimArray for sensitivity with correct units
            sens_unit = result_unit / x_unit
            sensitivities[name] = DimArray._from_data_and_unit(
                sensitivity, sens_unit
            )
        else:
            sensitivities[name] = sensitivity

        # Compute uncertainty contribution: u_i(y) = |c_i| * u(x_i)
        contribution = np.abs(sensitivity) * x_uncertainty
        contributions[name] = contribution

    # Compute combined standard uncertainty (quadrature sum)
    # u_c = √(Σ u_i²)
    total_variance = sum(
        np.asarray(c) ** 2 for c in contributions.values()
    )
    total_uncertainty = np.sqrt(total_variance)

    # Create and return ErrorBudget
    return ErrorBudget(
        result=result,
        total_uncertainty=total_uncertainty,
        contributions=contributions,
        sensitivities=sensitivities,
    )
