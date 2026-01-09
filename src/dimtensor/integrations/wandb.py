"""Weights & Biases integration for DimArray.

Log unit-aware tensors to W&B experiments with automatic unit tracking
and preservation in experiment metadata.

Examples:
    >>> import wandb
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.integrations.wandb import log_dimarray, DimWandbCallback

    >>> # Initialize W&B run
    >>> wandb.init(project="physics-ml")

    >>> # Log a DimArray
    >>> velocity = DimArray([10.0, 20.0], units.m / units.s)
    >>> log_dimarray("velocity", velocity)

    >>> # Use callback during training
    >>> callback = DimWandbCallback()
    >>> callback.log_epoch({"loss": loss, "lr": learning_rate}, epoch=0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit


def _check_wandb() -> Any:
    """Check if wandb is available and return the module.

    Returns:
        wandb module.

    Raises:
        ImportError: If wandb is not installed.
    """
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError(
            "wandb is required for Weights & Biases integration. "
            "Install with: pip install wandb"
        )


def _serialize_unit(unit: Unit) -> dict[str, Any]:
    """Serialize a Unit to a dictionary for W&B metadata.

    Args:
        unit: Unit to serialize.

    Returns:
        Dictionary with unit information.
    """
    return {
        "symbol": unit.symbol,
        "scale": unit.scale,
        "dimension": {
            "length": float(unit.dimension.length),
            "mass": float(unit.dimension.mass),
            "time": float(unit.dimension.time),
            "current": float(unit.dimension.current),
            "temperature": float(unit.dimension.temperature),
            "amount": float(unit.dimension.amount),
            "luminosity": float(unit.dimension.luminosity),
        },
    }


def log_dimarray(
    name: str,
    value: DimArray,
    step: Optional[int] = None,
    commit: Optional[bool] = None,
) -> None:
    """Log a DimArray to Weights & Biases with unit metadata.

    The DimArray's values and unit information are logged separately.
    Scalar values are logged directly, while arrays are logged as histograms.

    Args:
        name: Name for the logged metric.
        value: DimArray to log.
        step: Step number for the log entry.
        commit: Whether to commit the log immediately.

    Raises:
        ImportError: If wandb is not installed.

    Examples:
        >>> import wandb
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.integrations.wandb import log_dimarray

        >>> wandb.init(project="test")
        >>> velocity = DimArray([10.0], units.m / units.s)
        >>> log_dimarray("velocity", velocity)
    """
    wandb = _check_wandb()

    # Prepare log data
    log_data: dict[str, Any] = {}

    # Log the numerical values
    if value.size == 1:
        # Scalar: log directly
        log_data[name] = float(value._data.item())
    else:
        # Array: log as histogram
        log_data[name] = wandb.Histogram(value._data)

    # Log unit metadata
    unit_info = _serialize_unit(value.unit)
    log_data[f"{name}_unit"] = unit_info["symbol"]
    log_data[f"{name}_unit_metadata"] = unit_info

    # Log uncertainty if present
    if value.has_uncertainty and value._uncertainty is not None:
        if value.size == 1:
            log_data[f"{name}_uncertainty"] = float(value._uncertainty.item())
        else:
            log_data[f"{name}_uncertainty"] = wandb.Histogram(value._uncertainty)

    # Log to wandb
    wandb.log(log_data, step=step, commit=commit)


def log_config_with_units(config: dict[str, Union[DimArray, Any]]) -> None:
    """Log configuration parameters to W&B, extracting units from DimArrays.

    For DimArray values in the config, logs both the value and unit separately.
    Non-DimArray values are logged as-is.

    Args:
        config: Configuration dictionary that may contain DimArrays.

    Raises:
        ImportError: If wandb is not installed.

    Examples:
        >>> import wandb
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.integrations.wandb import log_config_with_units

        >>> wandb.init(project="test")
        >>> config = {
        ...     "learning_rate": DimArray([0.001], 1 / units.s),
        ...     "batch_size": 32,
        ...     "max_velocity": DimArray([100.0], units.m / units.s),
        ... }
        >>> log_config_with_units(config)
    """
    wandb = _check_wandb()

    flat_config: dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, DimArray):
            # Extract scalar value (or array representation)
            if value.size == 1:
                flat_config[key] = float(value._data.item())
            else:
                flat_config[key] = value._data.tolist()

            # Add unit as separate config entry
            flat_config[f"{key}_unit"] = value.unit.symbol
            flat_config[f"{key}_unit_metadata"] = _serialize_unit(value.unit)
        else:
            # Non-DimArray values pass through
            flat_config[key] = value

    wandb.config.update(flat_config)


def create_dimarray_table(
    data: List[Dict[str, Union[DimArray, Any]]],
    columns: Optional[List[str]] = None,
) -> Any:
    """Create a W&B Table with DimArray support.

    Creates a wandb.Table where DimArray values are converted to numeric
    values, and unit information is added to column names.

    Args:
        data: List of dictionaries containing data to tabulate.
        columns: Optional list of column names. If None, inferred from data.

    Returns:
        wandb.Table with unit-annotated columns.

    Raises:
        ImportError: If wandb is not installed.

    Examples:
        >>> import wandb
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.integrations.wandb import create_dimarray_table

        >>> wandb.init(project="test")
        >>> data = [
        ...     {"x": DimArray([1.0], units.m), "y": DimArray([2.0], units.s)},
        ...     {"x": DimArray([2.0], units.m), "y": DimArray([3.0], units.s)},
        ... ]
        >>> table = create_dimarray_table(data)
        >>> wandb.log({"results": table})
    """
    wandb = _check_wandb()

    if not data:
        return wandb.Table(columns=columns or [], data=[])

    # Determine columns if not provided
    if columns is None:
        columns = list(data[0].keys())

    # Determine units for each column
    column_units: dict[str, str] = {}
    for row in data:
        for col in columns:
            if col in row and isinstance(row[col], DimArray):
                column_units[col] = row[col].unit.symbol
                break

    # Create column names with units
    table_columns = [
        f"{col} [{column_units[col]}]" if col in column_units else col
        for col in columns
    ]

    # Convert data rows
    table_data = []
    for row in data:
        table_row = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, DimArray):
                # Extract scalar or first element
                if value.size == 1:
                    table_row.append(float(value._data.item()))
                else:
                    # For arrays, take mean or first element
                    table_row.append(float(value._data.flat[0]))
            else:
                table_row.append(value)
        table_data.append(table_row)

    return wandb.Table(columns=table_columns, data=table_data)


class DimWandbCallback:
    """Callback for logging DimArrays during training with W&B.

    Automatically detects DimArray values in logged metrics and preserves
    their unit information in W&B experiments.

    Examples:
        >>> import wandb
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.integrations.wandb import DimWandbCallback

        >>> wandb.init(project="training")
        >>> callback = DimWandbCallback()

        >>> for epoch in range(10):
        ...     loss = DimArray([0.5 / (epoch + 1)], units.J)
        ...     lr = DimArray([0.001], 1 / units.s)
        ...     callback.log_epoch({"loss": loss, "lr": lr}, epoch=epoch)
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize the callback.

        Args:
            prefix: Optional prefix for all logged metric names.
        """
        _check_wandb()  # Verify wandb is available
        self.prefix = prefix
        self._logged_units: dict[str, str] = {}

    def log_epoch(
        self,
        metrics: dict[str, Union[DimArray, float, int]],
        epoch: int,
        commit: bool = True,
    ) -> None:
        """Log metrics for a training epoch.

        Args:
            metrics: Dictionary of metric name to value. Values can be
                DimArray, float, or int.
            epoch: Epoch number.
            commit: Whether to commit the log immediately.
        """
        wandb = _check_wandb()

        log_data: dict[str, Any] = {"epoch": epoch}

        for name, value in metrics.items():
            full_name = f"{self.prefix}{name}" if self.prefix else name

            if isinstance(value, DimArray):
                # Log DimArray with unit metadata
                if value.size == 1:
                    log_data[full_name] = float(value._data.item())
                else:
                    # For arrays, log mean/histogram
                    log_data[full_name] = float(np.mean(value._data))
                    log_data[f"{full_name}_hist"] = wandb.Histogram(value._data)

                # Log unit on first occurrence
                unit_symbol = value.unit.symbol
                if full_name not in self._logged_units:
                    self._logged_units[full_name] = unit_symbol
                    unit_info = _serialize_unit(value.unit)
                    # Store unit metadata in wandb.summary for reference
                    wandb.run.summary[f"{full_name}_unit"] = unit_symbol
                    wandb.run.summary[f"{full_name}_unit_metadata"] = unit_info
            else:
                # Log regular numeric values
                log_data[full_name] = value

        wandb.log(log_data, step=epoch, commit=commit)

    def log_batch(
        self,
        metrics: dict[str, Union[DimArray, float, int]],
        step: int,
        commit: bool = True,
    ) -> None:
        """Log metrics for a training batch/step.

        Args:
            metrics: Dictionary of metric name to value.
            step: Global step number.
            commit: Whether to commit the log immediately.
        """
        wandb = _check_wandb()

        log_data: dict[str, Any] = {}

        for name, value in metrics.items():
            full_name = f"{self.prefix}{name}" if self.prefix else name

            if isinstance(value, DimArray):
                if value.size == 1:
                    log_data[full_name] = float(value._data.item())
                else:
                    log_data[full_name] = float(np.mean(value._data))

                # Log unit metadata on first occurrence
                unit_symbol = value.unit.symbol
                if full_name not in self._logged_units:
                    self._logged_units[full_name] = unit_symbol
                    unit_info = _serialize_unit(value.unit)
                    wandb.run.summary[f"{full_name}_unit"] = unit_symbol
                    wandb.run.summary[f"{full_name}_unit_metadata"] = unit_info
            else:
                log_data[full_name] = value

        wandb.log(log_data, step=step, commit=commit)

    def finalize(self, summary: Optional[dict[str, Union[DimArray, Any]]] = None) -> None:
        """Finalize logging and optionally add summary metrics.

        Args:
            summary: Optional dictionary of final summary metrics.
        """
        wandb = _check_wandb()

        if summary:
            for name, value in summary.items():
                full_name = f"{self.prefix}{name}" if self.prefix else name

                if isinstance(value, DimArray):
                    if value.size == 1:
                        wandb.run.summary[full_name] = float(value._data.item())
                    else:
                        wandb.run.summary[full_name] = value._data.tolist()

                    wandb.run.summary[f"{full_name}_unit"] = value.unit.symbol
                    wandb.run.summary[f"{full_name}_unit_metadata"] = _serialize_unit(value.unit)
                else:
                    wandb.run.summary[full_name] = value


__all__ = [
    "log_dimarray",
    "log_config_with_units",
    "create_dimarray_table",
    "DimWandbCallback",
]
