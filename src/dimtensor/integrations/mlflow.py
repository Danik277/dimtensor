"""MLflow integration for dimension-aware experiment tracking.

Provides utilities to log DimArrays with MLflow, preserving unit information
as tags and metadata for experiment tracking and comparison.

Example:
    >>> import mlflow
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.integrations.mlflow import log_dimarray, log_unit_param
    >>>
    >>> with mlflow.start_run():
    ...     # Log training parameters with units
    ...     lr = DimArray(0.001, units.Hz)  # 1/s
    ...     log_unit_param("learning_rate", lr)
    ...
    ...     # Log metrics with units
    ...     loss = DimArray(0.123, units.J)
    ...     log_dimarray("train_loss", loss, step=0)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def _check_mlflow() -> None:
    """Check that MLflow is available."""
    if not HAS_MLFLOW:
        raise ImportError(
            "MLflow is required for experiment tracking integration. "
            "Install with: pip install mlflow"
        )


def _unit_to_string(unit: Unit) -> str:
    """Convert a Unit to a string representation.
    
    Args:
        unit: The unit to convert.
        
    Returns:
        String representation of the unit.
    """
    if unit == dimensionless:
        return "dimensionless"
    return str(unit)


def _extract_scalar(value: DimArray | float | int) -> float:
    """Extract a scalar value from a DimArray or number.
    
    Args:
        value: DimArray, float, or int to extract from.
        
    Returns:
        Scalar float value.
        
    Raises:
        ValueError: If value is not a scalar.
    """
    if isinstance(value, DimArray):
        if value.data.size != 1:
            raise ValueError(
                f"Expected scalar DimArray, got shape {value.shape}. "
                "Use log_dimarray for arrays."
            )
        return float(value.data.item())
    return float(value)


def log_dimarray(
    name: str,
    value: DimArray,
    step: int | None = None,
) -> None:
    """Log a DimArray to MLflow with unit metadata.
    
    For scalar DimArrays, logs as a metric with unit stored in tags.
    For array DimArrays, logs statistical summaries (mean, std, min, max).
    
    Args:
        name: Name of the metric/array.
        value: DimArray to log.
        step: Optional step number for time series tracking.
        
    Example:
        >>> with mlflow.start_run():
        ...     loss = DimArray(0.123, units.J)
        ...     log_dimarray("loss", loss, step=0)
        ...     # Logs: metric "loss" = 0.123, tag "unit.loss" = "J"
    """
    _check_mlflow()
    
    unit_str = _unit_to_string(value.unit)
    
    # Store unit as tag
    mlflow.set_tag(f"unit.{name}", unit_str)
    
    if value.data.size == 1:
        # Scalar: log as metric
        scalar_value = float(value.data.item())
        mlflow.log_metric(name, scalar_value, step=step)
    else:
        # Array: log statistics
        mlflow.log_metric(f"{name}_mean", float(np.mean(value.data)), step=step)
        mlflow.log_metric(f"{name}_std", float(np.std(value.data)), step=step)
        mlflow.log_metric(f"{name}_min", float(np.min(value.data)), step=step)
        mlflow.log_metric(f"{name}_max", float(np.max(value.data)), step=step)
        
        # Store shape info
        mlflow.set_tag(f"shape.{name}", str(value.shape))


def log_unit_param(
    name: str,
    value: DimArray | float | int,
) -> None:
    """Log a parameter with unit metadata to MLflow.
    
    Args:
        name: Parameter name.
        value: Parameter value (DimArray or scalar).
        
    Example:
        >>> with mlflow.start_run():
        ...     lr = DimArray(0.001, units.Hz)  # 1/s
        ...     log_unit_param("learning_rate", lr)
        ...     # Logs: param "learning_rate" = "0.001", tag "unit.learning_rate" = "Hz"
    """
    _check_mlflow()
    
    if isinstance(value, DimArray):
        scalar_value = _extract_scalar(value)
        unit_str = _unit_to_string(value.unit)
        mlflow.set_tag(f"unit.{name}", unit_str)
    else:
        scalar_value = float(value)
        mlflow.set_tag(f"unit.{name}", "dimensionless")
    
    mlflow.log_param(name, scalar_value)


def log_metric_with_unit(
    name: str,
    value: float,
    unit: Unit,
    step: int | None = None,
) -> None:
    """Log a metric with explicit unit specification.
    
    Convenience function for logging raw floats with unit metadata.
    
    Args:
        name: Metric name.
        value: Metric value (raw float).
        unit: Physical unit of the value.
        step: Optional step number.
        
    Example:
        >>> with mlflow.start_run():
        ...     log_metric_with_unit("energy", 42.0, units.J, step=0)
    """
    _check_mlflow()
    
    unit_str = _unit_to_string(unit)
    mlflow.set_tag(f"unit.{name}", unit_str)
    mlflow.log_metric(name, value, step=step)


def compare_metrics_with_units(
    run_id_1: str,
    run_id_2: str,
    metric_name: str,
) -> dict[str, Any]:
    """Compare a metric across two runs, handling unit conversions.
    
    Args:
        run_id_1: First run ID.
        run_id_2: Second run ID.
        metric_name: Name of metric to compare.
        
    Returns:
        Dictionary with comparison results including values and units.
        
    Example:
        >>> result = compare_metrics_with_units(run1, run2, "loss")
        >>> print(f"Run 1: {result['value_1']} {result['unit_1']}")
        >>> print(f"Run 2: {result['value_2']} {result['unit_2']}")
        >>> print(f"Difference: {result['difference']}")
    """
    _check_mlflow()
    
    client = mlflow.tracking.MlflowClient()
    
    # Get run data
    run1 = client.get_run(run_id_1)
    run2 = client.get_run(run_id_2)
    
    # Get metric values
    metric_1 = run1.data.metrics.get(metric_name)
    metric_2 = run2.data.metrics.get(metric_name)
    
    if metric_1 is None or metric_2 is None:
        raise ValueError(f"Metric '{metric_name}' not found in one or both runs")
    
    # Get units from tags
    unit_tag_name = f"unit.{metric_name}"
    unit_1 = run1.data.tags.get(unit_tag_name, "dimensionless")
    unit_2 = run2.data.tags.get(unit_tag_name, "dimensionless")
    
    # Check unit compatibility
    if unit_1 != unit_2:
        return {
            "value_1": metric_1,
            "unit_1": unit_1,
            "value_2": metric_2,
            "unit_2": unit_2,
            "compatible": False,
            "difference": None,
            "warning": f"Units differ: {unit_1} vs {unit_2}. Cannot compute difference.",
        }
    
    # Units match - compute difference
    difference = metric_2 - metric_1
    relative_change = (difference / metric_1) * 100 if metric_1 != 0 else float('inf')
    
    return {
        "value_1": metric_1,
        "unit_1": unit_1,
        "value_2": metric_2,
        "unit_2": unit_2,
        "compatible": True,
        "difference": difference,
        "relative_change_percent": relative_change,
    }


class DimMLflowCallback:
    """Callback for automatic logging of dimensional quantities during training.
    
    This callback can be integrated into training loops to automatically log
    metrics and parameters with their units preserved.
    
    Attributes:
        log_every_n_steps: Log metrics every N steps (default: 1).
        logged_params: Set of parameters already logged.
        
    Example:
        >>> callback = DimMLflowCallback(log_every_n_steps=10)
        >>>
        >>> with mlflow.start_run():
        ...     for step in range(100):
        ...         loss = DimArray(compute_loss(), units.J)
        ...         callback.on_step(step, {"loss": loss})
    """
    
    def __init__(self, log_every_n_steps: int = 1):
        """Initialize callback.
        
        Args:
            log_every_n_steps: Frequency of logging (default: every step).
        """
        _check_mlflow()
        self.log_every_n_steps = log_every_n_steps
        self.logged_params: set[str] = set()
    
    def on_train_begin(self, params: dict[str, DimArray | float | int]) -> None:
        """Log training parameters at the start of training.
        
        Args:
            params: Dictionary of parameter names to values.
        """
        for name, value in params.items():
            if name not in self.logged_params:
                log_unit_param(name, value)
                self.logged_params.add(name)
    
    def on_step(
        self,
        step: int,
        metrics: dict[str, DimArray | float],
    ) -> None:
        """Log metrics at each training step.
        
        Args:
            step: Current training step.
            metrics: Dictionary of metric names to values.
        """
        if step % self.log_every_n_steps != 0:
            return
        
        for name, value in metrics.items():
            if isinstance(value, DimArray):
                log_dimarray(name, value, step=step)
            else:
                mlflow.log_metric(name, float(value), step=step)
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, DimArray | float],
    ) -> None:
        """Log metrics at the end of an epoch.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values.
        """
        for name, value in metrics.items():
            metric_name = f"epoch_{name}"
            if isinstance(value, DimArray):
                log_dimarray(metric_name, value, step=epoch)
            else:
                mlflow.log_metric(metric_name, float(value), step=epoch)
    
    def on_train_end(self, final_metrics: dict[str, DimArray | float]) -> None:
        """Log final metrics at the end of training.
        
        Args:
            final_metrics: Dictionary of final metric names to values.
        """
        for name, value in final_metrics.items():
            metric_name = f"final_{name}"
            if isinstance(value, DimArray):
                log_dimarray(metric_name, value)
            else:
                mlflow.log_metric(metric_name, float(value))


# Convenience function for PyTorch integration
def create_mlflow_logger(log_every_n_steps: int = 1) -> DimMLflowCallback:
    """Create an MLflow callback for training loops.
    
    Args:
        log_every_n_steps: Logging frequency.
        
    Returns:
        Configured DimMLflowCallback instance.
    """
    return DimMLflowCallback(log_every_n_steps=log_every_n_steps)


__all__ = [
    "log_dimarray",
    "log_unit_param",
    "log_metric_with_unit",
    "compare_metrics_with_units",
    "DimMLflowCallback",
    "create_mlflow_logger",
    "HAS_MLFLOW",
]
