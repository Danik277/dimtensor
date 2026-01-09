"""Integrations with external ML/data science tools.

Provides dimension-aware wrappers for popular ML platforms and tools.

Examples:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.integrations.mlflow import log_dimarray
    >>>
    >>> import mlflow
    >>> with mlflow.start_run():
    ...     loss = DimArray(0.123, units.dimensionless)
    ...     log_dimarray("loss", loss)

    >>> from dimtensor.integrations.wandb import DimWandbCallback
    >>>
    >>> import wandb
    >>> wandb.init(project="physics-ml")
    >>> callback = DimWandbCallback()
    >>> callback.log_epoch({"loss": loss}, epoch=0)
"""

__all__ = []

# Conditional imports - only expose if dependencies are available
try:
    from .mlflow import log_dimarray as mlflow_log_dimarray
    from .mlflow import DimMLflowCallback
    __all__.extend(["mlflow_log_dimarray", "DimMLflowCallback"])
except ImportError:
    pass

try:
    from .wandb import log_dimarray as wandb_log_dimarray
    from .wandb import DimWandbCallback, log_config_with_units, create_dimarray_table
    __all__.extend([
        "wandb_log_dimarray",
        "DimWandbCallback",
        "log_config_with_units",
        "create_dimarray_table",
    ])
except ImportError:
    pass
