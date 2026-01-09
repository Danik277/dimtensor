"""Dataset registry for physics-aware machine learning.

Provides a registry of datasets with dimensional metadata for
training physics-informed neural networks.

Example:
    >>> from dimtensor.datasets import list_datasets, get_dataset_info
    >>>
    >>> # List all datasets
    >>> datasets = list_datasets()
    >>> for ds in datasets[:3]:
    ...     print(f"{ds.name}: {ds.description}")
    >>>
    >>> # Get specific dataset
    >>> info = get_dataset_info("pendulum")
"""

from .registry import (
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    load_dataset,
    register_dataset,
)

__all__ = [
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    "register_dataset",
]
