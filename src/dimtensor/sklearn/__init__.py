"""Scikit-learn integration for dimension-aware transformers.

Provides sklearn-compatible transformers that preserve units.

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.sklearn import DimStandardScaler
    >>>
    >>> X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
    >>> scaler = DimStandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
"""

from .transformers import (
    DimStandardScaler,
    DimMinMaxScaler,
    DimTransformerMixin,
)

__all__ = [
    "DimStandardScaler",
    "DimMinMaxScaler",
    "DimTransformerMixin",
]
