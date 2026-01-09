"""Dimension-aware scikit-learn transformers.

Provides sklearn-compatible transformers that preserve physical units.
"""
# mypy: disable-error-code="type-arg,misc,no-any-return"

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Provide stub classes
    class BaseEstimator:  # type: ignore
        pass
    class TransformerMixin:  # type: ignore
        pass


def _check_sklearn() -> None:
    """Check that sklearn is available."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for transformers. "
            "Install with: pip install scikit-learn"
        )


class DimTransformerMixin:
    """Mixin for dimension-aware transformers.

    Provides unit handling for sklearn-compatible transformers.
    """

    _unit: Unit | None = None

    def _extract_data(self, X: DimArray | np.ndarray) -> np.ndarray:
        """Extract numpy array from DimArray, storing unit."""
        if isinstance(X, DimArray):
            self._unit = X.unit
            return np.asarray(X.data)
        return np.asarray(X)

    def _wrap_result(self, X: np.ndarray) -> DimArray | np.ndarray:
        """Wrap numpy array back to DimArray if we have a unit."""
        if self._unit is not None:
            return DimArray._from_data_and_unit(X, self._unit)
        return X


class DimStandardScaler(BaseEstimator, TransformerMixin, DimTransformerMixin):
    """Dimension-aware standard scaler.

    Standardizes features by removing mean and scaling to unit variance,
    while preserving physical units.

    Note: The scaled values are dimensionless (z-scores), but
    inverse_transform restores the original units.

    Attributes:
        with_mean: If True, center data before scaling.
        with_std: If True, scale data to unit variance.
        mean_: Mean of each feature (DimArray).
        scale_: Standard deviation of each feature (DimArray).
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """Initialize scaler.

        Args:
            with_mean: If True, center data before scaling.
            with_std: If True, scale data to unit variance.
        """
        _check_sklearn()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._unit: Unit | None = None

    def fit(self, X: DimArray | np.ndarray, y: Any = None) -> "DimStandardScaler":
        """Compute mean and std for scaling.

        Args:
            X: Training data with shape (n_samples, n_features).
            y: Ignored.

        Returns:
            self
        """
        X_data = self._extract_data(X)

        if self.with_mean:
            self.mean_ = np.mean(X_data, axis=0)
        else:
            self.mean_ = np.zeros(X_data.shape[1] if X_data.ndim > 1 else 1)

        if self.with_std:
            self.scale_ = np.std(X_data, axis=0)
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X_data.shape[1] if X_data.ndim > 1 else 1)

        return self

    def transform(self, X: DimArray | np.ndarray) -> np.ndarray:
        """Scale features.

        Args:
            X: Data to transform.

        Returns:
            Scaled data (dimensionless numpy array).
        """
        X_data = self._extract_data(X)

        if self.mean_ is None:
            raise RuntimeError("Must call fit() before transform()")

        if self.with_mean:
            X_data = X_data - self.mean_
        if self.with_std:
            X_data = X_data / self.scale_

        return X_data

    def inverse_transform(self, X: np.ndarray) -> DimArray | np.ndarray:
        """Undo scaling.

        Args:
            X: Scaled data.

        Returns:
            Original-scale data (DimArray if original had units).
        """
        if self.mean_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")

        X_data = np.asarray(X)

        if self.with_std:
            X_data = X_data * self.scale_
        if self.with_mean:
            X_data = X_data + self.mean_

        return self._wrap_result(X_data)

    def fit_transform(
        self, X: DimArray | np.ndarray, y: Any = None
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Data to fit and transform.
            y: Ignored.

        Returns:
            Scaled data.
        """
        return self.fit(X, y).transform(X)


class DimMinMaxScaler(BaseEstimator, TransformerMixin, DimTransformerMixin):
    """Dimension-aware min-max scaler.

    Scales features to a given range, while preserving physical units.

    Attributes:
        feature_range: Desired range of transformed data.
        min_: Minimum of each feature (in original units).
        max_: Maximum of each feature (in original units).
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        """Initialize scaler.

        Args:
            feature_range: Desired range of transformed data.
        """
        _check_sklearn()
        self.feature_range = feature_range
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self._unit: Unit | None = None

    def fit(self, X: DimArray | np.ndarray, y: Any = None) -> "DimMinMaxScaler":
        """Compute min and max for scaling.

        Args:
            X: Training data.
            y: Ignored.

        Returns:
            self
        """
        X_data = self._extract_data(X)

        self.min_ = np.min(X_data, axis=0)
        self.max_ = np.max(X_data, axis=0)

        return self

    def transform(self, X: DimArray | np.ndarray) -> np.ndarray:
        """Scale features to range.

        Args:
            X: Data to transform.

        Returns:
            Scaled data (dimensionless).
        """
        X_data = self._extract_data(X)

        if self.min_ is None:
            raise RuntimeError("Must call fit() before transform()")

        # Scale to [0, 1]
        range_ = self.max_ - self.min_
        range_ = np.where(range_ == 0, 1.0, range_)  # Avoid division by zero
        X_scaled = (X_data - self.min_) / range_

        # Scale to feature_range
        min_val, max_val = self.feature_range
        X_scaled = X_scaled * (max_val - min_val) + min_val

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> DimArray | np.ndarray:
        """Undo scaling.

        Args:
            X: Scaled data.

        Returns:
            Original-scale data (DimArray if original had units).
        """
        if self.min_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")

        X_data = np.asarray(X)

        # Undo feature_range scaling
        min_val, max_val = self.feature_range
        X_data = (X_data - min_val) / (max_val - min_val)

        # Undo [0, 1] scaling
        range_ = self.max_ - self.min_
        X_data = X_data * range_ + self.min_

        return self._wrap_result(X_data)

    def fit_transform(
        self, X: DimArray | np.ndarray, y: Any = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
