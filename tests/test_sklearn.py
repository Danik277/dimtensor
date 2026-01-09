"""Tests for scikit-learn integration."""

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, units

pytest.importorskip("sklearn")

from dimtensor.sklearn import DimStandardScaler, DimMinMaxScaler


class TestDimStandardScaler:
    """Tests for DimStandardScaler."""

    def test_fit_transform(self):
        """Test standard scaler fit_transform."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimStandardScaler()

        X_scaled = scaler.fit_transform(X)

        # Should be standardized (zero mean, unit variance)
        assert isinstance(X_scaled, np.ndarray)
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0])
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), [1, 1])

    def test_inverse_transform(self):
        """Test inverse transform restores units."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimStandardScaler()

        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert isinstance(X_restored, DimArray)
        assert X_restored.unit.dimension == Dimension(length=1)
        np.testing.assert_array_almost_equal(X_restored.data, X.data)

    def test_without_mean(self):
        """Test scaler without centering."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimStandardScaler(with_mean=False)

        X_scaled = scaler.fit_transform(X)

        # Should not be centered
        assert not np.allclose(np.mean(X_scaled, axis=0), [0, 0])

    def test_without_std(self):
        """Test scaler without variance scaling."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimStandardScaler(with_std=False)

        X_scaled = scaler.fit_transform(X)

        # Should be centered but not scaled
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0])
        # Std should not be 1
        assert not np.allclose(np.std(X_scaled, axis=0), [1, 1])

    def test_numpy_input(self):
        """Test with plain numpy array."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = DimStandardScaler()

        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert isinstance(X_restored, np.ndarray)
        np.testing.assert_array_almost_equal(X_restored, X)


class TestDimMinMaxScaler:
    """Tests for DimMinMaxScaler."""

    def test_fit_transform(self):
        """Test min-max scaler fit_transform."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimMinMaxScaler()

        X_scaled = scaler.fit_transform(X)

        # Should be in [0, 1]
        assert isinstance(X_scaled, np.ndarray)
        assert np.min(X_scaled) >= 0
        assert np.max(X_scaled) <= 1

    def test_inverse_transform(self):
        """Test inverse transform restores units."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimMinMaxScaler()

        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert isinstance(X_restored, DimArray)
        assert X_restored.unit.dimension == Dimension(length=1)
        np.testing.assert_array_almost_equal(X_restored.data, X.data)

    def test_custom_range(self):
        """Test custom feature range."""
        X = DimArray([[1, 2], [3, 4], [5, 6]], units.m)
        scaler = DimMinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler.fit_transform(X)

        # Should be in [-1, 1]
        assert np.min(X_scaled) >= -1
        assert np.max(X_scaled) <= 1

    def test_numpy_input(self):
        """Test with plain numpy array."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = DimMinMaxScaler()

        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert isinstance(X_restored, np.ndarray)
        np.testing.assert_array_almost_equal(X_restored, X)
