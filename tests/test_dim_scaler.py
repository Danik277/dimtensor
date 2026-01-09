"""Tests for DimScaler and MultiScaler."""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from dimtensor import DimArray, Dimension, DimensionError, units
from dimtensor.torch import DimScaler, DimTensor, MultiScaler


class TestDimScaler:
    """Tests for DimScaler."""

    def test_characteristic_scaling(self):
        """Test characteristic value scaling."""
        scaler = DimScaler(method="characteristic")
        data = DimArray([10, 100, 1000], units.m / units.s)
        scaler.fit(data)

        scaled = scaler.transform(data)

        # Max absolute value is 1000, so scaled should be in [-1, 1]
        assert torch.max(torch.abs(scaled)).item() <= 1.0
        assert torch.allclose(scaled, torch.tensor([0.01, 0.1, 1.0]))

    def test_standard_scaling(self):
        """Test z-score normalization."""
        scaler = DimScaler(method="standard")
        data = DimArray([100, 200, 300], units.K)
        scaler.fit(data)

        scaled = scaler.transform(data)

        # Should have mean ~0
        assert abs(scaled.mean().item()) < 0.01

    def test_minmax_scaling(self):
        """Test min-max scaling."""
        scaler = DimScaler(method="minmax")
        data = DimArray([10, 50, 100], units.Pa)
        scaler.fit(data)

        scaled = scaler.transform(data)

        # Should be in [0, 1]
        assert scaled.min().item() >= -0.01
        assert scaled.max().item() <= 1.01

    def test_inverse_transform(self):
        """Test round-trip transform."""
        scaler = DimScaler(method="characteristic")
        data = DimArray([10, 100, 1000], units.m / units.s)
        scaler.fit(data)

        scaled = scaler.transform(data)
        recovered = scaler.inverse_transform(scaled, units.m / units.s)

        assert isinstance(recovered, DimTensor)
        assert recovered.dimension == Dimension(length=1, time=-1)
        assert torch.allclose(
            recovered.data,
            torch.tensor([10.0, 100.0, 1000.0]),
        )

    def test_multiple_dimensions(self):
        """Test fitting with multiple dimensions."""
        scaler = DimScaler()

        velocity = DimArray([10, 20, 30], units.m / units.s)
        temperature = DimArray([300, 400, 500], units.K)

        scaler.fit(velocity, temperature)

        # Should handle both dimensions
        v_scaled = scaler.transform(velocity)
        T_scaled = scaler.transform(temperature)

        assert v_scaled.shape == (3,)
        assert T_scaled.shape == (3,)

    def test_unknown_dimension_error(self):
        """Test error on unknown dimension."""
        scaler = DimScaler()
        scaler.fit(DimArray([1, 2, 3], units.m))

        with pytest.raises(DimensionError):
            scaler.transform(DimArray([1, 2, 3], units.s))

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        scaler = DimScaler()

        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(DimArray([1, 2, 3], units.m))

    def test_fit_empty_error(self):
        """Test error on empty fit."""
        scaler = DimScaler()

        with pytest.raises(ValueError):
            scaler.fit()

    def test_dimtensor_input(self):
        """Test with DimTensor input."""
        scaler = DimScaler()
        data = DimTensor(torch.tensor([10.0, 100.0, 1000.0]), units.m)
        scaler.fit(data)

        scaled = scaler.transform(data)
        assert isinstance(scaled, torch.Tensor)

    def test_get_scale_and_offset(self):
        """Test getting scale and offset."""
        scaler = DimScaler(method="characteristic")
        data = DimArray([10, 100, 1000], units.m / units.s)
        scaler.fit(data)

        dim = Dimension(length=1, time=-1)
        assert scaler.get_scale(dim) == 1000.0
        assert scaler.get_offset(dim) == 0.0

    def test_dimensions_property(self):
        """Test dimensions property."""
        scaler = DimScaler()
        scaler.fit(
            DimArray([1, 2, 3], units.m),
            DimArray([100, 200, 300], units.K),
        )

        dims = scaler.dimensions
        assert len(dims) == 2
        assert Dimension(length=1) in dims
        assert Dimension(temperature=1) in dims

    def test_repr(self):
        """Test string representation."""
        scaler = DimScaler(method="standard")
        assert "fitted=False" in repr(scaler)

        scaler.fit(DimArray([1, 2, 3], units.m))
        assert "standard" in repr(scaler)

    def test_zero_scale_handling(self):
        """Test handling of constant (zero variance) data."""
        scaler = DimScaler(method="characteristic")
        data = DimArray([5.0, 5.0, 5.0], units.m)
        scaler.fit(data)

        # Should not divide by zero
        scaled = scaler.transform(data)
        assert not torch.any(torch.isnan(scaled))


class TestMultiScaler:
    """Tests for MultiScaler."""

    def test_add_and_transform(self):
        """Test adding quantities and transforming."""
        scaler = MultiScaler()

        position = DimArray([0, 10, 100], units.m)
        velocity = DimArray([1, 10, 100], units.m / units.s)

        scaler.add("position", position)
        scaler.add("velocity", velocity)

        x_scaled = scaler.transform("position", position)
        v_scaled = scaler.transform("velocity", velocity)

        assert x_scaled.shape == (3,)
        assert v_scaled.shape == (3,)

    def test_inverse_transform(self):
        """Test inverse transformation."""
        scaler = MultiScaler()
        scaler.add("temperature", DimArray([300, 400, 500], units.K))

        T = DimArray([350, 450], units.K)
        T_scaled = scaler.transform("temperature", T)
        T_recovered = scaler.inverse_transform("temperature", T_scaled, units.K)

        assert isinstance(T_recovered, DimTensor)
        assert T_recovered.dimension == Dimension(temperature=1)

    def test_unknown_quantity_error(self):
        """Test error on unknown quantity name."""
        scaler = MultiScaler()
        scaler.add("position", DimArray([1, 2, 3], units.m))

        with pytest.raises(KeyError):
            scaler.transform("velocity", DimArray([1, 2, 3], units.m / units.s))

    def test_get_scaler(self):
        """Test getting individual scaler."""
        scaler = MultiScaler()
        scaler.add("position", DimArray([1, 2, 3], units.m))

        pos_scaler = scaler.get_scaler("position")
        assert isinstance(pos_scaler, DimScaler)

    def test_quantities_property(self):
        """Test quantities property."""
        scaler = MultiScaler()
        scaler.add("position", DimArray([1, 2, 3], units.m))
        scaler.add("velocity", DimArray([1, 2, 3], units.m / units.s))

        assert scaler.quantities == ["position", "velocity"]

    def test_chaining(self):
        """Test method chaining."""
        scaler = (
            MultiScaler()
            .add("x", DimArray([1, 2, 3], units.m))
            .add("v", DimArray([10, 20, 30], units.m / units.s))
        )

        assert len(scaler.quantities) == 2

    def test_repr(self):
        """Test string representation."""
        scaler = MultiScaler()
        scaler.add("position", DimArray([1, 2, 3], units.m))

        assert "position" in repr(scaler)
