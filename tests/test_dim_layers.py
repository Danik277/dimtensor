"""Tests for dimension-aware neural network layers."""

import pytest

torch = pytest.importorskip("torch")

from dimtensor import Dimension, DimensionError, units
from dimtensor.torch import (
    DimConv1d,
    DimConv2d,
    DimLayer,
    DimLinear,
    DimSequential,
    DimTensor,
)


class TestDimLinear:
    """Tests for DimLinear layer."""

    def test_basic_forward(self):
        """Test basic forward pass with dimension tracking."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )
        x = DimTensor(torch.randn(32, 10), units.m)
        y = layer(x)

        assert y.shape == (32, 5)
        assert y.dimension == Dimension(length=1, time=-1)

    def test_dimension_validation(self):
        """Test that input dimension is validated."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )
        # Wrong input dimension
        x = DimTensor(torch.randn(32, 10), units.s)

        with pytest.raises(DimensionError):
            layer(x)

    def test_skip_validation(self):
        """Test that validation can be disabled."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
            validate_input=False,
        )
        # Wrong dimension but validation disabled
        x = DimTensor(torch.randn(32, 10), units.s)
        y = layer(x)

        # Should still get output dimension
        assert y.dimension == Dimension(length=1, time=-1)

    def test_raw_tensor_input(self):
        """Test that raw tensors work without validation."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )
        x = torch.randn(32, 10)
        y = layer(x)

        assert isinstance(y, DimTensor)
        assert y.dimension == Dimension(length=1, time=-1)

    def test_no_bias(self):
        """Test layer without bias."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1),
            bias=False,
        )
        assert layer.linear.bias is None

        x = DimTensor(torch.randn(32, 10), units.m)
        y = layer(x)
        assert y.shape == (32, 5)

    def test_gradient_flow(self):
        """Test that gradients flow through."""
        layer = DimLinear(
            in_features=3,
            out_features=2,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1),
        )
        x = DimTensor(torch.randn(4, 3, requires_grad=True), units.m)
        y = layer(x)
        loss = y.data.sum()
        loss.backward()

        assert layer.linear.weight.grad is not None


class TestDimConv1d:
    """Tests for DimConv1d layer."""

    def test_basic_forward(self):
        """Test basic 1D convolution."""
        layer = DimConv1d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1),
        )
        x = DimTensor(torch.randn(32, 1, 100), units.m)
        y = layer(x)

        assert y.shape == (32, 4, 98)
        assert y.dimension == Dimension(length=1)

    def test_with_padding(self):
        """Test convolution with padding."""
        layer = DimConv1d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            padding=1,
            input_dim=Dimension(length=1, time=-1),
            output_dim=Dimension(length=1, time=-1),
        )
        x = DimTensor(torch.randn(16, 2, 50), units.m / units.s)
        y = layer(x)

        assert y.shape == (16, 4, 50)  # Same length due to padding

    def test_dimension_change(self):
        """Test dimension transformation through conv."""
        layer = DimConv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=5,
            input_dim=Dimension(length=1),  # meters
            output_dim=Dimension(length=1, time=-1),  # velocity
        )
        x = DimTensor(torch.randn(8, 1, 100), units.m)
        y = layer(x)

        assert y.dimension == Dimension(length=1, time=-1)


class TestDimConv2d:
    """Tests for DimConv2d layer."""

    def test_basic_forward(self):
        """Test basic 2D convolution."""
        layer = DimConv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            input_dim=Dimension(temperature=1),  # temperature
            output_dim=Dimension(temperature=1),
        )
        x = DimTensor(torch.randn(16, 1, 64, 64), units.K)
        y = layer(x)

        assert y.shape == (16, 8, 62, 62)
        assert y.dimension == Dimension(temperature=1)

    def test_with_padding_and_stride(self):
        """Test conv with padding and stride."""
        layer = DimConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1),
        )
        x = DimTensor(torch.randn(8, 3, 32, 32), units.m)
        y = layer(x)

        assert y.shape == (8, 16, 16, 16)  # Downsampled by stride


class TestDimSequential:
    """Tests for DimSequential container."""

    def test_basic_chain(self):
        """Test chaining layers."""
        L = Dimension(length=1)
        V = Dimension(length=1, time=-1)
        A = Dimension(length=1, time=-2)

        model = DimSequential(
            DimLinear(3, 16, input_dim=L, output_dim=V),
            DimLinear(16, 3, input_dim=V, output_dim=A),
        )

        x = DimTensor(torch.randn(8, 3), units.m)
        y = model(x)

        assert y.shape == (8, 3)
        assert y.dimension == A

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        L = Dimension(length=1)
        V = Dimension(length=1, time=-1)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            DimSequential(
                DimLinear(3, 16, input_dim=L, output_dim=L),  # Output L
                DimLinear(16, 3, input_dim=V, output_dim=V),  # Expects V
            )

    def test_len_and_getitem(self):
        """Test container interface."""
        model = DimSequential(
            DimLinear(3, 16),
            DimLinear(16, 8),
            DimLinear(8, 3),
        )

        assert len(model) == 3
        assert isinstance(model[1], DimLinear)

    def test_input_output_dims(self):
        """Test chain input/output dimension properties."""
        L = Dimension(length=1)
        V = Dimension(length=1, time=-1)
        A = Dimension(length=1, time=-2)

        model = DimSequential(
            DimLinear(3, 16, input_dim=L, output_dim=V),
            DimLinear(16, 3, input_dim=V, output_dim=A),
        )

        assert model.input_dim == L
        assert model.output_dim == A


class TestDimLayerRepr:
    """Test string representations."""

    def test_linear_repr(self):
        """Test DimLinear repr."""
        layer = DimLinear(
            in_features=10,
            out_features=5,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )
        repr_str = repr(layer)
        assert "in_features=10" in repr_str
        assert "out_features=5" in repr_str

    def test_conv_repr(self):
        """Test DimConv repr."""
        layer = DimConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
        )
        repr_str = repr(layer)
        assert "in_channels=3" in repr_str
        assert "out_channels=16" in repr_str
