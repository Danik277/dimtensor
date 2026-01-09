"""Tests for dimension-aware loss functions."""

import pytest

torch = pytest.importorskip("torch")

from dimtensor import Dimension, DimensionError, units
from dimtensor.torch import (
    CompositeLoss,
    DimHuberLoss,
    DimL1Loss,
    DimMSELoss,
    DimTensor,
    PhysicsLoss,
)


class TestDimMSELoss:
    """Tests for DimMSELoss."""

    def test_basic_loss(self):
        """Test basic MSE loss computation."""
        loss_fn = DimMSELoss()
        pred = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
        target = DimTensor(torch.tensor([1.1, 1.9, 3.2]), units.m)

        loss = loss_fn(pred, target)

        assert isinstance(loss, DimTensor)
        # MSE has dimension of input squared
        assert loss.dimension == Dimension(length=2)
        assert loss.data.item() > 0

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        loss_fn = DimMSELoss()
        pred = DimTensor(torch.tensor([1.0, 2.0]), units.m)
        target = DimTensor(torch.tensor([1.0, 2.0]), units.s)

        with pytest.raises(DimensionError):
            loss_fn(pred, target)

    def test_skip_check(self):
        """Test that unit checking can be disabled."""
        loss_fn = DimMSELoss(check_units=False)
        pred = DimTensor(torch.tensor([1.0, 2.0]), units.m)
        target = DimTensor(torch.tensor([1.0, 2.0]), units.s)

        # Should not raise
        loss = loss_fn(pred, target)
        assert isinstance(loss, DimTensor)

    def test_raw_tensor(self):
        """Test with raw tensors."""
        loss_fn = DimMSELoss()
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.2])

        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
        target = DimTensor(torch.tensor([1.1, 1.9, 3.2]), units.m)

        # Mean reduction (default)
        loss_mean = DimMSELoss(reduction="mean")(pred, target)
        assert loss_mean.data.ndim == 0

        # Sum reduction
        loss_sum = DimMSELoss(reduction="sum")(pred, target)
        assert loss_sum.data.ndim == 0

        # No reduction
        loss_none = DimMSELoss(reduction="none")(pred, target)
        assert loss_none.data.shape == (3,)


class TestDimL1Loss:
    """Tests for DimL1Loss."""

    def test_basic_loss(self):
        """Test basic L1 loss."""
        loss_fn = DimL1Loss()
        pred = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m / units.s)
        target = DimTensor(torch.tensor([1.5, 2.5, 2.5]), units.m / units.s)

        loss = loss_fn(pred, target)

        assert isinstance(loss, DimTensor)
        # L1 preserves dimension
        assert loss.dimension == Dimension(length=1, time=-1)

    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        loss_fn = DimL1Loss()
        pred = DimTensor(torch.tensor([1.0]), units.m)
        target = DimTensor(torch.tensor([1.0]), units.kg)

        with pytest.raises(DimensionError):
            loss_fn(pred, target)


class TestDimHuberLoss:
    """Tests for DimHuberLoss."""

    def test_basic_loss(self):
        """Test basic Huber loss."""
        loss_fn = DimHuberLoss(delta=1.0)
        pred = DimTensor(torch.tensor([1.0, 2.0, 10.0]), units.J)
        target = DimTensor(torch.tensor([1.1, 1.9, 3.0]), units.J)

        loss = loss_fn(pred, target)

        assert isinstance(loss, DimTensor)
        assert loss.dimension == Dimension(mass=1, length=2, time=-2)  # Energy


class TestPhysicsLoss:
    """Tests for PhysicsLoss."""

    def test_conservation_satisfied(self):
        """Test loss when conservation is satisfied."""
        loss_fn = PhysicsLoss(rtol=0.01)

        E_initial = DimTensor(torch.tensor([100.0, 200.0]), units.J)
        E_final = DimTensor(torch.tensor([100.0, 200.0]), units.J)

        loss = loss_fn(E_initial, E_final)

        assert loss.item() == pytest.approx(0.0)

    def test_conservation_violated(self):
        """Test loss when conservation is violated."""
        loss_fn = PhysicsLoss(rtol=0.001)

        E_initial = DimTensor(torch.tensor([100.0]), units.J)
        E_final = DimTensor(torch.tensor([90.0]), units.J)  # 10% loss

        loss = loss_fn(E_initial, E_final)

        assert loss.item() > 0

    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        loss_fn = PhysicsLoss()

        with pytest.raises(DimensionError):
            loss_fn(
                DimTensor(torch.tensor([100.0]), units.J),
                DimTensor(torch.tensor([100.0]), units.m),
            )

    def test_reduction_modes(self):
        """Test reduction modes."""
        E_initial = DimTensor(torch.tensor([100.0, 200.0, 300.0]), units.J)
        E_final = DimTensor(torch.tensor([95.0, 190.0, 285.0]), units.J)

        loss_none = PhysicsLoss(reduction="none")(E_initial, E_final)
        assert loss_none.shape == (3,)

        loss_sum = PhysicsLoss(reduction="sum")(E_initial, E_final)
        assert loss_sum.ndim == 0


class TestCompositeLoss:
    """Tests for CompositeLoss."""

    def test_data_loss_only(self):
        """Test with only data loss."""
        loss_fn = CompositeLoss(data_loss=DimMSELoss())

        pred = DimTensor(torch.tensor([1.0, 2.0]), units.m)
        target = DimTensor(torch.tensor([1.1, 1.9]), units.m)

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_with_physics_loss(self):
        """Test with physics loss term."""
        loss_fn = CompositeLoss(
            data_loss=DimMSELoss(),
            physics_losses={"energy": (PhysicsLoss(), 0.1)},
        )

        pred = DimTensor(torch.tensor([1.0, 2.0]), units.m)
        target = DimTensor(torch.tensor([1.1, 1.9]), units.m)

        E_i = DimTensor(torch.tensor([100.0]), units.J)
        E_f = DimTensor(torch.tensor([95.0]), units.J)

        loss = loss_fn(pred, target, physics_terms={"energy": (E_i, E_f)})

        # Should be data loss + 0.1 * physics loss
        assert loss.item() > 0

    def test_multiple_physics_terms(self):
        """Test with multiple physics terms."""
        loss_fn = CompositeLoss(
            data_loss=DimMSELoss(),
            physics_losses={
                "energy": (PhysicsLoss(), 0.1),
                "momentum": (PhysicsLoss(), 0.05),
            },
        )

        pred = DimTensor(torch.randn(10), units.m)
        target = DimTensor(torch.randn(10), units.m)

        physics = {
            "energy": (
                DimTensor(torch.tensor([100.0]), units.J),
                DimTensor(torch.tensor([100.0]), units.J),
            ),
            "momentum": (
                DimTensor(torch.tensor([10.0]), units.kg * units.m / units.s),
                DimTensor(torch.tensor([10.0]), units.kg * units.m / units.s),
            ),
        }

        loss = loss_fn(pred, target, physics_terms=physics)
        assert loss.item() >= 0
