"""Tests for Weights & Biases integration.

These tests require wandb to be installed and will be skipped if it's not available.
"""

from __future__ import annotations

import pytest

# Skip all tests if wandb is not available
wandb = pytest.importorskip("wandb", reason="wandb not installed")

import numpy as np

from dimtensor import DimArray, units
from dimtensor.integrations.wandb import (
    log_dimarray,
    log_config_with_units,
    create_dimarray_table,
    DimWandbCallback,
    _serialize_unit,
)


@pytest.fixture
def wandb_run():
    """Create a wandb run in offline mode for testing."""
    # Initialize in offline mode to avoid actual uploads
    run = wandb.init(mode="offline", project="test-dimtensor")
    yield run
    run.finish()


class TestSerializeUnit:
    """Tests for unit serialization."""

    def test_serialize_base_unit(self):
        """Test serializing a base SI unit."""
        unit = units.m
        result = _serialize_unit(unit)

        assert result["symbol"] == "m"
        assert result["scale"] == 1.0
        assert result["dimension"]["length"] == 1.0
        assert result["dimension"]["mass"] == 0.0

    def test_serialize_derived_unit(self):
        """Test serializing a derived unit."""
        unit = units.m / units.s
        result = _serialize_unit(unit)

        assert "m" in result["symbol"]
        assert "s" in result["symbol"]
        assert result["dimension"]["length"] == 1.0
        assert result["dimension"]["time"] == -1.0

    def test_serialize_scaled_unit(self):
        """Test serializing a unit with non-unity scale."""
        unit = units.km
        result = _serialize_unit(unit)

        assert result["symbol"] == "km"
        assert result["scale"] == 1000.0
        assert result["dimension"]["length"] == 1.0


class TestLogDimarray:
    """Tests for log_dimarray function."""

    def test_log_scalar_dimarray(self, wandb_run):
        """Test logging a scalar DimArray."""
        value = DimArray([10.0], units.m)
        log_dimarray("distance", value)

        # Check that wandb.run has the logged data
        # Note: In offline mode, we can't easily verify the exact logged values
        # but we can verify the function runs without errors
        assert wandb_run is not None

    def test_log_array_dimarray(self, wandb_run):
        """Test logging an array DimArray."""
        value = DimArray([1.0, 2.0, 3.0, 4.0], units.m / units.s)
        log_dimarray("velocities", value)

        assert wandb_run is not None

    def test_log_with_uncertainty(self, wandb_run):
        """Test logging a DimArray with uncertainty."""
        value = DimArray(
            [10.0, 20.0],
            units.m,
            uncertainty=[0.1, 0.2],
        )
        log_dimarray("measurement", value)

        assert wandb_run is not None

    def test_log_with_step(self, wandb_run):
        """Test logging with a step parameter."""
        value = DimArray([5.0], units.s)
        log_dimarray("time", value, step=42)

        assert wandb_run is not None


class TestLogConfigWithUnits:
    """Tests for log_config_with_units function."""

    def test_log_config_with_dimarrays(self, wandb_run):
        """Test logging config with DimArray values."""
        config = {
            "learning_rate": DimArray([0.001], 1 / units.s),
            "batch_size": 32,
            "max_velocity": DimArray([100.0], units.m / units.s),
        }
        log_config_with_units(config)

        # Verify config was updated
        assert "learning_rate" in wandb_run.config
        assert "batch_size" in wandb_run.config
        assert wandb_run.config["batch_size"] == 32

    def test_log_config_scalar_extraction(self, wandb_run):
        """Test that scalar DimArrays are extracted correctly."""
        config = {
            "param": DimArray([42.0], units.m),
        }
        log_config_with_units(config)

        assert wandb_run.config["param"] == 42.0
        assert "param_unit" in wandb_run.config
        assert wandb_run.config["param_unit"] == "m"

    def test_log_config_array_values(self, wandb_run):
        """Test logging config with array DimArrays."""
        config = {
            "values": DimArray([1.0, 2.0, 3.0], units.kg),
        }
        log_config_with_units(config)

        assert "values" in wandb_run.config
        assert wandb_run.config["values"] == [1.0, 2.0, 3.0]
        assert wandb_run.config["values_unit"] == "kg"


class TestCreateDimarrayTable:
    """Tests for create_dimarray_table function."""

    def test_create_table_with_dimarrays(self, wandb_run):
        """Test creating a table with DimArray values."""
        data = [
            {"x": DimArray([1.0], units.m), "y": DimArray([2.0], units.s)},
            {"x": DimArray([2.0], units.m), "y": DimArray([3.0], units.s)},
            {"x": DimArray([3.0], units.m), "y": DimArray([4.0], units.s)},
        ]
        table = create_dimarray_table(data)

        assert table is not None
        assert len(table.columns) == 2
        assert "m" in table.columns[0]  # x [m]
        assert "s" in table.columns[1]  # y [s]

    def test_create_table_mixed_types(self, wandb_run):
        """Test creating a table with mixed DimArray and regular values."""
        data = [
            {"x": DimArray([1.0], units.m), "label": "first"},
            {"x": DimArray([2.0], units.m), "label": "second"},
        ]
        table = create_dimarray_table(data)

        assert table is not None
        assert len(table.columns) == 2

    def test_create_empty_table(self, wandb_run):
        """Test creating an empty table."""
        data = []
        table = create_dimarray_table(data)

        assert table is not None

    def test_create_table_with_columns(self, wandb_run):
        """Test creating a table with explicit columns."""
        data = [
            {"x": DimArray([1.0], units.m), "y": DimArray([2.0], units.s)},
        ]
        table = create_dimarray_table(data, columns=["x", "y"])

        assert table is not None
        assert len(table.columns) == 2


class TestDimWandbCallback:
    """Tests for DimWandbCallback class."""

    def test_callback_initialization(self, wandb_run):
        """Test callback initialization."""
        callback = DimWandbCallback()
        assert callback.prefix == ""
        assert callback._logged_units == {}

    def test_callback_with_prefix(self, wandb_run):
        """Test callback with prefix."""
        callback = DimWandbCallback(prefix="train_")
        assert callback.prefix == "train_"

    def test_log_epoch_with_dimarrays(self, wandb_run):
        """Test logging epoch metrics with DimArrays."""
        callback = DimWandbCallback()

        metrics = {
            "loss": DimArray([0.5], units.J),
            "accuracy": 0.95,
        }
        callback.log_epoch(metrics, epoch=0)

        # Verify units were tracked
        assert "loss" in callback._logged_units

    def test_log_epoch_scalar_dimarray(self, wandb_run):
        """Test logging scalar DimArray in epoch."""
        callback = DimWandbCallback()

        loss = DimArray([0.123], units.dimensionless)
        callback.log_epoch({"loss": loss}, epoch=1)

        assert "loss" in callback._logged_units

    def test_log_epoch_array_dimarray(self, wandb_run):
        """Test logging array DimArray in epoch."""
        callback = DimWandbCallback()

        gradients = DimArray([0.1, 0.2, 0.3], units.dimensionless)
        callback.log_epoch({"gradients": gradients}, epoch=0)

        assert "gradients" in callback._logged_units

    def test_log_batch(self, wandb_run):
        """Test logging batch metrics."""
        callback = DimWandbCallback()

        metrics = {
            "batch_loss": DimArray([0.3], units.J),
            "batch_size": 32,
        }
        callback.log_batch(metrics, step=0)

        assert "batch_loss" in callback._logged_units

    def test_log_multiple_epochs(self, wandb_run):
        """Test logging multiple epochs."""
        callback = DimWandbCallback()

        for epoch in range(3):
            loss = DimArray([0.5 / (epoch + 1)], units.J)
            callback.log_epoch({"loss": loss}, epoch=epoch)

        # Unit should only be logged once
        assert "loss" in callback._logged_units
        assert callback._logged_units["loss"] == "J"

    def test_finalize_with_summary(self, wandb_run):
        """Test finalizing with summary metrics."""
        callback = DimWandbCallback()

        summary = {
            "final_loss": DimArray([0.01], units.J),
            "final_accuracy": 0.98,
        }
        callback.finalize(summary)

        # Check that summary was added to wandb.run.summary
        assert "final_loss" in wandb_run.summary
        assert wandb_run.summary["final_accuracy"] == 0.98

    def test_finalize_without_summary(self, wandb_run):
        """Test finalizing without summary metrics."""
        callback = DimWandbCallback()
        callback.finalize()  # Should not raise an error

    def test_callback_prefix_usage(self, wandb_run):
        """Test that prefix is correctly applied."""
        callback = DimWandbCallback(prefix="val_")

        metrics = {"loss": DimArray([0.2], units.J)}
        callback.log_epoch(metrics, epoch=0)

        assert "val_loss" in callback._logged_units


class TestIntegrationErrors:
    """Tests for error handling."""

    def test_import_error_without_wandb(self, monkeypatch):
        """Test that appropriate error is raised when wandb is not available."""
        # This test temporarily makes wandb unavailable
        import sys

        # Save original wandb module
        original_wandb = sys.modules.get("wandb")

        try:
            # Remove wandb from modules
            if "wandb" in sys.modules:
                del sys.modules["wandb"]

            # Mock import to raise ImportError
            def mock_import(name, *args, **kwargs):
                if name == "wandb" or name.startswith("wandb."):
                    raise ImportError("No module named 'wandb'")
                return original_import(name, *args, **kwargs)

            original_import = __builtins__.__import__
            __builtins__.__import__ = mock_import

            # Now test that our integration raises proper error
            from dimtensor.integrations import wandb as wandb_integration

            with pytest.raises(ImportError, match="wandb is required"):
                wandb_integration._check_wandb()

        finally:
            # Restore original state
            __builtins__.__import__ = original_import
            if original_wandb is not None:
                sys.modules["wandb"] = original_wandb


class TestRealWorldUsage:
    """Integration tests simulating real-world usage patterns."""

    def test_training_loop_simulation(self, wandb_run):
        """Simulate a simple training loop with DimArrays."""
        callback = DimWandbCallback()

        # Simulate training
        for epoch in range(5):
            # Physics-informed loss with units
            loss = DimArray([1.0 / (epoch + 1)], units.J)
            velocity = DimArray([10.0 + epoch], units.m / units.s)

            callback.log_epoch(
                {
                    "loss": loss,
                    "mean_velocity": velocity,
                    "epoch_num": epoch,
                },
                epoch=epoch,
            )

        # Final summary
        callback.finalize(
            {
                "best_loss": DimArray([0.1], units.J),
            }
        )

        assert "loss" in callback._logged_units
        assert "mean_velocity" in callback._logged_units

    def test_config_and_metrics_together(self, wandb_run):
        """Test logging both config and metrics."""
        # Log config
        config = {
            "learning_rate": DimArray([0.001], 1 / units.s),
            "max_force": DimArray([100.0], units.N),
            "batch_size": 32,
        }
        log_config_with_units(config)

        # Log metrics
        callback = DimWandbCallback()
        for step in range(3):
            force = DimArray([50.0 + step * 10], units.N)
            callback.log_batch({"applied_force": force}, step=step)

        assert "applied_force" in callback._logged_units

    def test_table_logging(self, wandb_run):
        """Test creating and logging a table with results."""
        results = [
            {
                "time": DimArray([i], units.s),
                "distance": DimArray([i * 10], units.m),
                "velocity": DimArray([10], units.m / units.s),
            }
            for i in range(5)
        ]

        table = create_dimarray_table(results)
        wandb.log({"results": table})

        # Table should have been created successfully
        assert table is not None
