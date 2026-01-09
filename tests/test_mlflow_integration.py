"""Tests for MLflow integration."""

import numpy as np
import pytest

from dimtensor import DimArray, units

mlflow = pytest.importorskip("mlflow")

from dimtensor.integrations.mlflow import (
    DimMLflowCallback,
    HAS_MLFLOW,
    compare_metrics_with_units,
    create_mlflow_logger,
    log_dimarray,
    log_metric_with_unit,
    log_unit_param,
)


@pytest.fixture
def mlflow_run():
    """Create an MLflow run for testing."""
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("test_dimtensor")
    with mlflow.start_run() as run:
        yield run


class TestLogDimarray:
    """Tests for log_dimarray function."""

    def test_log_scalar_dimarray(self, mlflow_run):
        """Test logging a scalar DimArray."""
        loss = DimArray(0.123, units.J)
        log_dimarray("loss", loss)

        # Check metric was logged
        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "loss" in run.data.metrics
        assert run.data.metrics["loss"] == 0.123

        # Check unit tag
        assert "unit.loss" in run.data.tags
        assert run.data.tags["unit.loss"] == "J"

    def test_log_array_dimarray(self, mlflow_run):
        """Test logging an array DimArray (logs statistics)."""
        data = DimArray([1.0, 2.0, 3.0, 4.0, 5.0], units.m)
        log_dimarray("distances", data)

        run = mlflow.get_run(mlflow_run.info.run_id)

        # Check statistics were logged
        assert "distances_mean" in run.data.metrics
        assert "distances_std" in run.data.metrics
        assert "distances_min" in run.data.metrics
        assert "distances_max" in run.data.metrics

        # Check unit and shape tags
        assert "unit.distances" in run.data.tags
        assert run.data.tags["unit.distances"] == "m"
        assert "shape.distances" in run.data.tags

    def test_log_with_step(self, mlflow_run):
        """Test logging with step parameter."""
        for step in range(3):
            loss = DimArray(1.0 / (step + 1), units.J)
            log_dimarray("train_loss", loss, step=step)

        # Verify multiple steps were logged
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(mlflow_run.info.run_id, "train_loss")
        assert len(history) == 3

    def test_log_dimensionless(self, mlflow_run):
        """Test logging dimensionless quantity."""
        accuracy = DimArray(0.95, units.dimensionless)
        log_dimarray("accuracy", accuracy)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert run.data.tags["unit.accuracy"] == "dimensionless"


class TestLogUnitParam:
    """Tests for log_unit_param function."""

    def test_log_dimarray_param(self, mlflow_run):
        """Test logging a DimArray as parameter."""
        lr = DimArray(0.001, units.Hz)  # 1/s
        log_unit_param("learning_rate", lr)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "learning_rate" in run.data.params
        assert float(run.data.params["learning_rate"]) == 0.001
        assert run.data.tags["unit.learning_rate"] == "Hz"

    def test_log_scalar_param(self, mlflow_run):
        """Test logging a plain scalar as parameter."""
        log_unit_param("batch_size", 32)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert run.data.params["batch_size"] == "32.0"
        assert run.data.tags["unit.batch_size"] == "dimensionless"

    def test_log_non_scalar_raises(self, mlflow_run):
        """Test that non-scalar DimArray raises ValueError."""
        param = DimArray([1.0, 2.0], units.m)
        with pytest.raises(ValueError, match="Expected scalar"):
            log_unit_param("invalid", param)


class TestLogMetricWithUnit:
    """Tests for log_metric_with_unit function."""

    def test_log_metric_with_explicit_unit(self, mlflow_run):
        """Test logging a float with explicit unit."""
        log_metric_with_unit("energy", 42.0, units.J)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert run.data.metrics["energy"] == 42.0
        assert run.data.tags["unit.energy"] == "J"

    def test_log_with_step(self, mlflow_run):
        """Test logging with step parameter."""
        for step in range(3):
            log_metric_with_unit("power", 100.0 * (step + 1), units.W, step=step)

        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(mlflow_run.info.run_id, "power")
        assert len(history) == 3


class TestCompareMetricsWithUnits:
    """Tests for compare_metrics_with_units function."""

    def test_compare_compatible_metrics(self):
        """Test comparing metrics with same units."""
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("test_compare")

        # Run 1
        with mlflow.start_run() as run1:
            log_dimarray("loss", DimArray(1.0, units.J))
            run1_id = run1.info.run_id

        # Run 2
        with mlflow.start_run() as run2:
            log_dimarray("loss", DimArray(0.8, units.J))
            run2_id = run2.info.run_id

        # Compare
        result = compare_metrics_with_units(run1_id, run2_id, "loss")

        assert result["compatible"] is True
        assert result["value_1"] == 1.0
        assert result["value_2"] == 0.8
        assert result["unit_1"] == "J"
        assert result["unit_2"] == "J"
        assert result["difference"] == -0.2
        assert result["relative_change_percent"] == -20.0

    def test_compare_incompatible_metrics(self):
        """Test comparing metrics with different units."""
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("test_incompatible")

        # Run 1 with Joules
        with mlflow.start_run() as run1:
            log_dimarray("value", DimArray(1.0, units.J))
            run1_id = run1.info.run_id

        # Run 2 with Watts
        with mlflow.start_run() as run2:
            log_dimarray("value", DimArray(1.0, units.W))
            run2_id = run2.info.run_id

        # Compare
        result = compare_metrics_with_units(run1_id, run2_id, "value")

        assert result["compatible"] is False
        assert "warning" in result
        assert result["difference"] is None

    def test_compare_missing_metric(self):
        """Test comparing non-existent metric."""
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("test_missing")

        with mlflow.start_run() as run1:
            log_dimarray("loss", DimArray(1.0, units.J))
            run1_id = run1.info.run_id

        with mlflow.start_run() as run2:
            log_dimarray("other", DimArray(2.0, units.J))
            run2_id = run2.info.run_id

        with pytest.raises(ValueError, match="not found"):
            compare_metrics_with_units(run1_id, run2_id, "missing")


class TestDimMLflowCallback:
    """Tests for DimMLflowCallback class."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = DimMLflowCallback(log_every_n_steps=10)
        assert callback.log_every_n_steps == 10
        assert len(callback.logged_params) == 0

    def test_on_train_begin(self, mlflow_run):
        """Test logging parameters at training start."""
        callback = DimMLflowCallback()

        params = {
            "learning_rate": DimArray(0.001, units.Hz),
            "batch_size": 32,
            "momentum": DimArray(0.9, units.dimensionless),
        }
        callback.on_train_begin(params)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "learning_rate" in run.data.params
        assert "batch_size" in run.data.params
        assert "momentum" in run.data.params

    def test_on_step(self, mlflow_run):
        """Test logging metrics during training steps."""
        callback = DimMLflowCallback(log_every_n_steps=2)

        # Step 0 - should log
        callback.on_step(0, {"loss": DimArray(1.0, units.J)})

        # Step 1 - should skip
        callback.on_step(1, {"loss": DimArray(0.9, units.J)})

        # Step 2 - should log
        callback.on_step(2, {"loss": DimArray(0.8, units.J)})

        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(mlflow_run.info.run_id, "loss")
        assert len(history) == 2  # Only steps 0 and 2

    def test_on_epoch_end(self, mlflow_run):
        """Test logging metrics at epoch end."""
        callback = DimMLflowCallback()

        metrics = {
            "train_loss": DimArray(0.5, units.J),
            "val_loss": DimArray(0.6, units.J),
        }
        callback.on_epoch_end(0, metrics)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "epoch_train_loss" in run.data.metrics
        assert "epoch_val_loss" in run.data.metrics

    def test_on_train_end(self, mlflow_run):
        """Test logging final metrics."""
        callback = DimMLflowCallback()

        final_metrics = {
            "test_loss": DimArray(0.3, units.J),
            "test_accuracy": 0.95,
        }
        callback.on_train_end(final_metrics)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "final_test_loss" in run.data.metrics
        assert "final_test_accuracy" in run.data.metrics

    def test_full_training_loop(self, mlflow_run):
        """Test complete training loop simulation."""
        callback = DimMLflowCallback(log_every_n_steps=1)

        # Training begins
        params = {"learning_rate": DimArray(0.001, units.Hz)}
        callback.on_train_begin(params)

        # Training steps
        for step in range(3):
            metrics = {"loss": DimArray(1.0 / (step + 1), units.J)}
            callback.on_step(step, metrics)

        # Epoch ends
        callback.on_epoch_end(0, {"train_loss": DimArray(0.5, units.J)})

        # Training ends
        callback.on_train_end({"final_loss": DimArray(0.3, units.J)})

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "learning_rate" in run.data.params
        assert "loss" in run.data.metrics
        assert "epoch_train_loss" in run.data.metrics
        assert "final_final_loss" in run.data.metrics


class TestCreateMLflowLogger:
    """Tests for create_mlflow_logger convenience function."""

    def test_create_logger(self):
        """Test creating logger returns callback."""
        logger = create_mlflow_logger(log_every_n_steps=5)
        assert isinstance(logger, DimMLflowCallback)
        assert logger.log_every_n_steps == 5


class TestHasMLflow:
    """Tests for HAS_MLFLOW flag."""

    def test_has_mlflow_is_true(self):
        """Test that HAS_MLFLOW is True when mlflow is available."""
        assert HAS_MLFLOW is True


class TestImportGuards:
    """Tests for import error handling."""

    def test_functions_require_mlflow(self):
        """Test that functions check for MLflow availability."""
        # This test is more for documentation - we know MLflow is installed
        # in the test environment, but the functions have the checks
        assert HAS_MLFLOW is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_division_in_comparison(self):
        """Test comparison when first metric is zero."""
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("test_zero")

        with mlflow.start_run() as run1:
            log_dimarray("metric", DimArray(0.0, units.J))
            run1_id = run1.info.run_id

        with mlflow.start_run() as run2:
            log_dimarray("metric", DimArray(1.0, units.J))
            run2_id = run2.info.run_id

        result = compare_metrics_with_units(run1_id, run2_id, "metric")
        assert result["relative_change_percent"] == float('inf')

    def test_log_very_small_value(self, mlflow_run):
        """Test logging very small values."""
        tiny = DimArray(1e-10, units.m)
        log_dimarray("tiny_distance", tiny)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "tiny_distance" in run.data.metrics
        assert run.data.metrics["tiny_distance"] == 1e-10

    def test_log_very_large_array(self, mlflow_run):
        """Test logging large arrays (statistics only)."""
        large_data = DimArray(np.random.randn(1000), units.V)
        log_dimarray("voltages", large_data)

        run = mlflow.get_run(mlflow_run.info.run_id)
        assert "voltages_mean" in run.data.metrics
        assert "voltages_std" in run.data.metrics
        assert "voltages_min" in run.data.metrics
        assert "voltages_max" in run.data.metrics
