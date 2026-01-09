"""Example: Using dimtensor with Weights & Biases.

This example demonstrates how to log unit-aware tensors to W&B experiments.

Installation:
    pip install dimtensor[wandb]

Run this example:
    python examples/wandb_integration_example.py
"""

import numpy as np

from dimtensor import DimArray, units
from dimtensor.integrations.wandb import (
    log_dimarray,
    log_config_with_units,
    create_dimarray_table,
    DimWandbCallback,
)


def basic_logging_example():
    """Basic example of logging DimArrays to W&B."""
    import wandb

    # Initialize W&B (use offline mode for testing)
    wandb.init(project="dimtensor-demo", mode="offline")

    # Log individual DimArrays
    velocity = DimArray([10.0, 20.0, 30.0], units.m / units.s)
    log_dimarray("velocity", velocity)

    # Log scalar measurements
    temperature = DimArray([298.15], units.K)
    log_dimarray("temperature", temperature, step=0)

    wandb.finish()
    print("✓ Basic logging example completed")


def config_example():
    """Example of logging configuration with units."""
    import wandb

    wandb.init(project="dimtensor-demo", mode="offline")

    # Define hyperparameters with units
    config = {
        "learning_rate": DimArray([0.001], 1 / units.s),
        "max_velocity": DimArray([100.0], units.m / units.s),
        "max_force": DimArray([1000.0], units.N),
        "batch_size": 32,
        "epochs": 100,
    }

    # Log config with unit information
    log_config_with_units(config)

    wandb.finish()
    print("✓ Config logging example completed")


def table_example():
    """Example of creating W&B tables with DimArrays."""
    import wandb

    wandb.init(project="dimtensor-demo", mode="offline")

    # Create experimental results with units
    results = [
        {
            "time": DimArray([i], units.s),
            "distance": DimArray([i * 10.5], units.m),
            "velocity": DimArray([10.5], units.m / units.s),
            "acceleration": DimArray([0.0], units.m / units.s**2),
        }
        for i in range(10)
    ]

    # Create and log table
    table = create_dimarray_table(results)
    wandb.log({"kinematics_results": table})

    wandb.finish()
    print("✓ Table logging example completed")


def training_loop_example():
    """Example of using DimWandbCallback in a training loop."""
    import wandb

    wandb.init(project="dimtensor-demo", mode="offline")

    # Create callback
    callback = DimWandbCallback(prefix="train_")

    # Simulate physics-informed training
    for epoch in range(10):
        # Physics-informed loss (energy units)
        loss = DimArray([1.0 / (epoch + 1)], units.J)

        # Learning rate decay
        lr = DimArray([0.001 * 0.9**epoch], 1 / units.s)

        # Mean velocity in batch
        mean_velocity = DimArray([10.0 + np.random.randn()], units.m / units.s)

        # Log all metrics
        callback.log_epoch(
            {
                "loss": loss,
                "learning_rate": lr,
                "mean_velocity": mean_velocity,
            },
            epoch=epoch,
        )

    # Add final summary
    callback.finalize(
        {
            "best_loss": DimArray([0.05], units.J),
            "final_velocity": DimArray([12.3], units.m / units.s),
        }
    )

    wandb.finish()
    print("✓ Training loop example completed")


def batch_logging_example():
    """Example of logging per-batch metrics."""
    import wandb

    wandb.init(project="dimtensor-demo", mode="offline")

    callback = DimWandbCallback()

    # Simulate batch training
    for epoch in range(3):
        for batch in range(5):
            step = epoch * 5 + batch

            # Batch metrics with units
            batch_loss = DimArray([0.5 - step * 0.01], units.J)
            batch_force = DimArray([100.0 + np.random.randn() * 10], units.N)

            callback.log_batch(
                {
                    "batch_loss": batch_loss,
                    "applied_force": batch_force,
                },
                step=step,
            )

    wandb.finish()
    print("✓ Batch logging example completed")


def main():
    """Run all examples."""
    try:
        import wandb
    except ImportError:
        print("Error: wandb is not installed.")
        print("Install with: pip install wandb")
        return

    print("Running Weights & Biases integration examples...\n")

    basic_logging_example()
    config_example()
    table_example()
    training_loop_example()
    batch_logging_example()

    print("\n✓ All examples completed successfully!")
    print(f"\nNote: Runs were created in offline mode.")
    print("To sync to W&B cloud, run: wandb sync <run-directory>")


if __name__ == "__main__":
    main()
