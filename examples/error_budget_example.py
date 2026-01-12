"""Example: Error Budget Calculator for Uncertainty Analysis

This example demonstrates how to use the error budget calculator to identify
which input uncertainties contribute most to the total output uncertainty.
"""

import numpy as np
from dimtensor import DimArray, units
from dimtensor.uncertainty import compute_error_budget


def example_pendulum_period():
    """Example: Pendulum period uncertainty budget.

    For a simple pendulum, T = 2π√(L/g), where:
    - L is the length of the pendulum
    - g is gravitational acceleration

    This example shows how to determine whether uncertainty in length
    measurement or gravitational acceleration dominates the period uncertainty.
    """
    print("=" * 70)
    print("Example 1: Pendulum Period Uncertainty Budget")
    print("=" * 70)

    # Define the computation
    def pendulum_period(inputs):
        L = inputs['length']
        g = inputs['gravity']
        return 2 * np.pi * (L / g) ** 0.5

    # Input values with uncertainties
    inputs = {
        'length': DimArray(1.0, units.m, uncertainty=0.01),  # ±1 cm
        'gravity': DimArray(9.8, units.m/units.s**2, uncertainty=0.1),  # ±0.1 m/s²
    }

    # Compute error budget
    budget = compute_error_budget(pendulum_period, inputs)

    # Display results
    print(f"\nNominal period: {budget.result}")
    print(f"Total uncertainty: {budget.total_uncertainty:.6f} s")
    print("\nUncertainty contributions:")
    for name, contrib in budget.contributions.items():
        percent = budget.percent_contributions[name]
        print(f"  {name:10s}: {contrib:.6f} s  ({percent:.1f}%)")

    print("\nSensitivity coefficients:")
    for name, sens in budget.sensitivities.items():
        print(f"  ∂T/∂{name}: {sens}")

    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        budget.plot_pie(ax=ax1)
        budget.plot_bar(ax=ax2)
        plt.tight_layout()
        plt.savefig('pendulum_error_budget.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to: pendulum_error_budget.png")
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")

    # Export to DataFrame
    try:
        df = budget.to_dataframe()
        print("\nError Budget Table:")
        print(df.to_string(index=False))
    except ImportError:
        print("\nPandas not available - skipping table export")


def example_ohms_law_power():
    """Example: Ohm's law power dissipation uncertainty.

    For P = V²/R, where:
    - V is voltage
    - R is resistance

    Since voltage appears squared, its uncertainty contribution dominates.
    """
    print("\n" + "=" * 70)
    print("Example 2: Ohm's Law Power Dissipation")
    print("=" * 70)

    # Define the computation
    def power_dissipation(inputs):
        V = inputs['voltage']
        R = inputs['resistance']
        return V**2 / R

    # Input values with uncertainties
    inputs = {
        'voltage': DimArray(12.0, units.V, uncertainty=0.5),  # ±0.5 V (4.2%)
        'resistance': DimArray(10.0, units.ohm, uncertainty=0.2),  # ±0.2 Ω (2%)
    }

    # Compute error budget
    budget = compute_error_budget(power_dissipation, inputs)

    # Display results
    print(f"\nNominal power: {budget.result}")
    print(f"Total uncertainty: {budget.total_uncertainty:.4f} W")
    print("\nUncertainty contributions:")
    for name, contrib in budget.contributions.items():
        percent = budget.percent_contributions[name]
        print(f"  {name:12s}: {contrib:.4f} W  ({percent:.1f}%)")

    print("\nObservation: Voltage dominates because it appears squared in P = V²/R")
    print("Even though relative uncertainties are similar (4.2% vs 2%),")
    print("voltage contributes ~83% of total uncertainty.")


def example_kinetic_energy():
    """Example: Kinetic energy uncertainty budget.

    For KE = 0.5*m*v², where:
    - m is mass
    - v is velocity

    Velocity uncertainty typically dominates due to squared term.
    """
    print("\n" + "=" * 70)
    print("Example 3: Kinetic Energy")
    print("=" * 70)

    # Define the computation
    def kinetic_energy(inputs):
        m = inputs['mass']
        v = inputs['velocity']
        return 0.5 * m * v**2

    # Input values with uncertainties
    inputs = {
        'mass': DimArray(2.0, units.kg, uncertainty=0.01),  # ±0.01 kg (0.5%)
        'velocity': DimArray(10.0, units.m/units.s, uncertainty=0.5),  # ±0.5 m/s (5%)
    }

    # Compute error budget
    budget = compute_error_budget(kinetic_energy, inputs)

    # Display results
    print(f"\nNominal kinetic energy: {budget.result}")
    print(f"Total uncertainty: {budget.total_uncertainty:.4f} J")
    print("\nUncertainty contributions:")
    for name, contrib in budget.contributions.items():
        percent = budget.percent_contributions[name]
        print(f"  {name:10s}: {contrib:.4f} J  ({percent:.1f}%)")

    print("\nInsight: To reduce total uncertainty, focus on improving velocity")
    print("measurement accuracy rather than mass measurement.")


def example_comparison():
    """Example: Comparing different measurement scenarios.

    Shows how improving specific measurements affects total uncertainty.
    """
    print("\n" + "=" * 70)
    print("Example 4: Scenario Comparison")
    print("=" * 70)

    def compute_velocity(inputs):
        """v = d/t"""
        return inputs['distance'] / inputs['time']

    print("\nScenario A: Both measurements have similar relative uncertainty")
    inputs_a = {
        'distance': DimArray(100.0, units.m, uncertainty=1.0),  # ±1%
        'time': DimArray(10.0, units.s, uncertainty=0.1),  # ±1%
    }
    budget_a = compute_error_budget(compute_velocity, inputs_a)
    print(f"Total uncertainty: {budget_a.total_uncertainty:.4f} m/s")
    print(f"  Distance: {budget_a.percent_contributions['distance']:.1f}%")
    print(f"  Time:     {budget_a.percent_contributions['time']:.1f}%")

    print("\nScenario B: Improved distance measurement (10x better)")
    inputs_b = {
        'distance': DimArray(100.0, units.m, uncertainty=0.1),  # ±0.1%
        'time': DimArray(10.0, units.s, uncertainty=0.1),  # ±1%
    }
    budget_b = compute_error_budget(compute_velocity, inputs_b)
    print(f"Total uncertainty: {budget_b.total_uncertainty:.4f} m/s")
    print(f"  Distance: {budget_b.percent_contributions['distance']:.1f}%")
    print(f"  Time:     {budget_b.percent_contributions['time']:.1f}%")

    improvement = (budget_a.total_uncertainty - budget_b.total_uncertainty) / budget_a.total_uncertainty * 100
    print(f"\nImproving distance measurement by 10x reduces total uncertainty by {improvement:.1f}%")
    print("This demonstrates the value of error budget analysis for experimental design.")


if __name__ == '__main__':
    # Run all examples
    example_pendulum_period()
    example_ohms_law_power()
    example_kinetic_energy()
    example_comparison()

    print("\n" + "=" * 70)
    print("Error Budget Analysis Complete!")
    print("=" * 70)
