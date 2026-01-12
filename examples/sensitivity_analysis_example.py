"""Example: Sensitivity Analysis with dimtensor

This example demonstrates how to perform sensitivity analysis on unit-aware
functions using dimtensor's analysis tools.
"""

from dimtensor import DimArray, units
from dimtensor.analysis import (
    local_sensitivity,
    sensitivity_matrix,
    rank_parameters,
    tornado_diagram_data,
)


def main():
    print("=" * 60)
    print("Sensitivity Analysis Example")
    print("=" * 60)
    print()

    # Example 1: Kinetic Energy
    print("Example 1: Kinetic Energy E = 0.5 * m * v^2")
    print("-" * 60)

    def kinetic_energy(mass, velocity):
        """Kinetic energy of an object."""
        return 0.5 * mass * velocity**2

    # Define parameters
    mass = DimArray(2.0, units.kg)
    velocity = DimArray(10.0, units.m / units.s)

    # Compute local sensitivities
    print(f"Mass: {mass}")
    print(f"Velocity: {velocity}")
    print()

    dE_dm = local_sensitivity(kinetic_energy, mass, args=(velocity,))
    dE_dv = local_sensitivity(kinetic_energy, velocity, args=(mass,))

    print(f"∂E/∂m = {dE_dm}")
    print(f"∂E/∂v = {dE_dv}")
    print()

    # Compute sensitivity matrix
    params = {"mass": mass, "velocity": velocity}
    sens_matrix = sensitivity_matrix(kinetic_energy, params)

    print("Sensitivity Matrix:")
    for param_name, sens in sens_matrix.items():
        print(f"  ∂E/∂{param_name} = {sens}")
    print()

    # Rank parameters by importance
    result = rank_parameters(kinetic_energy, params, normalization="relative")

    print("Parameter Importance Ranking (relative sensitivity):")
    for param_name, importance in result.ranking:
        print(f"  {param_name}: {importance:.4f}")
    print()

    print(f"Energy at evaluation point: {result.output}")
    print()

    # Example 2: Gravitational Force
    print("Example 2: Gravitational Force F = G * m1 * m2 / r^2")
    print("-" * 60)

    def gravitational_force(m1, m2, r):
        """Gravitational force between two masses."""
        G = DimArray(6.674e-11, units.m**3 / (units.kg * units.s**2))
        return G * m1 * m2 / r**2

    # Define parameters (Earth-Moon system, simplified)
    m1 = DimArray(5.972e24, units.kg)  # Earth mass
    m2 = DimArray(7.342e22, units.kg)  # Moon mass
    r = DimArray(3.844e8, units.m)     # Earth-Moon distance

    params_grav = {"m1": m1, "m2": m2, "r": r}

    result_grav = rank_parameters(
        gravitational_force, params_grav, normalization="relative"
    )

    print(f"m1 (Earth): {m1}")
    print(f"m2 (Moon): {m2}")
    print(f"r (distance): {r}")
    print()

    print("Parameter Importance Ranking:")
    for param_name, importance in result_grav.ranking:
        print(f"  {param_name}: {importance:.4f}")
    print()

    print(f"Force at evaluation point: {result_grav.output:.3e}")
    print()

    # Example 3: Tornado Diagram Data
    print("Example 3: Tornado Diagram (One-at-a-Time Sensitivity)")
    print("-" * 60)

    tornado_data = tornado_diagram_data(
        kinetic_energy,
        {"mass": mass, "velocity": velocity},
        relative_variation=0.1  # ±10% variation
    )

    print("Output variation for ±10% parameter changes:")
    for param_name, data in tornado_data.items():
        print(f"  {param_name}:")
        print(f"    Low:  {data['low']:.2f}")
        print(f"    High: {data['high']:.2f}")
        print(f"    Range: {data['range']:.2f}")
    print()

    # Example 4: Pressure from Ideal Gas Law
    print("Example 4: Ideal Gas Law P = nRT/V")
    print("-" * 60)

    def pressure_from_ideal_gas(n, T, V):
        """Pressure from ideal gas law."""
        R = DimArray(8.314, units.J / (units.mol * units.K))
        return n * R * T / V

    n = DimArray(10.0, units.mol)
    T = DimArray(300.0, units.K)
    V = DimArray(0.1, units.m**3)

    params_gas = {"n": n, "T": T, "V": V}

    result_gas = rank_parameters(
        pressure_from_ideal_gas, params_gas, normalization="relative"
    )

    print(f"Amount of substance (n): {n}")
    print(f"Temperature (T): {T}")
    print(f"Volume (V): {V}")
    print()

    print("Parameter Importance Ranking:")
    for param_name, importance in result_gas.ranking:
        print(f"  {param_name}: {importance:.4f}")
    print()

    print(f"Pressure at evaluation point: {result_gas.output:.2f}")
    print()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
