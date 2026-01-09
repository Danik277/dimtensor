"""Entry point for python -m dimtensor."""

from __future__ import annotations

import sys


def show_help() -> int:
    """Show help message."""
    print("Usage: python -m dimtensor <command> [options]")
    print()
    print("Commands:")
    print("  lint       Check files for dimensional issues")
    print("  convert    Convert between units")
    print("  equations  Browse physics equations")
    print("  datasets   List available datasets")
    print("  constants  List physical constants")
    print("  plugins    Manage unit plugins")
    print("  info       Show information about dimtensor")
    print()
    print("Run 'python -m dimtensor <command> --help' for more info.")
    return 0


def cmd_convert() -> int:
    """Convert between units."""
    import argparse

    from dimtensor import units

    parser = argparse.ArgumentParser(
        prog="dimtensor convert",
        description="Convert values between units",
    )
    parser.add_argument("value", type=float, help="Value to convert")
    parser.add_argument("from_unit", help="Source unit (e.g., km, kg, s)")
    parser.add_argument("to_unit", help="Target unit (e.g., m, g, ms)")

    args = parser.parse_args()

    # Get units
    try:
        from_u = getattr(units, args.from_unit)
        to_u = getattr(units, args.to_unit)
    except AttributeError as e:
        print(f"Unknown unit: {e}")
        return 1

    # Check compatibility
    if from_u.dimension != to_u.dimension:
        print(f"Incompatible dimensions: {from_u.dimension} vs {to_u.dimension}")
        return 1

    # Convert
    result = args.value * (from_u.scale / to_u.scale)
    print(f"{args.value} {args.from_unit} = {result} {args.to_unit}")
    return 0


def cmd_equations() -> int:
    """Browse physics equations."""
    import argparse

    from dimtensor.equations import get_equations, list_domains, search_equations

    parser = argparse.ArgumentParser(
        prog="dimtensor equations",
        description="Browse physics equations database",
    )
    parser.add_argument("--domain", "-d", help="Filter by domain")
    parser.add_argument("--search", "-s", help="Search equations")
    parser.add_argument("--list-domains", action="store_true", help="List domains")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    args = parser.parse_args()

    if args.list_domains:
        print("Available domains:")
        for domain in list_domains():
            count = len(get_equations(domain=domain))
            print(f"  {domain}: {count} equations")
        return 0

    if args.search:
        equations = search_equations(args.search)
    else:
        equations = get_equations(domain=args.domain)

    if not equations:
        print("No equations found.")
        return 0

    for eq in equations:
        if args.verbose:
            print(f"\n{eq.name}")
            print(f"  Formula: {eq.formula}")
            print(f"  Domain: {eq.domain}")
            if eq.description:
                print(f"  Description: {eq.description}")
            if eq.tags:
                print(f"  Tags: {', '.join(eq.tags)}")
        else:
            print(f"{eq.name}: {eq.formula}")

    return 0


def cmd_datasets() -> int:
    """List available datasets."""
    import argparse

    from dimtensor.datasets import list_datasets

    parser = argparse.ArgumentParser(
        prog="dimtensor datasets",
        description="List available physics datasets",
    )
    parser.add_argument("--domain", "-d", help="Filter by domain")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    args = parser.parse_args()

    datasets = list_datasets(domain=args.domain)

    if not datasets:
        print("No datasets found.")
        return 0

    for ds in datasets:
        if args.verbose:
            print(f"\n{ds.name}")
            print(f"  Domain: {ds.domain}")
            if ds.description:
                print(f"  Description: {ds.description}")
            if ds.features:
                print(f"  Features: {', '.join(ds.features.keys())}")
            if ds.targets:
                print(f"  Targets: {', '.join(ds.targets.keys())}")
            if ds.tags:
                print(f"  Tags: {', '.join(ds.tags)}")
        else:
            desc = ds.description[:50] + "..." if len(ds.description) > 50 else ds.description
            print(f"{ds.name}: {desc}")

    return 0


def cmd_constants() -> int:
    """List physical constants."""
    import argparse

    from dimtensor import constants

    parser = argparse.ArgumentParser(
        prog="dimtensor constants",
        description="List physical constants",
    )
    parser.add_argument("--search", "-s", help="Search constants by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    args = parser.parse_args()

    # Get all constants
    all_constants = [
        ("c", constants.c, "Speed of light"),
        ("G", constants.G, "Gravitational constant"),
        ("h", constants.h, "Planck constant"),
        ("hbar", constants.hbar, "Reduced Planck constant"),
        ("e", constants.e, "Elementary charge"),
        ("epsilon_0", constants.epsilon_0, "Vacuum permittivity"),
        ("mu_0", constants.mu_0, "Vacuum permeability"),
        ("k_B", constants.k_B, "Boltzmann constant"),
        ("N_A", constants.N_A, "Avogadro constant"),
        ("R", constants.R, "Gas constant"),
        ("m_e", constants.m_e, "Electron mass"),
        ("m_p", constants.m_p, "Proton mass"),
        ("alpha", constants.alpha, "Fine-structure constant"),
        ("a_0", constants.a_0, "Bohr radius"),
        ("sigma", constants.sigma, "Stefan-Boltzmann constant"),
    ]

    if args.search:
        search = args.search.lower()
        all_constants = [
            (name, val, desc) for name, val, desc in all_constants
            if search in name.lower() or search in desc.lower()
        ]

    if not all_constants:
        print("No constants found.")
        return 0

    for name, val, desc in all_constants:
        if args.verbose:
            print(f"\n{name}: {desc}")
            print(f"  Value: {val.value}")
            print(f"  Unit: {val.unit}")
            print(f"  Dimension: {val.dimension}")
        else:
            print(f"{name} = {val.value:.6e} ({desc})")

    return 0


def cmd_plugins() -> int:
    """Manage plugins."""
    import argparse

    from dimtensor import plugins

    parser = argparse.ArgumentParser(
        prog="dimtensor plugins",
        description="Manage dimtensor unit plugins",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Plugin subcommands")

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List available plugins")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    # Info subcommand
    info_parser = subparsers.add_parser("info", help="Show plugin information")
    info_parser.add_argument("name", help="Plugin name")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "list":
        try:
            plugin_names = plugins.list_plugins()
        except Exception as e:
            print(f"Error discovering plugins: {e}")
            return 1

        if not plugin_names:
            print("No plugins found.")
            print()
            print("To create a plugin, see the documentation:")
            print("  https://dimtensor.readthedocs.io/en/latest/guide/plugins.html")
            return 0

        print(f"Found {len(plugin_names)} plugin(s):")
        for name in plugin_names:
            if args.verbose:
                try:
                    plugin = plugins.load_plugin(name)
                    print(f"\n{name}:")
                    print(f"  Version: {plugin.version}")
                    print(f"  Author: {plugin.author}")
                    print(f"  Description: {plugin.description}")
                    print(f"  Units: {', '.join(plugin.units.keys())}")
                except Exception as e:
                    print(f"\n{name}: (error loading - {e})")
            else:
                print(f"  {name}")
        return 0

    elif args.subcommand == "info":
        try:
            plugin = plugins.load_plugin(args.name)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error loading plugin: {e}")
            return 1

        print(f"Plugin: {plugin.name}")
        print(f"Version: {plugin.version}")
        print(f"Author: {plugin.author}")
        print(f"Description: {plugin.description}")
        print(f"\nUnits ({len(plugin.units)}):")
        for unit_name, unit in plugin.units.items():
            print(f"  {unit_name}: {unit.symbol} ({unit.dimension})")
        return 0

    return 0


def cmd_info() -> int:
    """Show dimtensor info."""
    from dimtensor import __version__
    from dimtensor._rust import HAS_RUST_BACKEND

    print(f"dimtensor {__version__}")
    print(f"Rust backend: {'available' if HAS_RUST_BACKEND else 'not available'}")
    print()
    print("Modules:")
    print("  core: Dimension, Unit, DimArray")
    print("  torch: DimTensor for PyTorch")
    print("  jax: DimArray for JAX")
    print("  constants: Physical constants (CODATA 2022)")
    print("  equations: Physics equation database")
    print("  datasets: Physics dataset registry")
    print("  hub: Model registry")
    print("  plugins: Unit plugin system")
    return 0


def main() -> int:
    """Main entry point for dimtensor CLI."""
    if len(sys.argv) < 2:
        return show_help()

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'dimtensor' from argv

    if command == "lint":
        from dimtensor.cli.lint import main as lint_main
        return lint_main()

    elif command == "convert":
        return cmd_convert()

    elif command == "equations":
        return cmd_equations()

    elif command == "datasets":
        return cmd_datasets()

    elif command == "constants":
        return cmd_constants()

    elif command == "plugins":
        return cmd_plugins()

    elif command == "info":
        return cmd_info()

    elif command in ("--help", "-h"):
        return show_help()

    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m dimtensor --help' for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
