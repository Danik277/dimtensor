"""COMSOL Multiphysics results loader.

Loads FEM simulation results from COMSOL Multiphysics exports (TXT/CSV format).
Provides automatic unit inference based on physics module and field names.

Example:
    >>> from dimtensor.datasets.loaders.comsol import load_comsol_csv, PhysicsModule
    >>> # Load structural mechanics results
    >>> data = load_comsol_csv("displacement.csv", physics_module=PhysicsModule.STRUCTURAL)
    >>> coords = data["coordinates"]  # DimArray with units of m
    >>> disp_x = data["u"]  # DimArray with units of m
    >>>
    >>> # Load thermal results with explicit units
    >>> data = load_comsol_csv(
    ...     "thermal.csv",
    ...     units={"T": "K", "ht.ntflux": "W/m**2"},
    ...     coord_columns=["x", "y", "z"]
    ... )
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.units import (
    meter,
    kg,
    second,
    kelvin,
    pascal,
    volt,
    ampere,
    tesla,
    watt,
    mole,
    dimensionless,
)
from .base import CSVLoader


class PhysicsModule(Enum):
    """COMSOL physics module types for automatic unit inference.

    Each physics module has standard field names and associated units.
    """

    STRUCTURAL = "structural"  # Solid mechanics, structural analysis
    THERMAL = "thermal"  # Heat transfer
    ELECTROMAGNETIC = "electromagnetic"  # Electromagnetics, RF
    FLUID = "fluid"  # CFD, fluid dynamics
    ACOUSTIC = "acoustic"  # Acoustics
    CHEMICAL = "chemical"  # Chemical reaction engineering
    MULTIPHYSICS = "multiphysics"  # Mixed physics (manual units recommended)
    UNKNOWN = "unknown"  # Unknown module (manual units required)


class COMSOLLoader(CSVLoader):
    """Loader for COMSOL Multiphysics simulation results.

    Supports TXT and CSV exports from COMSOL with automatic unit inference
    based on physics module and field naming conventions.

    Attributes:
        cache_enabled: Whether to use caching (default: True).
        cache_dir: Directory for cached files.

    Example:
        >>> loader = COMSOLLoader()
        >>> data = loader.load("results.txt", physics_module=PhysicsModule.STRUCTURAL)
        >>> print(data.keys())  # dict_keys(['coordinates', 'u', 'v', 'w', ...])
    """

    # Physics module → field name pattern → unit
    # These are common COMSOL field naming conventions
    UNIT_MAPPING = {
        PhysicsModule.STRUCTURAL: {
            # Displacement components
            r"^u$": meter,
            r"^v$": meter,
            r"^w$": meter,
            r"^disp[xyz]?$": meter,
            r"^displacement": meter,
            # Stress (solid.sx, solid.sy, etc.)
            r"solid\.s[xyz]": pascal,
            r"solid\.sxy": pascal,
            r"solid\.sxz": pascal,
            r"solid\.syz": pascal,
            r"^stress": pascal,
            r"^sigma": pascal,
            # Strain (dimensionless)
            r"solid\.e[xyz]": dimensionless,
            r"solid\.exy": dimensionless,
            r"^strain": dimensionless,
            r"^epsilon": dimensionless,
            # von Mises stress
            r"solid\.mises": pascal,
            r"^vonmises": pascal,
        },
        PhysicsModule.THERMAL: {
            # Temperature
            r"^T$": kelvin,
            r"^temp": kelvin,
            r"^temperature": kelvin,
            # Heat flux (W/m²)
            r"ht\.ntflux": watt / meter**2,
            r"ht\.q[xyz]": watt / meter**2,
            r"^heatflux": watt / meter**2,
            r"^q[xyz]?$": watt / meter**2,
            # Heat source (W/m³)
            r"ht\.Q": watt / meter**3,
            r"^heatsource": watt / meter**3,
        },
        PhysicsModule.ELECTROMAGNETIC: {
            # Electric field (V/m)
            r"emw\.E[xyz]": volt / meter,
            r"^E[xyz]$": volt / meter,
            r"^efield": volt / meter,
            # Magnetic field (T = tesla)
            r"emw\.B[xyz]": tesla,
            r"^B[xyz]$": tesla,
            r"^bfield": tesla,
            # Electric potential (V)
            r"^V$": volt,
            r"^phi$": volt,
            r"^potential": volt,
            # Current density (A/m²)
            r"emw\.J[xyz]": ampere / meter**2,
            r"^J[xyz]$": ampere / meter**2,
            r"^current_density": ampere / meter**2,
        },
        PhysicsModule.FLUID: {
            # Velocity components (m/s)
            r"^u$": meter / second,
            r"^v$": meter / second,
            r"^w$": meter / second,
            r"^vel[xyz]?$": meter / second,
            r"^velocity": meter / second,
            r"spf\.U": meter / second,
            # Pressure (Pa)
            r"^p$": pascal,
            r"^pressure": pascal,
            r"spf\.p": pascal,
            # Density (kg/m³)
            r"spf\.rho": kg / meter**3,
            r"^rho$": kg / meter**3,
            r"^density": kg / meter**3,
            # Viscosity (Pa·s)
            r"^mu$": pascal * second,
            r"^viscosity": pascal * second,
        },
        PhysicsModule.ACOUSTIC: {
            # Pressure (Pa)
            r"^p$": pascal,
            r"^pressure": pascal,
            r"acpr\.p": pascal,
            # Sound pressure level (dimensionless, dB)
            r"^spl$": dimensionless,
            r"acpr\.Lp": dimensionless,
            # Particle velocity (m/s)
            r"acpr\.u[xyz]": meter / second,
        },
        PhysicsModule.CHEMICAL: {
            # Concentration (mol/m³)
            r"^c$": mole / meter**3,
            r"^conc": mole / meter**3,
            r"chds\.c": mole / meter**3,
            # Reaction rate (mol/(m³·s))
            r"^R$": mole / (meter**3 * second),
            r"^rate": mole / (meter**3 * second),
        },
    }

    def load(
        self,
        filepath: str | Path,
        physics_module: PhysicsModule | None = None,
        units: dict[str, str] | None = None,
        coord_columns: list[str] | None = None,
        field_columns: list[str] | None = None,
        delimiter: str | None = None,
        skip_rows: int = 0,
        **kwargs: Any,
    ) -> dict[str, DimArray]:
        """Load COMSOL results from TXT or CSV file.

        Args:
            filepath: Path to COMSOL export file (.txt or .csv).
            physics_module: Physics module for automatic unit inference.
                If None, uses UNKNOWN (manual units required).
            units: Explicit unit mapping for fields (overrides automatic inference).
                Format: {"field_name": "unit_str"} where unit_str like "m", "Pa", "V/m".
            coord_columns: List of coordinate column names (e.g., ["x", "y", "z"]).
                If None, auto-detects columns named x, y, z.
            field_columns: List of field column names to load.
                If None, loads all non-coordinate columns.
            delimiter: Column delimiter (default: auto-detect comma or whitespace).
            skip_rows: Number of header rows to skip (default: 0).
            **kwargs: Additional arguments (unused, for compatibility).

        Returns:
            Dictionary mapping field names to DimArrays:
            - "coordinates": Combined (x, y, z) array with shape (N, 3) in meters
            - "x", "y", "z": Individual coordinate arrays (if present)
            - Other fields: Field data arrays with appropriate units

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is invalid or units cannot be inferred.

        Example:
            >>> loader = COMSOLLoader()
            >>> data = loader.load("results.csv", physics_module=PhysicsModule.THERMAL)
            >>> print(data["T"])  # Temperature in kelvin
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = self._detect_delimiter(filepath)

        # Parse the file
        lines = filepath.read_text().strip().split("\n")

        # Skip header rows
        if skip_rows > 0:
            lines = lines[skip_rows:]

        # Find header line (may have units in brackets)
        header_line = lines[0]
        data_lines = lines[1:]

        # Parse header to extract column names and units
        column_info = self._parse_header(header_line, delimiter)

        # Detect coordinate columns
        if coord_columns is None:
            coord_columns = self._detect_coord_columns(column_info)

        # Detect field columns
        if field_columns is None:
            field_columns = [
                name for name in column_info.keys() if name not in coord_columns
            ]

        # Parse data rows
        data_dict = self._parse_data_rows(
            data_lines, column_info, delimiter
        )

        # Convert to DimArrays with appropriate units
        result = {}

        # Process coordinates
        coord_arrays = []
        for coord_name in ["x", "y", "z"]:
            if coord_name in coord_columns and coord_name in data_dict:
                coord_data = data_dict[coord_name]
                # Coordinates are always in meters
                result[coord_name] = DimArray(coord_data, unit=meter)
                coord_arrays.append(coord_data)

        # Combine coordinates into single array
        if coord_arrays:
            result["coordinates"] = DimArray(
                np.column_stack(coord_arrays), unit=meter
            )

        # Process field data
        for field_name in field_columns:
            if field_name not in data_dict:
                continue

            field_data = data_dict[field_name]

            # Determine unit for this field
            if units and field_name in units:
                # Explicit unit provided
                unit = self._parse_unit_string(units[field_name])
            elif field_name in column_info and column_info[field_name] is not None:
                # Unit specified in header
                unit = self._parse_unit_string(column_info[field_name])
            elif physics_module is not None:
                # Infer from physics module
                unit = self._infer_unit(field_name, physics_module)
            else:
                # No unit information available
                raise ValueError(
                    f"Cannot determine unit for field '{field_name}'. "
                    f"Provide physics_module or explicit units."
                )

            result[field_name] = DimArray(field_data, unit=unit)

        return result

    def _detect_delimiter(self, filepath: Path) -> str:
        """Detect delimiter (comma or whitespace) from file.

        Args:
            filepath: Path to file.

        Returns:
            Delimiter string ("," or whitespace pattern).
        """
        first_line = filepath.read_text().split("\n")[0]

        # Check for comma
        if "," in first_line:
            return ","

        # Check for tab
        if "\t" in first_line:
            return "\t"

        # Default to whitespace
        return r"\s+"

    def _parse_header(
        self, header_line: str, delimiter: str
    ) -> dict[str, str | None]:
        """Parse header line to extract column names and units.

        COMSOL headers typically look like:
        - "x [m], y [m], T [K]"  (CSV with units in brackets)
        - "x y z u v w"  (space-separated without units)
        - "x [m] y [m] z [m] u v w"  (space-separated with units)

        Args:
            header_line: Header line string.
            delimiter: Column delimiter.

        Returns:
            Dictionary mapping column name to unit string (or None if no unit).
        """
        # Remove comment markers
        header_line = header_line.strip()
        if header_line.startswith("%"):
            header_line = header_line[1:].strip()
        if header_line.startswith("#"):
            header_line = header_line[1:].strip()

        column_info = {}

        # Special handling for whitespace delimiter with units
        # Pattern: "name [unit]" where spaces separate columns
        if delimiter == r"\s+":
            # Use regex to find "name [unit]" patterns or standalone names
            # Match either "word [unit]" or just "word"
            pattern = r"(\w+(?:\.\w+)?)\s*(?:\[([^\]]+)\])?"
            matches = re.findall(pattern, header_line)

            for name, unit_str in matches:
                if name:
                    column_info[name] = unit_str if unit_str else None
        else:
            # CSV-style delimiter: split and parse each column
            columns = [col.strip() for col in header_line.split(delimiter)]

            for col in columns:
                if not col:
                    continue

                # Check for unit in brackets: "name [unit]"
                match = re.match(r"^(.+?)\s*\[([^\]]+)\]$", col)
                if match:
                    name = match.group(1).strip()
                    unit_str = match.group(2).strip()
                    column_info[name] = unit_str
                else:
                    # No unit in header
                    column_info[col] = None

        return column_info

    def _detect_coord_columns(self, column_info: dict[str, str | None]) -> list[str]:
        """Detect coordinate column names from header.

        Args:
            column_info: Column name to unit mapping.

        Returns:
            List of coordinate column names (subset of ["x", "y", "z"]).
        """
        coord_columns = []
        for name in ["x", "y", "z"]:
            if name in column_info:
                coord_columns.append(name)
        return coord_columns

    def _parse_data_rows(
        self,
        data_lines: list[str],
        column_info: dict[str, str | None],
        delimiter: str,
    ) -> dict[str, np.ndarray]:
        """Parse data rows into numpy arrays.

        Args:
            data_lines: List of data line strings.
            column_info: Column name to unit mapping (for column count).
            delimiter: Column delimiter.

        Returns:
            Dictionary mapping column names to numpy arrays.
        """
        column_names = list(column_info.keys())
        n_cols = len(column_names)

        # Parse all data rows
        data_rows = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue

            if delimiter == r"\s+":
                values = re.split(r"\s+", line)
            else:
                values = [v.strip() for v in line.split(delimiter)]

            # Convert to floats
            try:
                row = [float(v) for v in values if v]
                if len(row) == n_cols:
                    data_rows.append(row)
            except ValueError:
                # Skip malformed rows
                continue

        # Convert to numpy array
        data_array = np.array(data_rows)

        # Handle single row case (1D array)
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)

        # Split into columns
        result = {}
        for i, col_name in enumerate(column_names):
            result[col_name] = data_array[:, i]

        return result

    def _parse_unit_string(self, unit_str: str) -> Any:
        """Parse unit string to dimtensor Unit object.

        Supports common unit strings like:
        - "m", "kg", "s", "K", "Pa", "V", "A", "T"
        - "m/s", "kg/m^3", "W/m^2", "V/m"
        - "m**2", "m^2"

        Args:
            unit_str: Unit string.

        Returns:
            Unit object.

        Raises:
            ValueError: If unit string cannot be parsed.
        """
        # Import units locally to avoid circular dependency
        from ...core.units import (
            meter, kg, second, kelvin, pascal, volt, ampere, tesla,
            newton, joule, watt, coulomb, dimensionless, mole
        )

        # Basic unit mapping
        unit_map = {
            "m": meter,
            "kg": kg,
            "s": second,
            "K": kelvin,
            "Pa": pascal,
            "V": volt,
            "A": ampere,
            "T": tesla,
            "N": newton,
            "J": joule,
            "W": watt,
            "C": coulomb,
            "mol": mole,
            "1": dimensionless,
            "": dimensionless,
        }

        # Try exact match first
        if unit_str in unit_map:
            return unit_map[unit_str]

        # Handle compound units
        # Replace ^ with **
        unit_str = unit_str.replace("^", "**")

        # Try to evaluate as Python expression
        # This handles cases like "W/m**2", "kg/m**3", etc.
        try:
            # Build namespace with units
            namespace = {
                "m": meter,
                "kg": kg,
                "s": second,
                "K": kelvin,
                "Pa": pascal,
                "V": volt,
                "A": ampere,
                "T": tesla,
                "N": newton,
                "J": joule,
                "W": watt,
                "C": coulomb,
                "mol": mole,
            }
            unit = eval(unit_str, {"__builtins__": {}}, namespace)
            return unit
        except Exception:
            raise ValueError(f"Cannot parse unit string: {unit_str}")

    def _infer_unit(self, field_name: str, physics_module: PhysicsModule) -> Any:
        """Infer unit for field based on physics module and field name.

        Args:
            field_name: Field name (e.g., "u", "solid.sx", "T").
            physics_module: Physics module enum.

        Returns:
            Unit object.

        Raises:
            ValueError: If unit cannot be inferred.
        """
        if physics_module not in self.UNIT_MAPPING:
            raise ValueError(
                f"No unit mapping for physics module: {physics_module}. "
                f"Use explicit units parameter."
            )

        patterns = self.UNIT_MAPPING[physics_module]

        # Try each pattern
        for pattern, unit in patterns.items():
            if re.search(pattern, field_name, re.IGNORECASE):
                return unit

        # No match found
        raise ValueError(
            f"Cannot infer unit for field '{field_name}' "
            f"in physics module {physics_module}. "
            f"Provide explicit units."
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def load_comsol_txt(
    filepath: str | Path,
    physics_module: PhysicsModule | None = None,
    **kwargs: Any,
) -> dict[str, DimArray]:
    """Load COMSOL results from TXT file.

    Convenience function that wraps COMSOLLoader.load().

    Args:
        filepath: Path to COMSOL TXT export file.
        physics_module: Physics module for automatic unit inference.
        **kwargs: Additional arguments passed to COMSOLLoader.load().

    Returns:
        Dictionary mapping field names to DimArrays.

    Example:
        >>> data = load_comsol_txt("results.txt", physics_module=PhysicsModule.THERMAL)
        >>> print(data["T"])  # Temperature field
    """
    loader = COMSOLLoader()
    return loader.load(filepath, physics_module=physics_module, **kwargs)


def load_comsol_csv(
    filepath: str | Path,
    physics_module: PhysicsModule | None = None,
    **kwargs: Any,
) -> dict[str, DimArray]:
    """Load COMSOL results from CSV file.

    Convenience function that wraps COMSOLLoader.load().

    Args:
        filepath: Path to COMSOL CSV export file.
        physics_module: Physics module for automatic unit inference.
        **kwargs: Additional arguments passed to COMSOLLoader.load().

    Returns:
        Dictionary mapping field names to DimArrays.

    Example:
        >>> data = load_comsol_csv("structural.csv", physics_module=PhysicsModule.STRUCTURAL)
        >>> disp = data["u"]  # Displacement field in meters
    """
    loader = COMSOLLoader()
    return loader.load(filepath, physics_module=physics_module, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "COMSOLLoader",
    "PhysicsModule",
    "load_comsol_txt",
    "load_comsol_csv",
]
