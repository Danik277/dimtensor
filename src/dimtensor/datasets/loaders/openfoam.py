"""OpenFOAM CFD results loader.

Loads OpenFOAM simulation results (velocity, pressure, temperature, turbulence)
and converts them to dimensionally-aware DimArrays.

OpenFOAM is a widely-used open-source CFD (Computational Fluid Dynamics) toolkit
that produces simulation results in a specific directory structure with field data
files containing dimensional information.

Example:
    >>> from dimtensor.datasets.loaders import OpenFOAMLoader
    >>> loader = OpenFOAMLoader()
    >>> # Load velocity field at time t=0.5
    >>> U = loader.load_field("./cavity", time="0.5", field="U")
    >>> # Load multiple fields
    >>> data = loader.load_fields("./cavity", time="0.5", fields=["U", "p"])
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import Dimension
from ...core.units import Unit, kelvin, meter, pascal, second
from .base import BaseLoader

# Try to import foamlib for binary format support
try:
    import foamlib  # type: ignore

    HAS_FOAMLIB = True
except ImportError:
    HAS_FOAMLIB = False


# Common CFD units
m_per_s = meter / second  # Velocity
m2_per_s2 = (meter**2) / (second**2)  # Kinematic pressure, turbulent kinetic energy
m2_per_s3 = (meter**2) / (second**3)  # Turbulent dissipation rate


def _openfoam_to_dimtensor(of_dims: list[int]) -> Dimension:
    """Convert OpenFOAM dimension array to dimtensor Dimension.

    OpenFOAM uses 7-element arrays representing powers of SI base units:
    [kg, m, s, K, mol, A, cd] = [mass, length, time, temperature, amount, current, luminosity]

    Args:
        of_dims: OpenFOAM dimension array (7 integers).

    Returns:
        Corresponding dimtensor Dimension.

    Example:
        >>> _openfoam_to_dimtensor([0, 1, -1, 0, 0, 0, 0])  # m/s
        Dimension(length=1, time=-1)
    """
    if len(of_dims) != 7:
        raise ValueError(f"OpenFOAM dimensions must have 7 elements, got {len(of_dims)}")

    # OpenFOAM order: [kg, m, s, K, mol, A, cd]
    # dimtensor order: length, mass, time, current, temperature, amount, luminosity
    return Dimension(
        mass=of_dims[0],
        length=of_dims[1],
        time=of_dims[2],
        temperature=of_dims[3],
        amount=of_dims[4],
        current=of_dims[5],
        luminosity=of_dims[6],
    )


def _parse_openfoam_dimensions(dim_str: str) -> list[int]:
    """Parse OpenFOAM dimension string.

    Args:
        dim_str: Dimension string like "[0 1 -1 0 0 0 0]".

    Returns:
        List of 7 integer exponents.

    Example:
        >>> _parse_openfoam_dimensions("[0 1 -1 0 0 0 0]")
        [0, 1, -1, 0, 0, 0, 0]
    """
    # Remove brackets and split by whitespace
    dim_str = dim_str.strip().strip("[]")
    parts = dim_str.split()

    if len(parts) != 7:
        raise ValueError(f"Expected 7 dimension values, got {len(parts)}")

    return [int(x) for x in parts]


def _parse_openfoam_field_ascii(field_path: Path) -> dict[str, Any]:
    """Parse OpenFOAM field file in ASCII format (pure Python fallback).

    Args:
        field_path: Path to OpenFOAM field file.

    Returns:
        Dictionary with:
            - dimensions: OpenFOAM dimension array [kg, m, s, K, mol, A, cd]
            - internal_field: NumPy array of field values
            - class_type: Field class (volScalarField, volVectorField, etc.)

    Raises:
        FileNotFoundError: If field file doesn't exist.
        ValueError: If file format is invalid or binary.
    """
    if not field_path.exists():
        raise FileNotFoundError(f"Field file not found: {field_path}")

    content = field_path.read_text()

    # Check if binary format
    if "format" in content and "binary" in content:
        raise ValueError(
            f"Binary format detected in {field_path.name}. "
            "Install foamlib for binary support: pip install foamlib"
        )

    # Parse dimensions
    dim_match = re.search(r"dimensions\s*\[([\d\s\-]+)\]", content)
    if not dim_match:
        raise ValueError(f"Could not find dimensions in {field_path.name}")

    dimensions = _parse_openfoam_dimensions(dim_match.group(1))

    # Parse class type
    class_match = re.search(r'class\s+(\w+);', content)
    class_type = class_match.group(1) if class_match else "unknown"

    # Parse internalField
    # Look for pattern: internalField   uniform <value>; or internalField   nonuniform List<Type> ...
    internal_match = re.search(
        r"internalField\s+(\w+)\s+(.+?)(?:;|\n\n)",
        content,
        re.DOTALL
    )

    if not internal_match:
        raise ValueError(f"Could not find internalField in {field_path.name}")

    field_type = internal_match.group(1)  # uniform or nonuniform
    field_data_str = internal_match.group(2)

    # Parse field data
    if field_type == "uniform":
        # Uniform field: single value for all cells
        # Can be scalar: "uniform 0;"
        # Or vector: "uniform (0 0 0);"
        uniform_match = re.search(r"(\([^\)]+\)|[\d\.\-eE]+)", field_data_str)
        if not uniform_match:
            raise ValueError(f"Could not parse uniform field value in {field_path.name}")

        value_str = uniform_match.group(1)

        if value_str.startswith("("):
            # Vector value
            vector_str = value_str.strip("()")
            values = [float(x) for x in vector_str.split()]
            internal_field = np.array([values])  # Shape (1, 3) for single vector
        else:
            # Scalar value
            internal_field = np.array([float(value_str)])  # Shape (1,)

    elif field_type == "nonuniform":
        # Nonuniform field: list of values
        # Format: nonuniform List<scalar> n ( v1 v2 v3 ... )
        # or: nonuniform List<vector> n ( (x1 y1 z1) (x2 y2 z2) ... )

        # Find the list size
        size_match = re.search(r"List<\w+>\s*\n?\s*(\d+)", field_data_str)
        if not size_match:
            raise ValueError(f"Could not find list size in {field_path.name}")

        list_size = int(size_match.group(1))

        # Find the data block
        data_block_match = re.search(r"\(\s*\n(.*?)\n\s*\)", field_data_str, re.DOTALL)
        if not data_block_match:
            raise ValueError(f"Could not find data block in {field_path.name}")

        data_block = data_block_match.group(1)

        # Check if vector field (contains parentheses)
        if "(" in data_block:
            # Vector field: extract all (x y z) tuples
            vector_matches = re.findall(r"\(([^\)]+)\)", data_block)
            if len(vector_matches) != list_size:
                raise ValueError(
                    f"Expected {list_size} vectors, found {len(vector_matches)} in {field_path.name}"
                )

            vectors = []
            for vec_str in vector_matches:
                values = [float(x) for x in vec_str.split()]
                vectors.append(values)

            internal_field = np.array(vectors)  # Shape (n, 3)
        else:
            # Scalar field: split by whitespace
            values = []
            for line in data_block.split("\n"):
                line = line.strip()
                if line:
                    values.extend([float(x) for x in line.split()])

            if len(values) != list_size:
                raise ValueError(
                    f"Expected {list_size} values, found {len(values)} in {field_path.name}"
                )

            internal_field = np.array(values)  # Shape (n,)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return {
        "dimensions": dimensions,
        "internal_field": internal_field,
        "class_type": class_type,
    }


def _parse_openfoam_field_foamlib(field_path: Path) -> dict[str, Any]:
    """Parse OpenFOAM field file using foamlib library.

    Args:
        field_path: Path to OpenFOAM field file.

    Returns:
        Dictionary with dimensions, internal_field, class_type.

    Raises:
        ImportError: If foamlib not installed.
    """
    if not HAS_FOAMLIB:
        raise ImportError(
            "foamlib required for this operation. Install with: pip install foamlib"
        )

    # Use foamlib to parse the field
    field = foamlib.FoamFile(field_path)

    # Extract dimensions
    dimensions = field["dimensions"]

    # Extract internal field
    internal_field = np.array(field["internalField"])

    # Extract class type
    class_type = field.get("class", "unknown")

    return {
        "dimensions": dimensions,
        "internal_field": internal_field,
        "class_type": class_type,
    }


class OpenFOAMLoader(BaseLoader):
    """Loader for OpenFOAM CFD simulation results.

    Loads OpenFOAM case directories and extracts field data (velocity,
    pressure, temperature, turbulence) as dimensionally-aware DimArrays.

    Supports ASCII format natively with pure Python parser. For binary
    format support, install foamlib: pip install foamlib

    Example:
        >>> loader = OpenFOAMLoader()
        >>> # Load single field
        >>> U = loader.load_field("./cavity", time="0.5", field="U")
        >>> # Load multiple fields
        >>> data = loader.load_fields("./cavity", "0.5", ["U", "p"])
        >>> # List available times and fields
        >>> times = loader.list_times("./cavity")
        >>> fields = loader.list_fields("./cavity", "0.5")
    """

    def __init__(self, cache: bool = True, use_foamlib: bool = True):
        """Initialize OpenFOAM loader.

        Args:
            cache: Whether to enable caching (default: True).
            use_foamlib: Use foamlib if available for better performance (default: True).
        """
        super().__init__(cache=cache)
        self.use_foamlib = use_foamlib and HAS_FOAMLIB
        self._field_cache: dict[str, dict[str, Any]] = {}

    def load(self, **kwargs: Any) -> Any:
        """Load OpenFOAM dataset.

        This is the main BaseLoader.load() interface. Forwards to load_fields().

        Args:
            case_path: Path to OpenFOAM case directory.
            time: Time directory to load (e.g., "0", "0.5").
            fields: List of field names to load (e.g., ["U", "p"]).

        Returns:
            Dictionary mapping field names to DimArrays.

        Example:
            >>> loader = OpenFOAMLoader()
            >>> data = loader.load(case_path="./cavity", time="0.5", fields=["U", "p"])
        """
        case_path = kwargs.get("case_path")
        time = kwargs.get("time")
        fields = kwargs.get("fields")

        if not case_path or not time or not fields:
            raise ValueError(
                "OpenFOAMLoader.load() requires case_path, time, and fields arguments"
            )

        return self.load_fields(case_path, time, fields)

    def load_field(
        self,
        case_path: str | Path,
        time: str,
        field: str,
    ) -> DimArray:
        """Load a single OpenFOAM field as a DimArray.

        Args:
            case_path: Path to OpenFOAM case directory.
            time: Time directory (e.g., "0", "0.5", "100").
            field: Field name (e.g., "U", "p", "T", "k", "epsilon").

        Returns:
            DimArray with field data and proper units.

        Raises:
            FileNotFoundError: If case or field file not found.
            ValueError: If field format is invalid.

        Example:
            >>> loader = OpenFOAMLoader()
            >>> U = loader.load_field("./cavity", "0.5", "U")
            >>> print(U.shape, U.unit)
            (400, 3) m/s
        """
        case_path = Path(case_path)
        field_path = case_path / time / field

        if not case_path.exists():
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        if not field_path.exists():
            raise FileNotFoundError(
                f"Field {field} not found at time {time} in {case_path}"
            )

        # Check cache
        cache_key = f"{case_path}:{time}:{field}"
        if self.cache_enabled and cache_key in self._field_cache:
            cached = self._field_cache[cache_key]
            return DimArray(cached["data"], unit=cached["unit"])

        # Parse field file
        if self.use_foamlib:
            try:
                parsed = _parse_openfoam_field_foamlib(field_path)
            except Exception as e:
                # Fall back to ASCII parser
                if "binary" not in str(e).lower():
                    # If not a binary format issue, re-raise
                    raise
                parsed = _parse_openfoam_field_ascii(field_path)
        else:
            parsed = _parse_openfoam_field_ascii(field_path)

        # Convert dimensions to dimtensor Dimension and Unit
        dimension = _openfoam_to_dimtensor(parsed["dimensions"])
        unit = Unit(symbol=field, dimension=dimension, scale=1.0)

        # Apply common unit conventions
        unit = self._apply_unit_conventions(field, unit, dimension)

        # Create DimArray
        data = parsed["internal_field"]
        dimarray = DimArray(data, unit=unit)

        # Cache result
        if self.cache_enabled:
            self._field_cache[cache_key] = {"data": data, "unit": unit}

        return dimarray

    def load_fields(
        self,
        case_path: str | Path,
        time: str,
        fields: list[str],
    ) -> dict[str, DimArray]:
        """Load multiple OpenFOAM fields.

        Args:
            case_path: Path to OpenFOAM case directory.
            time: Time directory (e.g., "0", "0.5").
            fields: List of field names to load.

        Returns:
            Dictionary mapping field names to DimArrays.

        Example:
            >>> loader = OpenFOAMLoader()
            >>> data = loader.load_fields("./cavity", "0.5", ["U", "p", "T"])
            >>> U, p, T = data["U"], data["p"], data["T"]
        """
        result = {}
        for field in fields:
            result[field] = self.load_field(case_path, time, field)
        return result

    def list_times(self, case_path: str | Path) -> list[str]:
        """List available time directories in an OpenFOAM case.

        Args:
            case_path: Path to OpenFOAM case directory.

        Returns:
            Sorted list of time directory names.

        Example:
            >>> loader = OpenFOAMLoader()
            >>> times = loader.list_times("./cavity")
            >>> print(times)
            ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
        """
        case_path = Path(case_path)

        if not case_path.exists():
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        time_dirs = []
        for item in case_path.iterdir():
            if item.is_dir():
                # Check if directory name is numeric (time directory)
                try:
                    float(item.name)
                    time_dirs.append(item.name)
                except ValueError:
                    # Not a time directory (e.g., "constant", "system")
                    continue

        # Sort numerically
        time_dirs.sort(key=float)
        return time_dirs

    def list_fields(self, case_path: str | Path, time: str) -> list[str]:
        """List available fields in a time directory.

        Args:
            case_path: Path to OpenFOAM case directory.
            time: Time directory to inspect.

        Returns:
            List of field names available at that time.

        Example:
            >>> loader = OpenFOAMLoader()
            >>> fields = loader.list_fields("./cavity", "0.5")
            >>> print(fields)
            ['U', 'p', 'phi']
        """
        case_path = Path(case_path)
        time_dir = case_path / time

        if not time_dir.exists():
            raise FileNotFoundError(f"Time directory not found: {time_dir}")

        fields = []
        for item in time_dir.iterdir():
            if item.is_file():
                # Skip hidden files and common non-field files
                if not item.name.startswith(".") and item.name not in ["uniform"]:
                    fields.append(item.name)

        return sorted(fields)

    def _apply_unit_conventions(
        self,
        field_name: str,
        unit: Unit,
        dimension: Dimension,
    ) -> Unit:
        """Apply common CFD field unit conventions.

        Maps field names to standard units based on typical OpenFOAM conventions.

        Args:
            field_name: Name of the field.
            unit: Original unit from dimension parsing.
            dimension: Field dimension.

        Returns:
            Unit with appropriate symbol and scale.
        """
        # Velocity: U, Ua, Ub, etc.
        if field_name.startswith("U") and dimension == Dimension(length=1, time=-1):
            return m_per_s

        # Pressure (kinematic): p, p_rgh
        if field_name in ["p", "p_rgh"] and dimension == Dimension(length=2, time=-2):
            return Unit(symbol="m²/s²", dimension=dimension, scale=1.0)

        # Pressure (static): Pa
        if dimension == Dimension(mass=1, length=-1, time=-2):
            return pascal

        # Temperature: T
        if field_name == "T" and dimension == Dimension(temperature=1):
            return kelvin

        # Turbulent kinetic energy: k
        if field_name == "k" and dimension == Dimension(length=2, time=-2):
            return m2_per_s2

        # Turbulent dissipation rate: epsilon
        if field_name in ["epsilon", "omega"] and dimension == Dimension(length=2, time=-3):
            return m2_per_s3

        # Default: use parsed unit
        return unit
