"""Polars DataFrame integration for DimArray.

Provides conversion between DimArray and Polars DataFrames with unit metadata.
"""
# mypy: disable-error-code="type-arg"

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit
from ..core.dimensions import Dimension

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _check_polars() -> None:
    """Check that polars is available."""
    if not HAS_POLARS:
        raise ImportError(
            "polars is required for Polars integration. "
            "Install with: pip install polars"
        )


def to_polars(
    arrays: dict[str, DimArray],
    include_units: bool = True,
) -> "pl.DataFrame":
    """Convert DimArrays to a Polars DataFrame.

    Args:
        arrays: Dict mapping column names to DimArrays.
        include_units: If True, add unit info to column names.

    Returns:
        Polars DataFrame with columns for each array.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.polars import to_polars
        >>>
        >>> data = {
        ...     "distance": DimArray([1, 2, 3], units.m),
        ...     "time": DimArray([0.5, 1.0, 1.5], units.s),
        ... }
        >>> df = to_polars(data)
    """
    _check_polars()

    columns = {}
    unit_info = {}

    for name, arr in arrays.items():
        if isinstance(arr, DimArray):
            data = np.asarray(arr.data).flatten()
            unit_info[name] = str(arr.unit)
            if include_units:
                col_name = f"{name} [{arr.unit}]"
            else:
                col_name = name
            columns[col_name] = data
        else:
            columns[name] = np.asarray(arr).flatten()

    df = pl.DataFrame(columns)

    # Store unit info as metadata (if supported)
    # Note: Polars doesn't have native metadata support like pandas,
    # but we can return the unit_info alongside the DataFrame
    return df


def from_polars(
    df: "pl.DataFrame",
    units_map: dict[str, Unit] | None = None,
) -> dict[str, DimArray]:
    """Convert Polars DataFrame to DimArrays.

    Args:
        df: Polars DataFrame.
        units_map: Dict mapping column names to Units.

    Returns:
        Dict mapping column names to DimArrays.

    Example:
        >>> from dimtensor import units
        >>> from dimtensor.io.polars import from_polars
        >>> import polars as pl
        >>>
        >>> df = pl.DataFrame({"distance": [1, 2, 3], "time": [0.5, 1.0, 1.5]})
        >>> arrays = from_polars(df, units_map={
        ...     "distance": units.m,
        ...     "time": units.s,
        ... })
    """
    _check_polars()

    result = {}
    units_map = units_map or {}

    for col in df.columns:
        data = df[col].to_numpy()

        if col in units_map:
            unit = units_map[col]
            result[col] = DimArray._from_data_and_unit(data, unit)
        else:
            # Try to parse unit from column name [unit]
            if "[" in col and col.endswith("]"):
                base_name = col[:col.index("[")].strip()
                # Just use dimensionless for unparseable units
                result[base_name] = DimArray._from_data_and_unit(
                    data, Unit("1", Dimension(), 1.0)
                )
            else:
                result[col] = DimArray._from_data_and_unit(
                    data, Unit("1", Dimension(), 1.0)
                )

    return result


def save_polars(
    arrays: dict[str, DimArray],
    path: str,
    format: str = "parquet",
) -> None:
    """Save DimArrays to a file via Polars.

    Args:
        arrays: Dict mapping column names to DimArrays.
        path: Output file path.
        format: Output format ('parquet', 'csv', 'json').

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.polars import save_polars
        >>>
        >>> data = {"x": DimArray([1, 2, 3], units.m)}
        >>> save_polars(data, "data.parquet")
    """
    _check_polars()

    df = to_polars(arrays, include_units=True)

    if format == "parquet":
        df.write_parquet(path)
    elif format == "csv":
        df.write_csv(path)
    elif format == "json":
        df.write_json(path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_polars(
    path: str,
    units_map: dict[str, Unit] | None = None,
    format: str | None = None,
) -> dict[str, DimArray]:
    """Load DimArrays from a file via Polars.

    Args:
        path: Input file path.
        units_map: Dict mapping column names to Units.
        format: Input format (auto-detected if None).

    Returns:
        Dict mapping column names to DimArrays.

    Example:
        >>> from dimtensor import units
        >>> from dimtensor.io.polars import load_polars
        >>>
        >>> arrays = load_polars("data.parquet", units_map={"x": units.m})
    """
    _check_polars()

    # Auto-detect format
    if format is None:
        if path.endswith(".parquet"):
            format = "parquet"
        elif path.endswith(".csv"):
            format = "csv"
        elif path.endswith(".json"):
            format = "json"
        else:
            raise ValueError(f"Cannot detect format from path: {path}")

    if format == "parquet":
        df = pl.read_parquet(path)
    elif format == "csv":
        df = pl.read_csv(path)
    elif format == "json":
        df = pl.read_json(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return from_polars(df, units_map=units_map)
