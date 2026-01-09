"""Tests for Polars integration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, units

pytest.importorskip("polars")
import polars as pl

from dimtensor.io.polars import to_polars, from_polars, save_polars, load_polars


class TestToPolars:
    """Tests for to_polars conversion."""

    def test_basic_conversion(self):
        """Test basic DimArray to Polars conversion."""
        data = {
            "distance": DimArray([1, 2, 3], units.m),
            "time": DimArray([0.5, 1.0, 1.5], units.s),
        }
        df = to_polars(data)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert len(df.columns) == 2

    def test_include_units_in_names(self):
        """Test unit info in column names."""
        data = {"x": DimArray([1, 2, 3], units.m)}
        df = to_polars(data, include_units=True)

        assert "[" in df.columns[0]

    def test_exclude_units_from_names(self):
        """Test excluding units from column names."""
        data = {"x": DimArray([1, 2, 3], units.m)}
        df = to_polars(data, include_units=False)

        assert df.columns[0] == "x"


class TestFromPolars:
    """Tests for from_polars conversion."""

    def test_with_units_map(self):
        """Test conversion with explicit units map."""
        df = pl.DataFrame({
            "distance": [1, 2, 3],
            "time": [0.5, 1.0, 1.5],
        })
        arrays = from_polars(df, units_map={
            "distance": units.m,
            "time": units.s,
        })

        assert "distance" in arrays
        assert "time" in arrays
        assert arrays["distance"].unit.dimension == Dimension(length=1)
        assert arrays["time"].unit.dimension == Dimension(time=1)

    def test_without_units_map(self):
        """Test conversion without units map."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        arrays = from_polars(df)

        assert "x" in arrays
        assert isinstance(arrays["x"], DimArray)


class TestRoundtrip:
    """Tests for roundtrip conversion."""

    def test_polars_roundtrip(self):
        """Test DimArray -> Polars -> DimArray roundtrip."""
        original = {
            "x": DimArray([1, 2, 3], units.m),
            "v": DimArray([10, 20, 30], units.m / units.s),
        }

        df = to_polars(original, include_units=False)
        restored = from_polars(df, units_map={
            "x": units.m,
            "v": units.m / units.s,
        })

        np.testing.assert_array_almost_equal(
            restored["x"].data, original["x"].data
        )
        np.testing.assert_array_almost_equal(
            restored["v"].data, original["v"].data
        )


class TestFileIO:
    """Tests for file I/O with Polars."""

    def test_save_load_parquet(self):
        """Test save and load as Parquet."""
        data = {"x": DimArray([1, 2, 3], units.m)}

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            save_polars(data, path, format="parquet")
            loaded = load_polars(path, units_map={"x [m]": units.m})

            assert len(loaded) == 1
        finally:
            Path(path).unlink()

    def test_save_load_csv(self):
        """Test save and load as CSV."""
        data = {"x": DimArray([1, 2, 3], units.m)}

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            save_polars(data, path, format="csv")
            loaded = load_polars(path, units_map={"x [m]": units.m})

            assert len(loaded) == 1
        finally:
            Path(path).unlink()

    def test_auto_detect_format(self):
        """Test auto-detection of file format."""
        data = {"x": DimArray([1, 2, 3], units.m)}

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            save_polars(data, path)
            loaded = load_polars(path)  # Auto-detect from extension

            assert len(loaded) == 1
        finally:
            Path(path).unlink()
