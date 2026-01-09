"""Tests for the plugin system."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dimtensor.core.dimensions import Dimension
from dimtensor.core.units import Unit
from dimtensor.plugins import (
    PluginMetadata,
    PluginRegistry,
    discover_plugins,
    get_unit,
    list_plugins,
    load_plugin,
    plugin_info,
)
from dimtensor.plugins.loader import (
    discover_entry_points,
    load_plugin_from_entry_point,
)
from dimtensor.plugins.validation import validate_plugin, validate_plugin_name


# =============================================================================
# Mock Plugin Data
# =============================================================================

def create_mock_nuclear_plugin() -> PluginMetadata:
    """Create a mock nuclear physics plugin."""
    MeV = Unit("MeV", Dimension(mass=1, length=2, time=-2), 1.602176634e-13)
    barn = Unit("barn", Dimension(length=2), 1e-28)
    becquerel = Unit("Bq", Dimension(time=-1), 1.0)

    return PluginMetadata(
        name="nuclear",
        version="0.1.0",
        author="Test Author",
        description="Nuclear physics units for testing",
        units={
            "MeV": MeV,
            "barn": barn,
            "becquerel": becquerel,
        },
    )


def create_mock_geophysics_plugin() -> PluginMetadata:
    """Create a mock geophysics plugin."""
    milligal = Unit("mGal", Dimension(length=1, time=-2), 1e-5)
    nanotesla = Unit("nT", Dimension(mass=1, time=-2, current=-1), 1e-9)

    return PluginMetadata(
        name="geophysics",
        version="1.0.0",
        author="Geo Team",
        description="Geophysics units",
        units={
            "milligal": milligal,
            "nanotesla": nanotesla,
        },
    )


# =============================================================================
# Validation Tests
# =============================================================================

def test_validate_plugin_valid():
    """Test validation of a valid plugin."""
    plugin = create_mock_nuclear_plugin()
    valid, error = validate_plugin(plugin)
    assert valid
    assert error == ""


def test_validate_plugin_not_metadata():
    """Test validation fails for non-PluginMetadata objects."""
    valid, error = validate_plugin("not a plugin")
    assert not valid
    assert "PluginMetadata instance" in error


def test_validate_plugin_missing_name():
    """Test validation fails for missing name."""
    plugin = create_mock_nuclear_plugin()
    plugin.name = ""
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "name is required" in error


def test_validate_plugin_missing_version():
    """Test validation fails for missing version."""
    plugin = create_mock_nuclear_plugin()
    plugin.version = ""
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "version is required" in error


def test_validate_plugin_missing_author():
    """Test validation fails for missing author."""
    plugin = create_mock_nuclear_plugin()
    plugin.author = ""
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "author is required" in error


def test_validate_plugin_missing_description():
    """Test validation fails for missing description."""
    plugin = create_mock_nuclear_plugin()
    plugin.description = ""
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "description is required" in error


def test_validate_plugin_units_not_dict():
    """Test validation fails if units is not a dict."""
    plugin = create_mock_nuclear_plugin()
    plugin.units = "not a dict"
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "must be dict" in error


def test_validate_plugin_empty_units():
    """Test validation fails for empty units dict."""
    plugin = create_mock_nuclear_plugin()
    plugin.units = {}
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "at least one unit" in error


def test_validate_plugin_invalid_unit():
    """Test validation fails for invalid unit object."""
    plugin = create_mock_nuclear_plugin()
    plugin.units["bad_unit"] = "not a unit"
    valid, error = validate_plugin(plugin)
    assert not valid
    assert "must be Unit instance" in error


def test_validate_plugin_name_valid():
    """Test plugin name validation for valid names."""
    assert validate_plugin_name("nuclear")[0]
    assert validate_plugin_name("geo_physics")[0]
    assert validate_plugin_name("astro-chem")[0]
    assert validate_plugin_name("plugin123")[0]


def test_validate_plugin_name_invalid():
    """Test plugin name validation for invalid names."""
    assert not validate_plugin_name("")[0]
    assert not validate_plugin_name("has space")[0]
    assert not validate_plugin_name("has@special")[0]


# =============================================================================
# Loader Tests
# =============================================================================

def test_load_plugin_from_entry_point_success():
    """Test loading plugin from entry point string."""
    # Create a mock module with plugin
    mock_module = MagicMock()
    mock_module.plugin = create_mock_nuclear_plugin()

    with patch("importlib.import_module", return_value=mock_module):
        plugin = load_plugin_from_entry_point("nuclear", "mock_module:plugin")
        assert isinstance(plugin, PluginMetadata)
        assert plugin.name == "nuclear"


def test_load_plugin_from_entry_point_invalid_format():
    """Test loading fails with invalid entry point format."""
    with pytest.raises(ValueError, match="Invalid entry point format"):
        load_plugin_from_entry_point("test", "invalid_format")


def test_load_plugin_from_entry_point_import_error():
    """Test loading fails when module import fails."""
    with patch("importlib.import_module", side_effect=ImportError("Module not found")):
        with pytest.raises(ImportError, match="Failed to import module"):
            load_plugin_from_entry_point("test", "nonexistent:plugin")


def test_load_plugin_from_entry_point_attribute_error():
    """Test loading fails when attribute not found."""
    mock_module = MagicMock(spec=[])  # Empty spec means no attributes
    with patch("importlib.import_module", return_value=mock_module):
        with pytest.raises(AttributeError, match="has no attribute"):
            load_plugin_from_entry_point("test", "mock_module:nonexistent")


# =============================================================================
# Registry Tests
# =============================================================================

def test_registry_discover_plugins_empty():
    """Test discovering plugins when none are installed."""
    registry = PluginRegistry()
    with patch("dimtensor.plugins.registry.discover_entry_points", return_value={}):
        plugins = registry.discover_plugins()
        assert plugins == []


def test_registry_discover_plugins_multiple():
    """Test discovering multiple plugins."""
    registry = PluginRegistry()
    mock_eps = {
        "nuclear": "mock_nuclear:plugin",
        "geophysics": "mock_geo:plugin",
    }
    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        plugins = registry.discover_plugins()
        assert sorted(plugins) == ["geophysics", "nuclear"]


def test_registry_discover_plugins_cached():
    """Test that discovery results are cached."""
    registry = PluginRegistry()
    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps) as mock_discover:
        # First call
        plugins1 = registry.discover_plugins()
        # Second call
        plugins2 = registry.discover_plugins()

        # discover_entry_points should only be called once (cached)
        assert mock_discover.call_count == 1
        assert plugins1 == plugins2


def test_registry_load_plugin_success():
    """Test loading a plugin successfully."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            plugin = registry.load_plugin("nuclear")
            assert isinstance(plugin, PluginMetadata)
            assert plugin.name == "nuclear"
            assert len(plugin.units) == 3


def test_registry_load_plugin_cached():
    """Test that loaded plugins are cached."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()
    mock_module = MagicMock()
    mock_module.plugin = mock_plugin

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin) as mock_load:
            # First load
            plugin1 = registry.load_plugin("nuclear")
            # Second load
            plugin2 = registry.load_plugin("nuclear")

            # Import should only be called once (cached)
            assert mock_load.call_count == 1
            assert plugin1 is plugin2  # Same instance


def test_registry_load_plugin_not_found():
    """Test loading a non-existent plugin."""
    registry = PluginRegistry()
    with patch("dimtensor.plugins.registry.discover_entry_points", return_value={}):
        with pytest.raises(ValueError, match="not found"):
            registry.load_plugin("nonexistent")


def test_registry_load_plugin_invalid_name():
    """Test loading plugin with invalid name."""
    registry = PluginRegistry()
    with pytest.raises(ValueError, match="Invalid plugin name"):
        registry.load_plugin("has space")


def test_registry_load_plugin_validation_fails():
    """Test loading plugin that fails validation."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()
    mock_plugin.units = {}  # Invalid: empty units

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            with pytest.raises(ValueError, match="validation failed"):
                registry.load_plugin("nuclear")


def test_registry_load_plugin_name_mismatch_warning():
    """Test warning when plugin name doesn't match entry point."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()
    mock_plugin.name = "different_name"  # Doesn't match entry point "nuclear"

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            with pytest.warns(UserWarning, match="does not match metadata name"):
                plugin = registry.load_plugin("nuclear")
                # Should still load, using entry point name
                assert plugin is not None


def test_registry_get_unit_success():
    """Test getting a unit from a plugin."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            unit = registry.get_unit("nuclear", "MeV")
            assert isinstance(unit, Unit)
            assert unit.symbol == "MeV"


def test_registry_get_unit_not_found():
    """Test getting a non-existent unit."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            with pytest.raises(ValueError, match="not found in plugin"):
                registry.get_unit("nuclear", "nonexistent")


def test_registry_is_loaded():
    """Test checking if plugin is loaded."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            assert not registry.is_loaded("nuclear")
            registry.load_plugin("nuclear")
            assert registry.is_loaded("nuclear")


def test_registry_get_loaded_plugins():
    """Test getting list of loaded plugins."""
    registry = PluginRegistry()
    mock_plugin1 = create_mock_nuclear_plugin()
    mock_plugin2 = create_mock_geophysics_plugin()

    mock_eps = {
        "nuclear": "mock_nuclear:plugin",
        "geophysics": "mock_geo:plugin",
    }

    def mock_load(name, entry_point):
        if name == "nuclear":
            return mock_plugin1
        elif name == "geophysics":
            return mock_plugin2
        raise ValueError(f"Unknown plugin: {name}")

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", side_effect=mock_load):
            assert registry.get_loaded_plugins() == []
            registry.load_plugin("nuclear")
            assert registry.get_loaded_plugins() == ["nuclear"]
            registry.load_plugin("geophysics")
            assert sorted(registry.get_loaded_plugins()) == ["geophysics", "nuclear"]


def test_registry_clear_cache():
    """Test clearing the registry cache."""
    registry = PluginRegistry()
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            # Load plugin
            registry.load_plugin("nuclear")
            assert registry.is_loaded("nuclear")

            # Clear cache
            registry.clear_cache()
            assert not registry.is_loaded("nuclear")
            assert registry.get_loaded_plugins() == []


# =============================================================================
# Global API Tests
# =============================================================================

def test_global_discover_plugins():
    """Test global discover_plugins function."""
    mock_eps = {"nuclear": "mock_nuclear:plugin"}
    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        plugins = discover_plugins()
        assert plugins == ["nuclear"]


def test_global_list_plugins():
    """Test global list_plugins function."""
    mock_eps = {"nuclear": "mock_nuclear:plugin"}
    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        plugins = list_plugins()
        assert plugins == ["nuclear"]


def test_global_load_plugin():
    """Test global load_plugin function."""
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            plugin = load_plugin("nuclear")
            assert isinstance(plugin, PluginMetadata)
            assert plugin.name == "nuclear"


def test_global_get_unit():
    """Test global get_unit function."""
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            unit = get_unit("nuclear", "MeV")
            assert isinstance(unit, Unit)
            assert unit.symbol == "MeV"


def test_global_plugin_info():
    """Test global plugin_info function."""
    mock_plugin = create_mock_nuclear_plugin()

    mock_eps = {"nuclear": "mock_nuclear:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", return_value=mock_plugin):
            info = plugin_info("nuclear")
            assert isinstance(info, PluginMetadata)
            assert info.name == "nuclear"
            assert info.version == "0.1.0"


# =============================================================================
# PluginMetadata Tests
# =============================================================================

def test_plugin_metadata_repr():
    """Test PluginMetadata string representation."""
    plugin = create_mock_nuclear_plugin()
    repr_str = repr(plugin)
    assert "PluginMetadata" in repr_str
    assert "name='nuclear'" in repr_str
    assert "version='0.1.0'" in repr_str
    assert "3 units" in repr_str


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_registry_load_plugin_import_error_cached():
    """Test that failed plugin loads are cached to avoid repeated errors."""
    registry = PluginRegistry()
    mock_eps = {"broken": "broken_module:plugin"}

    with patch("dimtensor.plugins.registry.discover_entry_points", return_value=mock_eps):
        with patch("dimtensor.plugins.registry.load_plugin_from_entry_point", side_effect=ImportError("Module not found")):
            # First attempt
            with pytest.raises(ValueError, match="Failed to load"):
                registry.load_plugin("broken")

            # Second attempt should fail immediately with cached error
            with pytest.raises(ValueError, match="previously failed to load"):
                registry.load_plugin("broken")
