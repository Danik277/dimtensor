"""Plugin registry for managing loaded plugins."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .loader import discover_entry_points, load_plugin_from_entry_point
from .validation import validate_plugin, validate_plugin_name

if TYPE_CHECKING:
    from .metadata import PluginMetadata
    from ..core.units import Unit


class PluginRegistry:
    """Registry for managing dimtensor plugins.

    This class provides plugin discovery, loading, and management.
    Plugins are loaded lazily (on first access) to avoid unnecessary imports.
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._discovered: dict[str, str] | None = None  # name -> entry_point
        self._loaded: dict[str, PluginMetadata] = {}  # name -> plugin
        self._failed: dict[str, str] = {}  # name -> error_message

    def discover_plugins(self) -> list[str]:
        """Discover all available plugins.

        Scans installed packages for dimtensor plugins using entry points.
        Results are cached for performance.

        Returns:
            List of plugin names.
        """
        if self._discovered is None:
            self._discovered = discover_entry_points()
        return sorted(self._discovered.keys())

    def list_plugins(self) -> list[str]:
        """List all available plugins.

        Returns:
            List of plugin names.
        """
        return self.discover_plugins()

    def load_plugin(self, name: str) -> PluginMetadata:
        """Load a specific plugin.

        Loads the plugin from its entry point and validates it.
        Results are cached, so subsequent calls return the same instance.

        Args:
            name: Name of the plugin to load.

        Returns:
            Plugin metadata object.

        Raises:
            ValueError: If plugin not found or invalid.
        """
        # Validate plugin name
        valid, error = validate_plugin_name(name)
        if not valid:
            raise ValueError(f"Invalid plugin name: {error}")

        # Check if already loaded
        if name in self._loaded:
            return self._loaded[name]

        # Check if loading previously failed
        if name in self._failed:
            raise ValueError(
                f"Plugin {name!r} previously failed to load: {self._failed[name]}"
            )

        # Discover plugins if not done yet
        if self._discovered is None:
            self.discover_plugins()

        # Check if plugin exists
        if name not in self._discovered:
            available = ", ".join(self._discovered.keys()) if self._discovered else "none"
            raise ValueError(
                f"Plugin {name!r} not found. Available plugins: {available}"
            )

        # Load the plugin
        entry_point = self._discovered[name]
        try:
            plugin = load_plugin_from_entry_point(name, entry_point)
        except (ImportError, AttributeError, ValueError) as e:
            error_msg = str(e)
            self._failed[name] = error_msg
            raise ValueError(f"Failed to load plugin {name!r}: {error_msg}") from e

        # Validate the plugin
        valid, error = validate_plugin(plugin)
        if not valid:
            error_msg = f"Invalid plugin structure: {error}"
            self._failed[name] = error_msg
            raise ValueError(f"Plugin {name!r} validation failed: {error_msg}")

        # Warn if plugin name doesn't match metadata
        if plugin.name != name:
            warnings.warn(
                f"Plugin entry point name {name!r} does not match metadata name "
                f"{plugin.name!r}. Using entry point name.",
                UserWarning,
                stacklevel=2,
            )

        # Cache and return
        self._loaded[name] = plugin
        return plugin

    def get_unit(self, plugin_name: str, unit_name: str) -> Unit:
        """Get a specific unit from a plugin.

        Args:
            plugin_name: Name of the plugin.
            unit_name: Name of the unit.

        Returns:
            Unit object.

        Raises:
            ValueError: If plugin or unit not found.
        """
        # Load the plugin (will raise if not found)
        plugin = self.load_plugin(plugin_name)

        # Get the unit
        if unit_name not in plugin.units:
            available = ", ".join(plugin.units.keys())
            raise ValueError(
                f"Unit {unit_name!r} not found in plugin {plugin_name!r}. "
                f"Available units: {available}"
            )

        return plugin.units[unit_name]

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded.

        Args:
            name: Plugin name.

        Returns:
            True if plugin is loaded.
        """
        return name in self._loaded

    def get_loaded_plugins(self) -> list[str]:
        """Get list of currently loaded plugins.

        Returns:
            List of loaded plugin names.
        """
        return sorted(self._loaded.keys())

    def clear_cache(self) -> None:
        """Clear the plugin cache.

        This forces re-discovery and re-loading on next access.
        """
        self._discovered = None
        self._loaded.clear()
        self._failed.clear()
