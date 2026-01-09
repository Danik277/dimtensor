"""Plugin system for custom unit collections.

This module provides a plugin system that allows users to create, distribute, and
share custom unit definitions as PyPI packages, enabling a community ecosystem of
domain-specific unit collections.

Example:
    >>> from dimtensor.plugins import list_plugins, load_plugin
    >>> # List all available plugins
    >>> list_plugins()
    ['nuclear', 'geophysics']
    >>> # Load a specific plugin
    >>> plugin = load_plugin('nuclear')
    >>> # Use units from the plugin
    >>> from dimtensor_units_nuclear import MeV
    >>> energy = DimArray([1.0], MeV)

Plugin Structure:
    Plugins are discovered via setuptools entry points. In your plugin's pyproject.toml:

    [project.entry-points."dimtensor.plugins"]
    nuclear = "dimtensor_units_nuclear:plugin"

    Your plugin module must export a PluginMetadata object named 'plugin'.
"""

from __future__ import annotations

from .metadata import PluginMetadata
from .registry import PluginRegistry

# Global registry singleton
_global_registry = PluginRegistry()


def discover_plugins() -> list[str]:
    """Discover all available plugins.

    Scans installed packages for dimtensor plugins using entry points.

    Returns:
        List of plugin names.
    """
    return _global_registry.discover_plugins()


def list_plugins() -> list[str]:
    """List all available plugins.

    Returns:
        List of plugin names.
    """
    return _global_registry.list_plugins()


def load_plugin(name: str) -> PluginMetadata:
    """Load a specific plugin.

    Args:
        name: Name of the plugin to load.

    Returns:
        Plugin metadata object.

    Raises:
        ValueError: If plugin not found or invalid.
    """
    return _global_registry.load_plugin(name)


def get_unit(plugin_name: str, unit_name: str):
    """Get a specific unit from a plugin.

    Args:
        plugin_name: Name of the plugin.
        unit_name: Name of the unit.

    Returns:
        Unit object.

    Raises:
        ValueError: If plugin or unit not found.
    """
    return _global_registry.get_unit(plugin_name, unit_name)


def plugin_info(name: str) -> PluginMetadata:
    """Get information about a plugin.

    Args:
        name: Name of the plugin.

    Returns:
        Plugin metadata object.

    Raises:
        ValueError: If plugin not found.
    """
    plugin = _global_registry.load_plugin(name)
    return plugin


__all__ = [
    "PluginMetadata",
    "PluginRegistry",
    "discover_plugins",
    "list_plugins",
    "load_plugin",
    "get_unit",
    "plugin_info",
]
