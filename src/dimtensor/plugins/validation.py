"""Plugin validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metadata import PluginMetadata


def validate_plugin(plugin: Any) -> tuple[bool, str]:
    """Validate a plugin object.

    Args:
        plugin: Object to validate (should be PluginMetadata).

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.
    """
    from .metadata import PluginMetadata
    from ..core.units import Unit

    # Check if it's a PluginMetadata instance
    if not isinstance(plugin, PluginMetadata):
        return False, f"Plugin must be PluginMetadata instance, got {type(plugin).__name__}"

    # Check required attributes
    if not plugin.name:
        return False, "Plugin name is required"

    if not plugin.version:
        return False, "Plugin version is required"

    if not plugin.author:
        return False, "Plugin author is required"

    if not plugin.description:
        return False, "Plugin description is required"

    if not isinstance(plugin.units, dict):
        return False, f"Plugin units must be dict, got {type(plugin.units).__name__}"

    if not plugin.units:
        return False, "Plugin must define at least one unit"

    # Validate each unit
    for unit_name, unit in plugin.units.items():
        if not isinstance(unit_name, str):
            return False, f"Unit name must be string, got {type(unit_name).__name__}"

        if not isinstance(unit, Unit):
            return False, f"Unit {unit_name!r} must be Unit instance, got {type(unit).__name__}"

    return True, ""


def validate_plugin_name(name: str) -> tuple[bool, str]:
    """Validate a plugin name.

    Args:
        name: Plugin name to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not name:
        return False, "Plugin name cannot be empty"

    if not isinstance(name, str):
        return False, f"Plugin name must be string, got {type(name).__name__}"

    # Check for valid Python identifier (basic check)
    if not name.replace("_", "").replace("-", "").isalnum():
        return False, f"Plugin name must be alphanumeric (with _ or -): {name!r}"

    return True, ""
