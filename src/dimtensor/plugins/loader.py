"""Plugin loader using importlib.metadata entry points."""

from __future__ import annotations

import importlib
import sys
from typing import Any

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points


ENTRY_POINT_GROUP = "dimtensor.plugins"


def discover_entry_points() -> dict[str, str]:
    """Discover all dimtensor plugins via entry points.

    Returns:
        Dictionary mapping plugin names to entry point strings.
    """
    discovered = {}

    # Get entry points for dimtensor.plugins group
    if sys.version_info >= (3, 10):
        eps = entry_points(group=ENTRY_POINT_GROUP)
    else:
        eps = entry_points().get(ENTRY_POINT_GROUP, [])

    for ep in eps:
        discovered[ep.name] = f"{ep.value}"

    return discovered


def load_plugin_from_entry_point(name: str, entry_point_value: str) -> Any:
    """Load a plugin from an entry point string.

    Args:
        name: Plugin name.
        entry_point_value: Entry point value (e.g., "module:attribute").

    Returns:
        Plugin object (should be PluginMetadata).

    Raises:
        ImportError: If module cannot be imported.
        AttributeError: If plugin attribute not found.
        ValueError: If entry point format is invalid.
    """
    # Parse entry point: "module.path:attribute"
    if ":" not in entry_point_value:
        raise ValueError(
            f"Invalid entry point format for plugin {name!r}: {entry_point_value!r}. "
            f"Expected format: 'module.path:attribute'"
        )

    module_path, attr_name = entry_point_value.split(":", 1)

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module {module_path!r} for plugin {name!r}: {e}"
        ) from e

    # Get the plugin attribute
    try:
        plugin = getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module {module_path!r} has no attribute {attr_name!r} for plugin {name!r}: {e}"
        ) from e

    return plugin
