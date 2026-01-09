"""Plugin metadata structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.units import Unit


@dataclass
class PluginMetadata:
    """Metadata for a dimtensor plugin.

    Attributes:
        name: Plugin name (e.g., 'nuclear', 'geophysics').
        version: Plugin version string (e.g., '0.1.0').
        author: Plugin author or organization.
        description: Brief description of the plugin.
        units: Dictionary mapping unit names to Unit objects.
    """

    name: str
    version: str
    author: str
    description: str
    units: dict[str, Unit]

    def __repr__(self) -> str:
        """String representation."""
        num_units = len(self.units)
        return (
            f"PluginMetadata(name={self.name!r}, version={self.version!r}, "
            f"author={self.author!r}, units={num_units} units)"
        )
