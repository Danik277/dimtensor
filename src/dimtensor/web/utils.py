"""Utility functions for the web dashboard."""

from __future__ import annotations

from typing import Any

from ..core.dimensions import Dimension


def format_dimension(dim: Dimension) -> str:
    """Format a Dimension object as a readable string.

    Args:
        dim: Dimension to format.

    Returns:
        String representation (e.g., "L¹T⁻²" for acceleration).
    """
    return str(dim)


def format_dimensions_dict(dims: dict[str, Dimension]) -> str:
    """Format a dictionary of dimensions as a readable string.

    Args:
        dims: Dictionary mapping names to dimensions.

    Returns:
        Formatted string (e.g., "x: L, v: LT⁻¹").
    """
    if not dims:
        return "None"
    return ", ".join(f"{k}: {format_dimension(v)}" for k, v in dims.items())


def generate_model_code(name: str, info: Any) -> str:
    """Generate code snippet for loading a model.

    Args:
        name: Model name.
        info: ModelInfo object.

    Returns:
        Python code snippet as string.
    """
    code = f"""# Load model: {name}
from dimtensor.hub import load_model

model = load_model("{name}")

# Model information:
# - Domain: {info.domain}
# - Architecture: {info.architecture if info.architecture else 'N/A'}
# - Input dimensions: {format_dimensions_dict(info.input_dims)}
# - Output dimensions: {format_dimensions_dict(info.output_dims)}
"""
    return code


def generate_dataset_code(name: str, info: Any) -> str:
    """Generate code snippet for loading a dataset.

    Args:
        name: Dataset name.
        info: DatasetInfo object.

    Returns:
        Python code snippet as string.
    """
    code = f"""# Load dataset: {name}
from dimtensor.datasets import load_dataset

try:
    data = load_dataset("{name}")
    print(data)
except RuntimeError:
    # Some datasets don't have loaders yet
    # Use get_dataset_info to see metadata
    from dimtensor.datasets import get_dataset_info
    info = get_dataset_info("{name}")
    print(info)

# Dataset information:
# - Domain: {info.domain}
# - Features: {format_dimensions_dict(info.features)}
# - Targets: {format_dimensions_dict(info.targets)}
"""
    if info.size:
        code += f"# - Size: {info.size} samples\n"
    return code


def generate_equation_code(name: str, equation: Any) -> str:
    """Generate code snippet for using an equation.

    Args:
        name: Equation name.
        equation: Equation object.

    Returns:
        Python code snippet as string.
    """
    code = f"""# Use equation: {name}
from dimtensor.equations import get_equation

eq = get_equation("{name}")

# Equation information:
# - Formula: {equation.formula}
# - Domain: {equation.domain}
# - Variables: {format_dimensions_dict(equation.variables)}

# Verify dimensions in your calculation
# Example: Check if F = ma is dimensionally consistent
from dimtensor import Dimension
F_dim = Dimension(mass=1, length=1, time=-2)  # Force
m_dim = Dimension(mass=1)  # Mass
a_dim = Dimension(length=1, time=-2)  # Acceleration

# Check: F should equal m * a dimensionally
assert F_dim == m_dim * a_dim
"""
    return code


def search_items(items: list[Any], query: str, fields: list[str]) -> list[Any]:
    """Search items by query across multiple fields.

    Args:
        items: List of items to search.
        query: Search query (case-insensitive).
        fields: List of field names to search in.

    Returns:
        Filtered list of items matching the query.
    """
    if not query:
        return items

    query_lower = query.lower()
    results = []

    for item in items:
        for field in fields:
            value = getattr(item, field, "")
            if isinstance(value, str) and query_lower in value.lower():
                results.append(item)
                break
            elif isinstance(value, list):
                # Search in list fields (like tags)
                if any(query_lower in str(v).lower() for v in value):
                    results.append(item)
                    break
            elif isinstance(value, dict):
                # Search in dict keys
                if any(query_lower in k.lower() for k in value.keys()):
                    results.append(item)
                    break

    return results


def filter_by_domain(items: list[Any], domain: str | None) -> list[Any]:
    """Filter items by domain.

    Args:
        items: List of items to filter.
        domain: Domain to filter by, or None for all.

    Returns:
        Filtered list of items.
    """
    if not domain or domain == "All":
        return items
    return [item for item in items if getattr(item, "domain", "") == domain]


def filter_by_tags(items: list[Any], tags: list[str]) -> list[Any]:
    """Filter items by tags (must have all tags).

    Args:
        items: List of items to filter.
        tags: List of tags to filter by.

    Returns:
        Filtered list of items.
    """
    if not tags:
        return items
    return [
        item for item in items
        if all(tag in getattr(item, "tags", []) for tag in tags)
    ]


def get_all_tags(items: list[Any]) -> list[str]:
    """Extract all unique tags from items.

    Args:
        items: List of items with tags attribute.

    Returns:
        Sorted list of unique tags.
    """
    tags = set()
    for item in items:
        tags.update(getattr(item, "tags", []))
    return sorted(tags)


def get_all_domains(items: list[Any]) -> list[str]:
    """Extract all unique domains from items.

    Args:
        items: List of items with domain attribute.

    Returns:
        Sorted list of unique domains.
    """
    domains = set(getattr(item, "domain", "general") for item in items)
    return sorted(domains)
