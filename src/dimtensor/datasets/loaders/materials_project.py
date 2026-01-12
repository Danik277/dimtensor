"""Materials Project dataset loader.

Loads crystal structures and material properties from the Materials Project
database via the mp-api library.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.units import angstrom, electronvolt
from ...domains.materials import GPa
from .base import BaseLoader

try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False


class MaterialsProjectLoader(BaseLoader):
    """Loader for Materials Project crystal structures and properties.

    Queries the Materials Project database for material properties including
    band gap, formation energy, elastic constants, and crystal structures.

    Requires mp-api library: pip install mp-api
    API key required (free registration at materialsproject.org)

    Example:
        >>> loader = MaterialsProjectLoader(api_key="your_api_key")
        >>> # Or use environment variable: MATERIALS_PROJECT_API_KEY
        >>> loader = MaterialsProjectLoader()
        >>>
        >>> # Query by material ID
        >>> data = loader.load(material_ids=["mp-149"])  # Silicon
        >>>
        >>> # Query by formula
        >>> data = loader.load(formula="Fe2O3")
        >>>
        >>> # Query by chemical system
        >>> data = loader.load(chemsys="Si-O")
    """

    def __init__(self, api_key: str | None = None, cache: bool = True):
        """Initialize Materials Project loader.

        Args:
            api_key: Materials Project API key. If None, reads from
                MATERIALS_PROJECT_API_KEY environment variable.
            cache: Whether to enable caching (default: True).

        Raises:
            ImportError: If mp-api library not installed.
            ValueError: If no API key provided or found in environment.
        """
        super().__init__(cache=cache)

        if not HAS_MP_API:
            raise ImportError(
                "mp-api library required for Materials Project loader. "
                "Install with: pip install mp-api"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("MATERIALS_PROJECT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Materials Project API key required. "
                "Provide via api_key parameter or MATERIALS_PROJECT_API_KEY "
                "environment variable. Get free key at: materialsproject.org"
            )

    def load(
        self,
        material_ids: list[str] | None = None,
        formula: str | None = None,
        chemsys: str | None = None,
        include_structure: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load materials data from Materials Project.

        Query by material ID, formula, or chemical system. Returns material
        properties as DimArrays with proper units.

        Args:
            material_ids: List of material IDs (e.g., ["mp-149", "mp-1234"]).
            formula: Chemical formula (e.g., "Fe2O3", "SiO2").
            chemsys: Chemical system (e.g., "Si-O", "Fe-O").
            include_structure: Include raw structure data (default: False).
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary with keys:
                - material_ids: List of material IDs
                - formulas: List of chemical formulas
                - band_gap: Band gap energies (eV)
                - formation_energy_per_atom: Formation energy per atom (eV/atom)
                - bulk_modulus: Bulk modulus (GPa), if available
                - shear_modulus: Shear modulus (GPa), if available
                - structures: Raw structure data (if include_structure=True)

        Example:
            >>> loader = MaterialsProjectLoader()
            >>> data = loader.load(material_ids=["mp-149"])
            >>> print(data["band_gap"])  # Band gap in eV
            >>> print(data["formulas"])  # ["Si"]

        Raises:
            ValueError: If no query parameter provided.
            RuntimeError: If API request fails.
        """
        # Validate query parameters
        if not any([material_ids, formula, chemsys]):
            raise ValueError(
                "Must provide at least one query parameter: "
                "material_ids, formula, or chemsys"
            )

        # Query Materials Project API
        try:
            materials = self._query_materials(
                material_ids=material_ids,
                formula=formula,
                chemsys=chemsys,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query Materials Project API: {e}"
            ) from e

        if not materials:
            return {
                "material_ids": [],
                "formulas": [],
                "band_gap": DimArray(np.array([]), unit=electronvolt),
                "formation_energy_per_atom": DimArray(np.array([]), unit=electronvolt),
            }

        # Convert to DimArrays
        result = self._convert_to_dimarrays(materials, include_structure)

        return result

    def _query_materials(
        self,
        material_ids: list[str] | None = None,
        formula: str | None = None,
        chemsys: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query Materials Project API.

        Args:
            material_ids: List of material IDs.
            formula: Chemical formula.
            chemsys: Chemical system.

        Returns:
            List of material data dictionaries from API.
        """
        with MPRester(self.api_key) as mpr:
            # Query based on provided parameters
            if material_ids:
                # Query by material IDs
                materials = []
                for mat_id in material_ids:
                    try:
                        # Get summary document for material
                        doc = mpr.materials.summary.search(
                            material_ids=[mat_id],
                        )
                        if doc:
                            materials.extend(doc)
                    except Exception:
                        # Skip materials that fail to load
                        continue

            elif formula:
                # Query by formula
                materials = mpr.materials.summary.search(
                    formula=formula,
                )

            elif chemsys:
                # Query by chemical system
                materials = mpr.materials.summary.search(
                    chemsys=chemsys,
                )
            else:
                materials = []

        # Convert to list of dicts
        result = []
        for mat in materials:
            # Extract relevant properties
            mat_dict = {
                "material_id": mat.material_id,
                "formula": str(mat.formula_pretty),
                "band_gap": mat.band_gap,
                "formation_energy_per_atom": mat.formation_energy_per_atom,
                "structure": mat.structure if hasattr(mat, "structure") else None,
            }

            # Add elastic properties if available
            if hasattr(mat, "bulk_modulus") and mat.bulk_modulus is not None:
                mat_dict["bulk_modulus"] = mat.bulk_modulus.vrh  # Voigt-Reuss-Hill average

            if hasattr(mat, "shear_modulus") and mat.shear_modulus is not None:
                mat_dict["shear_modulus"] = mat.shear_modulus.vrh

            result.append(mat_dict)

        return result

    def _convert_to_dimarrays(
        self,
        materials: list[dict[str, Any]],
        include_structure: bool = False,
    ) -> dict[str, Any]:
        """Convert materials data to DimArrays.

        Args:
            materials: List of material data dictionaries.
            include_structure: Whether to include raw structure data.

        Returns:
            Dictionary of DimArrays and metadata.
        """
        # Extract data lists
        material_ids = []
        formulas = []
        band_gaps = []
        formation_energies = []
        bulk_moduli = []
        shear_moduli = []
        structures = []

        for mat in materials:
            material_ids.append(mat["material_id"])
            formulas.append(mat["formula"])

            # Band gap (eV) - use 0.0 if None
            band_gaps.append(mat.get("band_gap", 0.0) or 0.0)

            # Formation energy per atom (eV/atom)
            formation_energies.append(
                mat.get("formation_energy_per_atom", np.nan) or np.nan
            )

            # Elastic properties (GPa) - may be None
            bulk_moduli.append(mat.get("bulk_modulus", np.nan))
            shear_moduli.append(mat.get("shear_modulus", np.nan))

            # Structure (optional)
            if include_structure:
                structures.append(mat.get("structure"))

        # Build result dictionary
        result: dict[str, Any] = {
            "material_ids": material_ids,
            "formulas": formulas,
            "band_gap": DimArray(np.array(band_gaps), unit=electronvolt),
            "formation_energy_per_atom": DimArray(
                np.array(formation_energies), unit=electronvolt
            ),
        }

        # Add elastic properties if any are available
        if any(not np.isnan(v) for v in bulk_moduli):
            result["bulk_modulus"] = DimArray(np.array(bulk_moduli), unit=GPa)

        if any(not np.isnan(v) for v in shear_moduli):
            result["shear_modulus"] = DimArray(np.array(shear_moduli), unit=GPa)

        # Add structures if requested
        if include_structure:
            result["structures"] = structures

        return result
