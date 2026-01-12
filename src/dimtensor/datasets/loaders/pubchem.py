"""PubChem compound data loader.

Loads molecular properties from the PubChem PUG REST API with proper
chemistry units (dalton, angstrom, kelvin, pascal).

Example:
    >>> from dimtensor.datasets.loaders import PubChemLoader
    >>> loader = PubChemLoader()
    >>> # Fetch aspirin by CID
    >>> compound = loader.get_compound_by_cid(2244)
    >>> print(compound['MolecularWeight'])  # DimArray in dalton
    >>> # Search by name
    >>> caffeine = loader.get_compound_by_name('caffeine')
    >>> print(caffeine['MolecularFormula'])

PubChem API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...core.dimarray import DimArray
from ...domains.chemistry import angstrom, dalton
from ...core.units import kelvin, meter, kg, pascal
from .base import BaseLoader

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class PubChemLoader(BaseLoader):
    """Loader for PubChem compound data via PUG REST API.

    Fetches molecular properties with automatic unit conversion to DimArrays.
    Supports lookup by CID, compound name, InChI, and SMILES.

    Attributes:
        cache_enabled: Whether to cache API responses.
        rate_limit: Minimum seconds between requests (default: 0.2 for 5 req/sec).
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Property name to unit mapping
    PROPERTY_UNITS = {
        'MolecularWeight': dalton,
        'Volume3D': angstrom**3,
        'Density': kg / (meter**3),
        # Note: PubChem returns temperatures in Celsius in some contexts,
        # but we'll apply kelvin units and document this
    }

    # Common properties to fetch by default
    DEFAULT_PROPERTIES = [
        'MolecularFormula',
        'MolecularWeight',
        'CanonicalSMILES',
        'IsomericSMILES',
        'InChI',
        'InChIKey',
        'IUPACName',
        'XLogP',
        'TPSA',
        'Complexity',
        'HBondDonorCount',
        'HBondAcceptorCount',
        'RotatableBondCount',
        'HeavyAtomCount',
    ]

    def __init__(self, cache: bool = True, rate_limit: float = 0.2):
        """Initialize PubChem loader.

        Args:
            cache: Whether to enable response caching.
            rate_limit: Minimum seconds between requests (default: 0.2 = 5 req/sec).
        """
        super().__init__(cache=cache)
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for PubChem loader. "
                "Install with: pip install requests"
            )

    def _rate_limit_wait(self) -> None:
        """Ensure rate limit is respected between API calls."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        self._last_request_time = time.time()

    def _make_request(self, url: str, cache_key: str | None = None) -> dict[str, Any]:
        """Make API request with rate limiting and caching.

        Args:
            url: Full URL to request.
            cache_key: Optional cache identifier.

        Returns:
            Parsed JSON response as dict.

        Raises:
            RuntimeError: If request fails.
        """
        # Check cache first
        if cache_key and self.cache_enabled:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())

        # Apply rate limiting
        self._rate_limit_wait()

        # Make request
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"PubChem API request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PubChem response: {e}") from e

        # Cache response
        if cache_key and self.cache_enabled:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.write_text(json.dumps(data, indent=2))

        return data

    def get_properties(
        self,
        identifiers: int | str | list[int] | list[str],
        properties: list[str] | None = None,
        namespace: str = "cid",
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Get compound properties by identifier.

        Args:
            identifiers: Single or list of CIDs, names, InChI, or SMILES.
            properties: List of property names (default: common properties).
            namespace: Identifier type - 'cid', 'name', 'inchi', 'smiles'.

        Returns:
            Dict of properties (single compound) or list of dicts (multiple).
            Properties with known units are returned as DimArrays.

        Example:
            >>> loader = PubChemLoader()
            >>> props = loader.get_properties(2244, namespace='cid')
            >>> print(props['MolecularWeight'])  # DimArray in dalton
        """
        if properties is None:
            properties = self.DEFAULT_PROPERTIES

        # Handle single vs multiple identifiers
        if isinstance(identifiers, (int, str)):
            identifiers = [identifiers]
            single_result = True
        else:
            single_result = False

        # Build URL
        id_str = ",".join(str(i) for i in identifiers)
        prop_str = ",".join(properties)
        url = f"{self.BASE_URL}/compound/{namespace}/{id_str}/property/{prop_str}/JSON"

        # Create cache key
        cache_key = f"pubchem_{namespace}_{id_str}_{prop_str}".replace(",", "_")[:100]

        # Make request
        data = self._make_request(url, cache_key=cache_key)

        # Parse response
        if 'PropertyTable' not in data or 'Properties' not in data['PropertyTable']:
            raise RuntimeError(f"Unexpected PubChem response format: {data}")

        results = []
        for compound_data in data['PropertyTable']['Properties']:
            parsed = self._parse_properties(compound_data)
            results.append(parsed)

        return results[0] if single_result else results

    def _parse_properties(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Parse raw property data and convert to DimArrays where appropriate.

        Args:
            raw_data: Raw property dict from PubChem API.

        Returns:
            Dict with DimArray values for properties with known units.
        """
        import numpy as np

        parsed = {}

        for key, value in raw_data.items():
            # Skip null values
            if value is None:
                parsed[key] = None
                continue

            # Check if property has a known unit
            if key in self.PROPERTY_UNITS:
                unit = self.PROPERTY_UNITS[key]
                # Convert to numpy array and wrap in DimArray
                if isinstance(value, (int, float)):
                    parsed[key] = DimArray(np.array([value]), unit=unit)
                else:
                    # Try to convert to float
                    try:
                        parsed[key] = DimArray(np.array([float(value)]), unit=unit)
                    except (ValueError, TypeError):
                        # If conversion fails, store as-is
                        parsed[key] = value
            else:
                # Store non-unit properties as-is
                parsed[key] = value

        return parsed

    def get_compound_by_cid(self, cid: int) -> dict[str, Any]:
        """Get compound properties by PubChem Compound ID (CID).

        Args:
            cid: PubChem Compound ID (integer).

        Returns:
            Dict of compound properties with DimArray values.

        Example:
            >>> loader = PubChemLoader()
            >>> aspirin = loader.get_compound_by_cid(2244)
            >>> print(aspirin['MolecularWeight'])
            >>> print(aspirin['MolecularFormula'])
        """
        return self.get_properties(cid, namespace='cid')

    def get_compound_by_name(self, name: str) -> dict[str, Any]:
        """Get compound properties by compound name.

        Args:
            name: Common or IUPAC compound name.

        Returns:
            Dict of compound properties with DimArray values.

        Note:
            If multiple compounds match the name, returns the first match.
            Use search_compounds() to find all matches.

        Example:
            >>> loader = PubChemLoader()
            >>> caffeine = loader.get_compound_by_name('caffeine')
            >>> print(caffeine['MolecularFormula'])
        """
        return self.get_properties(name, namespace='name')

    def search_compounds(self, query: str, max_results: int = 10) -> list[int]:
        """Search for compounds by name and return matching CIDs.

        Args:
            query: Search query (compound name or formula).
            max_results: Maximum number of CIDs to return.

        Returns:
            List of PubChem CIDs matching the query.

        Example:
            >>> loader = PubChemLoader()
            >>> cids = loader.search_compounds('glucose')
            >>> print(f"Found {len(cids)} compounds")
        """
        # Use PubChem's compound search endpoint
        url = f"{self.BASE_URL}/compound/name/{query}/cids/JSON"
        cache_key = f"pubchem_search_{query}".replace(" ", "_")[:100]

        try:
            data = self._make_request(url, cache_key=cache_key)

            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                cids = data['IdentifierList']['CID']
                return cids[:max_results]
            else:
                return []
        except RuntimeError:
            # Search failed, return empty list
            return []

    def get_compound_by_inchi(self, inchi: str) -> dict[str, Any]:
        """Get compound properties by InChI identifier.

        Args:
            inchi: InChI string (e.g., 'InChI=1S/C9H8O4/c1-6(10)...').

        Returns:
            Dict of compound properties with DimArray values.

        Example:
            >>> loader = PubChemLoader()
            >>> inchi = 'InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)'
            >>> compound = loader.get_compound_by_inchi(inchi)
        """
        return self.get_properties(inchi, namespace='inchi')

    def get_compound_by_smiles(self, smiles: str) -> dict[str, Any]:
        """Get compound properties by SMILES string.

        Args:
            smiles: SMILES notation (e.g., 'CC(=O)Oc1ccccc1C(=O)O' for aspirin).

        Returns:
            Dict of compound properties with DimArray values.

        Example:
            >>> loader = PubChemLoader()
            >>> smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # aspirin
            >>> compound = loader.get_compound_by_smiles(smiles)
        """
        return self.get_properties(smiles, namespace='smiles')

    def load(self, **kwargs: Any) -> dict[str, Any]:
        """Load compound data (required by BaseLoader interface).

        This is a generic interface. Use specific methods like
        get_compound_by_cid() or get_compound_by_name() instead.

        Args:
            **kwargs: Must include either 'cid', 'name', 'inchi', or 'smiles'.

        Returns:
            Dict of compound properties with DimArray values.

        Raises:
            ValueError: If no valid identifier provided.
        """
        if 'cid' in kwargs:
            return self.get_compound_by_cid(kwargs['cid'])
        elif 'name' in kwargs:
            return self.get_compound_by_name(kwargs['name'])
        elif 'inchi' in kwargs:
            return self.get_compound_by_inchi(kwargs['inchi'])
        elif 'smiles' in kwargs:
            return self.get_compound_by_smiles(kwargs['smiles'])
        else:
            raise ValueError(
                "Must provide one of: cid, name, inchi, or smiles. "
                "Use get_compound_by_cid() or get_compound_by_name() instead."
            )
