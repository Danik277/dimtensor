"""CERN Open Data Portal loader for particle physics datasets.

Provides access to CERN Open Data Portal (opendata.cern.ch) datasets from LHC
experiments (CMS, ATLAS, LHCb, ALICE). Supports NanoAOD format ROOT files with
automatic unit conversion to DimArrays (GeV, MeV, barn, etc.).

Example:
    >>> from dimtensor.datasets.loaders import CERNOpenDataLoader
    >>> loader = CERNOpenDataLoader()
    >>> # Load NanoAOD file
    >>> events = loader.load_nanoaod(
    ...     "root://eospublic.cern.ch//eos/opendata/cms/...",
    ...     max_events=1000
    ... )
    >>> electrons = events['Electron']
    >>> print(electrons['pt'])  # Transverse momentum in GeV

Dependencies:
    - uproot: Pure Python ROOT I/O (required)
    - awkward: Jagged array support (required)
    - cernopendata-client: Metadata queries (optional)

Reference:
    - CERN Open Data Portal: https://opendata.cern.ch/
    - CMS NanoAOD Guide: https://opendata.cern.ch/docs/cms-getting-started-nanoaod
    - uproot documentation: https://uproot.readthedocs.io/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import DIMENSIONLESS, Dimension
from ...core.units import Unit, cm
from ...domains.nuclear import GeV, MeV, barn, mb
from .base import BaseLoader

# Check for required dependencies
try:
    import uproot

    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

try:
    import awkward as ak

    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False

try:
    from cernopendata_client.api import get_record, search_records

    HAS_CERNOPENDATA_CLIENT = True
except ImportError:
    HAS_CERNOPENDATA_CLIENT = False


class CERNOpenDataLoader(BaseLoader):
    """Loader for CERN Open Data Portal particle physics datasets.

    Downloads and parses NanoAOD ROOT files from CERN Open Data Portal,
    converting particle physics data to DimArrays with proper units.

    Supports:
        - CMS NanoAOD format (2015+)
        - Energy/momentum in GeV or MeV
        - Cross-sections in barn/mb
        - Metadata queries via cernopendata-client (optional)

    Attributes:
        cache_enabled: Whether to use caching (default: True).
        cache_dir: Directory for cached files.
    """

    # Common NanoAOD branches and their units
    # pt, energy, mass → GeV
    # eta, phi → dimensionless (pseudorapidity, azimuthal angle)
    # charge → dimensionless (elementary charge units)
    NANOAOD_UNITS = {
        "pt": GeV,  # Transverse momentum
        "eta": DIMENSIONLESS,  # Pseudorapidity
        "phi": DIMENSIONLESS,  # Azimuthal angle
        "mass": GeV,  # Invariant mass
        "energy": GeV,  # Energy
        "charge": DIMENSIONLESS,  # Electric charge (in units of e)
        "pdgId": DIMENSIONLESS,  # Particle Data Group ID
        "dxy": cm,  # Impact parameter (cm)
        "dz": cm,  # Longitudinal impact (cm)
        "pfRelIso03_all": DIMENSIONLESS,  # Particle flow isolation
        "pfRelIso04_all": DIMENSIONLESS,  # Particle flow isolation
        "jetIdx": DIMENSIONLESS,  # Jet index
        "genPartIdx": DIMENSIONLESS,  # Generator particle index
    }

    def __init__(self, cache: bool = True):
        """Initialize the CERN Open Data loader.

        Args:
            cache: Whether to enable caching of downloaded files.

        Raises:
            ImportError: If uproot or awkward libraries are not installed.
        """
        super().__init__(cache=cache)

        if not HAS_UPROOT:
            raise ImportError(
                "uproot library required for ROOT file parsing. "
                "Install with: pip install uproot"
            )

        if not HAS_AWKWARD:
            raise ImportError(
                "awkward library required for jagged array support. "
                "Install with: pip install awkward"
            )

    def load(self, **kwargs: Any) -> dict[str, Any]:
        """Load CERN Open Data dataset.

        This is a generic interface. Use load_nanoaod() for NanoAOD files
        or query_metadata() for dataset discovery.

        Args:
            **kwargs: Loader-specific arguments.

        Returns:
            Dictionary of DimArrays with physics data.

        Raises:
            NotImplementedError: Use specific loading methods instead.
        """
        raise NotImplementedError(
            "Use load_nanoaod() to load ROOT files or "
            "query_metadata() to search for datasets."
        )

    def load_nanoaod(
        self,
        file_path: str | Path,
        tree_name: str = "Events",
        max_events: int | None = None,
        branches: list[str] | None = None,
        flatten: bool = True,
    ) -> dict[str, dict[str, DimArray | list]]:
        """Load NanoAOD ROOT file and convert to DimArrays.

        Args:
            file_path: Path or URL to ROOT file (supports XRootD URLs).
            tree_name: Name of TTree to read (default: "Events").
            max_events: Maximum number of events to read (None = all).
            branches: List of branch patterns to read (None = all).
                     Supports wildcards, e.g., ["Electron_*", "Muon_*"].
            flatten: Whether to flatten jagged arrays to 1D (default: True).
                    If False, returns awkward arrays for variable-length collections.

        Returns:
            Dictionary of physics objects, each containing DimArrays:
                {
                    "Electron": {"pt": DimArray, "eta": DimArray, ...},
                    "Muon": {"pt": DimArray, "eta": DimArray, ...},
                    "Jet": {"pt": DimArray, "eta": DimArray, ...},
                    ...
                }

        Example:
            >>> loader = CERNOpenDataLoader()
            >>> events = loader.load_nanoaod(
            ...     "root://eospublic.cern.ch//eos/opendata/cms/...",
            ...     max_events=10000
            ... )
            >>> electrons = events['Electron']
            >>> print(f"Loaded {len(electrons['pt'])} electrons")
            >>> print(f"Mean pt: {electrons['pt'].mean()}")

        Notes:
            - NanoAOD stores collections as flat arrays with one entry per object
            - Event structure is implicitly defined by size metadata
            - XRootD URLs (root://) are supported for remote access
        """
        # Handle both local paths and URLs
        if isinstance(file_path, Path):
            file_path = str(file_path)

        # If it's a URL and caching is enabled, download first
        if file_path.startswith(("http://", "https://")):
            if self.cache_enabled:
                cache_key = f"cern_root_{hash(file_path) & 0xFFFFFFFF}"
                file_path = str(self.download(file_path, cache_key=cache_key))

        # Open ROOT file with uproot
        try:
            root_file = uproot.open(file_path)
            tree = root_file[tree_name]
        except Exception as e:
            raise RuntimeError(
                f"Failed to open ROOT file {file_path}: {e}. "
                "Ensure uproot is installed and file exists."
            ) from e

        # Determine which branches to read
        if branches is None:
            # Read all branches
            branch_names = tree.keys()
        else:
            # Expand wildcards and filter
            branch_names = []
            all_branches = tree.keys()
            for pattern in branches:
                if "*" in pattern:
                    # Simple wildcard matching
                    prefix = pattern.split("*")[0]
                    branch_names.extend([b for b in all_branches if b.startswith(prefix)])
                else:
                    if pattern in all_branches:
                        branch_names.append(pattern)

        # Read data as awkward arrays
        try:
            arrays = tree.arrays(
                branch_names,
                entry_stop=max_events,
                library="ak",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read branches from ROOT file: {e}") from e

        # Group branches by physics object (Electron, Muon, Jet, etc.)
        physics_objects: dict[str, dict[str, Any]] = {}

        for branch_name in arrays.fields:
            # Parse branch name: "Object_property" or "property"
            if "_" in branch_name:
                obj_name, prop_name = branch_name.split("_", 1)
            else:
                # Global event-level property
                obj_name = "Event"
                prop_name = branch_name

            # Initialize object dict if needed
            if obj_name not in physics_objects:
                physics_objects[obj_name] = {}

            # Get the data
            data = arrays[branch_name]

            # Flatten jagged arrays if requested
            if flatten and hasattr(data, "layout"):
                # Check if it's a jagged array (ListOffsetArray, etc.)
                if "ListOffset" in type(data.layout).__name__:
                    data = ak.flatten(data)

            # Convert awkward array to numpy (if possible)
            try:
                data_np = ak.to_numpy(data)
            except Exception:
                # Keep as awkward array for complex structures
                data_np = data

            # Assign units based on property name
            unit = self._get_unit_for_property(prop_name)

            # Convert to DimArray if numeric
            if isinstance(data_np, np.ndarray) and np.issubdtype(data_np.dtype, np.number):
                physics_objects[obj_name][prop_name] = DimArray(data_np, unit=unit)
            else:
                # Keep as raw data for non-numeric or complex types
                physics_objects[obj_name][prop_name] = data_np

        return physics_objects

    def _get_unit_for_property(self, prop_name: str) -> Unit:
        """Determine the appropriate unit for a NanoAOD property.

        Args:
            prop_name: Property name (e.g., "pt", "eta", "mass").

        Returns:
            Unit object appropriate for this property.
        """
        # Check common properties
        if prop_name in self.NANOAOD_UNITS:
            return self.NANOAOD_UNITS[prop_name]

        # Heuristics for other properties
        if "pt" in prop_name.lower() or "energy" in prop_name.lower():
            return GeV
        elif "mass" in prop_name.lower():
            return GeV  # In particle physics, mass is often in GeV/c²
        elif "eta" in prop_name.lower() or "phi" in prop_name.lower():
            return Unit("dimensionless", DIMENSIONLESS, 1.0)
        elif "charge" in prop_name.lower():
            return Unit("dimensionless", DIMENSIONLESS, 1.0)
        else:
            # Default to dimensionless for unknown properties
            return Unit("dimensionless", DIMENSIONLESS, 1.0)

    def extract_physics_objects(
        self,
        events: dict[str, dict[str, DimArray | list]],
        object_type: str,
        pt_min: float | None = None,
        eta_max: float | None = None,
    ) -> dict[str, DimArray]:
        """Extract and filter physics objects (electrons, muons, jets, etc.).

        Args:
            events: Dictionary from load_nanoaod().
            object_type: Type of object to extract ("Electron", "Muon", "Jet", etc.).
            pt_min: Minimum transverse momentum in GeV (None = no cut).
            eta_max: Maximum absolute pseudorapidity (None = no cut).

        Returns:
            Dictionary of DimArrays for the selected objects after cuts.

        Example:
            >>> events = loader.load_nanoaod("data.root")
            >>> electrons = loader.extract_physics_objects(
            ...     events, "Electron", pt_min=25.0, eta_max=2.5
            ... )
            >>> print(f"Selected {len(electrons['pt'])} electrons")
        """
        if object_type not in events:
            raise ValueError(
                f"Object type '{object_type}' not found in events. "
                f"Available: {list(events.keys())}"
            )

        obj_data = events[object_type]

        # Apply cuts
        mask = np.ones(len(obj_data["pt"]), dtype=bool)

        if pt_min is not None and "pt" in obj_data:
            pt_values = obj_data["pt"].data  # Get raw numpy array
            mask &= pt_values >= pt_min

        if eta_max is not None and "eta" in obj_data:
            eta_values = obj_data["eta"].data
            mask &= np.abs(eta_values) <= eta_max

        # Apply mask to all properties
        filtered = {}
        for prop_name, prop_data in obj_data.items():
            if isinstance(prop_data, DimArray):
                filtered[prop_name] = DimArray._from_data_and_unit(
                    prop_data.data[mask],
                    prop_data.unit,
                )
            elif isinstance(prop_data, np.ndarray):
                filtered[prop_name] = prop_data[mask]
            else:
                # Skip non-array data
                continue

        return filtered

    def query_metadata(
        self,
        experiment: str | None = None,
        year: int | None = None,
        search_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query CERN Open Data Portal for available datasets.

        Args:
            experiment: Filter by experiment ("CMS", "ATLAS", "LHCb", "ALICE").
            year: Filter by year (e.g., 2015, 2016).
            search_query: Free-text search query.

        Returns:
            List of dataset metadata dictionaries.

        Example:
            >>> loader = CERNOpenDataLoader()
            >>> datasets = loader.query_metadata(
            ...     experiment="CMS",
            ...     year=2015,
            ...     search_query="NanoAOD"
            ... )
            >>> for ds in datasets[:5]:
            ...     print(ds['title'])

        Raises:
            ImportError: If cernopendata-client is not installed.
            RuntimeError: If API query fails.

        Note:
            This requires cernopendata-client:
                pip install cernopendata-client
        """
        if not HAS_CERNOPENDATA_CLIENT:
            raise ImportError(
                "cernopendata-client required for metadata queries. "
                "Install with: pip install cernopendata-client"
            )

        # Build search filters
        filters = []
        if experiment:
            filters.append(f"experiment:{experiment}")
        if year:
            filters.append(f"year:{year}")

        # Combine with search query
        if search_query:
            query = f"{search_query} {' '.join(filters)}"
        else:
            query = " ".join(filters) if filters else ""

        try:
            # Search records via cernopendata-client
            results = search_records(query)
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to query CERN Open Data Portal: {e}") from e

    def get_record_info(self, record_id: int) -> dict[str, Any]:
        """Get detailed information about a specific dataset record.

        Args:
            record_id: CERN Open Data record ID (e.g., 12341).

        Returns:
            Dictionary with record metadata (title, description, files, etc.).

        Example:
            >>> loader = CERNOpenDataLoader()
            >>> info = loader.get_record_info(12341)
            >>> print(info['title'])
            >>> print(info['files'][0]['uri'])

        Raises:
            ImportError: If cernopendata-client is not installed.
            RuntimeError: If API query fails.
        """
        if not HAS_CERNOPENDATA_CLIENT:
            raise ImportError(
                "cernopendata-client required for metadata queries. "
                "Install with: pip install cernopendata-client"
            )

        try:
            record = get_record(record_id)
            return record
        except Exception as e:
            raise RuntimeError(f"Failed to get record {record_id}: {e}") from e

    def cache_nanoaod_metadata(
        self,
        file_path: str | Path,
        metadata: dict[str, Any],
    ) -> None:
        """Cache metadata for a NanoAOD file.

        Args:
            file_path: Path to the ROOT file.
            metadata: Metadata dictionary to cache.

        Example:
            >>> loader = CERNOpenDataLoader()
            >>> metadata = {
            ...     "experiment": "CMS",
            ...     "year": 2015,
            ...     "record_id": 12341,
            ...     "tree_name": "Events",
            ...     "branches": ["Electron_pt", "Muon_pt"],
            ... }
            >>> loader.cache_nanoaod_metadata("data.root", metadata)
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        cache_key = f"cern_metadata_{hash(file_path) & 0xFFFFFFFF}"
        metadata_file = self.cache_dir / f"{cache_key}.json"

        metadata_file.write_text(json.dumps(metadata, indent=2))

    def get_cached_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Retrieve cached metadata for a NanoAOD file.

        Args:
            file_path: Path to the ROOT file.

        Returns:
            Cached metadata dictionary, or None if not cached.
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        cache_key = f"cern_metadata_{hash(file_path) & 0xFFFFFFFF}"
        metadata_file = self.cache_dir / f"{cache_key}.json"

        if not metadata_file.exists():
            return None

        return json.loads(metadata_file.read_text())
