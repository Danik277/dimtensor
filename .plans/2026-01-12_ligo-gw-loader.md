# Plan: LIGO Gravitational Wave Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Create a loader module for gravitational wave data from GWOSC (Gravitational Wave Open Science Center), enabling users to download strain data and event catalogs with proper dimensional units (strain, Hz, seconds).

---

## Background

GWOSC provides open-access gravitational wave data from LIGO, Virgo, and GEO detectors. Gravitational wave research requires:
- Event catalogs (GWTC-1, GWTC-2, GWTC-3, GWTC-4) with metadata
- Strain time series data h(t) from detectors (dimensionless)
- GPS timestamps and sample rates (Hz)
- Integration with scientific Python ecosystem (gwpy, numpy, h5py)

dimtensor can provide dimensional safety for GW analysis by ensuring:
- Strain data is properly marked as dimensionless
- Time coordinates have proper time units
- Frequency data has Hz units
- Event parameters (mass, distance) have correct dimensions

---

## Approach

### Option A: Direct GWOSC API Integration
- Use the `gwosc` Python package (official GWOSC client)
- Wrap gwosc.datasets and gwosc.api functions
- Convert gwpy TimeSeries to DimArray
- Pros:
  - Official API client is maintained and documented
  - Handles authentication, caching, and error handling
  - Already integrated with gwpy ecosystem
  - Supports all GWOSC features (events, runs, datasets)
- Cons:
  - Adds dependency on gwosc package
  - Additional layer of abstraction

### Option B: Raw HTTP API
- Implement direct HTTP calls to GWOSC REST API
- Parse JSON/HDF5 responses manually
- Build our own caching layer
- Pros:
  - No external dependencies beyond requests
  - Full control over implementation
- Cons:
  - Reimplementing functionality that gwosc provides
  - More maintenance burden
  - API changes require updates

### Decision: Option A (gwosc package)

Use the official `gwosc` Python client because:
1. It's the recommended approach by GWOSC documentation
2. Handles API versioning (v1 → v2 migration)
3. Provides robust error handling and retry logic
4. Already integrates with gwpy for strain data access
5. dimtensor's role is dimensional safety, not HTTP client implementation

We'll make `gwosc` an optional dependency (in `[all]` extra) since not all users need GW data.

---

## Implementation Steps

1. [ ] Add `gwosc>=0.8.0` to optional dependencies in pyproject.toml
2. [ ] Create `src/dimtensor/datasets/loaders/gravitational_wave.py`
3. [ ] Implement `GWOSCEventLoader` class (inherits from `BaseLoader`)
   - `load_catalog(catalog='GWTC-3')` → dict of event parameters
   - Convert masses to solar mass units
   - Convert distances to megaparsecs
   - Include GPS times as time dimension
4. [ ] Implement `GWOSCStrainLoader` class (inherits from `BaseLoader`)
   - `load_strain(event, detector, duration, sample_rate)` → DimArray
   - Convert gwpy TimeSeries to DimArray with dimensionless strain
   - Add time axis with GPS time origin
   - Include metadata (detector, event name, quality flags)
5. [ ] Add gravitational wave units to module
   - Define `strain = dimensionless` with symbol "strain"
   - Add solar_mass, parsec units if not in astronomy module
   - Define hertz unit for sample rates
6. [ ] Update `src/dimtensor/datasets/loaders/__init__.py` exports
7. [ ] Create tests in `tests/test_dataset_loaders.py`
   - Mock gwosc API calls to avoid network dependencies
   - Test event catalog parsing with sample JSON
   - Test strain data conversion with synthetic data
   - Test error handling (missing gwosc package, network errors)
8. [ ] Add usage examples to docstrings
9. [ ] Update CONTINUITY.md with completion status

---

## Files to Modify

| File | Change |
|------|--------|
| pyproject.toml | Add `gwosc>=0.8.0` to `all` optional dependencies |
| src/dimtensor/datasets/loaders/gravitational_wave.py | New file: GWOSCEventLoader, GWOSCStrainLoader classes |
| src/dimtensor/datasets/loaders/__init__.py | Export new loaders |
| tests/test_dataset_loaders.py | Add TestGWOSCEventLoader, TestGWOSCStrainLoader classes |
| src/dimtensor/domains/units/astronomy.py (if exists) | Add solar_mass, parsec if not present |

---

## Testing Strategy

### Unit Tests
- [ ] Test GWOSCEventLoader.load_catalog() with mocked gwosc responses
  - Verify dict keys (name, mass_1, mass_2, distance, GPS, etc.)
  - Check dimensional units (solar mass, Mpc, seconds)
  - Handle missing/null values gracefully
- [ ] Test GWOSCStrainLoader.load_strain() with synthetic data
  - Mock gwpy TimeSeries → DimArray conversion
  - Verify strain is dimensionless
  - Check time axis has time dimension
  - Validate metadata preservation
- [ ] Test graceful failure when gwosc not installed
  - Should raise ImportError with helpful message
- [ ] Test caching behavior
  - Verify strain data cached correctly
  - Test force re-download flag

### Integration Tests (marked with @pytest.mark.network)
- [ ] Real download of GW150914 event metadata
- [ ] Small strain data segment download (1 second max)
- [ ] Verify compatibility with gwpy data

### Manual Verification
- [ ] Load GWTC-3 catalog and plot mass distribution
- [ ] Download GW150914 strain and reproduce chirp plot
- [ ] Verify dimensional operations (e.g., FFT, filtering)

---

## Risks / Edge Cases

- **Risk**: gwosc API changes or deprecations
  - **Mitigation**: Pin to gwosc>=0.8.0 which uses API v2. Document version requirements.

- **Risk**: Large strain data downloads (hours of data at 16384 Hz)
  - **Mitigation**: Require explicit duration parameter. Warn if download >10MB. Default to 32 seconds around event.

- **Risk**: HDF5 file format changes
  - **Mitigation**: Use gwosc's built-in parsers rather than raw h5py access.

- **Edge case**: Missing detector data for some events
  - **Handling**: Check detector availability via gwosc.datasets before attempting download. Return None or raise clear error.

- **Edge case**: Event names vs GPS times
  - **Handling**: Support both event names ("GW150914") and GPS times. Convert names to GPS using gwosc.datasets.event_gps().

- **Edge case**: Data quality flags
  - **Handling**: Include DQFlags in metadata but don't filter by default. Let users apply quality cuts.

- **Edge case**: Different sample rates (4096 vs 16384 Hz)
  - **Handling**: Make sample_rate an explicit parameter. Default to highest available.

---

## Definition of Done

- [ ] GWOSCEventLoader loads event catalogs with proper units
- [ ] GWOSCStrainLoader converts gwpy strain data to DimArray
- [ ] All tests pass (including mocked network tests)
- [ ] Docstrings include usage examples
- [ ] gwosc added as optional dependency
- [ ] Integration with existing loaders/ module structure
- [ ] CONTINUITY.md updated with completion

---

## Notes / Log

**2026-01-12 13:00** - Research complete. Key findings:
- GWOSC has official Python client (gwosc package)
- Strain is dimensionless (fractional length change)
- Event catalogs available as GWTC-1 through GWTC-4
- gwpy provides TimeSeries class that we'll convert to DimArray
- API is read-only GET requests, no authentication needed

**Key GWOSC API Endpoints:**
- `/api/v1/catalog/` - List available catalogs
- `/api/v1/catalog/{name}/` - Get catalog events
- `/api/v1/events/{event}/` - Get single event details
- Strain data via gwosc.datasets.fetch_urls() → gwpy.TimeSeries.fetch()

**Dimensional Units Needed:**
- `strain` = dimensionless (represented as "1" or "strain")
- `solar_mass` = 1.989e30 kg
- `megaparsec` = 3.0857e22 m
- `hertz` = 1/second

**Integration Points:**
- Follow BaseLoader pattern from existing loaders (NIST, NASA, NOAA)
- Use download() method for caching
- Return dict[str, DimArray] format for consistency
- Support both cache=True and force_download flags

---

## References

### GWOSC Resources
- [GWOSC Public API Documentation](https://gwosc.org/api/v1/docs/)
- [GWOSC API](https://gwosc.org/api/)
- [gwosc Python Package Docs](https://gwosc.readthedocs.io/en/stable/)

### GWpy Integration
- [Accessing Open Data from GWOSC - GWpy](https://gwpy.github.io/docs/stable/timeseries/opendata/)
- [GWpy TimeSeries Documentation](https://gwpy.github.io/docs/stable/timeseries/)

### Example Usage Pattern
```python
from dimtensor.datasets.loaders import GWOSCEventLoader, GWOSCStrainLoader

# Load event catalog
event_loader = GWOSCEventLoader()
events = event_loader.load_catalog(catalog='GWTC-3')
print(f"Mass of primary: {events['mass_1_source'][0]}")  # DimArray in solar masses

# Load strain data
strain_loader = GWOSCStrainLoader()
strain = strain_loader.load_strain(
    event='GW150914',
    detector='H1',  # Hanford
    duration=32,    # seconds around event
    sample_rate=4096  # Hz
)
print(f"Strain shape: {strain.shape}")
print(f"Strain unit: {strain.unit}")  # dimensionless
```

---
