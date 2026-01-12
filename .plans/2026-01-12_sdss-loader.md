# Plan: SDSS (Sloan Digital Sky Survey) Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a data loader for the Sloan Digital Sky Survey (SDSS) that provides access to galaxy catalogs and astronomical data through the SDSS SkyServer API and CasJobs interface, returning data as DimArrays with appropriate astronomical units.

---

## Background

The SDSS is one of the most comprehensive astronomical surveys, containing imaging and spectroscopic data for hundreds of millions of celestial objects. Scientists need to query this data programmatically and work with it in a physics-aware framework. This loader will enable users to:

1. Query galaxy catalogs (positions, redshifts, magnitudes, colors)
2. Retrieve spectroscopic data with proper units
3. Access imaging metadata with dimensional quantities
4. Build common astronomical queries without SQL knowledge

The loader will follow the existing pattern established by NASAExoplanetLoader and integrate with the astronomy units module.

---

## Approach

### Option A: Direct SkyServer SQL Queries

- Description: Use SDSS SkyServer's public SQL interface to query the database directly
- Pros:
  - Direct access to full catalog
  - Most flexible for custom queries
  - No rate limits on public queries
  - Can combine multiple tables (PhotoObj, SpecObj, etc.)
- Cons:
  - Requires SQL knowledge for custom queries
  - API could change between data releases
  - Need to handle large result sets

### Option B: Pre-defined Query Templates

- Description: Provide high-level functions (get_galaxies, get_quasars) that abstract SQL
- Pros:
  - Easier for users unfamiliar with SDSS schema
  - Can optimize common queries
  - Less error-prone
- Cons:
  - Limited to pre-defined queries
  - Less flexible
  - Need to maintain templates for each data release

### Option C: Hybrid Approach (Selected)

- Description: Provide both query builder for common patterns AND raw SQL execution capability
- Pros:
  - User-friendly defaults with query builder methods
  - Power users can write custom SQL
  - Can evolve query templates over time
  - Follows pattern from similar astronomy tools (astroquery)
- Cons:
  - Slightly more complex implementation
  - Need to test both paths

### Decision: Option C - Hybrid Approach

We'll implement a hybrid approach that provides:
1. High-level query builder methods for common use cases (get_galaxies, radial_search, spectroscopic_sample)
2. Raw SQL execution method for advanced users (execute_query)
3. Helper methods to construct common WHERE clauses (magnitude_range, redshift_range, etc.)

This matches the pattern used by astroquery and provides the best balance of usability and flexibility.

---

## Implementation Steps

1. [ ] Create base SDSSLoader class extending CSVLoader
   - Implement download and caching for query results
   - Add data release selection (DR17 default, configurable)
   - Implement execute_query() for raw SQL

2. [ ] Implement query builder methods
   - get_galaxies(ra, dec, radius, redshift_range, magnitude_range)
   - get_spectroscopy(obj_ids or coordinates)
   - radial_search(ra, dec, radius) - cone search
   - rectangular_search(ra_min, ra_max, dec_min, dec_max)

3. [ ] Create unit mapping for SDSS columns
   - ra, dec -> arcsec (or degrees as dimensionless radians)
   - z (redshift) -> dimensionless
   - u, g, r, i, z magnitudes -> dimensionless (magnitudes are logarithmic)
   - petroRad, deVRad -> arcsec
   - distances -> Mpc (if calculated)
   - stellar masses -> solar_mass
   - velocities -> km/s

4. [ ] Implement data parsing and DimArray conversion
   - Parse CSV results from SkyServer
   - Handle missing values (NaN, -9999, etc.)
   - Convert to appropriate DimArrays with astronomy units
   - Support both PhotoObj and SpecObj tables

5. [ ] Add helper functions for common filters
   - magnitude_range(band, min_mag, max_mag) -> SQL WHERE clause
   - redshift_range(z_min, z_max) -> SQL WHERE clause
   - clean_sample() -> standard quality cuts
   - main_galaxy_sample() -> SDSS MGS criteria

6. [ ] Create example queries and documentation
   - Document available columns and their units
   - Provide example notebooks
   - Link to SDSS schema browser

7. [ ] Register dataset in registry.py
   - Add sdss_galaxies dataset entry
   - Add sdss_spectroscopy dataset entry
   - Include proper citations (SDSS collaboration papers)

8. [ ] Write tests
   - Test query builder methods
   - Test unit conversions
   - Test error handling (empty results, invalid SQL)
   - Mock HTTP requests for CI/CD

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/datasets/loaders/sdss.py | NEW: Create SDSSLoader class with query builder |
| src/dimtensor/datasets/loaders/__init__.py | Add SDSSLoader to exports |
| src/dimtensor/datasets/registry.py | Register sdss_galaxies and sdss_spectroscopy datasets |
| src/dimtensor/domains/astronomy.py | Add magnitude unit if needed (dimensionless but special) |
| tests/datasets/loaders/test_sdss.py | NEW: Create comprehensive tests |
| docs/examples/sdss_queries.ipynb | NEW: Example notebook (optional) |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for query builder
  - Test SQL generation for each query method
  - Verify WHERE clause helpers produce correct SQL
  - Test parameter validation

- [ ] Integration tests with mocked HTTP
  - Mock SkyServer responses
  - Test parsing of real SDSS CSV format
  - Verify DimArray creation with correct units
  - Test caching behavior

- [ ] Manual verification with real queries
  - Execute small cone search (< 100 objects)
  - Verify results match SDSS Navigate tool
  - Check units match expected values
  - Verify redshifts and magnitudes are reasonable

- [ ] Edge case tests
  - Empty result sets
  - Very large result sets (pagination)
  - Invalid coordinates (RA > 360)
  - Missing/null values in results
  - Network errors and timeouts

---

## Risks / Edge Cases

- **Risk: API changes between data releases**
  - Mitigation: Make data release configurable, default to stable DR17
  - Document which DR is supported
  - Add version checking in code

- **Risk: Large query results exhaust memory**
  - Mitigation: Warn users about query size
  - Implement row limit with default (e.g., 10000)
  - Consider pagination for very large queries
  - Use streaming for huge datasets

- **Risk: Rate limiting or API timeouts**
  - Mitigation: Add retry logic with exponential backoff
  - Cache all results aggressively
  - Implement timeout parameter (default 60s)

- **Edge case: Objects near RA=0/360 boundary**
  - Handling: Use SDSS's built-in fGetNearbyObjEq function for cone searches
  - Document coordinate wrapping behavior

- **Edge case: Magnitude values are logarithmic (not linear physical units)**
  - Handling: Store as dimensionless, document that they're in AB magnitude system
  - Consider adding magnitude <-> flux conversion utilities
  - Note: SDSS uses asinh magnitudes for faint objects

- **Edge case: Missing spectroscopy**
  - Handling: PhotoObj queries always work, but SpecObj may be empty
  - Provide clear error messages when spectroscopy unavailable
  - Document spectroscopic completeness

- **Edge case: Different coordinate systems (J2000 vs Galactic)**
  - Handling: Default to J2000 (SDSS standard)
  - Document coordinate system in metadata
  - Could add coordinate transformation helpers later

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] SDSSLoader class works with both query builder and raw SQL
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests work with mocked data
- [ ] Manual verification with 3+ real queries successful
- [ ] Documentation includes example usage
- [ ] Dataset registered in registry
- [ ] CONTINUITY.md updated

---

## Notes / Log

### API Endpoints

**SkyServer SQL Search** (primary interface):
- Base URL: `http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch`
- Parameters: `?cmd=<SQL>&format=csv`
- Example: `SELECT TOP 100 ra, dec, u, g, r, i, z FROM PhotoObj WHERE ...`

**CasJobs** (for large batch queries):
- Not implementing in v1 - requires authentication
- Future enhancement for million+ object queries

**Key SDSS Tables**:
- `PhotoObj` - Photometric measurements (positions, magnitudes)
- `SpecObj` - Spectroscopic data (redshifts, spectral classifications)
- `Galaxy` - Galaxy-specific measurements (often join PhotoObj)
- `galSpecExtra` - Additional derived galaxy properties

**Important Columns**:
- `objID` - Unique object identifier (64-bit integer)
- `ra`, `dec` - Right ascension, declination (degrees, J2000)
- `u`, `g`, `r`, `i`, `z` - SDSS magnitudes (AB system)
- `z`, `zErr` - Redshift and uncertainty (dimensionless)
- `petroRad_r` - Petrosian radius in r-band (arcsec)
- `specClass` - Spectral classification (1=galaxy, 2=star, 3=QSO)

### SDSS Citation

Users of this loader should cite:
```
Abdurro'uf et al., "The Seventeenth Data Release of the Sloan Digital Sky Surveys"
ApJS 259 35 (2022). DOI: 10.3847/1538-4365/ac4414
```

---
