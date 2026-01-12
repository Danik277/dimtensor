# Plan: Materials Project Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a dataset loader for the Materials Project database that fetches crystal structures and material properties (band gap, formation energy, elastic constants) via the mp-api client library and converts them to DimArray format with proper units.

---

## Background

The Materials Project (materialsproject.org) is a comprehensive database of computed material properties using density functional theory (DFT). It provides access to crystal structures (CIF format), electronic properties (band gap), thermodynamic data (formation energy in eV/atom), and mechanical properties (elastic tensor, bulk/shear moduli in GPa). The mp-api Python library provides the MPRester client for querying this data.

Key resources:
- [Materials Project API Documentation](https://docs.materialsproject.org/downloading-data/using-the-api)
- [mp-api PyPI package](https://pypi.org/project/mp-api/)
- [Materials Project Electronic Structure Methodology](https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure)
- [Materials Project FAQ](https://docs.materialsproject.org/frequently-asked-questions)

dimtensor already has loaders for NASA Exoplanet Archive (astronomy.py), NIST CODATA (nist.py), and climate data. The Materials Project loader will follow similar patterns using BaseLoader with caching support.

---

## Approach

### Option A: Minimal loader (properties only, no structure)

- Query basic material properties only (band gap, formation energy, bulk/shear moduli)
- Return dict of DimArrays with proper units
- No crystal structure parsing (simpler, faster)
- Pros: Easy to implement, sufficient for ML workflows needing property data
- Cons: Missing structural information which is valuable for many use cases

### Option B: Full loader (properties + crystal structure)

- Query both properties and crystal structures
- Parse structure data (lattice, atomic positions, composition)
- Return both DimArrays (properties with units) and structure objects
- Pros: Complete dataset, enables structure-property ML
- Cons: More complex, requires structure parsing

### Option C: Hybrid with optional structure

- Query properties by default
- Optional flag to include structures
- Structures returned as JSON/dict format (let user parse with pymatgen if needed)
- Pros: Flexible, doesn't add pymatgen as dependency
- Cons: User needs to handle structure parsing

### Decision: Option C (Hybrid)

Start with Option C for maximum flexibility without adding heavy dependencies. Load properties as DimArrays with units, optionally include raw structure data as JSON. Users who need advanced structure analysis can use pymatgen separately.

Key architectural decisions:
1. Use BaseLoader for caching support (similar to NASAExoplanetLoader, NISTCODATALoader)
2. API key authentication via environment variable (MATERIALS_PROJECT_API_KEY) or constructor
3. Query by material ID, formula, or chemical system
4. Return properties as DimArrays with proper units (eV, GPa, angstrom)
5. mp-api as optional dependency (extras_require)

---

## Implementation Steps

1. [ ] Add mp-api to setup.py extras_require (materials group)
2. [ ] Create src/dimtensor/datasets/loaders/materials_project.py
3. [ ] Implement MaterialsProjectLoader(BaseLoader):
   - [ ] __init__ with API key (from env or param)
   - [ ] load() method with query parameters
   - [ ] _query_materials() helper (calls mp-api)
   - [ ] _convert_to_dimarrays() helper (parse properties)
   - [ ] Optional structure loading flag
4. [ ] Define unit mappings for Materials Project data:
   - [ ] eV for energies (already in core.units)
   - [ ] angstrom for lengths (create if missing)
   - [ ] GPa for elastic constants (already in domains.materials)
   - [ ] dimensionless for band gap (eV)
5. [ ] Add to src/dimtensor/datasets/loaders/__init__.py exports
6. [ ] Write tests/test_materials_project_loader.py:
   - [ ] Test with mock API responses (avoid real API calls in tests)
   - [ ] Test unit conversions
   - [ ] Test caching behavior
   - [ ] Test API key handling
   - [ ] Test optional structure loading
7. [ ] Update documentation/examples

---

## Files to Modify

| File | Change |
|------|--------|
| setup.py | Add mp-api to extras_require (materials group) |
| src/dimtensor/datasets/loaders/materials_project.py | Create new loader class |
| src/dimtensor/datasets/loaders/__init__.py | Export MaterialsProjectLoader |
| src/dimtensor/core/units.py OR domains/materials.py | Add angstrom unit if missing |
| tests/test_materials_project_loader.py | Create comprehensive test suite |
| README.md OR docs/ | Add Materials Project loader example |

---

## Testing Strategy

### Unit tests
- [ ] Test API key loading from environment variable
- [ ] Test query by material ID (e.g., "mp-149" for Si)
- [ ] Test query by formula (e.g., "Fe2O3")
- [ ] Test query by chemical system (e.g., "Si-O")
- [ ] Test property conversion to DimArrays with correct units
- [ ] Test optional structure loading flag
- [ ] Test caching mechanism (check cache hit/miss)
- [ ] Test error handling (missing API key, invalid material ID, API timeout)

### Integration tests (manual, with real API key)
- [ ] Query real material (e.g., "mp-149" for Silicon)
- [ ] Verify band gap in eV
- [ ] Verify formation energy in eV/atom
- [ ] Verify elastic constants in GPa
- [ ] Verify structure data (if loaded)
- [ ] Verify caching works across multiple runs

### Mock API responses for automated tests
- Create fixtures with representative MP API response JSON
- Test parsing without hitting real API

---

## Risks / Edge Cases

**Risk 1**: mp-api API changes or rate limiting
- Mitigation: Use caching aggressively, document API key setup clearly, handle HTTP errors gracefully

**Risk 2**: Missing data (not all materials have all properties)
- Mitigation: Return None or np.nan for missing properties, document clearly

**Risk 3**: API key security
- Mitigation: Load from environment variable (never hardcode), add to .gitignore examples, document best practices

**Risk 4**: Large query results (thousands of materials)
- Mitigation: Support pagination or limit results, document memory considerations

**Risk 5**: Unit confusion (eV vs J, GPa vs Pa)
- Mitigation: Clear documentation, test conversions thoroughly, use descriptive variable names

**Edge case 1**: Materials with multiple polymorphs (same formula, different structures)
- Handling: Return list of materials, let user filter by specific material ID

**Edge case 2**: Deprecated or updated material IDs
- Handling: Document that mp-api handles redirects, test with known deprecated ID

**Edge case 3**: Experimental vs theoretical properties
- Handling: Materials Project is primarily DFT computed data, document data source

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (both unit tests with mocks and manual integration test)
- [ ] Documentation updated with usage example
- [ ] Can query materials by ID, formula, and chemical system
- [ ] Properties returned as DimArrays with correct units (eV, GPa, angstrom)
- [ ] Caching works correctly
- [ ] API key authentication works via environment variable
- [ ] Optional structure loading implemented
- [ ] CONTINUITY.md updated

---

## Notes / Log

**[2026-01-12]** - Plan created by planner agent

Key findings from research:
- mp-api is the official Python client (version 0.45.15 as of 2026-01)
- MPRester is the main API client class
- API key required (free registration at materialsproject.org)
- Band gaps systematically underestimated by ~40% (PBE limitation)
- Formation energies include MP2020 correction scheme
- Elastic tensor available as full tensor or derived properties (K, G moduli)
- Structures exported as JSON, CIF, or POSCAR formats
- dimtensor already has GPa unit in domains.materials module
- dimtensor already has eV unit in core.units module
- Need to add angstrom unit (1 Å = 1e-10 m)

Integration pattern follows existing loaders:
- BaseLoader provides caching via self.download()
- Return dict[str, DimArray] for properties
- Handle missing values gracefully (np.nan)
- Optional force_download kwarg

---
