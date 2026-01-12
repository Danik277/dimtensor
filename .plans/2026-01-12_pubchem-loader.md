# Plan: PubChem Compound Data Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Create a loader module for PubChem chemical compound data that fetches molecular properties via the PUG REST API and converts them to DimArrays with appropriate chemistry units (dalton, angstrom, kelvin, pascal).

---

## Background

PubChem is NCBI's open chemistry database containing millions of chemical compounds with computed properties. Having a loader enables dimtensor users to:
- Fetch molecular properties (MW, boiling point, density, etc.) with proper units
- Integrate real chemical data into scientific ML workflows
- Demonstrate dimtensor's chemistry domain capabilities

The PUG REST API provides programmatic access with simple URL-based queries.

---

## Approach

### Option A: Thin wrapper around raw REST API
- Description: Direct HTTP calls, minimal abstractions, return raw dicts with DimArrays
- Pros:
  - Simple implementation
  - Flexible - users can request any property
  - Low maintenance
- Cons:
  - Less user-friendly API
  - No validation of property names
  - Users need to understand PubChem API

### Option B: High-level interface with typed methods
- Description: Methods like `get_compound()`, `get_molecular_weight()`, with predefined property mappings
- Pros:
  - Cleaner API for common use cases
  - Type hints and validation
  - Better error messages
- Cons:
  - More code to maintain
  - Less flexible - limited to predefined properties
  - Redundant with PubChem's own API design

### Decision: **Hybrid Approach (A with B's best features)**

Provide a `PubChemLoader` that:
1. Has a generic `get_properties(identifiers, properties)` method (Option A flexibility)
2. Includes convenience methods `get_compound_by_cid()`, `get_compound_by_name()` (Option B usability)
3. Maps common properties to DimArray units automatically
4. Returns dict with both raw values and DimArray objects where appropriate

This balances flexibility with usability, following the pattern established in `NISTCODATALoader`.

---

## Implementation Steps

1. [ ] Create `src/dimtensor/datasets/loaders/pubchem.py` with `PubChemLoader` class
2. [ ] Implement base URL construction and request handling with rate limiting (5 req/sec max)
3. [ ] Add `get_properties()` method supporting CID, name, InChI, SMILES identifiers
4. [ ] Create property-to-unit mapping for common properties:
   - MolecularWeight → dalton
   - Volume → angstrom³
   - MeltingPoint, BoilingPoint → kelvin
   - Density → kg/m³
   - Pressure → pascal
5. [ ] Implement convenience methods:
   - `get_compound_by_cid(cid: int)` → dict with common properties
   - `get_compound_by_name(name: str)` → dict with common properties
   - `search_compounds(query: str, max_results: int)` → list of CIDs
6. [ ] Add JSON response parsing with error handling for missing properties
7. [ ] Integrate caching via `BaseLoader.download()` mechanism
8. [ ] Add `PubChemLoader` to `loaders/__init__.py` exports
9. [ ] Update docstrings with examples
10. [ ] Create comprehensive tests in `tests/datasets/loaders/test_pubchem.py`

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/loaders/pubchem.py` | Create new loader class |
| `src/dimtensor/datasets/loaders/__init__.py` | Add `PubChemLoader` to exports |
| `tests/datasets/loaders/test_pubchem.py` | Create test suite |
| `docs/examples/chemistry_loader.py` | (Optional) Add usage example |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for URL construction and property mapping
- [ ] Integration test fetching real compound (e.g., CID=2244 for aspirin)
- [ ] Test multiple identifiers (CID list, compound names)
- [ ] Test error handling (invalid CID, network failure, missing properties)
- [ ] Test caching mechanism (verify file cached, reuse on second call)
- [ ] Test rate limiting (mock to verify delay between requests)
- [ ] Verify DimArray units for all supported properties
- [ ] Test with missing/null properties (some compounds lack certain data)

---

## Risks / Edge Cases

- **Risk**: PubChem API rate limit (5 req/sec). **Mitigation**: Implement request throttling with time tracking between calls.
- **Risk**: API downtime or network failures. **Mitigation**: Use try/except, return fallback/cached data, clear error messages.
- **Risk**: Property values may be null/missing for some compounds. **Mitigation**: Return None for missing properties, document this behavior.
- **Edge case**: Compound names can be ambiguous (multiple CIDs). **Handling**: `get_compound_by_name()` returns first match, add `search_compounds()` for disambiguation.
- **Edge case**: Temperature properties may be in Celsius or Kelvin. **Handling**: Document that PubChem returns Celsius; convert to Kelvin when creating DimArray.
- **Edge case**: Some properties are dimensionless (pH, LogP). **Handling**: Return as regular Python floats or dimensionless DimArray.
- **Risk**: JSON response format changes. **Mitigation**: Add version checking, robust parsing with fallbacks.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass with >90% coverage for pubchem.py
- [ ] Documentation includes usage examples with real compound
- [ ] CONTINUITY.md updated with task completion
- [ ] Can successfully fetch and parse properties for test compounds
- [ ] Rate limiting verified to respect 5 req/sec limit
- [ ] Caching works correctly (no redundant API calls)

---

## Notes / Log

**Research Notes:**

PubChem PUG REST API Structure:
- Base URL: `https://pubchem.ncbi.nlm.nih.gov/rest/pug`
- Format: `/{input_type}/{identifier}/{operation}/{output_format}`
- Example: `/compound/cid/2244/property/MolecularWeight,MolecularFormula/JSON`

Key Properties Available:
- MolecularFormula, MolecularWeight
- CanonicalSMILES, IsomericSMILES
- InChI, InChIKey
- IUPACName
- XLogP, TPSA, Complexity
- HBondDonorCount, HBondAcceptorCount
- HeavyAtomCount, RotatableBondCount
- Volume, MeltingPoint, BoilingPoint, Density

Supported Input Types:
- CID (Compound ID) - numeric identifier
- Name - common/IUPAC name (requires lookup)
- InChI, InChIKey - structure identifiers
- SMILES - structure notation

Chemistry Units Available in dimtensor:
- dalton (atomic mass unit) - for MolecularWeight
- angstrom - for molecular dimensions/volume (as angstrom³)
- kelvin - for temperature (melting/boiling points)
- pascal - for pressure properties
- molar - for concentration (not directly from PubChem, but useful for solutions)

Pattern from NISTCODATALoader:
- Inherits from BaseLoader
- Uses self.download() for caching
- Returns dict[str, DimArray]
- Has fallback for offline usage
- Parses response and maps to units

---
