# Plan: CERN Open Data Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Create a dataset loader for CERN Open Data Portal that downloads and parses particle physics datasets (CMS, ATLAS, LHCb) in ROOT/NanoAOD format, converting them into DimArrays with proper units (GeV, MeV, barn, etc.) for ML and analysis.

---

## Background

CERN Open Data Portal (opendata.cern.ch) provides open access to particle physics datasets from LHC experiments. These datasets contain collision event data with particle properties (energy, momentum, mass, cross-sections) that are ideal for physics-aware ML research. The data is typically in ROOT format (analysis framework used by particle physics) or simplified NanoAOD format.

Key characteristics:
- **Data formats**: ROOT TTrees, NanoAOD (simplified ROOT), some CSV/JSON metadata
- **Experiments**: CMS, ATLAS, LHCb, ALICE
- **Units**: GeV/MeV for energy/momentum/mass, barn/mb for cross-sections
- **Access**: Public API via cernopendata-client, no authentication required
- **File sizes**: Can be very large (GB-scale ROOT files)

dimtensor already has:
- Energy units (GeV, MeV, eV, TeV) in domains/natural.py and domains/nuclear.py
- Cross-section units (barn, mb, ub) in domains/nuclear.py
- Loader infrastructure (BaseLoader, CSVLoader) in datasets/loaders/base.py
- Caching system (~/.dimtensor/cache/)

---

## Approach

### Option A: NanoAOD Focus (Simplified Format)

Use NanoAOD files as the primary target since they:
- Are standard C++ types in ROOT TTrees (easier to parse)
- Don't require CMS VM or Docker containers
- Can be read with uproot (pure Python ROOT I/O)
- Contain common physics objects ready for analysis

**Pros:**
- No C++ dependencies or VMs required
- uproot is pure Python, well-maintained (Scikit-HEP)
- NanoAOD is designed for analysis (not raw detector data)
- Smaller file sizes than full AOD

**Cons:**
- NanoAOD only available for recent runs (2015+)
- Less detailed than full AOD format
- Requires uproot dependency

### Option B: Full AOD Support (Complex Format)

Support full AOD (Analysis Object Data) files with CMS-specific C++ classes.

**Pros:**
- Access to all CERN Open Data releases (2010+)
- More detailed physics information

**Cons:**
- Requires CMS software environment or Docker containers
- Much more complex to parse
- Large file sizes (100s of GB)
- Not practical for most users

### Option C: Metadata + CSV Exports Only

Only support CSV/JSON exports and metadata from the API.

**Pros:**
- Simple implementation
- No ROOT dependencies

**Cons:**
- Most interesting data is in ROOT format
- Very limited physics content
- Not useful for real particle physics ML

### Decision: **Option A (NanoAOD + metadata)**

Focus on NanoAOD files using uproot for ROOT I/O. This provides the best balance of:
- Practical usability (no VMs/containers)
- Rich physics content (particle 4-vectors, jet properties, etc.)
- Python-native tools (uproot, awkward arrays)
- Reasonable file sizes

We'll also support metadata queries via cernopendata-client for dataset discovery.

---

## Implementation Steps

1. [ ] Add optional dependencies: `uproot`, `awkward`, `cernopendata-client`
2. [ ] Create `CERNOpenDataLoader` class in `datasets/loaders/cern.py`
   - Inherit from `BaseLoader`
   - Implement metadata query methods (list datasets, get record info)
   - Implement NanoAOD file download and caching
3. [ ] Implement NanoAOD parsing with unit conversion
   - Use uproot to read ROOT TTrees
   - Extract particle physics branches (pt, eta, phi, mass, energy)
   - Convert to DimArrays with proper units:
     - `pt`, `energy`, `mass` → GeV or MeV
     - Cross-sections → barn/mb
     - Angles (eta, phi) → dimensionless
4. [ ] Add physics object extractors
   - Electrons, muons, jets, photons, MET (missing energy)
   - Each returns dict of DimArrays with proper dimensions
5. [ ] Implement smart caching
   - Cache metadata queries (JSON)
   - Cache downloaded ROOT files (with size limits?)
   - Store parsed arrays in HDF5/NPZ for faster reload
6. [ ] Add example usage and documentation
   - Jupyter notebook showing CMS NanoAOD analysis
   - Integration with dimtensor ML models
7. [ ] Update package dependencies and __init__.py
8. [ ] Write comprehensive tests

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/loaders/cern.py` | **CREATE**: Main loader implementation |
| `src/dimtensor/datasets/loaders/__init__.py` | Add CERNOpenDataLoader export |
| `pyproject.toml` | Add optional dependencies: uproot, awkward, cernopendata-client |
| `README.md` | Add CERN Open Data loader to features list |
| `tests/datasets/test_cern_loader.py` | **CREATE**: Unit tests with mocked data |
| `examples/notebooks/cern_nanoaod_analysis.ipynb` | **CREATE** (optional): Example usage |

---

## Testing Strategy

### Unit Tests
- [ ] Test metadata queries (mock cernopendata-client responses)
- [ ] Test NanoAOD parsing (use small test ROOT file or mock)
- [ ] Test unit conversion (GeV, MeV, barn)
- [ ] Test caching behavior (download, reuse, clear)
- [ ] Test physics object extraction (electrons, jets, etc.)

### Integration Tests
- [ ] Download real (small) NanoAOD file from CERN portal
- [ ] Verify parsed data has correct dimensions
- [ ] Verify DimArray operations work (energy sums, mass calculations)

### Manual Verification
- [ ] Load CMS NanoAOD dataset
- [ ] Plot particle momentum distributions
- [ ] Calculate invariant masses (Z → ee, etc.)
- [ ] Verify units convert correctly to SI

---

## Risks / Edge Cases

**Risk 1: Large file sizes**
- NanoAOD files can be 1-10 GB each
- **Mitigation**:
  - Implement download progress bars
  - Add file size warnings
  - Support chunk/event range loading
  - Document cache directory cleanup

**Risk 2: uproot/awkward version compatibility**
- ROOT file formats evolve, uproot compatibility varies
- **Mitigation**:
  - Pin known-good versions in pyproject.toml
  - Document supported ROOT versions
  - Add version checking in loader

**Risk 3: Complex jagged arrays**
- Particle physics data has variable-length collections (jets per event)
- **Mitigation**:
  - Use awkward arrays for jagged data
  - Flatten arrays when converting to DimArray (or support ragged)
  - Document limitations clearly

**Risk 4: cernopendata-client API changes**
- API is external dependency
- **Mitigation**:
  - Make API client optional (direct HTTP fallback)
  - Cache metadata aggressively
  - Document manual download procedure

**Edge Case: Missing branches**
- Not all NanoAOD files have same structure
- **Handling**: Graceful fallback, return None for missing branches

**Edge Case: Natural units vs SI**
- Particle physics often uses natural units (c=ℏ=1)
- **Handling**:
  - Default to SI-compatible units (GeV → J equivalent via scale factor)
  - Support natural units mode via flag
  - Document unit conventions clearly

**Edge Case: Unit inference ambiguity**
- Some quantities (eta, phi) are dimensionless but have meaning
- **Handling**: Use descriptive variable names, document conventions

---

## Definition of Done

- [ ] CERNOpenDataLoader class implemented and tested
- [ ] Can download and parse NanoAOD files to DimArrays
- [ ] Units correctly assigned (GeV, MeV, barn)
- [ ] Physics objects (electrons, jets) extractable
- [ ] Tests pass (unit + integration)
- [ ] Documentation includes usage example
- [ ] Optional dependencies documented
- [ ] CONTINUITY.md updated

---

## Notes / Log

**Research findings:**

1. **CERN Open Data Portal API**
   - REST API via Invenio framework
   - cernopendata-client provides Python CLI
   - No authentication required (public data)
   - Metadata in JSON format

2. **Data formats**
   - AOD: Full format, requires CMS software (complex)
   - NanoAOD: Simplified, standard C++ types (preferred)
   - ROOT TTrees: Hierarchical data structure
   - uproot: Pure Python ROOT I/O (Scikit-HEP)

3. **Units in particle physics**
   - Energy/momentum/mass: GeV, MeV, TeV
   - Cross-sections: barn = 10⁻²⁸ m²
   - Natural units: c=ℏ=1 (mass~energy~momentum)
   - Already available in dimtensor.domains.nuclear and .natural

4. **Existing dimtensor patterns**
   - BaseLoader provides download/cache infrastructure
   - CSVLoader shows parsing pattern
   - NASAExoplanetLoader is good reference for astronomy data

5. **Scikit-HEP ecosystem**
   - uproot: ROOT I/O
   - awkward: Jagged/nested arrays
   - vector: Lorentz vectors (could integrate later)
   - hepunits: Units library (overlap with dimtensor)

---

## References

- [CERN Open Data Portal](https://opendata.cern.ch/)
- [cernopendata-client docs](https://cernopendata-client.readthedocs.io/en/latest/)
- [CMS NanoAOD Getting Started](https://opendata.cern.ch/docs/cms-getting-started-nanoaod)
- [uproot documentation](https://uproot.readthedocs.io/)
- [Scikit-HEP project](https://scikit-hep.org/)
- [Particle physics units (PDG)](https://pdg.lbl.gov/)

---
