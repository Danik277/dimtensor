# Plan: OpenFOAM Results Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a loader module to parse OpenFOAM CFD simulation results and convert field data (velocity, pressure, temperature, turbulence) into dimensionally-aware DimArrays, enabling seamless integration of CFD data with dimtensor workflows.

---

## Background

OpenFOAM is a widely-used open-source CFD toolkit that produces simulation results in a specific directory structure. Engineers and researchers need to post-process this data with proper dimensional tracking for analysis, visualization, and machine learning applications. This loader will enable dimtensor users to directly load OpenFOAM results with automatic unit conversion and dimension verification.

### OpenFOAM File Structure

Based on research:
- **Time directories** (e.g., `0/`, `100/`, `500/`): Contain field data files
- **constant/polyMesh/**: Contains mesh geometry (points, faces, cells, boundary)
- **system/**: Contains solver configuration (controlDict, fvSchemes, fvSolution)

### OpenFOAM Field Format

- **Dimensions**: 7-element array `[0 1 -1 0 0 0 0]` representing powers of SI base units
- **Field structure**: Header with dimensions, internalField (cell values), boundaryField (boundary patches)
- **Formats**: ASCII (human-readable) and binary (compact)
- **Common fields**:
  - `U`: Velocity [0 1 -1 0 0 0 0] (m/s)
  - `p`: Pressure [0 2 -2 0 0 0 0] (m²/s², kinematic) or [1 -1 -2 0 0 0 0] (Pa)
  - `T`: Temperature [0 0 0 1 0 0 0] (K)
  - `k`: Turbulent kinetic energy [0 2 -2 0 0 0 0] (m²/s²)
  - `epsilon`: Turbulent dissipation rate [0 2 -3 0 0 0 0] (m²/s³)

---

## Approach

### Option A: Pure Python Parser (ofpp-style)

- Description: Implement custom ASCII/binary parser using only stdlib and NumPy
- Pros:
  - Minimal dependencies
  - Full control over parsing logic
  - Fast for simple ASCII cases
- Cons:
  - Complex binary format handling
  - Must implement OpenFOAM-specific parsing from scratch
  - More maintenance burden

### Option B: PyFoam Integration

- Description: Use PyFoam library (legacy, last updated 2023.7) as parsing backend
- Pros:
  - Mature, well-tested parser
  - Handles edge cases
- Cons:
  - Legacy library, potential Python 3.12+ compatibility issues
  - Heavy dependency
  - Less active maintenance

### Option C: foamlib Integration

- Description: Use modern foamlib library (2025+) as parsing backend
- Pros:
  - Modern Python (type-hinted, async support)
  - Transparent binary format support
  - Active maintenance, compatible with current Python versions
  - Clean API design
  - Performance optimized
- Cons:
  - Newer library (less battle-tested than PyFoam)
  - Additional dependency

### Decision: Option C (foamlib) with fallback parser

Use **foamlib** as the primary parsing backend with an optional pure-Python fallback for ASCII-only cases. This provides:
- Best user experience (modern, fast, type-safe)
- Binary format support out of the box
- Clean integration with existing loader patterns
- Optional dependency: graceful degradation if not installed

---

## Implementation Steps

1. [ ] Create `src/dimtensor/datasets/loaders/openfoam.py` module
2. [ ] Implement `OpenFOAMLoader(BaseLoader)` class structure
3. [ ] Add dimension mapping from OpenFOAM format to dimtensor Dimension
4. [ ] Implement ASCII field parser (pure Python fallback)
5. [ ] Integrate foamlib for full format support (optional dependency)
6. [ ] Add field extraction methods:
   - [ ] `load_field(case_path, time, field_name)` → DimArray
   - [ ] `load_fields(case_path, time, fields)` → dict[str, DimArray]
   - [ ] `list_times(case_path)` → list of available time directories
   - [ ] `list_fields(case_path, time)` → list of available fields
7. [ ] Create unit mapping for common CFD fields
8. [ ] Add boundary field extraction (optional, for advanced users)
9. [ ] Implement caching for mesh data (constant/polyMesh/)
10. [ ] Write comprehensive tests
11. [ ] Update documentation
12. [ ] Add to `__init__.py` exports

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/loaders/openfoam.py` | Create new loader module |
| `src/dimtensor/datasets/loaders/__init__.py` | Add OpenFOAMLoader export |
| `tests/test_openfoam_loader.py` | Create comprehensive test suite |
| `tests/fixtures/openfoam/` | Add sample OpenFOAM case data |
| `setup.py` or `pyproject.toml` | Add foamlib as optional dependency `[openfoam]` |
| `docs/loaders.md` (if exists) | Document OpenFOAM loader usage |

---

## Testing Strategy

### Unit Tests

- [ ] Test dimension mapping (OpenFOAM → dimtensor)
  - Velocity: `[0 1 -1 0 0 0 0]` → `Dimension(length=1, time=-1)`
  - Pressure (kinematic): `[0 2 -2 0 0 0 0]` → `Dimension(length=2, time=-2)`
  - Pressure (static): `[1 -1 -2 0 0 0 0]` → `Dimension(mass=1, length=-1, time=-2)`
  - Temperature: `[0 0 0 1 0 0 0]` → `Dimension(temperature=1)`
- [ ] Test ASCII field parsing with sample data
- [ ] Test field list extraction
- [ ] Test time directory discovery
- [ ] Test error handling (missing files, malformed data)

### Integration Tests

- [ ] Load complete sample case (e.g., cavity tutorial)
- [ ] Verify units match expected values
- [ ] Test with/without foamlib installed
- [ ] Test binary format (if foamlib available)

### Sample Data

Create minimal OpenFOAM case in `tests/fixtures/openfoam/cavity/`:
```
cavity/
├── 0/
│   ├── U        # Velocity field
│   ├── p        # Pressure field
│   └── T        # Temperature field (optional)
├── constant/
│   └── polyMesh/
│       ├── points
│       ├── faces
│       ├── owner
│       ├── neighbour
│       └── boundary
└── system/
    └── controlDict
```

---

## Risks / Edge Cases

### Risk 1: OpenFOAM Dimension Format Ambiguity
OpenFOAM uses 7 dimensions `[kg, m, s, K, mol, A, cd]`, which maps directly to SI but can be interpreted differently (e.g., kinematic vs static pressure).
- **Mitigation**: Document common field conventions. Add `pressure_type` parameter to disambiguate. Default to field name convention (`p` = kinematic, `p_rgh` = kinematic, `p_static` = static).

### Risk 2: Binary Format Complexity
Binary OpenFOAM files require complex parsing logic.
- **Mitigation**: Use foamlib for binary support. Provide clear error message if binary file encountered without foamlib installed.

### Risk 3: Large Case Files
CFD cases can have millions of cells and time steps.
- **Mitigation**:
  - Load only requested fields (not entire case)
  - Support time range filtering
  - Consider memory-mapped arrays for very large cases (future enhancement)
  - Cache mesh data (constant across time steps)

### Risk 4: Boundary Field Handling
Boundary patches have different dimensions/types than internal fields.
- **Mitigation**: Focus on internal field first. Add boundary extraction as optional advanced feature with separate methods.

### Edge Case 1: Non-standard Dimensions
Custom solvers may use non-standard dimensions.
- **Handling**: Support custom dimension specification via parameter. Fall back to dimensionless if cannot map.

### Edge Case 2: Decomposed Cases (Parallel Runs)
OpenFOAM parallel cases have `processor*` directories.
- **Handling**: Phase 2 feature. Document limitation for MVP. Future: Auto-detect and reconstruct.

### Edge Case 3: Missing Files
Case directories may be incomplete or corrupted.
- **Handling**: Robust error checking with informative messages. Verify directory structure before parsing.

---

## Definition of Done

- [x] Plan created and reviewed
- [ ] All implementation steps complete
- [ ] OpenFOAMLoader class functional
- [ ] ASCII parsing works without foamlib
- [ ] foamlib integration functional (when installed)
- [ ] Tests pass with >90% coverage
- [ ] Sample data fixtures created
- [ ] Documentation complete with examples
- [ ] Integration with existing loaders/ module
- [ ] CONTINUITY.md updated

---

## Notes / Log

### Research Summary (2026-01-12)

**OpenFOAM Structure** (from official docs):
- Standard case structure: time dirs, constant/, system/
- Field format: dictionary with dimensions, internalField, boundaryField
- Dimensions: 7-element array for SI base units

**Python Libraries**:
1. **PyFoam** - Legacy (2023.7), stable but older
2. **foamlib** - Modern (2025+), type-hinted, binary support, async
3. **ofpp** - Lightweight, NumPy-focused parser

**Decision**: Use foamlib as primary with pure-Python ASCII fallback.

### Implementation Notes

Key design decisions:
- Inherit from `BaseLoader` for caching support
- Dimension mapping function: `_openfoam_to_dimtensor(of_dims: list[int]) -> Dimension`
- Field cache by case path + time to avoid re-parsing
- Memory-efficient: load only requested fields
- Clear error messages for common issues

### API Design

```python
from dimtensor.datasets.loaders import OpenFOAMLoader

loader = OpenFOAMLoader(cache=True)

# Load single field at specific time
U = loader.load_field("./cavity", time="0.5", field="U")
# Returns: DimArray with shape (n_cells, 3) and m/s units

# Load multiple fields
data = loader.load_fields("./cavity", time="0.5", fields=["U", "p", "T"])
# Returns: {"U": DimArray, "p": DimArray, "T": DimArray}

# Discover available data
times = loader.list_times("./cavity")  # ["0", "0.1", "0.2", ..., "0.5"]
fields = loader.list_fields("./cavity", "0.5")  # ["U", "p", "T", "phi"]
```

---

## References

- [OpenFOAM File Structure](https://www.openfoam.com/documentation/user-guide/2-openfoam-cases/2.1-file-structure-of-openfoam-cases)
- [OpenFOAM I/O Format](https://www.openfoam.com/documentation/user-guide/2-openfoam-cases/2-2-basic-inputoutput-file-format)
- [foamlib GitHub](https://github.com/gerlero/foamlib)
- [PyFoam Wiki](https://wiki.openfoam.com/PyFoam_and_swakFoam_by_Bernhard_Gschaider)
- [ofpp GitHub](https://github.com/xu-xianghua/ofpp)
