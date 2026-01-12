# Plan: COMSOL Multiphysics Results Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a loader module for COMSOL Multiphysics FEM simulation results that can parse exported data files (TXT, CSV, VTU) and convert mesh, field, and multi-physics data into DimArrays with appropriate units based on the physics module used.

---

## Background

COMSOL Multiphysics is a widely-used finite element analysis (FEA) software for multi-physics simulations in engineering and science. Users need to:
- Load FEM results into dimtensor for post-processing and analysis
- Preserve dimensional units from various physics modules (structural, thermal, electromagnetics, fluid dynamics, etc.)
- Work with mesh data (nodes, elements, connectivity) alongside field data (displacement, temperature, stress, etc.)
- Handle multi-physics coupled simulations with different unit systems

COMSOL exports data in several formats:
- **TXT/CSV**: Text-based field data with coordinate and value columns
- **VTU**: VTK unstructured grid format (XML-based), includes mesh topology and field data
- **MPH**: Native COMSOL format (binary + text) - out of scope for MVP

Reference:
- [COMSOL File Formats](https://www.comsol.com/fileformats)
- [COMSOL Mesh Import/Export Guide](https://www.comsol.com/model/download/1273541/COMSOL_MeshImportExportGuide.pdf)
- [VTK Format Documentation](https://docs.pyvista.org/user-guide/data_model.html)

---

## Approach

### Option A: Text-Only Parser (TXT/CSV)
- Description: Parse simple text/CSV exports only, extract column data
- Pros:
  - Simple implementation, no dependencies
  - Works with most basic COMSOL exports
  - Easy to debug and maintain
- Cons:
  - No mesh topology information
  - Limited metadata about physics modules
  - Cannot handle complex multi-field exports

### Option B: VTU-Focused with meshio/PyVista
- Description: Use meshio or PyVista to parse VTU files, extract mesh and field data
- Pros:
  - Full mesh topology (nodes, elements, connectivity)
  - Preserves field data on nodes/cells
  - Standard format with good Python tooling
  - Handles complex multi-field data
- Cons:
  - Requires external dependencies (meshio or pyvista)
  - More complex implementation
  - VTU export must be enabled in COMSOL

### Option C: Hybrid Approach
- Description: Support both TXT/CSV for simple field data and VTU for full mesh+field
- Pros:
  - Flexible for different COMSOL workflows
  - Covers both simple and advanced use cases
  - Can start with text and add VTU later
- Cons:
  - More code to maintain
  - Needs careful API design for consistency

### Decision: Option C - Hybrid Approach

Start with TXT/CSV for MVP (simpler, no dependencies), design API to accommodate VTU support later. This aligns with dimtensor's pattern of optional dependencies and progressive enhancement.

**Implementation strategy:**
1. Core loader based on CSVLoader (like NIST loader)
2. Parse COMSOL-specific CSV format (coordinate columns + field columns)
3. Infer or accept physics module type to assign appropriate units
4. Design API to add VTU support later via `load_comsol_vtu()` function

---

## Implementation Steps

### Phase 1: Core TXT/CSV Loader (MVP)
1. [ ] Create `src/dimtensor/datasets/loaders/comsol.py`
2. [ ] Implement `COMSOLLoader` class extending `CSVLoader`
3. [ ] Add `load_comsol_txt()` function for text file parsing
4. [ ] Add `load_comsol_csv()` function for CSV parsing
5. [ ] Implement column detection (coordinates: x, y, z + fields)
6. [ ] Add physics module unit mapping (structural, thermal, electromagnetic, etc.)
7. [ ] Support multi-field exports (e.g., temperature + stress)

### Phase 2: Unit Inference System
1. [ ] Create physics module enum (STRUCTURAL, THERMAL, ELECTROMAGNETIC, FLUID, etc.)
2. [ ] Define unit mappings for common field names per physics module
3. [ ] Implement automatic unit inference from field names
4. [ ] Add manual unit override parameter for ambiguous cases
5. [ ] Document unit conventions in docstrings

### Phase 3: Testing & Documentation
1. [ ] Create test data files (synthetic COMSOL exports)
2. [ ] Write unit tests for each format (TXT, CSV)
3. [ ] Test multi-field loading
4. [ ] Test unit inference for different physics modules
5. [ ] Add usage examples in docstring
6. [ ] Update io/__init__.py exports

### Phase 4: VTU Support (Future Enhancement - Out of MVP Scope)
1. [ ] Add optional dependency: meshio (lightweight) or pyvista (full-featured)
2. [ ] Implement `load_comsol_vtu()` function
3. [ ] Parse mesh topology (nodes, elements, cell types)
4. [ ] Extract point data and cell data as separate DimArrays
5. [ ] Return structured dict with mesh + field data
6. [ ] Add VTU tests

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/loaders/comsol.py` | **CREATE** - Main loader implementation |
| `src/dimtensor/datasets/loaders/__init__.py` | Update to export COMSOLLoader |
| `src/dimtensor/datasets/__init__.py` | Add convenience functions (load_comsol, etc.) |
| `tests/datasets/loaders/test_comsol.py` | **CREATE** - Unit tests |
| `tests/datasets/loaders/fixtures/` | **CREATE** - Sample COMSOL export files |
| `src/dimtensor/domains/engineering.py` | Reference for engineering units (already exists) |
| `docs/examples/` | **CREATE** - COMSOL loader example notebook (optional) |

---

## Testing Strategy

### Unit Tests
- [ ] Test TXT parsing with single field (temperature)
- [ ] Test CSV parsing with multiple fields (displacement X, Y, Z)
- [ ] Test coordinate extraction (2D and 3D meshes)
- [ ] Test unit inference for structural mechanics (displacement → m, stress → Pa)
- [ ] Test unit inference for heat transfer (temperature → K, heat flux → W/m²)
- [ ] Test unit inference for electromagnetics (E-field → V/m, B-field → T)
- [ ] Test manual unit override
- [ ] Test error handling (malformed files, missing columns)

### Integration Tests
- [ ] Load real COMSOL export (if available) or realistic synthetic data
- [ ] Verify DimArray shapes match expected node count
- [ ] Verify units are correctly assigned
- [ ] Test multi-physics data (e.g., thermal-structural coupling)

### Manual Verification
- [ ] Compare loaded data against COMSOL GUI display
- [ ] Verify coordinate system orientation (COMSOL uses right-handed Cartesian)
- [ ] Check field value ranges are physically reasonable

---

## Risks / Edge Cases

### Risk 1: Format Variations
- **Issue**: COMSOL export format may vary by version or user settings
- **Mitigation**:
  - Support flexible column detection (regex patterns, case-insensitive)
  - Add format_version parameter for future compatibility
  - Document tested COMSOL versions (e.g., 5.x, 6.x)

### Risk 2: Unit Ambiguity
- **Issue**: Field names alone may not uniquely determine units (e.g., "u" could be displacement or velocity)
- **Mitigation**:
  - Require physics_module parameter for automatic unit inference
  - Allow explicit unit dictionary for full control
  - Log warnings when inference is ambiguous

### Risk 3: Large File Performance
- **Issue**: FEM results can be very large (millions of nodes)
- **Mitigation**:
  - Use numpy's efficient CSV loading (np.loadtxt or pd.read_csv)
  - Consider memory-mapped arrays for VTU support
  - Document memory requirements in docstring

### Risk 4: Coordinate System Conventions
- **Issue**: COMSOL may use different coordinate conventions than expected
- **Mitigation**:
  - Document that COMSOL uses right-handed Cartesian (x, y, z)
  - Provide coordinate_transform parameter for custom transformations
  - Add examples showing coordinate system handling

### Edge Case: Time-Dependent Data
- **Handling**: COMSOL can export time series - for MVP, load single timestep only. Document that user should export separate files per timestep.

### Edge Case: Parametric Sweeps
- **Handling**: COMSOL supports parameter sweeps - for MVP, load single parameter value. Document multi-file loading pattern.

### Edge Case: Mixed Element Types (VTU only)
- **Handling**: VTU files can have mixed element types (tets, hexes, prisms) - defer to Phase 4 (VTU support).

---

## Definition of Done

- [ ] All implementation steps in Phase 1-3 complete
- [ ] Tests pass with >90% coverage for comsol.py
- [ ] Docstrings include usage examples for each physics module
- [ ] Can load structural mechanics data (displacement, stress, strain)
- [ ] Can load heat transfer data (temperature, heat flux)
- [ ] Can load electromagnetics data (E-field, B-field, potential)
- [ ] Unit inference works for at least 5 physics modules
- [ ] README or docs explain COMSOL export process
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-12 - Initial Planning**

### COMSOL Export Format Details (from research):

**Supported Formats:**
- TXT/CSV: Plain text with columns (coordinates + field values)
- VTU: Unstructured VTK XML format with mesh topology
- MPH/MPHBIN/MPHTXT: Native COMSOL formats (binary/text) - complex, out of scope

**Typical CSV Export Structure:**
```
% x [m], y [m], z [m], T [K], u [m], v [m], w [m]
0.0, 0.0, 0.0, 293.15, 0.001, 0.0, 0.0
0.1, 0.0, 0.0, 295.20, 0.002, 0.0, 0.0
...
```

**VTU Format (via meshio/pyvista):**
- Points: Node coordinates (N x 3 array)
- Cells: Element connectivity (varies by element type)
- Point data: Field values at nodes (N x 1 arrays, one per field)
- Cell data: Field values at element centers (M x 1 arrays)

**Physics Module → Common Fields → Units:**

| Physics Module | Field Names | Units |
|----------------|-------------|-------|
| Structural Mechanics | u, v, w (displacement) | m |
| | solid.sx, solid.sy, solid.sz (stress) | Pa |
| | solid.ex, solid.ey, solid.ez (strain) | dimensionless |
| Heat Transfer | T (temperature) | K |
| | ht.ntflux (heat flux) | W/m² |
| | ht.Q (heat source) | W/m³ |
| Electromagnetics | emw.Ex, emw.Ey, emw.Ez (E-field) | V/m |
| | emw.Bx, emw.By, emw.Bz (B-field) | T |
| | V (electric potential) | V |
| Fluid Dynamics | u, v, w (velocity) | m/s |
| | p (pressure) | Pa |
| | spf.rho (density) | kg/m³ |

**Python Tools for VTU Parsing:**
- **meshio**: Lightweight, converts between many mesh formats
- **PyVista**: Full VTK wrapper, visualization capabilities
- Recommendation: Use meshio for MVP (smaller dependency)

---

## API Design Preview

```python
from dimtensor.datasets.loaders.comsol import (
    COMSOLLoader,
    load_comsol_txt,
    load_comsol_csv,
    PhysicsModule,
)

# Simple loading with automatic unit inference
loader = COMSOLLoader()
data = loader.load("results.txt", physics_module=PhysicsModule.STRUCTURAL)
# Returns: {"coordinates": DimArray, "displacement_x": DimArray, ...}

# Explicit unit mapping
data = load_comsol_csv(
    "thermal.csv",
    units={"T": "K", "q": "W/m**2"},
    coords=["x", "y", "z"],
    fields=["T", "q"],
)

# Future: VTU support
mesh_data = load_comsol_vtu("model.vtu", physics_module=PhysicsModule.THERMAL)
# Returns: {
#   "mesh": {"points": DimArray, "cells": dict},
#   "fields": {"temperature": DimArray, "heat_flux": DimArray}
# }
```

---
