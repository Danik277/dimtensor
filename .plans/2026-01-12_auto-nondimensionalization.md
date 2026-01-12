# Plan: Automatic Non-Dimensionalization System

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Build an automatic non-dimensionalization system that identifies characteristic scales from problem parameters, scales equations and data to dimensionless form, and computes common dimensionless numbers (Reynolds, Mach, etc.). This extends the existing `DimScaler` to support physics-based scaling strategies.

---

## Background

Non-dimensionalization is fundamental in physics and engineering:
- **Simplifies equations**: Reduces parameter count from N to M << N dimensionless groups
- **Reveals scaling laws**: Buckingham Pi theorem guarantees dimensionless formulation
- **Numerical stability**: Maps quantities to O(1) values for better conditioning
- **Similarity solutions**: Problems with same dimensionless parameters share solutions
- **Physical insight**: Dimensionless numbers (Re, Ma, Fr, etc.) characterize regime

Current state:
- `DimScaler` exists in `torch/scaler.py` with characteristic/standard/minmax scaling
- Works dimension-by-dimension but doesn't leverage **problem structure**
- No automatic identification of characteristic scales
- No dimensionless number database

This plan implements:
1. **Characteristic scale identification** from problem parameters
2. **Buckingham Pi theorem** application for dimensionless group generation
3. **Dimensionless number database** (Reynolds, Mach, Froude, etc.)
4. **Enhanced DimScaler** with physics-aware strategies

---

## Approach

### Option A: Physics-First Approach
- Build on `DimScaler` with new scaling strategies
- Add `CharacteristicScalesFinder` to analyze problem structure
- Implement Buckingham Pi algorithm for automatic group generation
- Create `DimensionlessNumbers` module with pre-defined groups

**Pros:**
- Leverages existing scaler infrastructure
- Physics-based, interpretable scalings
- Automatic dimensionless group generation via Buckingham Pi
- Extensible database of dimensionless numbers

**Cons:**
- More complex than simple heuristics
- Requires problem structure analysis

### Option B: Heuristic-Only Approach
- Just add min/max/mean heuristics to existing scaler
- Pre-defined dimensionless numbers only (no generation)

**Pros:**
- Simpler implementation
- Faster to ship

**Cons:**
- Less general, misses physics structure
- No automatic group generation
- Limited to known cases

### Decision: Option A (Physics-First)

**Rationale:**
- dimtensor's value proposition is physics correctness
- Buckingham Pi is fundamental, not a "nice-to-have"
- Characteristic scale identification is automatable from dimensional analysis
- Database of dimensionless numbers provides immediate value while generation adds power
- Implementation builds on existing `Dimension` algebra (already have the foundation)

---

## Implementation Steps

1. [ ] **Create `analysis/scales.py` module**
   - `CharacteristicScalesFinder` class
   - Algorithm: Given problem parameters {v, L, ρ, μ, ...}, find characteristic scales for each dimension
   - Strategy: For each dimension D, find max/representative value from parameters with that dimension
   - Handle composite dimensions (velocity = length/time → use separately or combined)

2. [ ] **Create `analysis/buckingham.py` module**
   - `buckingham_pi(variables: dict[str, Dimension]) -> list[DimensionlessGroup]`
   - Implements Buckingham Pi theorem using linear algebra on dimension matrix
   - Returns basis of dimensionless groups
   - Handle rank-deficient cases

3. [ ] **Create `analysis/dimensionless_numbers.py` module**
   - `DimensionlessNumber` dataclass (name, formula, variables, interpretation, regime_map)
   - Database of common numbers:
     - **Fluid mechanics**: Reynolds (Re), Mach (Ma), Froude (Fr), Weber (We), Prandtl (Pr), Nusselt (Nu), Grashof (Gr), Rayleigh (Ra), Eckert (Ec), Strouhal (St)
     - **Heat transfer**: Biot (Bi), Fourier (Fo), Peclet (Pe), Lewis (Le)
     - **Multiphase**: Capillary (Ca), Bond (Bo), Morton (Mo)
     - **Electromagnetics**: Fine structure constant (α), magnetic Reynolds (Rm)
     - **Relativity**: Schwarzschild (Rs/R), Eddington ratio
   - `compute(name: str, **params: DimArray) -> float` function
   - `infer_regime(name: str, value: float) -> str` (e.g., Re=100 → "laminar")

4. [ ] **Extend `torch/scaler.py` with new strategies**
   - Add `method='physics'` option to `DimScaler.__init__`
   - In `fit()`, use `CharacteristicScalesFinder` when method='physics'
   - Add `compute_dimensionless_numbers(**params)` method
   - Returns dict of applicable dimensionless numbers and their values

5. [ ] **Add `NonDimensionalizer` class to `analysis/` module**
   - High-level interface: `NonDimensionalizer.from_problem(**params)`
   - Automatically identifies scales, generates Pi groups, suggests scalings
   - Method: `nondimensionalize(data: dict[str, DimArray]) -> dict[str, np.ndarray]`
   - Method: `redimensionalize(data: dict[str, np.ndarray]) -> dict[str, DimArray]`
   - Integration with `DimScaler` for ML workflows

6. [ ] **Add unit tests**
   - `tests/analysis/test_scales.py`: Test characteristic scale identification
   - `tests/analysis/test_buckingham.py`: Test Pi theorem (classic examples: pendulum, drag)
   - `tests/analysis/test_dimensionless_numbers.py`: Test number computation and regime detection
   - `tests/torch/test_scaler_physics.py`: Test physics-aware scaling integration

7. [ ] **Add examples and documentation**
   - Example: Navier-Stokes non-dimensionalization
   - Example: Heat equation with Fourier number
   - Example: Using dimensionless numbers for regime detection
   - Documentation in `docs/guide/analysis.md`

8. [ ] **Update CHANGELOG.md and bump version**

---

## Files to Modify

| File | Change |
|------|--------|
| **NEW** `src/dimtensor/analysis/__init__.py` | New module for dimensional analysis tools |
| **NEW** `src/dimtensor/analysis/scales.py` | Characteristic scale identification |
| **NEW** `src/dimtensor/analysis/buckingham.py` | Buckingham Pi theorem implementation |
| **NEW** `src/dimtensor/analysis/dimensionless_numbers.py` | Database and computation of dimensionless numbers |
| `src/dimtensor/torch/scaler.py` | Add `method='physics'` strategy, integrate with scale finder |
| **NEW** `tests/analysis/test_scales.py` | Test scale identification |
| **NEW** `tests/analysis/test_buckingham.py` | Test Buckingham Pi |
| **NEW** `tests/analysis/test_dimensionless_numbers.py` | Test dimensionless number database |
| `tests/torch/test_scaler.py` | Add tests for physics-aware scaling |
| **NEW** `examples/06_nondimensionalization.ipynb` | Tutorial notebook |
| **NEW** `docs/guide/analysis.md` | Documentation for analysis tools |
| `docs/api/analysis.md` | API reference |
| `README.md` | Add analysis features to feature list |
| `CHANGELOG.md` | Document new features |

---

## Testing Strategy

### Unit Tests

1. **Characteristic Scales** (`test_scales.py`):
   - Test with simple problem: {velocity, length} → correct scales
   - Test with composite dimensions: Force = MLT⁻²
   - Test with multiple parameters of same dimension (take max)
   - Test edge case: no parameter for a dimension → None

2. **Buckingham Pi** (`test_buckingham.py`):
   - Classic pendulum: {L, m, g, T} → [T²g/L] (dimensionless)
   - Drag force: {F, ρ, v, L, μ} → [F/(ρv²L²), ρvL/μ] = [Cd, Re]
   - Test rank-deficient cases
   - Test with 7+ parameters (multiple Pi groups)

3. **Dimensionless Numbers** (`test_dimensionless_numbers.py`):
   - Test Reynolds: Re = ρvL/μ with known values
   - Test Mach: Ma = v/c with known values
   - Test regime inference: Re=10 → "Stokes", Re=1e5 → "turbulent"
   - Test all database entries compute correctly

4. **Physics Scaler** (`test_scaler_physics.py`):
   - Test `method='physics'` finds correct scales
   - Test integration with `NonDimensionalizer`
   - Compare physics vs characteristic scaling on real problem

### Integration Tests

- **Full pipeline**: Problem setup → scale identification → non-dimensionalization → ML training → re-dimensionalization
- **Notebook test**: Run `06_nondimensionalization.ipynb` without errors

### Manual Verification

- Verify Navier-Stokes non-dimensionalization produces standard form
- Verify dimensionless numbers match literature values (e.g., standard Re for pipe flow)

---

## Risks / Edge Cases

### Risk 1: Buckingham Pi Linear Algebra Instability
**Issue**: Dimension matrix can be near-singular, causing numerical issues

**Mitigation**:
- Use `scipy.linalg.null_space` with tolerance parameter
- Normalize dimension matrix before computation
- Validate results: verify Pi groups are actually dimensionless

### Risk 2: Non-Unique Characteristic Scales
**Issue**: Given {v₁, v₂} both with dimension velocity, which is characteristic?

**Mitigation**:
- Take maximum by default (conservative, keeps O(1))
- Allow user to specify `preferred_scales` dict
- Document that choice affects specific scaling, not dimensionless groups

### Risk 3: Dimensionless Number Applicability
**Issue**: Not all dimensionless numbers are relevant to all problems

**Mitigation**:
- Only compute numbers where ALL required parameters are present
- Return dict with only applicable numbers
- Add `filter_regime` option to exclude irrelevant numbers

### Edge Case 1: Dimensionless Input Parameters
**Issue**: Problem includes dimensionless parameters (e.g., ratio of lengths)

**Handling**:
- These are already scaled
- Do not transform them
- Include in Buckingham Pi analysis (they're Pi groups themselves)

### Edge Case 2: Zero or Near-Zero Characteristic Scale
**Issue**: Parameter is nearly zero, making scaling blow up

**Handling**:
- Same as current `DimScaler`: fallback to scale=1.0
- Warn user about potential numerical issues
- Consider robust scale estimation (e.g., median instead of max)

### Edge Case 3: Time-Dependent Problems
**Issue**: Characteristic scales may evolve (e.g., boundary layer thickness growing)

**Handling**:
- Phase 1: Use scales from initial/boundary conditions
- Document limitation
- Future: Adaptive scaling (beyond this plan)

---

## Definition of Done

- [ ] All modules created and implemented
- [ ] All unit tests pass (coverage >90% for new code)
- [ ] Integration test passes (full pipeline)
- [ ] Example notebook runs without errors
- [ ] Documentation complete (guide + API reference)
- [ ] CHANGELOG.md updated
- [ ] Code review passed
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-12 - Initial Planning**

Research findings:
- Existing `DimScaler` provides good foundation in `torch/scaler.py`
- `Dimension` class already supports algebra needed for Buckingham Pi
- No existing dimensional analysis tooling in codebase
- Similar libraries (Pint, Astropy) don't have Buckingham Pi or dimensionless number databases

Design decisions:
- Place in new `analysis/` module (not `torch/`) as it's framework-agnostic
- `DimScaler` gets physics integration but core logic is independent
- Dimensionless number database uses dataclass + registry pattern (like `equations/database.py`)
- Buckingham Pi uses SVD/null space of dimension matrix (standard algorithm)

Key insight: dimtensor already has all the primitives (Dimension algebra, Unit tracking). We're just building higher-level analysis tools on top.

Implementation complexity: MEDIUM
- Buckingham Pi is well-defined algorithm (~100 lines)
- Characteristic scale finder is heuristic-based (~150 lines)
- Dimensionless number database is data entry (~200 lines for 20-30 numbers)
- Integration with DimScaler is straightforward (~50 lines)

Total estimate: ~500-700 lines of new code, ~400 lines of tests.

---
