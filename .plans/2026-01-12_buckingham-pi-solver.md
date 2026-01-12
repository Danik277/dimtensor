# Plan: Buckingham Pi Theorem Solver

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Implement a dimensional analysis solver based on the Buckingham Pi theorem that automatically finds complete sets of dimensionless groups (Pi groups) from a list of physical variables. This enables automatic discovery of scaling laws and dimensionless relationships in physical systems.

---

## Background

The Buckingham Pi theorem is a fundamental result in dimensional analysis stating that if a physical problem involves n variables with k independent base dimensions, then the problem can be reduced to n-k dimensionless parameters (Pi groups). This is crucial for:

1. **Scaling laws**: Understanding how physical systems scale
2. **Experimental design**: Reducing the number of independent variables
3. **Similarity analysis**: Comparing systems of different scales
4. **Physical insight**: Revealing fundamental dimensionless numbers (Reynolds, Mach, etc.)

The solver uses linear algebra on the dimensional matrix to find the null space, which corresponds to dimensionless combinations of variables.

---

## Approach

### Option A: Pure Python with NumPy

- Description: Implement using NumPy's linear algebra (SVD or nullspace computation)
- Pros:
  - Integrates naturally with existing DimArray infrastructure
  - No additional dependencies beyond NumPy (already required)
  - Full control over algorithm and output formatting
  - Can leverage existing Dimension/Unit classes
- Cons:
  - NumPy's matrix_rank and nullspace may have numerical precision issues
  - Requires careful handling of rational arithmetic (Fraction) to avoid floating point errors

### Option B: SymPy-based symbolic approach

- Description: Use SymPy's Matrix.nullspace() for exact symbolic computation
- Pros:
  - Exact rational arithmetic (no floating point errors)
  - Symbolic manipulation of exponents
  - Clean mathematical representation
- Cons:
  - Requires SymPy as dependency (already optional)
  - Conversion between dimtensor Dimension objects and SymPy matrices
  - May be slower for large systems
  - Less control over output format

### Option C: Hybrid approach

- Description: Use NumPy for fast computation with Fraction-based validation
- Pros:
  - Best of both worlds: speed and accuracy
  - Can fall back to exact computation when needed
  - Works with or without SymPy
- Cons:
  - More complex implementation
  - Need to handle edge cases carefully

### Decision: Option C (Hybrid approach)

Use NumPy for the main algorithm with Fraction-based dimension representation to ensure exact results. This matches the existing architecture (Dimension uses Fraction exponents) and avoids adding hard dependencies. The algorithm will:

1. Build dimensional matrix from Dimension objects (as Fractions)
2. Convert to float for SVD computation
3. Find nullspace vectors
4. Clean up numerical artifacts (round near-zero values)
5. Convert back to Fraction for exact representation
6. Validate that combinations are truly dimensionless

---

## Implementation Steps

1. [ ] Create `src/dimtensor/analysis/buckingham.py` module
2. [ ] Implement `DimensionalMatrix` class to represent variable dimensions as matrix
3. [ ] Implement `nullspace_rational()` function using SVD with Fraction cleanup
4. [ ] Implement `buckingham_pi()` main function that:
   - Takes dict of {variable_name: Unit} or {variable_name: Dimension}
   - Builds dimensional matrix
   - Computes nullspace
   - Constructs Pi groups as dimensionless combinations
   - Returns list of Pi groups with human-readable names
5. [ ] Add `PiGroup` dataclass to represent dimensionless combinations
6. [ ] Implement pretty-printing for Pi groups (LaTeX and Unicode)
7. [ ] Add validation to ensure Pi groups are truly dimensionless
8. [ ] Create `tests/test_buckingham.py` with comprehensive test cases
9. [ ] Add utility function `suggest_pi_groups()` that provides physical interpretations
10. [ ] Integrate with existing `inference/` module for cross-validation
11. [ ] Add visualization support for dimensional matrix and Pi groups
12. [ ] Document common Pi groups (Reynolds number, Froude number, etc.)
13. [ ] Update CONTINUITY.md with completion

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/analysis/__init__.py | Create new module (if doesn't exist) |
| src/dimtensor/analysis/buckingham.py | New file: main implementation |
| tests/test_buckingham.py | New file: comprehensive tests |
| src/dimtensor/__init__.py | Export buckingham_pi function |
| docs/examples/ | Add example notebook (future) |

---

## Testing Strategy

### Unit Tests

- [ ] Test dimensional matrix construction from Units/Dimensions
- [ ] Test nullspace computation with known examples
- [ ] Test Pi group construction and validation

### Integration Tests

- [ ] **Fluid mechanics**: Drag force problem (Reynolds, Froude numbers)
  - Variables: force F, velocity v, length L, density ρ, viscosity μ, gravity g
  - Expected: Re = ρvL/μ, Fr = v/√(gL), Cd dimensionless

- [ ] **Pendulum**: Period of oscillation
  - Variables: period T, length L, mass m, gravity g
  - Expected: T√(g/L) dimensionless (mass drops out)

- [ ] **Heat transfer**: Convection problem
  - Variables: heat flux q, temperature difference ΔT, length L, thermal conductivity k, velocity v, density ρ, specific heat cp, viscosity μ
  - Expected: Nusselt, Reynolds, Prandtl numbers

- [ ] **Projectile motion**: Range of projectile
  - Variables: range R, initial velocity v₀, angle θ, gravity g
  - Expected: R/(v₀²/g) and θ as dimensionless parameters

### Edge Cases

- [ ] Verify behavior when variables span fewer dimensions than expected
- [ ] Handle case where no dimensionless groups exist (degenerate)
- [ ] Handle case where all variables have same dimension (trivial)
- [ ] Test with large number of variables (performance)
- [ ] Test with fractional dimension exponents

### Validation

- [ ] Manual verification that Pi groups are dimensionless
- [ ] Cross-check with known results from dimensional analysis literature
- [ ] Verify linear independence of Pi groups

---

## API Design

```python
from dimtensor.analysis import buckingham_pi
from dimtensor.core.units import m, s, kg

# Define the physical problem
variables = {
    'F': kg * m / s**2,      # Force
    'v': m / s,               # Velocity
    'L': m,                   # Length
    'rho': kg / m**3,         # Density
    'mu': kg / (m * s),       # Dynamic viscosity
}

# Find dimensionless groups
result = buckingham_pi(variables)

# Result structure:
# {
#   'pi_groups': [PiGroup(...), PiGroup(...), ...],
#   'rank': 3,  # Number of independent dimensions
#   'n_variables': 5,
#   'n_groups': 2,  # n - rank = 5 - 3 = 2
#   'base_dimensions': ['L', 'M', 'T'],
# }

# Each PiGroup has:
# - name: str (e.g., 'Π₁', 'Π₂', or 'Re' if recognized)
# - exponents: dict[str, Fraction]  # variable name -> exponent
# - expression: str  # Human-readable like 'ρvL/μ'
# - latex: str  # LaTeX representation
# - interpretation: str | None  # Physical meaning if known

for pi in result['pi_groups']:
    print(f"{pi.name} = {pi.expression}")
    # Π₁ = ρvL/μ (Reynolds number)
    # Π₂ = F/(ρv²L²) (Drag coefficient)
```

---

## Algorithm Details

### Dimensional Matrix Construction

For n variables and k base dimensions (L, M, T, I, Θ, N, J):

```
          v₁  v₂  v₃  ...  vₙ
    L  [  a₁  a₂  a₃  ...  aₙ ]
    M  [  b₁  b₂  b₃  ...  bₙ ]
    T  [  c₁  c₂  c₃  ...  cₙ ]
    I  [  d₁  d₂  d₃  ...  dₙ ]
    Θ  [  e₁  e₂  e₃  ...  eₙ ]
    N  [  f₁  f₂  f₃  ...  fₙ ]
    J  [  g₁  g₂  g₃  ...  gₙ ]
```

Where each column is the dimension exponents for one variable.

### Nullspace Computation

1. Compute SVD: A = UΣV^T
2. Rank r = number of non-zero singular values (with tolerance)
3. Nullspace = last (n - r) columns of V
4. Each nullspace vector gives one Pi group

### Pi Group Construction

For nullspace vector `w = [w₁, w₂, ..., wₙ]`:

```
Π = v₁^w₁ · v₂^w₂ · ... · vₙ^wₙ
```

Dimension of Π should be `1` (dimensionless).

### Cleanup and Normalization

1. Round near-zero exponents to exactly zero
2. Convert to Fraction for exact representation
3. Normalize so first non-zero exponent is positive
4. Simplify fractions using GCD

---

## Risks / Edge Cases

- **Risk 1**: Numerical precision in SVD may produce non-exact nullspace vectors
  - **Mitigation**: Use high tolerance threshold, convert to Fraction, validate dimensionlessness

- **Risk 2**: Non-unique choice of Pi groups (nullspace basis is not unique)
  - **Mitigation**: Normalize using consistent convention, document that choice is arbitrary

- **Edge case**: Variables with identical dimensions
  - **Handling**: Matrix rank will be less than number of variables, produces valid Pi groups

- **Edge case**: Redundant base dimensions (e.g., only using L and T, not M)
  - **Handling**: Automatic detection of active dimensions, reduce matrix size

- **Edge case**: Complex exponents (fractional or irrational)
  - **Handling**: Dimension already uses Fraction with limit_denominator, should work naturally

- **Risk 3**: Performance for large systems (100+ variables)
  - **Mitigation**: NumPy SVD is efficient, but document scalability limits

---

## Physical Interpretations

Include database of known dimensionless numbers:

- **Reynolds number**: Re = ρvL/μ (inertial forces / viscous forces)
- **Froude number**: Fr = v/√(gL) (inertial forces / gravitational forces)
- **Mach number**: Ma = v/c (velocity / speed of sound)
- **Prandtl number**: Pr = cpμ/k (momentum diffusivity / thermal diffusivity)
- **Nusselt number**: Nu = hL/k (convective / conductive heat transfer)
- **Rayleigh number**: Ra = gβΔTL³/(αν) (buoyancy / viscous forces)
- **Weber number**: We = ρv²L/σ (inertial forces / surface tension)
- **Strouhal number**: St = fL/v (frequency / characteristic velocity)

Pattern matching algorithm to recognize these from exponents.

---

## Visualization Support

1. **Dimensional Matrix Heatmap**: Visualize which dimensions each variable uses
2. **Pi Group Exponent Plot**: Show how variables combine in each group
3. **Dependency Graph**: Show relationships between variables through Pi groups

Integration with `dimtensor.visualization` module.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass with >95% coverage
- [ ] Documentation includes:
  - Docstrings with examples
  - Mathematical background
  - Interpretation guide for common Pi groups
- [ ] Integration tests verify known physical results
- [ ] CONTINUITY.md updated with task completion
- [ ] No regression in existing tests
- [ ] Performance acceptable for typical use cases (<1s for <20 variables)

---

## Notes / Log

**2026-01-12 14:54** - Plan created by planner agent. Key design decisions:
- Hybrid NumPy/Fraction approach for accuracy without new dependencies
- Rich return type with metadata and interpretations
- Integration with existing inference module
- Comprehensive test suite covering standard dimensional analysis problems

---
