# Plan: General Relativity Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add a comprehensive set of general relativity equations to the equation database, enabling dimensional analysis and validation for GR calculations in astrophysics, cosmology, and gravitational physics applications.

---

## Background

The existing equation database (src/dimtensor/equations/database.py) has 67 equations covering mechanics, thermodynamics, electromagnetism, fluid dynamics, special relativity, quantum mechanics, optics, and acoustics. However, it lacks general relativity equations, which are fundamental for:
- Astrophysics and cosmology research
- Gravitational wave physics
- Black hole calculations
- Spacetime physics and curved spacetime problems
- Physics-informed machine learning for GR applications

General relativity equations involve tensor quantities and geometric concepts, which pose unique challenges for dimensional analysis since many quantities are dimensionless (metric tensor components, Christoffel symbols) while others have specific dimensions (energy-momentum tensor, curvature scalars).

---

## Approach

### Option A: Simplified Scalar Representations
- Store GR equations in their scalar or simplified forms
- Focus on dimensionally meaningful quantities (energy, length, time)
- Avoid explicit tensor notation in formula strings
- Pros: Simple integration, works with existing Equation dataclass
- Cons: Loses tensor structure, less pedagogically complete

### Option B: Full Tensor Representation with Metadata
- Store full tensor equations with index notation
- Add metadata fields for tensor rank, symmetries, coordinate system
- Create specialized GREquation subclass
- Pros: Complete representation, better for education and validation
- Cons: More complex, requires new infrastructure

### Option C: Hybrid Approach (RECOMMENDED)
- Use existing Equation dataclass but add careful documentation
- Store equations in both coordinate-independent form (where possible) and specific coordinate systems
- Use assumptions field to document metric signature, coordinate system, and simplifications
- Add tensor structure information in description field
- Focus on scalar quantities with physical dimensions (invariants, contractions)
- Pros: Works with existing system, provides context, extensible
- Cons: Requires careful documentation discipline

### Decision: Option C - Hybrid Approach

This balances completeness with pragmatism. We can store GR equations using the existing infrastructure while providing rich context. Key scalar quantities (Ricci scalar, Schwarzschild radius, cosmological parameters) have clear dimensions, while tensor equations can be documented with coordinate-specific forms.

---

## Implementation Steps

1. [ ] Research and catalog GR equations to include:
   - Einstein field equations (EFE) - scalar form R - (1/2)Rg + Λg = (8πG/c⁴)T
   - Geodesic equation - coordinate form
   - Schwarzschild metric components (already have Schwarzschild radius)
   - Kerr metric (rotating black hole)
   - Friedmann equations (cosmology) - both equations
   - Friedmann acceleration equation
   - Ricci scalar for specific metrics
   - Energy-momentum tensor for perfect fluid
   - Kretschmann scalar (curvature invariant)
   - Gravitational wave strain
   - ADM mass
   - Bondi mass
   - Tolman-Oppenheimer-Volkoff (TOV) equation

2. [ ] Define dimensional analysis for key GR quantities:
   - Ricci scalar R: [T⁻²] (in natural units, relates to curvature)
   - Ricci tensor R_μν: [T⁻²]
   - Energy-momentum tensor T_μν: [M L⁻¹ T⁻²] (energy density)
   - Christoffel symbols Γ: [L⁻¹] (inverse length)
   - Metric tensor g_μν: dimensionless
   - Einstein tensor G_μν: [T⁻²]
   - Cosmological constant Λ: [L⁻²] or [T⁻²] depending on convention
   - Hubble parameter H: [T⁻¹]
   - Scale factor a: dimensionless (relative)
   - Energy density ρ: [M L⁻³]
   - Pressure p: [M L⁻¹ T⁻²]

3. [ ] Define standard dimension variables for GR:
   - _CURVATURE = Dimension(time=-2)  # Ricci scalar
   - _ENERGY_DENSITY = Dimension(mass=1, length=-3)
   - _HUBBLE = Dimension(time=-1)
   - _COSMOLOGICAL_CONST = Dimension(length=-2)
   - Reuse existing: _E, _L, _T, _M, _V, _PRESSURE, _DIMLESS

4. [ ] Create equations for Schwarzschild solution:
   - Already have Schwarzschild radius (r_s = 2GM/c²)
   - Add Schwarzschild metric time component: g_tt = -(1 - r_s/r)
   - Add Schwarzschild metric radial component: g_rr = 1/(1 - r_s/r)
   - Add gravitational time dilation: proper time vs coordinate time
   - Add gravitational redshift: z = 1/sqrt(1 - r_s/r) - 1

5. [ ] Create equations for Friedmann-Lemaître-Robertson-Walker (FLRW) cosmology:
   - First Friedmann equation: H² = (8πG/3)ρ - k/a² + Λ/3
   - Second Friedmann equation: ȧ/ä = -(4πG/3)(ρ + 3p) + Λ/3
   - Fluid equation: ρ̇ + 3H(ρ + p) = 0
   - Scale factor evolution (matter-dominated): a ∝ t^(2/3)
   - Scale factor evolution (radiation-dominated): a ∝ t^(1/2)
   - Scale factor evolution (Λ-dominated): a ∝ exp(Ht)
   - Critical density: ρ_c = 3H²/(8πG)
   - Density parameter: Ω = ρ/ρ_c

6. [ ] Create equations for Kerr metric (rotating black hole):
   - Kerr radius: r_K = GM/c² + sqrt((GM/c²)² - (J/Mc)²)
   - Angular momentum parameter: a = J/(Mc)
   - Ergosphere radius: r_e = GM/c² + sqrt((GM/c²)² - a²cos²θ)
   - Note: Full metric requires complex tensor notation in description

7. [ ] Create equations for gravitational waves:
   - Strain amplitude: h ~ (G/c⁴)(E/r)
   - Characteristic strain: h_c ~ (G/c³)(dE/df)^(1/2)/r
   - GW frequency (inspiral): f = (1/π)(G M_chirp/c³)^(-5/8) (πf)^(3/8)
   - Chirp mass: M_chirp = (m₁m₂)^(3/5)/(m₁+m₂)^(1/5)

8. [ ] Create equations for energy-momentum tensor:
   - Perfect fluid T_μν = (ρ + p/c²)u_μu_ν + pg_μν
   - Energy density: T_00 components
   - Note: Full tensor in description, scalar components with dimensions

9. [ ] Create curvature invariant equations:
   - Kretschmann scalar: K = R_αβγδ R^αβγδ [T⁻⁴]
   - Kretschmann for Schwarzschild: K = 48(GM)²/(c⁴r⁶) = 12r_s²/r⁶

10. [ ] Create compact object equations:
    - TOV equation for hydrostatic equilibrium
    - Maximum mass relations
    - Compactness parameter: C = GM/(Rc²)

11. [ ] Add all equations to database.py following existing patterns:
    - Domain: "general_relativity"
    - Appropriate tags: ["GR", "cosmology", "black_hole", "gravitational_waves", etc.]
    - Clear LaTeX representations
    - Document assumptions (metric signature (-+++), coordinate system, units)
    - Reference related equations

12. [ ] Update tests to cover new equations:
    - Test equation registration
    - Test domain filtering for "general_relativity"
    - Test search for GR-related terms
    - Test dimensional consistency of variables

13. [ ] Update documentation:
    - Add GR section to docs/guide/equations.md
    - Include example usage with GR constants (G, c)
    - Document coordinate conventions and signature conventions

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | Add ~20-25 GR equations after line 779 (after existing Schwarzschild radius) |
| tests/test_equations.py | Add test cases for GR equations (domain, search, dimensions) |
| docs/guide/equations.md | Add GR section with examples |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for equation registration and retrieval
  - Test `get_equations(domain="general_relativity")` returns all GR equations
  - Test `search_equations("Einstein")` finds Einstein field equations
  - Test `search_equations("Friedmann")` finds Friedmann equations
  - Test `search_equations("black hole")` finds relevant equations

- [ ] Dimensional consistency tests
  - Verify Ricci scalar has dimension [T⁻²]
  - Verify Hubble parameter has dimension [T⁻¹]
  - Verify energy-momentum tensor components have correct dimensions
  - Verify cosmological constant has dimension [L⁻²]
  - Test that equations with G, c maintain dimensional consistency

- [ ] Integration tests with constants
  - Use CODATA 2022 constants (G, c) from dimtensor.constants
  - Calculate Schwarzschild radius for solar mass
  - Calculate Hubble time from Hubble constant
  - Verify units propagate correctly through equations

- [ ] Documentation tests
  - Verify all equations have non-empty description
  - Verify all equations have LaTeX representation
  - Verify all equations have assumptions documented where needed
  - Check that related equations are cross-referenced

---

## Risks / Edge Cases

- **Risk 1: Tensor notation complexity**
  - Many GR equations are inherently tensorial with indices
  - Mitigation: Focus on scalar forms and coordinate-specific expressions with clear documentation. Use description field to explain tensor structure. Store component equations where appropriate.

- **Risk 2: Coordinate system ambiguity**
  - Metrics look different in different coordinates (Schwarzschild, Eddington-Finkelstein, Kruskal, etc.)
  - Mitigation: Always document coordinate system in assumptions field. Default to most common coordinate system for each metric.

- **Risk 3: Signature convention**
  - Two common conventions: (-+++) "mostly plus" and (+---) "mostly minus"
  - Mitigation: Use (-+++) convention (standard in relativity community). Document in assumptions field.

- **Risk 4: Unit convention ambiguity**
  - Some equations look different with G=c=1 vs SI units
  - Mitigation: Store all equations in SI units with explicit G and c. Document natural units form in description.

- **Risk 5: Dimensionless vs dimensional quantities**
  - Metric tensor components are dimensionless, but represent coordinate relationships
  - Mitigation: Be explicit about dimensionless quantities using _DIMLESS. Document physical interpretation.

- **Edge case: Cosmological equations with different matter content**
  - Friedmann equations change for radiation vs matter vs dark energy domination
  - Handling: Create separate equations for each era with clear assumptions documented.

- **Edge case: Geodesic equation coordinate dependence**
  - Geodesic equation has different forms in different coordinate systems
  - Handling: Provide general form with Christoffel symbols, plus specific examples (Schwarzschild geodesics).

- **Edge case: Kerr metric complexity**
  - Full Kerr metric is extremely complex with cross terms
  - Handling: Focus on key derived quantities (ergosphere, horizon radii) rather than full metric components.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] 20-25 GR equations added to database
- [ ] Tests pass (including new GR-specific tests)
- [ ] Coverage maintained above 85%
- [ ] Documentation updated with GR section
- [ ] All equations have proper LaTeX, descriptions, and assumptions
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

### Key Design Decisions

1. **Domain name**: Use "general_relativity" rather than "GR" or "relativity" (to distinguish from special relativity)

2. **Dimensional conventions for curvature**:
   - Ricci scalar R has dimensions [T⁻²] (curvature ~ 1/length² ~ 1/time²)
   - This is consistent with R appearing in equations like R ~ 8πGT/c² where T is energy-momentum tensor

3. **Focus areas**:
   - Schwarzschild solution (spherical, non-rotating black holes)
   - FLRW cosmology (expanding universe)
   - Gravitational waves (linearized GR)
   - Selected Kerr results (rotating black holes)
   - Curvature invariants

4. **What to exclude** (for now):
   - Full tensor equations with all components
   - Numerical relativity equations
   - Alternative theories of gravity
   - Quantum gravity approaches
   - Advanced differential geometry machinery
   - These can be added in future extensions

5. **Integration with existing code**:
   - Reuse physical constants from dimtensor.constants.universal (G, c)
   - Compatible with DimArray for calculations
   - Equations can be used in physics-informed ML for GR applications

---
