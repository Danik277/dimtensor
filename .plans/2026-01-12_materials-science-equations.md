# Plan: Materials Science Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive materials science equations module that provides standard equations for stress-strain relationships, fracture mechanics, fatigue, creep, diffusion, phase transformations, and heat treatment, following the existing `equations/database.py` pattern.

---

## Background

Materials science involves mechanical testing, failure analysis, and microstructural characterization. The current equations database has mechanics, thermodynamics, E&M, fluid dynamics, optics, acoustics, quantum, and relativity domains, but lacks materials-specific equations. Engineers need these for:

- Structural analysis and design
- Material selection and testing
- Failure prediction and lifetime analysis
- Process optimization (heat treatment, forming)
- Microstructure-property relationships

The materials.py units module already provides stress (MPa, GPa), strain (microstrain), hardness (HV, HRC), and fracture toughness (MPa·√m) units.

---

## Approach

### Option A: Single materials_science domain
- Description: Add all materials equations to equations/database.py under domain="materials_science"
- Pros:
  - Simple integration, follows existing pattern
  - All materials equations in one place
  - Easy to search and filter
- Cons:
  - Could become large over time
  - Mixes mechanical, thermal, and diffusion phenomena

### Option B: Separate domains (materials_mechanical, materials_thermal, etc.)
- Description: Split into materials_mechanical, materials_thermal, materials_diffusion domains
- Pros:
  - Better organization for large equation sets
  - Clearer categorization
- Cons:
  - More complex to navigate
  - Overkill for initial implementation

### Option C: Separate equations/materials.py file
- Description: Create a new file specifically for materials equations
- Pros:
  - Keeps database.py from becoming too large
  - Clear separation of concerns
- Cons:
  - Inconsistent with current pattern (all equations in database.py)
  - Requires changes to import structure

### Decision: Option A - Single materials_science domain

Use domain="materials_science" and add all equations to the existing database.py file. This follows the established pattern and keeps things simple. We can always refactor later if the equation count becomes unwieldy. Use descriptive tags for sub-categorization (stress-strain, fracture, fatigue, creep, diffusion, phase).

---

## Implementation Steps

1. [ ] Research and document standard materials science equations with proper dimensional analysis
2. [ ] Define dimension constants for materials science (stress tensor, strain tensor, etc.)
3. [ ] Implement stress-strain relationships
   - Hooke's Law (elastic, with Young's modulus)
   - True stress/strain conversion
   - Yield criteria (von Mises, Tresca)
4. [ ] Implement fracture mechanics equations
   - Griffith criterion
   - Stress intensity factor
   - Paris law (fatigue crack growth)
   - J-integral (elastic-plastic fracture)
5. [ ] Implement fatigue equations
   - Basquin's law (high-cycle fatigue)
   - Coffin-Manson (low-cycle fatigue)
   - Goodman diagram
6. [ ] Implement creep equations
   - Norton power law
   - Larson-Miller parameter
   - Monkman-Grant relation
7. [ ] Implement diffusion equations
   - Fick's first law (steady-state)
   - Fick's second law (transient)
   - Arrhenius equation for diffusivity
8. [ ] Implement phase transformation equations
   - Avrami equation (JMAK)
   - Lever rule
   - Tie line rule
9. [ ] Implement hardening mechanisms
   - Hall-Petch relation
   - Taylor hardening
10. [ ] Add comprehensive LaTeX representations for all equations
11. [ ] Add assumptions and related equations links
12. [ ] Register all equations in database
13. [ ] Write unit tests for dimensional correctness
14. [ ] Add examples to docstrings
15. [ ] Update CONTINUITY.md

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | Add materials science equations section with ~15-25 equations |
| tests/test_equations.py | Add test cases for materials science equations |
| CONTINUITY.md | Mark task as DONE |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for dimensional consistency of all equations
- [ ] Test that variables have correct dimensions (stress, strain, energy, etc.)
- [ ] Test search_equations() finds materials equations
- [ ] Test get_equations(domain="materials_science") returns all materials equations
- [ ] Test filtering by tags (fracture, fatigue, creep, etc.)
- [ ] Verify LaTeX rendering is correct
- [ ] Manual verification of equation formulas against standard references (ASM Handbook, Callister textbook)

---

## Risks / Edge Cases

- **Risk 1: Tensor notation complexity** - Stress and strain are rank-2 tensors, but equations use scalar/effective values
  - Mitigation: Use effective/von Mises stress, document tensor vs scalar distinction in descriptions

- **Risk 2: Engineering vs true stress/strain** - Different conventions in literature
  - Mitigation: Clearly document which convention each equation uses, provide conversion equations

- **Risk 3: Empirical constants** - Many equations have material-specific constants (n, k, C, etc.)
  - Mitigation: Mark constants as dimensionless or with proper dimensions, note they're material-dependent in assumptions

- **Edge case: Temperature-dependent properties** - Creep, diffusion equations involve temperature
  - Handling: Include temperature as a variable with proper dimension

- **Edge case: Strain rate terms** - Some plasticity equations involve strain rate (dimension: 1/time)
  - Handling: Define strain rate dimension explicitly

- **Risk 4: Complex multi-variable equations** - Some equations (e.g., yield surfaces) involve multiple stress components
  - Mitigation: Use principal stresses or effective stress formulations

- **Risk 5: Unit compatibility with existing domains** - Need to ensure materials units work with mechanics/thermo equations
  - Mitigation: Use consistent SI base dimensions (Pa = kg/(m·s²))

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] ~15-25 materials science equations added to database
- [ ] All equations have proper dimensional metadata
- [ ] Tests pass (dimensional consistency verified)
- [ ] LaTeX representations added
- [ ] Assumptions and related equations documented
- [ ] Examples added to docstrings
- [ ] CONTINUITY.md updated

---

## Key Equations to Include

### Stress-Strain (4-5 equations)
1. **Hooke's Law**: σ = E·ε (elastic region)
2. **True Stress/Strain**: σ_true = σ_eng(1 + ε_eng)
3. **Von Mises Yield Criterion**: σ_VM = √((σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²)/√2
4. **Tresca Yield Criterion**: σ_yield = max(|σ1-σ2|, |σ2-σ3|, |σ3-σ1|)/2
5. **Poisson's Ratio**: ε_trans = -ν·ε_axial

### Fracture Mechanics (4 equations)
1. **Griffith Criterion**: σ_f = √(2Eγ/πa) (brittle fracture)
2. **Stress Intensity Factor**: K_I = Yσ√(πa)
3. **Paris Law**: da/dN = C(ΔK)^m (fatigue crack growth)
4. **Fracture Toughness Relation**: K_IC = σ_f√(πa)

### Fatigue (3 equations)
1. **Basquin's Law**: Δσ/2 = σ'_f(2N_f)^b (high-cycle fatigue)
2. **Coffin-Manson**: Δε_p/2 = ε'_f(2N_f)^c (low-cycle fatigue)
3. **Goodman Relation**: σ_a/σ_e + σ_m/σ_u = 1 (mean stress effect)

### Creep (3 equations)
1. **Norton Power Law**: ε̇ = Aσ^n exp(-Q/RT)
2. **Larson-Miller Parameter**: LMP = T(C + log(t_r))
3. **Monkman-Grant**: ε̇_min·t_r = C (minimum creep rate)

### Diffusion (3 equations)
1. **Fick's First Law**: J = -D(dC/dx)
2. **Fick's Second Law**: ∂C/∂t = D(∂²C/∂x²)
3. **Arrhenius (Diffusivity)**: D = D₀ exp(-Q/RT)

### Phase Transformations (2 equations)
1. **Avrami Equation**: f = 1 - exp(-kt^n) (JMAK model)
2. **Lever Rule**: f_α = (C_β - C₀)/(C_β - C_α)

### Hardening (2 equations)
1. **Hall-Petch**: σ_y = σ₀ + k_y/√d
2. **Taylor Hardening**: σ_y = σ₀ + αGbρ^(1/2)

**Total: ~20-22 equations**

---

## Dimensional Analysis Notes

Key dimensions needed:
- **Stress**: σ [M L^-1 T^-2] = Pa
- **Strain**: ε [dimensionless]
- **Young's Modulus**: E [M L^-1 T^-2] = Pa
- **Fracture Energy**: γ [M T^-2] = J/m²
- **Stress Intensity**: K [M L^-1/2 T^-2] = Pa·√m
- **Strain Rate**: ε̇ [T^-1]
- **Diffusivity**: D [L^2 T^-1]
- **Concentration**: C [amount L^-3] = mol/m³
- **Flux**: J [amount L^-2 T^-1] = mol/(m²·s)
- **Activation Energy**: Q [M L^2 T^-2] = J/mol
- **Grain Size**: d [L]
- **Dislocation Density**: ρ [L^-2]
- **Burgers Vector**: b [L]
- **Shear Modulus**: G [M L^-1 T^-2] = Pa

---

## Notes / Log

**[Initial]** - Research phase complete. Identified 20-22 key equations covering stress-strain, fracture, fatigue, creep, diffusion, phase transformations, and hardening mechanisms.

**[Design]** - Decided on single materials_science domain following existing pattern. Will use descriptive tags for sub-categorization. All equations will be added to database.py following the established format.

---
