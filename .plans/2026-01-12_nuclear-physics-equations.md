# Plan: Nuclear Physics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive nuclear physics equations module covering binding energy (SEMF), radioactive decay, Q-values, cross-sections, and related nuclear phenomena. Integrate seamlessly with existing equation database infrastructure.

---

## Background

Nuclear physics equations are fundamental to nuclear engineering, particle physics, medical physics, and astrophysics applications. The dimtensor library already has:
- Nuclear physics units (MeV, barn, becquerel) in `domains/nuclear.py`
- Atomic constants (m_p, m_n, m_u) in `constants/atomic.py`
- An equation database infrastructure in `equations/database.py`
- Related physics equations (quantum mechanics, relativity)

Adding nuclear physics equations will enable dimensional validation for:
- Nuclear reactor calculations
- Particle physics experiments
- Medical radiation dosimetry
- Astrophysical nucleosynthesis simulations

---

## Approach

### Option A: Add to existing equations/database.py
- Description: Extend the current database.py file with nuclear physics section
- Pros: Single file, follows current pattern, easy discovery
- Cons: File is already 1150+ lines, would add ~400 more lines

### Option B: Create equations/nuclear.py module
- Description: New module that auto-registers equations on import
- Pros: Better organization, cleaner separation by domain, scalable pattern
- Cons: Needs infrastructure for auto-registration, slightly more complex

### Decision: Option B - Create equations/nuclear.py module

This follows best practices for code organization and allows future equation domains to follow the same pattern. Will implement auto-registration using `register_equation()` at module level (same as current database.py pattern).

**Key design decisions:**
- Use standard nuclear physics symbols (A, Z, N, BE, Q, σ, λ, t₁/₂)
- Express binding energy in MeV units in examples
- Include both practical forms (with numerical coefficients) and symbolic forms
- Link related equations (e.g., half-life ↔ decay constant ↔ mean lifetime)
- Follow CODATA 2022 and NNDC standards for mass-energy conventions
- Use atomic mass unit (u) dimension: same as mass (M)

---

## Implementation Steps

1. [x] Research nuclear physics equations and dimensional analysis
2. [ ] Create `src/dimtensor/equations/nuclear.py` with sections:
   - **Binding energy equations:**
     - Semi-empirical mass formula (SEMF/Weizsäcker formula) - full form with 5 terms
     - Mass defect equation: Δm = Zm_p + Nm_n - M_nucleus
     - Binding energy: BE = Δm · c²
     - Binding energy per nucleon: BE/A
   - **Radioactive decay equations:**
     - Exponential decay law: N(t) = N₀ · exp(-λt)
     - Activity: A(t) = λN(t) = A₀ · exp(-λt)
     - Half-life relation: t₁/₂ = ln(2)/λ
     - Mean lifetime: τ = 1/λ
     - Bateman equations for decay chains (simple 2-step case)
   - **Nuclear reaction equations:**
     - Q-value: Q = (Σm_reactants - Σm_products)c²
     - Threshold energy for endothermic reactions
     - Kinetic energy conservation with Q-value
   - **Cross-section equations:**
     - Breit-Wigner resonance formula
     - Reaction rate: R = n₁n₂σv (for two species)
     - Mean free path: λ_mfp = 1/(nσ)
   - **Nuclear shell model:**
     - Magic numbers (descriptive, not formula)
     - Shell closure energy (parametric form)
   - **Fission/Fusion energy:**
     - Energy release in fission (using Q-value)
     - Coulomb barrier for fusion
     - Gamow factor (tunnel probability)

3. [ ] Update `src/dimtensor/equations/__init__.py`:
   - Import nuclear module to trigger auto-registration
   - Add to docstring examples

4. [ ] Create comprehensive tests in `tests/test_equations.py`:
   - Add `TestNuclearEquations` class
   - Test all equation dimensions are correct
   - Test variable dimensions for each equation
   - Test that equations are searchable by domain="nuclear"
   - Test related equations linkages
   - Test tags (decay, binding, cross-section, fission, fusion)

5. [ ] Add integration example in docstring showing:
   - Calculating binding energy for Fe-56 (most stable nucleus)
   - Decay calculation with actual isotope
   - Q-value calculation for D-T fusion reaction

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/nuclear.py | CREATE - nuclear physics equations module (~400 lines) |
| src/dimtensor/equations/__init__.py | UPDATE - import nuclear module to trigger registration |
| tests/test_equations.py | UPDATE - add TestNuclearEquations class (~150 lines) |

---

## Testing Strategy

### Dimension validation tests:
- [ ] Test SEMF terms have energy dimension (M·L²·T⁻²)
- [ ] Test mass defect has mass dimension (M)
- [ ] Test binding energy has energy dimension (M·L²·T⁻²)
- [ ] Test decay constant has frequency dimension (T⁻¹)
- [ ] Test activity has frequency dimension (T⁻¹)
- [ ] Test half-life has time dimension (T)
- [ ] Test Q-value has energy dimension (M·L²·T⁻²)
- [ ] Test cross-section has area dimension (L²)
- [ ] Test reaction rate has rate dimension (T⁻¹)
- [ ] Test mean free path has length dimension (L)

### Variable dimension tests:
- [ ] Test SEMF variables (A: dimensionless, Z: dimensionless, coefficients: energy)
- [ ] Test decay law variables (N: dimensionless count, λ: T⁻¹, t: T)
- [ ] Test Q-value variables (masses: M, c: L·T⁻¹, Q: M·L²·T⁻²)
- [ ] Test Breit-Wigner variables (E: M·L²·T⁻², Γ: M·L²·T⁻², σ: L²)

### Registry tests:
- [ ] Test all nuclear equations registered in database
- [ ] Test retrievable by domain="nuclear"
- [ ] Test searchable by keywords (binding, decay, fission, cross-section)
- [ ] Test tags filter correctly (decay, resonance, fusion, etc.)

### Relationship tests:
- [ ] Test related equations link correctly (half-life ↔ decay constant ↔ mean lifetime)
- [ ] Test Q-value equations link to energy-mass equivalence
- [ ] Test SEMF links to binding energy

### Real-world validation examples:
- [ ] Calculate Fe-56 binding energy per nucleon (~8.79 MeV/nucleon)
- [ ] Verify C-14 half-life calculation (5730 years)
- [ ] Check D-T fusion Q-value (17.6 MeV)

---

## Risks / Edge Cases

**Risk 1:** Mass-energy units confusion (u vs kg vs MeV/c²)
- **Mitigation:** Use SI base units (kg) for dimensions. Document that atomic mass unit (u) has mass dimension. Provide examples showing conversion: 1 u = 931.494 MeV/c² = 1.66054e-27 kg.

**Risk 2:** SEMF coefficients have energy units but are typically given as pure numbers (e.g., aᵥ = 15.75 MeV)
- **Mitigation:** Define coefficients with energy dimension. Document that the formula as written assumes A, Z are dimensionless particle counts. Users can provide coefficients as DimArray with MeV units.

**Risk 3:** Decay constant λ has same dimension as activity (becquerel)
- **Mitigation:** Document clearly that λ is a rate constant (per atom) while activity is total rate (per sample). Both have T⁻¹ dimension but different physical meanings.

**Risk 4:** Q-value can be positive (exothermic) or negative (endothermic)
- **Mitigation:** Document sign convention clearly. Q > 0: energy released, Q < 0: energy required.

**Risk 5:** Breit-Wigner formula involves Lorentzian shape - dimensionless ratio appears in denominator
- **Mitigation:** Carefully define all variables. Energy difference (E - E_R) and width Γ both have energy dimension, so ratio is dimensionless as expected.

**Edge case:** Shell model equations are mostly qualitative/descriptive
- **Handling:** Include magic numbers as constants (descriptive), and parametric shell closure energy formula if found in literature.

**Edge case:** Decay chains (Bateman equations) involve complex exponentials
- **Handling:** Implement simple 2-step decay chain case (parent → daughter → granddaughter). Note that general N-step case exists but is complex.

**Edge case:** Coulomb barrier involves nuclear radii
- **Handling:** Link to nuclear radius formula R = r₀A^(1/3) where r₀ ≈ 1.2 fm.

---

## Definition of Done

- [ ] src/dimtensor/equations/nuclear.py created with all equations
- [ ] All equations have correct dimensions and variable definitions
- [ ] Module imports successfully and auto-registers equations
- [ ] Comprehensive docstring with examples and references
- [ ] LaTeX formulas provided for all equations
- [ ] Related equations properly linked
- [ ] Tags correctly assigned for searchability
- [ ] Tests pass for all dimension checks
- [ ] Tests pass for registry integration
- [ ] Real-world examples validated (Fe-56 BE, C-14 decay, D-T fusion)
- [ ] Domain="nuclear" filter returns all nuclear equations
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

### Nuclear Physics Equation Inventory

**Binding Energy:**
1. Semi-empirical mass formula (Weizsäcker, 1935): BE = aᵥA - aₛA^(2/3) - aᴄZ(Z-1)/A^(1/3) - aₐ(A-2Z)²/A + δ(A,Z)
   - Volume term: aᵥ = 15.75 MeV
   - Surface term: aₛ = 17.8 MeV
   - Coulomb term: aᴄ = 0.711 MeV
   - Asymmetry term: aₐ = 23.7 MeV
   - Pairing term δ: +12/A^(1/2) (even-even), 0 (odd-A), -12/A^(1/2) (odd-odd)

2. Mass defect: Δm = Zm_p + Nm_n - M_nucleus
3. Binding energy: BE = Δm · c²
4. Binding energy per nucleon: BE/A

**Decay Laws:**
1. Exponential decay: N(t) = N₀ · exp(-λt)
2. Activity: A(t) = λN(t) = A₀ · exp(-λt)
3. Half-life: t₁/₂ = ln(2)/λ ≈ 0.693/λ
4. Mean lifetime: τ = 1/λ = t₁/₂/ln(2) ≈ 1.44 · t₁/₂
5. Bateman equation (2-step): N₂(t) = N₁₀ · λ₁/(λ₂-λ₁) · [exp(-λ₁t) - exp(-λ₂t)]

**Q-value and Reactions:**
1. Q-value definition: Q = (Σm_reactants - Σm_products)c²
2. Q-value with kinetic energy: Q = ΣKE_products - ΣKE_reactants
3. Threshold energy (endothermic): K_th = -Q(1 + m_products/m_target + Q/(2m_target·c²))

**Cross-sections:**
1. Breit-Wigner resonance: σ(E) = πλ² · g · Γₙ·Γᵧ/[(E-E_R)² + (Γ/2)²]
   - λ = h/(2π·p) = reduced wavelength
   - g = (2J+1)/[(2Jₐ+1)(2Jᵦ+1)] = spin factor
   - Γ = total width, Γₙ = neutron width, Γᵧ = gamma width
2. Reaction rate: R = n₁·n₂·σ·v (non-relativistic)
3. Mean free path: λ_mfp = 1/(n·σ)

**Fusion/Fission:**
1. Coulomb barrier: E_C = Z₁·Z₂·e²/(4πε₀·r)
2. Gamow factor (penetration): P ∝ exp(-2πη) where η = Z₁·Z₂·e²/(ℏv)
3. Nuclear radius: R = r₀·A^(1/3) where r₀ ≈ 1.2 fm

### References
- [Semi-empirical mass formula - Wikipedia](https://en.wikipedia.org/wiki/Semi-empirical_mass_formula)
- [Binding energy and Semi-empirical mass formula - Physics LibreTexts](https://phys.libretexts.org/Bookshelves/Nuclear_and_Particle_Physics/Introduction_to_Applied_Nuclear_Physics_(Cappellaro)/01:_Introduction_to_Nuclear_Physics/1.02:_Binding_energy_and_Semi-empirical_mass_formula)
- [Half-life - Wikipedia](https://en.wikipedia.org/wiki/Half-life)
- [Radioactive decay - Wikipedia](https://en.wikipedia.org/wiki/Radioactive_decay)
- [Q value (nuclear science) - Wikipedia](https://en.wikipedia.org/wiki/Q_value_(nuclear_science))
- [Q-value - Energetics of Nuclear Reactions | nuclear-power.com](https://www.nuclear-power.com/nuclear-power/nuclear-reactions/q-value-energetics-nuclear-reactions/)
- [Resonances - Physics LibreTexts](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Introductory_Quantum_Mechanics_(Fitzpatrick)/14:_Scattering_Theory/14.08:_Resonances)
- CODATA 2022: https://physics.nist.gov/cuu/Constants/
- NNDC (National Nuclear Data Center): https://www.nndc.bnl.gov/

### Dimensional Analysis

**Key dimensions:**
- Mass: [M]
- Energy: [M·L²·T⁻²]
- Length: [L]
- Time: [T]
- Area (cross-section): [L²]
- Frequency/Activity/Decay constant: [T⁻¹]
- Dimensionless: A (mass number), Z (atomic number), N (neutron number), particle counts

**Energy units in nuclear physics:**
- SI: joule (J) = kg·m²/s²
- Nuclear: MeV = 1.602176634e-13 J
- Mass-energy: u = 931.494 MeV/c² = 1.66054e-27 kg

**Cross-section units:**
- SI: m²
- Nuclear: barn (b) = 1e-28 m²

---
