# Plan: Solid State Physics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add solid state physics equations to the equation database, covering electronic band structure, phonons, transport properties, and superconductivity. This enables dimensional validation for condensed matter physics, materials science, and semiconductor device modeling applications.

---

## Background

Solid state physics is a fundamental domain not yet represented in the equation database (currently covers mechanics, thermodynamics, electromagnetism, fluids, relativity, quantum, optics, acoustics). Solid state equations are essential for:
- Semiconductor device physics (band gaps, effective mass, carrier transport)
- Materials characterization (phonon dispersion, lattice dynamics)
- Electronic structure calculations (density of states, Fermi surfaces)
- Superconductivity and low-temperature physics
- Condensed matter theory and computational materials science

The equations span electronic properties (Bloch's theorem, band structure), lattice dynamics (phonons, Debye model), transport (Drude model, Hall effect), and many-body effects (BCS superconductivity).

---

## Approach

### Decision: Add solid_state domain to equations/database.py

Following the existing pattern in database.py where domains like "quantum", "optics", and "acoustics" are registered in the same file. This keeps all equations centralized and searchable.

**Key design decisions:**

1. **Crystal momentum (k)**: Use dimension of inverse length [L⁻¹] (wave vector). Note: Crystal momentum ℏk has momentum dimensions [M·L·T⁻¹].

2. **Energy gaps and effective mass**: Energy has standard dimension [M·L²·T⁻²]. Effective mass has dimension [M] but represents m* = m/ratio where ratio is dimensionless.

3. **Phonon frequencies**: Angular frequency [T⁻¹], relates to energy via ℏω.

4. **Density of states**: DOS has dimension [energy⁻¹] = [M⁻¹·L⁻²·T²] in 3D. DOS varies by dimensionality (1D, 2D, 3D).

5. **Conductivity and Hall coefficient**:
   - Conductivity σ: [M⁻¹·L⁻³·T³·I²]
   - Hall coefficient RH: [L³·I⁻¹·T⁻¹]

6. **BCS gap**: Energy dimension [M·L²·T⁻²], relates to critical temperature via kB.

7. **Dimensionless equations**: For Kronig-Penney and dispersion relations that produce dimensionless quantities (like cos(ka)), represent using the underlying dimensional quantities.

---

## Implementation Steps

1. [ ] Research and document all equation formulas with proper dimensional analysis
2. [ ] Define dimension constants for solid state physics:
   - Wave vector: _K = Dimension(length=-1)
   - DOS (3D): _DOS_3D = Dimension(mass=-1, length=-2, time=2)
   - Conductivity: _CONDUCTIVITY = Dimension(mass=-1, length=-3, time=3, current=2)
   - Hall coefficient: _HALL_COEFF = Dimension(length=3, current=-1, time=-1)

3. [ ] Add equations to database.py in new "solid_state" domain section:

   **Band Structure & Electronic Properties:**
   - [ ] Bloch's Theorem (wavefunction form: ψ_k(r) = e^(ik·r) u_k(r))
   - [ ] Kronig-Penney Model (dispersion relation for 1D periodic potential)
   - [ ] Effective Mass (m* = ℏ²/(d²E/dk²))
   - [ ] Fermi Energy (E_F = (ℏ²/2m)(3π²n)^(2/3))
   - [ ] Fermi Wavevector (k_F = (3π²n)^(1/3))

   **Density of States:**
   - [ ] DOS 3D Free Electrons (g(E) = (1/2π²)(2m/ℏ²)^(3/2) √E)
   - [ ] DOS 2D Free Electrons (g(E) = m/(πℏ²))
   - [ ] DOS at Fermi Level (g(E_F))

   **Lattice Dynamics & Phonons:**
   - [ ] Phonon Dispersion (ω = v_s |k| for Debye model)
   - [ ] Debye Frequency (ω_D)
   - [ ] Debye Temperature (Θ_D = ℏω_D/k_B)
   - [ ] Debye Heat Capacity (low-T: C_V ∝ T³)

   **Transport Properties:**
   - [ ] Drude Conductivity (σ = ne²τ/m)
   - [ ] Drude Mobility (μ = eτ/m)
   - [ ] Hall Coefficient (R_H = 1/(ne) for classical)
   - [ ] Hall Resistance (R_H = V_H/(I·B))
   - [ ] Mean Free Path (ℓ = v_F τ)

   **Superconductivity:**
   - [ ] BCS Energy Gap (Δ(T=0) = 1.76 k_B T_c)
   - [ ] BCS Gap-Temperature Relation (2Δ(0) = 3.52 k_B T_c)
   - [ ] Cooper Pair Binding Energy (E_b = 2Δ)
   - [ ] Coherence Length (ξ_0 = ℏv_F/(πΔ))
   - [ ] London Penetration Depth (λ_L = √(m/(μ₀ns e²)))

4. [ ] Add metadata for all equations:
   - LaTeX representations
   - Assumptions (e.g., "free electron approximation", "weak-coupling limit")
   - Related equations (cross-references within solid state and to quantum domain)
   - Tags for searchability

5. [ ] Add comprehensive tests in tests/test_equations.py:
   - [ ] Test that solid_state domain is registered
   - [ ] Test retrieval of solid state equations by domain
   - [ ] Test searching for solid state equations by tags
   - [ ] Test dimensional correctness for key equations
   - [ ] Test that related equations are properly linked

6. [ ] Update list_domains() output to include "solid_state"

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | UPDATE - Add solid_state domain section with ~20 equations |
| tests/test_equations.py | UPDATE - Add TestSolidStateEquations class with tests |

---

## Testing Strategy

### Equation registration tests:
- [ ] Test that all solid state equations are registered
- [ ] Test that get_equations(domain="solid_state") returns correct count
- [ ] Test that list_domains() includes "solid_state"

### Dimensional correctness tests:
- [ ] Test Fermi energy has energy dimension [M·L²·T⁻²]
- [ ] Test effective mass has mass dimension [M]
- [ ] Test wave vector has inverse length dimension [L⁻¹]
- [ ] Test DOS 3D has dimension [M⁻¹·L⁻²·T²]
- [ ] Test conductivity has correct dimension [M⁻¹·L⁻³·T³·I²]
- [ ] Test Hall coefficient has dimension [L³·I⁻¹·T⁻¹]
- [ ] Test phonon frequency has frequency dimension [T⁻¹]
- [ ] Test Debye temperature has temperature dimension [Θ]
- [ ] Test BCS gap has energy dimension [M·L²·T⁻²]
- [ ] Test coherence length has length dimension [L]

### Search and filter tests:
- [ ] Test search_equations("band") finds band-related equations
- [ ] Test search_equations("phonon") finds phonon equations
- [ ] Test search_equations("superconductivity") finds BCS equations
- [ ] Test get_equations(tags=["transport"]) filters correctly
- [ ] Test get_equations(tags=["fermi"]) returns Fermi-related equations

### Metadata tests:
- [ ] Test that LaTeX strings are present for all equations
- [ ] Test that assumptions are documented (e.g., "parabolic bands", "isotropic")
- [ ] Test that related equations link to quantum domain where appropriate

### Integration tests:
- [ ] Test that solid state equations can be retrieved alongside quantum equations
- [ ] Test cross-domain searches (quantum + solid_state)

---

## Risks / Edge Cases

**Risk 1:** Crystal momentum (ℏk) vs wave vector (k) confusion.
- **Mitigation:** Use k with dimension [L⁻¹] for wave vectors. Document clearly when ℏk (momentum dimension) vs k (inverse length) is used. Most equations use k directly.

**Risk 2:** Effective mass is dimensionally mass but represents a ratio to electron mass.
- **Mitigation:** Treat m* as having dimension [M]. In practice, effective mass is often quoted as a dimensionless ratio (m*/m_e), but in equations it appears with mass dimensions.

**Risk 3:** Density of states dimension varies with system dimensionality (1D/2D/3D).
- **Mitigation:** Specify dimensionality in equation names (DOS 3D, DOS 2D) and document the dimension for each.

**Risk 4:** BCS equations have multiple conventions (Δ vs 2Δ, numerical factors).
- **Mitigation:** Use standard weak-coupling BCS convention: 2Δ(0) = 3.52 k_B T_c. Document this is weak-coupling limit.

**Risk 5:** Hall effect sign convention varies (classical vs quantum Hall).
- **Mitigation:** Start with classical Hall effect (R_H = 1/ne). Note in assumptions this is classical, single-carrier.

**Edge case:** Kronig-Penney model produces transcendental equation, not closed form.
- **Handling:** Express dispersion relation as the condition equation: cos(ka) = cos(qₑa) + P·sin(qₑa)/(qₑa), where P is dimensionless parameter.

**Edge case:** Phonon dispersion is often multi-branch (acoustic, optical).
- **Handling:** Start with simple Debye model (ω = v_s|k|) for acoustic branch. Document as simplified model.

**Edge case:** Some equations (like Bloch's theorem) are structural, not calculational.
- **Handling:** Include them for completeness with formula field describing the form, even if not a simple algebraic relation.

---

## Definition of Done

- [ ] All 20+ solid state equations added to database.py
- [ ] All equations have correct dimensions
- [ ] All equations have LaTeX, assumptions, and related fields
- [ ] "solid_state" domain registered and searchable
- [ ] Tests added and passing (TestSolidStateEquations class)
- [ ] list_domains() returns "solid_state"
- [ ] search_equations() works for solid state keywords
- [ ] Documentation strings reference standard solid state physics sources
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-12** - Plan created after researching:
- Existing equation patterns in database.py
- Dimensional analysis for solid state quantities
- Standard solid state physics textbooks (Ashcroft & Mermin, Kittel)
- Online resources on Bloch theorem, Kronig-Penney, BCS theory

Key design decisions:
- Use wave vector k [L⁻¹] consistently (not crystal momentum ℏk)
- DOS dimension varies by dimensionality, specify in equation name
- BCS equations use weak-coupling limit (standard convention)
- Start with classical Hall effect (quantum Hall could be future extension)
- Debye model for phonons (simpler than full dispersion relations)
- Effective mass treated as having mass dimension [M] in equations

Sources consulted:
- Bloch's theorem - Wikipedia
- Kronig-Penney Model - IPLTS
- BCS theory - Wikipedia
- Drude model - Wikipedia
- Debye model - Wikipedia

Estimated implementation time: 2-3 hours (20 equations with full metadata)
Estimated complexity: MEDIUM (requires careful dimensional analysis but follows established patterns)

---

## Detailed Equation Specifications

### Band Structure & Electronic Properties

1. **Bloch's Theorem**
   - Formula: ψ_k(r) = e^(ik·r) u_k(r)
   - Variables: k (wave vector [L⁻¹]), r (position [L]), ψ (wavefunction [L⁻³/²]), u (periodic function [L⁻³/²])
   - LaTeX: \psi_k(\vec{r}) = e^{i\vec{k}\cdot\vec{r}} u_k(\vec{r})
   - Tags: band structure, fundamental, bloch
   - Assumptions: Periodic potential, single-particle approximation

2. **Kronig-Penney Dispersion**
   - Formula: cos(ka) = cos(q_E·a) + (P/q_E·a)·sin(q_E·a)
   - Variables: k [L⁻¹], a (lattice constant [L]), q_E = √(2mE)/ℏ [L⁻¹], P (dimensionless)
   - LaTeX: \cos(ka) = \cos(q_E a) + \frac{P}{q_E a}\sin(q_E a)
   - Tags: band structure, periodic potential, 1D, kronig penney
   - Assumptions: 1D periodic potential, delta function barriers

3. **Effective Mass**
   - Formula: 1/m* = (1/ℏ²)·d²E/dk²
   - Variables: m* [M], E [M·L²·T⁻²], k [L⁻¹], ℏ [M·L²·T⁻¹]
   - LaTeX: \frac{1}{m^*} = \frac{1}{\hbar^2}\frac{d^2E}{dk^2}
   - Tags: effective mass, band structure, semiconductor
   - Assumptions: Parabolic bands near extrema

4. **Fermi Energy (3D)**
   - Formula: E_F = (ℏ²/2m)·(3π²n)^(2/3)
   - Variables: E_F [M·L²·T⁻²], ℏ [M·L²·T⁻¹], m [M], n [L⁻³]
   - LaTeX: E_F = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}
   - Tags: fermi, energy, free electrons, 3D
   - Assumptions: Free electron gas, zero temperature

5. **Fermi Wavevector**
   - Formula: k_F = (3π²n)^(1/3)
   - Variables: k_F [L⁻¹], n [L⁻³]
   - LaTeX: k_F = (3\pi^2 n)^{1/3}
   - Tags: fermi, wavevector, free electrons
   - Related: Fermi Energy (3D)

### Density of States

6. **DOS 3D Free Electrons**
   - Formula: g(E) = (V/2π²)·(2m/ℏ²)^(3/2)·√E
   - Variables: g [M⁻¹·L⁻²·T²], V [L³], m [M], E [M·L²·T⁻²], ℏ [M·L²·T⁻¹]
   - LaTeX: g(E) = \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}
   - Tags: dos, density of states, 3D, free electrons

7. **DOS 2D Free Electrons**
   - Formula: g(E) = (A·m)/(πℏ²)
   - Variables: g [M⁻¹·L⁻²·T²], A [L²], m [M], ℏ [M·L²·T⁻¹]
   - LaTeX: g(E) = \frac{Am}{\pi\hbar^2}
   - Tags: dos, density of states, 2D, free electrons
   - Assumptions: Constant DOS, parabolic bands

### Phonons & Lattice Dynamics

8. **Phonon Dispersion (Debye)**
   - Formula: ω = v_s·|k|
   - Variables: ω [T⁻¹], v_s [L·T⁻¹], k [L⁻¹]
   - LaTeX: \omega = v_s |k|
   - Tags: phonon, dispersion, debye, acoustic
   - Assumptions: Linear dispersion, acoustic branch only

9. **Debye Temperature**
   - Formula: Θ_D = (ℏω_D)/k_B
   - Variables: Θ_D [Θ], ℏ [M·L²·T⁻¹], ω_D [T⁻¹], k_B [M·L²·T⁻²·Θ⁻¹]
   - LaTeX: \Theta_D = \frac{\hbar\omega_D}{k_B}
   - Tags: debye, temperature, phonon
   - Related: Phonon Dispersion (Debye)

10. **Debye Heat Capacity (low T)**
    - Formula: C_V = (12π⁴/5)·N·k_B·(T/Θ_D)³
    - Variables: C_V [M·L²·T⁻²·Θ⁻¹], N (dimensionless), T [Θ], Θ_D [Θ], k_B [M·L²·T⁻²·Θ⁻¹]
    - LaTeX: C_V = \frac{12\pi^4}{5}Nk_B\left(\frac{T}{\Theta_D}\right)^3
    - Tags: debye, heat capacity, low temperature
    - Assumptions: T << Θ_D

### Transport Properties

11. **Drude Conductivity**
    - Formula: σ = n·e²·τ/m
    - Variables: σ [M⁻¹·L⁻³·T³·I²], n [L⁻³], e [I·T], τ [T], m [M]
    - LaTeX: \sigma = \frac{ne^2\tau}{m}
    - Tags: drude, conductivity, transport
    - Assumptions: Classical free electrons, constant scattering time

12. **Drude Mobility**
    - Formula: μ = e·τ/m
    - Variables: μ [I·T²·M⁻¹], e [I·T], τ [T], m [M]
    - LaTeX: \mu = \frac{e\tau}{m}
    - Tags: drude, mobility, transport
    - Related: Drude Conductivity

13. **Hall Coefficient (classical)**
    - Formula: R_H = 1/(n·e)
    - Variables: R_H [L³·I⁻¹·T⁻¹], n [L⁻³], e [I·T]
    - LaTeX: R_H = \frac{1}{ne}
    - Tags: hall effect, transport, classical
    - Assumptions: Single carrier type, classical limit

14. **Hall Voltage**
    - Formula: V_H = (I·B·R_H)/d
    - Variables: V_H [M·L²·T⁻³·I⁻¹], I [I], B [M·T⁻²·I⁻¹], R_H [L³·I⁻¹·T⁻¹], d [L]
    - LaTeX: V_H = \frac{IBR_H}{d}
    - Tags: hall effect, voltage, transport

15. **Mean Free Path**
    - Formula: ℓ = v_F·τ
    - Variables: ℓ [L], v_F [L·T⁻¹], τ [T]
    - LaTeX: \ell = v_F\tau
    - Tags: transport, scattering, mean free path

### Superconductivity (BCS Theory)

16. **BCS Energy Gap (zero T)**
    - Formula: Δ(0) = 1.76·k_B·T_c
    - Variables: Δ [M·L²·T⁻²], k_B [M·L²·T⁻²·Θ⁻¹], T_c [Θ]
    - LaTeX: \Delta(0) = 1.76 k_B T_c
    - Tags: bcs, superconductivity, energy gap
    - Assumptions: Weak-coupling limit

17. **BCS Gap Ratio**
    - Formula: 2Δ(0)/(k_B·T_c) = 3.52
    - Dimensionless relation
    - LaTeX: \frac{2\Delta(0)}{k_B T_c} = 3.52
    - Tags: bcs, superconductivity, weak coupling
    - Related: BCS Energy Gap (zero T)

18. **Cooper Pair Binding Energy**
    - Formula: E_b = 2·Δ
    - Variables: E_b [M·L²·T⁻²], Δ [M·L²·T⁻²]
    - LaTeX: E_b = 2\Delta
    - Tags: bcs, cooper pairs, superconductivity

19. **BCS Coherence Length**
    - Formula: ξ_0 = ℏ·v_F/(π·Δ)
    - Variables: ξ_0 [L], ℏ [M·L²·T⁻¹], v_F [L·T⁻¹], Δ [M·L²·T⁻²]
    - LaTeX: \xi_0 = \frac{\hbar v_F}{\pi\Delta}
    - Tags: bcs, coherence length, superconductivity

20. **London Penetration Depth**
    - Formula: λ_L = √(m/(μ₀·n_s·e²))
    - Variables: λ_L [L], m [M], μ₀ [M·L·T⁻²·I⁻²], n_s [L⁻³], e [I·T]
    - LaTeX: \lambda_L = \sqrt{\frac{m}{\mu_0 n_s e^2}}
    - Tags: superconductivity, london, penetration depth

---
