# Plan: Plasma Physics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add comprehensive plasma physics equations to the equation database, covering MHD (ideal and resistive), kinetic theory (Vlasov equation), characteristic scales (Debye length, plasma frequency, gyroradius), wave physics (Alfvén velocity), and dimensionless parameters (Lundquist, magnetic Reynolds numbers).

---

## Background

Plasma physics is a fundamental domain for fusion energy, space physics, astrophysics, and industrial plasma applications. The equation database currently covers mechanics, thermodynamics, electromagnetism, quantum mechanics, relativity, optics, acoustics, and fluid dynamics, but lacks plasma-specific equations.

Plasma equations involve unique physical phenomena:
- **Magnetohydrodynamics (MHD)**: Describes plasma as a conducting fluid in magnetic fields
- **Kinetic theory**: Vlasov equation describes particle distribution functions
- **Characteristic scales**: Debye length (shielding), gyroradius (particle orbits), skin depth
- **Wave physics**: Alfvén waves, plasma oscillations
- **Dimensionless parameters**: Lundquist number, magnetic Reynolds number, beta parameter

These equations are essential for dimensional analysis in fusion reactor design, space plasma modeling, and plasma-assisted manufacturing.

---

## Approach

### Decision: Add to existing src/dimtensor/equations/database.py

Following the existing pattern in database.py where equations are organized by domain (mechanics, thermodynamics, electromagnetism, etc.). Add a new section for plasma physics equations at the end of the file, using the same Equation dataclass and register_equation() pattern.

**Key design decisions:**

1. **SI units by default**: Use SI dimensions throughout, but document CGS variants in notes/assumptions where relevant (especially for magnetic field)

2. **Organize by category**:
   - MHD equations (ideal and resistive)
   - Kinetic theory (Vlasov)
   - Characteristic scales (Debye length, gyroradius, skin depth)
   - Wave physics (Alfvén velocity, plasma frequency)
   - Dimensionless parameters (Lundquist, magnetic Reynolds, beta)

3. **CGS vs SI handling**: Note that plasma physics historically uses CGS-Gaussian units, but dimtensor uses SI. Document conversion factors in equation notes/assumptions where needed.

4. **Vector notation**: Use descriptive formula strings that indicate vector operations (e.g., "B x v" for cross products, "grad(P)" for gradients)

5. **Plasma-specific dimensions**: Introduce number density [L⁻³], particle distributions [L⁻⁶·T³], conductivity [M⁻¹·L⁻³·T³·I²]

---

## Implementation Steps

1. [ ] Research and document all plasma equations with proper references (NRL Plasma Formulary, Goedbloed & Poedts, Chen's "Introduction to Plasma Physics")

2. [ ] Add dimensional definitions for plasma-specific quantities:
   - Number density: `_N_DENSITY = Dimension(length=-3)`
   - Conductivity: `_CONDUCTIVITY = Dimension(mass=-1, length=-3, time=3, current=2)`
   - Distribution function: `_DIST_FUNC = Dimension(length=-6, time=3)` (phase space density)

3. [ ] Add MHD equations (ideal):
   - Ideal Ohm's Law: `E = -v x B`
   - Magnetic flux freezing: `dB/dt = curl(v x B)`
   - Ideal MHD equilibrium: `J x B = grad(P)`

4. [ ] Add MHD equations (resistive):
   - Resistive Ohm's Law: `E + v x B = eta*J`
   - Magnetic diffusion equation: `dB/dt = curl(v x B) + (eta/mu_0)*laplacian(B)`

5. [ ] Add kinetic theory equations:
   - Vlasov equation: `df/dt + v·grad(f) + (q/m)*(E + v x B)·grad_v(f) = 0`

6. [ ] Add characteristic scales:
   - Debye length: `lambda_D = sqrt(epsilon_0*k_B*T_e/(n_e*e^2))`
   - Plasma frequency: `omega_pe = sqrt(n_e*e^2/(epsilon_0*m_e))`
   - Electron gyrofrequency: `omega_ce = e*B/m_e`
   - Ion gyrofrequency: `omega_ci = Z*e*B/m_i`
   - Electron gyroradius: `r_Le = v_th_e/omega_ce`
   - Ion gyroradius: `r_Li = v_th_i/omega_ci`
   - Inertial length (c/omega_pe): `delta_e = c/omega_pe`

7. [ ] Add wave physics:
   - Alfvén velocity: `v_A = B/sqrt(mu_0*rho)`
   - Alfvén transit time: `tau_A = L/v_A`

8. [ ] Add dimensionless parameters:
   - Lundquist number: `S = tau_R/tau_A = mu_0*L*v_A/eta`
   - Magnetic Reynolds number: `R_m = mu_0*sigma*L*v`
   - Plasma beta: `beta = 2*mu_0*P/B^2 = P/(B^2/(2*mu_0))`
   - Magnetic mirror ratio: `R_m = B_max/B_min`

9. [ ] Add each equation using register_equation() with:
   - Descriptive name
   - Formula string
   - Variables dict with correct dimensions
   - domain="plasma"
   - Appropriate tags (mhd, kinetic, scale, wave, dimensionless)
   - Description text
   - LaTeX formula
   - Assumptions list (e.g., "quasi-neutral", "collisionless", "non-relativistic")
   - Related equations list

10. [ ] Update tests in `tests/test_equations.py`:
    - Test that plasma equations are registered
    - Test dimensional consistency for each equation
    - Test filtering by domain="plasma"
    - Test searching for plasma-related terms

11. [ ] Document CGS-SI conversion notes in equation assumptions where relevant

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | ADD - Plasma physics equations section at end (after acoustics/fluids) |
| tests/test_equations.py | UPDATE - Add tests for plasma equations |

---

## Testing Strategy

### Equation registration tests:
- [ ] Test all plasma equations are registered and retrievable
- [ ] Test `get_equations(domain="plasma")` returns all plasma equations
- [ ] Test `search_equations("plasma")` finds relevant equations
- [ ] Test `search_equations("Debye")` finds Debye length equation
- [ ] Test `search_equations("Alfven")` finds Alfvén velocity
- [ ] Test `search_equations("MHD")` finds MHD equations

### Dimensional consistency tests:
- [ ] Test Debye length has dimension [L]
- [ ] Test plasma frequency has dimension [T⁻¹]
- [ ] Test gyrofrequency has dimension [T⁻¹]
- [ ] Test gyroradius has dimension [L]
- [ ] Test Alfvén velocity has dimension [L·T⁻¹]
- [ ] Test Lundquist number is dimensionless
- [ ] Test magnetic Reynolds number is dimensionless
- [ ] Test plasma beta is dimensionless
- [ ] Test Vlasov equation variables are dimensionally consistent

### Variable dimension tests:
- [ ] Test number density has dimension [L⁻³]
- [ ] Test distribution function has dimension [L⁻⁶·T³]
- [ ] Test conductivity has dimension [M⁻¹·L⁻³·T³·I²]
- [ ] Test magnetic field in SI has dimension [M·T⁻²·I⁻¹]
- [ ] Test resistivity has dimension [M·L³·T⁻³·I⁻²]

### Equation relationships:
- [ ] Test that ideal Ohm's law and resistive Ohm's law are related
- [ ] Test that Debye length and plasma frequency are related
- [ ] Test that Alfvén velocity and Lundquist number are related
- [ ] Test that gyrofrequency and gyroradius are related

### LaTeX and metadata:
- [ ] Test all equations have non-empty LaTeX strings
- [ ] Test all equations have descriptions
- [ ] Test all equations have appropriate tags
- [ ] Test all MHD equations have "mhd" tag
- [ ] Test all dimensionless parameters have "dimensionless" tag

---

## Risks / Edge Cases

**Risk 1:** CGS vs SI unit confusion for magnetic field
- **Mitigation:** Use SI throughout (B in Tesla = kg/(A·s²)), document CGS conversions in assumptions. Note that CGS Gauss = 1e-4 Tesla. Add assumption field noting "SI units: B in Tesla" where relevant.

**Risk 2:** Plasma frequency formula varies by convention (factor of 2π)
- **Mitigation:** Use angular frequency ω_pe (rad/s) as standard, note that ordinary frequency f = ω/(2π). Document in description field.

**Risk 3:** Distribution function dimensions depend on normalization
- **Mitigation:** Use phase space density f(x,v) with dimension [L⁻⁶·T³] (6D phase space: 3 position + 3 velocity). Document normalization in assumptions.

**Risk 4:** Resistivity η and conductivity σ are inverses, dimension confusion
- **Mitigation:** Define both clearly: η [M·L³·T⁻³·I⁻²], σ [M⁻¹·L⁻³·T³·I²]. Test that η·σ = 1 dimensionally.

**Risk 5:** Thermal velocity appears in gyroradius but isn't uniquely defined
- **Mitigation:** Use v_th = sqrt(k_B*T/m) as standard (not RMS, not most probable). Document in assumptions as "thermal velocity = sqrt(k_B*T/m)".

**Edge case:** Quasi-neutrality (n_e ≈ Z*n_i) vs exact charge neutrality
- **Handling:** Note quasi-neutrality assumption where relevant (plasma frequency, Debye length use n_e, not total charge density).

**Edge case:** Single fluid vs two-fluid MHD
- **Handling:** Start with single-fluid MHD (most common), note that two-fluid effects (Hall term, electron inertia) can be added in future versions.

**Edge case:** Magnetic mirror equation has different conventions
- **Handling:** Use simple ratio R_m = B_max/B_min, document that loss cone angle θ_lc = arcsin(sqrt(1/R_m)).

**Edge case:** Lundquist number vs magnetic Prandtl number
- **Handling:** Define Lundquist number S (standard in fusion), note that magnetic Prandtl number Pm = ν/η is related but different.

---

## Definition of Done

- [ ] All plasma equations added to database.py
- [ ] All equations have correct dimensions and formulas
- [ ] Equations organized by category with clear section headers
- [ ] All equations use register_equation() pattern
- [ ] Tests added to test_equations.py for plasma domain
- [ ] All tests pass
- [ ] LaTeX formulas provided for all equations
- [ ] Descriptions and assumptions documented
- [ ] Related equations cross-referenced
- [ ] CGS-SI conversion notes added where relevant
- [ ] Code follows existing patterns in database.py
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

### Plasma Physics Equations to Implement

**MHD Equations (Ideal):**
1. Ideal Ohm's Law: E + v × B = 0
2. Magnetic flux freezing: ∂B/∂t = ∇×(v×B)
3. Ideal MHD equilibrium: J×B = ∇P

**MHD Equations (Resistive):**
4. Resistive Ohm's Law: E + v×B = ηJ
5. Magnetic diffusion: ∂B/∂t = ∇×(v×B) + (η/μ₀)∇²B

**Kinetic Theory:**
6. Vlasov equation: ∂f/∂t + v·∇f + (q/m)(E + v×B)·∇ᵥf = 0

**Characteristic Scales:**
7. Debye length: λD = √(ε₀kᵦTₑ/(nₑe²))
8. Plasma frequency (electron): ωₚₑ = √(nₑe²/(ε₀mₑ))
9. Plasma frequency (ion): ωₚᵢ = √(nᵢZ²e²/(ε₀mᵢ))
10. Electron gyrofrequency: ωcₑ = eB/mₑ
11. Ion gyrofrequency: ωcᵢ = ZeB/mᵢ
12. Electron gyroradius: rLₑ = vₜₕ,ₑ/ωcₑ = mₑvₜₕ,ₑ/(eB)
13. Ion gyroradius: rLᵢ = vₜₕ,ᵢ/ωcᵢ = mᵢvₜₕ,ᵢ/(ZeB)
14. Inertial length (skin depth): δₑ = c/ωₚₑ = √(ε₀mₑc²/(nₑe²))

**Wave Physics:**
15. Alfvén velocity: vA = B/√(μ₀ρ)
16. Alfvén transit time: τA = L/vA

**Dimensionless Parameters:**
17. Lundquist number: S = τR/τA = μ₀LvA/η = μ₀σLvA
18. Magnetic Reynolds number: Rm = μ₀σLv
19. Plasma beta: β = 2μ₀P/B² = P/(B²/(2μ₀))
20. Magnetic mirror ratio: Rm = Bmax/Bmin

### Dimensional Analysis Reference

SI Dimensions:
- Electric field E: [M·L·T⁻³·I⁻¹] (V/m)
- Magnetic field B: [M·T⁻²·I⁻¹] (T = kg/(A·s²))
- Current density J: [I·L⁻²] (A/m²)
- Resistivity η: [M·L³·T⁻³·I⁻²] (Ω·m)
- Conductivity σ: [M⁻¹·L⁻³·T³·I²] (S/m = (Ω·m)⁻¹)
- Number density n: [L⁻³] (m⁻³)
- Distribution function f: [L⁻⁶·T³] (m⁻⁶·s³ in phase space)
- Temperature T: [Θ] (K)
- Pressure P: [M·L⁻¹·T⁻²] (Pa)
- Velocity v: [L·T⁻¹] (m/s)
- Density ρ: [M·L⁻³] (kg/m³)

CGS-SI Conversion:
- B(Gauss) = 1e-4 × B(Tesla)
- CGS-Gaussian uses different electromagnetic dimensions (no separate current dimension)

### Key References
- NRL Plasma Formulary (2023): https://www.nrl.navy.mil/News-Media/Publications/nrl-plasma-formulary/
- Chen, "Introduction to Plasma Physics and Controlled Fusion"
- Goedbloed & Poedts, "Principles of Magnetohydrodynamics"
- Freidberg, "Ideal MHD"

---
