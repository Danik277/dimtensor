# Plan: Chemical Kinetics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add a comprehensive set of chemical kinetics equations to the equation database, enabling dimensional verification and physics-informed modeling of reaction rates, equilibria, and catalysis.

---

## Background

Chemical kinetics studies reaction rates and mechanisms. This is critical for:
- Chemical engineering (reactor design)
- Atmospheric chemistry (pollutant modeling)
- Biochemistry (enzyme kinetics, drug metabolism)
- Combustion modeling
- Materials synthesis

The equations database currently has mechanics, thermodynamics, electromagnetism, quantum, optics, and acoustics domains, but lacks kinetics.

---

## Approach

### Option A: Create separate domain "kinetics" or "chemical_kinetics"
- Description: Add new domain specifically for kinetics equations
- Pros: Clean separation, follows existing pattern (mechanics, thermodynamics, etc.)
- Cons: May overlap with thermodynamics (equilibrium constants)

### Option B: Add as "chemistry" domain
- Description: Group all chemistry equations under one domain
- Pros: Simple, groups all chemistry equations together
- Cons: Could be too broad (spectroscopy, electrochemistry also chemistry)

### Option C: Use "kinetics" domain, relate to "thermodynamics"
- Description: Specific domain for kinetics, use related field to link thermo equations
- Pros: Specific and clear, can link to related thermo equations via related field
- Cons: None significant

### Decision: Option C - Create "kinetics" domain

Use domain="kinetics" for reaction kinetics and catalysis. Use related fields to link to thermodynamics equations (equilibrium, entropy). This follows the pattern of having specific, well-defined domains.

---

## Implementation Steps

1. [ ] Add dimension constants for kinetics in database.py
   - Rate constants (dimension varies by order)
   - Activation energy (energy/amount for J/mol)
   - Pre-exponential factor (same as rate constant)
   - Concentration (amount/volume)

2. [ ] Add integrated rate laws (3 equations)
   - Zero-order: [A] = [A]0 - kt
   - First-order: [A] = [A]0 * exp(-kt)
   - Second-order: 1/[A] = 1/[A]0 + kt

3. [ ] Add differential rate laws (3 equations)
   - Zero-order: d[A]/dt = -k
   - First-order: d[A]/dt = -k[A]
   - Second-order: d[A]/dt = -k[A]^2

4. [ ] Add Arrhenius equation
   - k = A * exp(-Ea/(RT))
   - Variables: k (rate constant), A (pre-exponential), Ea (activation energy), R (gas constant), T (temperature)

5. [ ] Add transition state theory (Eyring equation)
   - k = (kB*T/h) * exp(-ΔG‡/(RT))
   - Variables: kB (Boltzmann), h (Planck), ΔG‡ (activation Gibbs energy)

6. [ ] Add equilibrium equations (2 equations)
   - K = k_forward / k_reverse
   - K = exp(-ΔG°/(RT)) (thermodynamic definition)

7. [ ] Add Van't Hoff equation
   - d(ln K)/dT = ΔH°/(RT^2)
   - Integrated form: ln(K2/K1) = (ΔH°/R) * (1/T1 - 1/T2)

8. [ ] Add catalysis equations (2 equations)
   - Langmuir-Hinshelwood: r = k*KA*KB*PA*PB / (1 + KA*PA + KB*PB)^2
   - Michaelis-Menten: v = Vmax*[S] / (Km + [S])

9. [ ] Add collision theory equation
   - k = Z*p*exp(-Ea/(RT))
   - Z = collision frequency, p = steric factor

10. [ ] Add half-life equations (2 equations)
    - First-order: t1/2 = ln(2)/k
    - Second-order: t1/2 = 1/(k[A]0)

11. [ ] Add tests in tests/test_equations.py
    - Test each equation is registered
    - Test dimensions are correct for rate constants (vary by order)
    - Test domain filtering works for kinetics
    - Test search functionality

12. [ ] Update documentation
    - Add kinetics section to docs/guide/equations.md

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | ADD - kinetics equations section (after acoustics, around line 1075) |
| tests/test_equations.py | ADD - test cases for kinetics equations |
| docs/guide/equations.md | UPDATE - add kinetics domain section with examples |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit test that all kinetics equations are registered (count >= 15)
- [ ] Unit test that get_equations(domain="kinetics") returns kinetics equations only
- [ ] Unit test that search_equations("Arrhenius") finds Arrhenius equation
- [ ] Dimensional analysis tests:
  - Verify zero-order rate constant has dimension M/T (concentration/time)
  - Verify first-order rate constant has dimension 1/T (1/time)
  - Verify second-order rate constant has dimension 1/(M*T)
  - Verify activation energy has dimension energy/amount (J/mol)
- [ ] Integration test: verify Arrhenius equation relates k, A, Ea, T with correct units
- [ ] Manual verification: check LaTeX rendering looks correct in documentation

---

## Risks / Edge Cases

- **Risk**: Rate constant dimensions vary by reaction order (zero, first, second)
  - **Mitigation**: Define separate equations for each order with correct dimensions
  - **Example**: k_zero has dim amount/(length^3 * time), k_first has dim 1/time, k_second has dim length^3/(amount * time)

- **Risk**: Concentration dimension (M) uses chemistry units (mol/L) vs SI (mol/m^3)
  - **Mitigation**: Use Dimension(amount=1, length=-3) for concentration, which is mol/m^3 in SI
  - **Note**: Chemistry mol/L = 1000 mol/m^3, but dimension is same
  - **Scale handled by**: Units in chemistry.py module (molar = 1000 mol/m^3)

- **Risk**: Activation energy Ea typically given per mole vs per molecule
  - **Mitigation**: Define Ea with dimension(mass=1, length=2, time=-2, amount=-1) for J/mol
  - **Rationale**: Arrhenius equation uses RT (gas constant * temp), so Ea should be per mole

- **Risk**: Pre-exponential factor A has same dimensions as rate constant
  - **Mitigation**: Document this clearly in description field
  - **Note**: A dimensions vary by order (same as k for that order)

- **Edge case**: Equilibrium constant K can be dimensionless or have dimensions
  - **Handling**: Define K as dimensionless (activity-based definition)
  - **Note**: Concentration-based K has dimensions that depend on stoichiometry (aA + bB -> cC + dD gives K with dim M^(c+d-a-b))
  - **Justification**: Activity-based is thermodynamically rigorous

- **Edge case**: Michaelis-Menten (enzyme kinetics) overlaps with biochemistry
  - **Handling**: Include in kinetics domain, tag with both "enzyme" and "biochemistry"
  - **Related**: Can link to biophysics domain if it exists

- **Edge case**: Collision theory requires molecular quantities
  - **Handling**: Define collision frequency Z with dimension length^3/(time) for bimolecular collisions per volume per time
  - **Note**: Steric factor p is dimensionless

- **Risk**: Langmuir-Hinshelwood equation is complex with many parameters
  - **Mitigation**: Start with simplified form for surface catalysis, add detailed notes about assumptions
  - **Assumptions**: List in equation assumptions field

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] At least 15 kinetics equations registered
- [ ] Tests pass (pytest tests/test_equations.py)
- [ ] Documentation updated with kinetics examples
- [ ] Can retrieve equations: get_equations(domain="kinetics")
- [ ] Can search: search_equations("Arrhenius") returns results
- [ ] list_domains() includes "kinetics"
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**Key Design Decisions:**

1. **Rate constant dimensions**: Create separate equations for zero, first, second order with correct dimensions
   - Zero: amount/(length^3 * time) = mol/(m^3 * s)
   - First: 1/time = 1/s
   - Second: length^3/(amount * time) = m^3/(mol * s)

2. **Concentration**: Use Dimension(amount=1, length=-3) representing mol/m^3
   - Note about mol/L vs mol/m^3 conversion in documentation

3. **Activation energy**: Use J/mol dimension (mass=1, length=2, time=-2, amount=-1)
   - Consistent with RT term in Arrhenius equation

4. **Domain**: Use "kinetics" not "chemistry" (more specific, follows pattern)

5. **Equilibrium constant**: Treat K as dimensionless (activity-based definition)
   - More thermodynamically rigorous

**Equations to implement** (15 total):

Rate Laws (6):
1. Rate Law (Zero Order) - integrated
2. Rate Law (First Order) - integrated
3. Rate Law (Second Order) - integrated
4. Rate Law (Zero Order) - differential
5. Rate Law (First Order) - differential
6. Rate Law (Second Order) - differential

Temperature Dependence (4):
7. Arrhenius Equation
8. Eyring Equation (Transition State Theory)
9. Van't Hoff Equation
10. Collision Theory

Equilibrium (2):
11. Equilibrium Constant (kinetic definition)
12. Equilibrium Constant (thermodynamic definition)

Special Cases (3):
13. Half-life (First Order)
14. Half-life (Second Order)
15. Michaelis-Menten Equation

Optional (if time permits):
16. Langmuir-Hinshelwood (simplified)

**Related equations already in database:**
- Ideal Gas Law (thermodynamics) - can relate to collision theory
- First Law of Thermodynamics - relates to enthalpy in Van't Hoff
- Entropy Change - relates to Gibbs energy in Eyring

---
