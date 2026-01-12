# Plan: Statistical Mechanics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive statistical mechanics equations module that extends the equation database with fundamental statistical mechanics formulas including distribution functions, partition functions, free energies, and ensemble-related equations.

---

## Background

dimtensor currently has a thermodynamics section in equations/database.py with basic thermodynamic equations (Ideal Gas Law, First Law, Entropy Change). However, it lacks statistical mechanics equations that bridge microscopic physics with macroscopic thermodynamics.

Statistical mechanics is critical for:
- Physics simulations (molecular dynamics, Monte Carlo)
- Materials science (phase transitions, critical phenomena)
- Machine learning (Boltzmann machines, energy-based models)
- Quantum computing (quantum statistical mechanics)
- Chemistry (chemical equilibria, reaction rates)

The task requires adding 8 categories of stat mech equations:
1. **Boltzmann distribution** - probability of states
2. **Partition functions** - canonical, grand canonical, microcanonical ensembles
3. **Fermi-Dirac distribution** - fermion occupation
4. **Bose-Einstein distribution** - boson occupation
5. **Free energies** - Helmholtz (A = U - TS), Gibbs (G = H - TS)
6. **Entropy formulas** - Boltzmann entropy (S = k ln W), Gibbs entropy
7. **Maxwell-Boltzmann distribution** - classical gas velocities
8. **Chemical potential** - μ = (∂G/∂N)

---

## Approach

### Option A: Add to Existing Thermodynamics Section
- Description: Append stat mech equations to the existing thermodynamics block in database.py
- Pros: Minimal file changes, keeps related physics together
- Cons: Thermodynamics section becomes very long (~40 equations), mixes macroscopic and microscopic perspectives

### Option B: Create Separate Statistical Mechanics Section
- Description: Add new dedicated section "STATISTICAL MECHANICS EQUATIONS" after thermodynamics
- Pros: Clear separation, better organization, easier to extend, follows established pattern (mechanics, thermo, EM all separate)
- Cons: Slight duplication (entropy appears in both), requires deciding boundary cases

### Option C: Create Separate Module File
- Description: New file src/dimtensor/equations/statistical_mechanics.py with auto-registration
- Pros: Best long-term scalability, keeps database.py manageable, follows "one domain = one file" pattern
- Cons: More structural change, requires module loading logic, may be premature optimization

### Decision: Option B - Separate Statistical Mechanics Section

Rationale:
1. Follows existing pattern (database.py has multiple domain sections)
2. Statistical mechanics is conceptually distinct from classical thermodynamics
3. Keeps all equations in one discoverable file (current user expectation)
4. Easy to refactor to Option C later if database.py grows too large
5. Clear domain="statistical_mechanics" enables targeted equation queries

---

## Implementation Steps

1. [ ] Define dimension constants for statistical mechanics
   - Number of states/microstates (dimensionless)
   - Occupation number (dimensionless)
   - Chemical potential (energy)
   - Phase space volume dimensions (if needed)

2. [ ] Implement Boltzmann Distribution equations
   - Canonical ensemble probability: P_i = exp(-E_i/kT) / Z
   - Boltzmann factor: exp(-E/kT)

3. [ ] Implement Partition Function equations
   - Canonical partition function: Z = sum(exp(-E_i/kT))
   - Grand canonical partition function: Ξ = sum(exp((μN - E)/kT))
   - Microcanonical partition function: Ω(E) = count of states

4. [ ] Implement Fermi-Dirac Distribution
   - Occupation number: n(E) = 1/(exp((E-μ)/kT) + 1)
   - Zero-temperature limit (Fermi function)

5. [ ] Implement Bose-Einstein Distribution
   - Occupation number: n(E) = 1/(exp((E-μ)/kT) - 1)
   - Planck distribution (photons, μ=0)

6. [ ] Implement Free Energy equations
   - Helmholtz free energy: A = U - TS
   - Gibbs free energy: G = H - TS = U + PV - TS
   - Relation to partition function: A = -kT ln(Z)

7. [ ] Implement Entropy formulas
   - Boltzmann entropy: S = k_B ln(Ω)
   - Gibbs entropy: S = -k_B sum(P_i ln(P_i))
   - Von Neumann entropy (quantum): S = -k_B Tr(ρ ln(ρ))

8. [ ] Implement Maxwell-Boltzmann Distribution
   - Speed distribution: f(v) ∝ v^2 exp(-mv^2/2kT)
   - Energy distribution: f(E) ∝ E^(1/2) exp(-E/kT)

9. [ ] Implement Chemical Potential equations
   - Thermodynamic definition: μ = (∂G/∂N)_{T,P}
   - Ideal gas: μ = kT ln(n/n_Q) where n_Q is quantum concentration

10. [ ] Add comprehensive tags for each equation
    - Domain: "statistical_mechanics"
    - Tags: ensemble type, distribution type, classical/quantum

11. [ ] Add assumptions and related equations
    - Link to existing thermodynamics equations where applicable
    - Note classical vs quantum regimes

12. [ ] Document integration with constants module
    - Use k_B from constants.physico_chemical
    - Note that h, hbar available for quantum distributions

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | ADD - New section "STATISTICAL MECHANICS EQUATIONS" with ~15-20 equations after thermodynamics section (line ~352) |
| tests/test_equations.py | UPDATE - Add tests for stat mech equations (domain filtering, tag queries) |
| docs/guide/equations.md | UPDATE - Add statistical_mechanics to domain list and examples |
| CHANGELOG.md | UPDATE - Note new statistical mechanics equations in next version section |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit test: Verify all stat mech equations are registered
- [ ] Unit test: Check dimensional correctness of each equation's variables
- [ ] Unit test: Filter equations by domain="statistical_mechanics"
- [ ] Unit test: Search for Boltzmann, Fermi, Bose distributions by name
- [ ] Unit test: Verify partition function equations have correct energy dimensions
- [ ] Integration test: Calculate Z from ensemble and verify dimension
- [ ] Integration test: Use Fermi-Dirac distribution at T=0 (step function)
- [ ] Integration test: Verify F = -kT ln(Z) dimensional consistency
- [ ] Manual verification: Check LaTeX rendering for all new equations
- [ ] Manual verification: Ensure related equations are cross-referenced

---

## Risks / Edge Cases

- **Risk 1**: Dimensional analysis of logarithms and exponentials
  - **Mitigation**: Ensure arguments to log/exp are dimensionless (E/kT, not E). Document this clearly in descriptions.

- **Risk 2**: Confusion between different ensemble conventions
  - **Mitigation**: Clearly label each equation with ensemble type (canonical, grand canonical, microcanonical) in tags and description

- **Risk 3**: Overlap with existing thermodynamics equations (entropy appears in both)
  - **Mitigation**: Make thermodynamic entropy macroscopic (dS = dQ/T), statistical entropy microscopic (S = k ln Ω). Link via "related" field.

- **Risk 4**: Discrete vs continuous distributions
  - **Mitigation**: Note in assumptions whether distribution is discrete (energy levels) or continuous (classical limit)

- **Risk 5**: Chemical potential dimensions may be confused with energy
  - **Mitigation**: Use clear variable names (μ, not E) and document that μ has energy dimensions but represents "energy per particle"

- **Edge Case**: Bose-Einstein distribution diverges for μ > E
  - **Handling**: Document assumption μ < E_min for bosons (except photons with μ=0)

- **Edge Case**: Partition function Z is dimensionless but sum is over states
  - **Handling**: Document that Z is dimensionless (count of accessible states weighted by Boltzmann factor)

- **Edge Case**: Zero temperature limits (T → 0)
  - **Handling**: Add separate equations for T=0 limits of Fermi-Dirac (step function)

---

## Definition of Done

- [ ] All implementation steps complete (15-20 equations added)
- [ ] Tests pass (domain filtering, dimensional correctness)
- [ ] All equations have proper LaTeX representation
- [ ] Assumptions clearly documented (ensemble, classical/quantum, temperature range)
- [ ] Cross-references added via "related" field
- [ ] Documentation updated (equations.md domain list)
- [ ] CONTINUITY.md updated with task completion

---

## Equations to Include

### 1. Distributions (4-5 equations)
- Boltzmann distribution (canonical ensemble)
- Fermi-Dirac distribution
- Fermi function (T=0 limit)
- Bose-Einstein distribution
- Planck distribution (photons)

### 2. Partition Functions (3-4 equations)
- Canonical partition function Z(T,V,N)
- Grand canonical partition function Ξ(T,V,μ)
- Microcanonical density of states Ω(E,V,N)
- Relation: A = -kT ln(Z)

### 3. Free Energies (3 equations)
- Helmholtz free energy A = U - TS
- Gibbs free energy G = H - TS
- Grand potential Ω = U - TS - μN

### 4. Entropy (3 equations)
- Boltzmann entropy S = k_B ln(Ω)
- Gibbs entropy S = -k_B Σ P_i ln(P_i)
- Von Neumann entropy (quantum)

### 5. Maxwell-Boltzmann (2 equations)
- Speed distribution
- Energy distribution

### 6. Chemical Potential (2 equations)
- Thermodynamic definition μ = (∂G/∂N)
- Ideal gas chemical potential

**Total: ~15-18 equations**

---

## Dimension Definitions Needed

```python
# Statistical mechanics dimensions
_DIMLESS = Dimension()  # already defined
_E = Dimension(mass=1, length=2, time=-2)  # energy, already defined
_TEMP = Dimension(temperature=1)  # already defined
_CHEMICAL_POTENTIAL = Dimension(mass=1, length=2, time=-2)  # energy per particle
_NUMBER_DENSITY = Dimension(length=-3)  # particles per volume
_PROBABILITY = _DIMLESS  # unitless
_MICROSTATES = _DIMLESS  # count of states
_PHASE_SPACE_VOL = Dimension(mass=1, length=3, time=-1)  # for classical partition function
```

---

## Example Equation Implementation

```python
# Boltzmann Distribution
register_equation(Equation(
    name="Boltzmann Distribution",
    formula="P_i = exp(-E_i/kT) / Z",
    variables={
        "P_i": _DIMLESS,  # probability
        "E_i": _E,  # energy of state i
        "k": Dimension(mass=1, length=2, time=-2, temperature=-1),  # Boltzmann constant
        "T": _TEMP,
        "Z": _DIMLESS  # partition function
    },
    domain="statistical_mechanics",
    tags=["boltzmann", "canonical", "ensemble", "probability", "fundamental"],
    description="Probability of finding system in state i in canonical ensemble",
    latex=r"P_i = \frac{e^{-E_i/k_B T}}{Z}",
    assumptions=["Canonical ensemble (N,V,T fixed)", "Thermal equilibrium"],
    related=["Canonical Partition Function", "Helmholtz Free Energy"],
))

# Fermi-Dirac Distribution
register_equation(Equation(
    name="Fermi-Dirac Distribution",
    formula="n(E) = 1/(exp((E-mu)/kT) + 1)",
    variables={
        "n": _DIMLESS,  # occupation number (0 to 1)
        "E": _E,  # single-particle energy
        "mu": _CHEMICAL_POTENTIAL,
        "k": Dimension(mass=1, length=2, time=-2, temperature=-1),
        "T": _TEMP
    },
    domain="statistical_mechanics",
    tags=["fermi-dirac", "fermion", "quantum", "distribution", "occupation"],
    description="Average occupation number for fermions (particles with half-integer spin)",
    latex=r"n(E) = \frac{1}{e^{(E-\mu)/k_B T} + 1}",
    assumptions=["Quantum statistics", "Non-interacting fermions", "Grand canonical ensemble"],
    related=["Bose-Einstein Distribution", "Chemical Potential"],
))

# Helmholtz Free Energy
register_equation(Equation(
    name="Helmholtz Free Energy",
    formula="A = U - TS",
    variables={
        "A": _E,  # free energy
        "U": _E,  # internal energy
        "T": _TEMP,
        "S": _ENTROPY
    },
    domain="statistical_mechanics",
    tags=["free energy", "helmholtz", "canonical", "thermodynamic potential"],
    description="Free energy for canonical ensemble (natural variables: T, V, N)",
    latex=r"A = U - TS",
    assumptions=["Equilibrium thermodynamics"],
    related=["Canonical Partition Function", "Gibbs Free Energy", "First Law of Thermodynamics"],
))

# Boltzmann Entropy
register_equation(Equation(
    name="Boltzmann Entropy",
    formula="S = k*ln(Omega)",
    variables={
        "S": _ENTROPY,
        "k": Dimension(mass=1, length=2, time=-2, temperature=-1),
        "Omega": _DIMLESS  # number of microstates
    },
    domain="statistical_mechanics",
    tags=["entropy", "boltzmann", "fundamental", "microcanonical"],
    description="Entropy as measure of number of accessible microstates",
    latex=r"S = k_B \ln \Omega",
    assumptions=["All microstates equally probable", "Microcanonical ensemble"],
    related=["Gibbs Entropy", "Entropy Change"],
))
```

---

## Integration with Existing Constants

The stat mech equations should reference the existing physical constants:

```python
from dimtensor.constants.physico_chemical import k_B  # Boltzmann constant
from dimtensor.constants.universal import h, hbar  # for quantum distributions
```

However, in the equation definitions, we define dimensions explicitly rather than importing constants. This keeps the equation database self-contained and focused on dimensional relationships.

---

## Documentation Updates

Update docs/guide/equations.md to include:

```markdown
## Statistical Mechanics Domain

The statistical mechanics domain includes equations for:
- **Distribution functions**: Boltzmann, Fermi-Dirac, Bose-Einstein
- **Partition functions**: Canonical, grand canonical, microcanonical
- **Free energies**: Helmholtz, Gibbs, grand potential
- **Entropy formulas**: Boltzmann, Gibbs, von Neumann
- **Classical distributions**: Maxwell-Boltzmann speed and energy

Example:
```python
from dimtensor.equations import get_equations, search_equations

# Get all statistical mechanics equations
stat_mech_eqs = get_equations(domain="statistical_mechanics")

# Find Fermi-Dirac distribution
fd = search_equations("fermi-dirac")[0]
print(f"{fd.name}: {fd.latex}")
print(f"Variables: {fd.variables}")
```

---

## Notes / Log

**2026-01-12** - Plan created after reviewing:
- src/dimtensor/equations/database.py (existing structure, 50+ equations)
- src/dimtensor/constants/physico_chemical.py (k_B, R already defined)
- .plans/2026-01-09_equations-guide.md (documentation patterns)
- CONTINUITY.md task #222

Key design decisions:
1. Use domain="statistical_mechanics" (separate from thermodynamics)
2. Ensure all exp/log arguments are dimensionless (E/kT not E)
3. Clearly distinguish ensemble types via tags
4. Cross-reference with thermodynamics equations where overlap exists
5. Include both classical and quantum distributions
6. Document temperature range assumptions (T→0, T→∞ limits)

Scope boundaries:
- **In scope**: Fundamental stat mech equations, standard ensembles, distribution functions
- **Out of scope**: Specific models (Ising, Heisenberg), phase transition equations, advanced topics (DMFT, QFT at finite T)
- **Future extensions**: Fluctuation-dissipation theorem, linear response, kinetic theory equations

---
