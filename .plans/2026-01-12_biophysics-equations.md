# Plan: Biophysics Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add biophysics and biochemistry equations to the equation database to enable dimensional analysis and validation for enzyme kinetics, membrane biophysics, diffusion, population dynamics, and electrophysiology applications.

---

## Background

Biophysics and biochemistry involve quantitative models that bridge molecular biology, chemistry, and physics. These include enzyme kinetics (Michaelis-Menten, Hill equation), membrane electrophysiology (Nernst, Goldman, Hodgkin-Huxley), diffusion (Fick's laws), transport equations (cable equation), and population dynamics. Adding these equations to the database will enable:
- Dimensional validation for computational biology simulations
- Integration with the existing biophysics units module (katal, molar, millivolt)
- Support for systems biology and quantitative physiology research
- Physics-informed ML for biological systems

The existing equation database (database.py) has mechanics, thermodynamics, EM, quantum, etc., but no biophysics domain. We'll follow the established pattern of registering equations with dimensional metadata.

---

## Approach

### Option A: Create separate biophysics module
- Description: Create `equations/biophysics.py` with imports in `equations/__init__.py`
- Pros: Organized, follows pattern of `domains/biophysics.py`
- Cons: Splits equation definitions across files

### Option B: Add to main database.py
- Description: Add biophysics equations directly to `equations/database.py` following existing pattern
- Pros: All equations in one place, consistent with current structure
- Cons: Database.py becomes very large

### Decision: Option B - Add to database.py

**Rationale:** The current codebase has all equations in `database.py` (~1100 lines). This is maintainable and keeps the equation registry centralized. We'll add a new section at the end with clear delimiting comments, following the established pattern for other domains.

**Key Design Decisions:**
- **Concentration units**: Use molar dimension (amount=1, length=-3) defined in chemistry/biophysics modules
- **Rate constants**: Dimensions depend on reaction order (see details below)
- **Voltage**: Use voltage dimension (mass=1, length=2, time=-3, current=-1) from existing EM section
- **Partial differential equations**: Represent with formula strings, note that full PDE solutions require numerical methods
- **Cross-references**: Link related equations (e.g., Nernst ↔ Goldman, Fick's 1st ↔ Fick's 2nd)

---

## Implementation Steps

1. [ ] Define dimension shortcuts for biophysics (following existing pattern):
   - `_CONC` = Dimension(amount=1, length=-3)  # molar concentration
   - `_RATE` = Dimension(amount=1, length=-3, time=-1)  # reaction rate
   - `_K1` = Dimension(time=-1)  # 1st order rate constant
   - `_K2` = Dimension(amount=-1, length=3, time=-1)  # 2nd order rate constant
   - `_CONDUCTANCE` = Dimension(mass=-1, length=-2, time=3, current=2)  # siemens
   - `_CAPACITANCE_DENSITY` = Dimension(mass=-1, time=4, current=2)  # F/m²
   - Use existing `_VOLTAGE`, `_CURRENT`, `_T`, `_L` from database.py

2. [ ] Add Enzyme Kinetics equations:
   - **Michaelis-Menten Equation**
     - Formula: `v = (Vmax * [S]) / (Km + [S])`
     - Variables: v (reaction rate), Vmax (max rate), [S] (substrate conc), Km (Michaelis constant, conc)
     - Tags: ["enzyme", "kinetics", "catalysis", "biochemistry"]
   - **Lineweaver-Burk Equation**
     - Formula: `1/v = (Km/Vmax)*(1/[S]) + 1/Vmax`
     - Variables: reciprocals of rate and concentration
     - Tags: ["enzyme", "kinetics", "linearization"]
   - **Hill Equation**
     - Formula: `v = Vmax * [S]^n / (K^n + [S]^n)`
     - Variables: n (Hill coefficient, dimensionless), K (half-saturation, conc)
     - Tags: ["enzyme", "cooperativity", "allosteric"]

3. [ ] Add Membrane Biophysics equations:
   - **Nernst Equation**
     - Formula: `E = (RT/zF) * ln([ion]_out/[ion]_in)`
     - Variables: E (voltage), R (gas const), T (temp), z (charge number, dimless), F (Faraday const), [ion] (conc)
     - Tags: ["membrane", "potential", "ion", "equilibrium"]
   - **Goldman-Hodgkin-Katz Equation**
     - Formula: `E = (RT/F) * ln((P_K[K+]_out + P_Na[Na+]_out + ...)/(P_K[K+]_in + P_Na[Na+]_in + ...))`
     - Variables: P (permeability, velocity dimension), concentrations
     - Tags: ["membrane", "potential", "ion", "permeability"]
   - **Hodgkin-Huxley Conductance**
     - Formula: `I = g_Na*m^3*h*(V - E_Na) + g_K*n^4*(V - E_K) + g_L*(V - E_L)`
     - Variables: I (current/area), g (conductance/area), m,h,n (gating vars, dimless), V (voltage), E (reversal)
     - Note: Full HH model is a system of ODEs; this is the current equation
     - Tags: ["hodgkin-huxley", "action-potential", "neuron", "ion-channel"]

4. [ ] Add Diffusion & Transport equations:
   - **Fick's First Law**
     - Formula: `J = -D * dC/dx`
     - Variables: J (flux, amount/(area*time)), D (diffusivity, area/time), C (conc), x (position)
     - Tags: ["diffusion", "transport", "fick"]
   - **Fick's Second Law**
     - Formula: `dC/dt = D * d²C/dx²`
     - Variables: time derivative of concentration
     - Tags: ["diffusion", "pde", "fick"]
   - **Cable Equation**
     - Formula: `lambda*d²V/dx² = tau*dV/dt + V - V_rest`
     - Variables: lambda (length constant), tau (time constant), V (membrane voltage)
     - Note: Simplified passive cable; active models include HH currents
     - Tags: ["neuron", "cable", "electrophysiology", "pde"]
   - **Einstein-Stokes Relation**
     - Formula: `D = kB*T/(6*pi*eta*r)`
     - Variables: D (diffusivity), kB (Boltzmann), eta (viscosity), r (radius)
     - Tags: ["diffusion", "stokes", "brownian"]

5. [ ] Add Population Dynamics equations:
   - **Logistic Growth**
     - Formula: `dN/dt = r*N*(1 - N/K)`
     - Variables: N (population, dimless count), r (growth rate, 1/time), K (carrying capacity, dimless)
     - Tags: ["population", "ecology", "growth"]
   - **Lotka-Volterra (Predator-Prey)**
     - Formula: `dx/dt = alpha*x - beta*x*y; dy/dt = delta*x*y - gamma*y`
     - Variables: x,y (populations), alpha,gamma (1/time), beta,delta (1/(population*time))
     - Note: Coupled ODEs
     - Tags: ["population", "ecology", "predator-prey", "ode"]

6. [ ] Add Biochemical Kinetics equations:
   - **Arrhenius Equation**
     - Formula: `k = A * exp(-Ea/(R*T))`
     - Variables: k (rate const, depends on order), A (pre-exponential, same as k), Ea (activation energy), R, T
     - Tags: ["kinetics", "temperature", "activation-energy"]
   - **Q10 Temperature Coefficient**
     - Formula: `k2 = k1 * Q10^((T2-T1)/10K)`
     - Variables: k (rate const), T (temp), Q10 (dimensionless, typically 2-3)
     - Tags: ["kinetics", "temperature", "biophysics"]

7. [ ] Update `src/dimtensor/equations/__init__.py` if needed (currently just re-exports database functions)

8. [ ] Add comprehensive tests in `tests/test_equations.py`:
   - Test each equation is registered and retrievable
   - Test equations have correct domain ("biophysics")
   - Test dimensional consistency of variables
   - Test search/filter by biophysics tags
   - Test that related equations are cross-referenced

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | ADD - Biophysics equations section (~300-400 lines) |
| tests/test_equations.py | ADD - Test suite for biophysics equations (~100 lines) |
| src/dimtensor/equations/__init__.py | NO CHANGE - already exports all database functions |

---

## Testing Strategy

How will we verify this works?

- [ ] **Equation registration**: Verify all biophysics equations are in registry
  ```python
  bio_eqs = get_equations(domain="biophysics")
  assert len(bio_eqs) >= 12  # We're adding ~12-14 equations
  ```

- [ ] **Dimension correctness**: Test each equation's variable dimensions
  ```python
  eq = get_equation("Michaelis-Menten Equation")
  assert eq.variables["v"].dimension == Dimension(amount=1, length=-3, time=-1)
  assert eq.variables["Km"].dimension == Dimension(amount=1, length=-3)
  ```

- [ ] **Search functionality**: Verify biophysics equations are searchable
  ```python
  enzyme_eqs = search_equations("enzyme")
  assert any("Michaelis" in eq.name for eq in enzyme_eqs)
  ```

- [ ] **Tag filtering**: Test tag-based queries
  ```python
  kinetics = get_equations(tags=["kinetics"])
  membrane = get_equations(tags=["membrane"])
  ```

- [ ] **Cross-references**: Verify related equations are linked
  ```python
  nernst = get_equation("Nernst Equation")
  assert "Goldman" in str(nernst.related)
  ```

- [ ] **Integration with biophysics units**: Dimensional validation with actual units
  ```python
  from dimtensor import DimArray
  from dimtensor.domains.biophysics import molar, millimolar, enzyme_unit
  from dimtensor.equations import get_equation

  eq = get_equation("Michaelis-Menten Equation")
  Km = DimArray([0.1], millimolar)
  assert Km.dimension == eq.variables["Km"]
  ```

---

## Risks / Edge Cases

- **Risk 1**: Partial differential equations (Fick's 2nd law, Cable equation) can't be directly computed, only validated dimensionally.
  - **Mitigation**: Document in equation description that PDEs require numerical solvers; the equation metadata is for dimensional validation only.

- **Risk 2**: Rate constant dimensions depend on reaction order (0th: conc/time, 1st: 1/time, 2nd: 1/(conc*time)).
  - **Mitigation**: Specify in description/assumptions the reaction order. For Michaelis-Menten, k_cat is 1st order (1/time), Km is a concentration.

- **Risk 3**: Gating variables (m, h, n) in Hodgkin-Huxley are dimensionless but satisfy auxiliary ODEs not captured by single equation.
  - **Mitigation**: Note in description that this is one component of HH model; full model includes gating variable dynamics.

- **Risk 4**: Population counts (N) are dimensionless, different from chemical amount (moles).
  - **Mitigation**: Clearly document N as dimensionless count. Use `_DIMLESS = Dimension()`.

- **Edge Case**: Nernst equation involves logarithm of concentration ratio (dimensionless).
  - **Handling**: Variables are individual concentrations with dimension, ratio is computed at runtime and is dimensionless.

- **Edge Case**: Hill coefficient n can be non-integer (e.g., 2.3), representing effective cooperativity.
  - **Handling**: Document n as dimensionless parameter, typically positive real number.

- **Edge Case**: Permeability P in Goldman equation has dimension of velocity (length/time).
  - **Handling**: Define P with Dimension(length=1, time=-1), document as membrane permeability.

---

## Definition of Done

- [ ] All implementation steps complete (12-14 equations added)
- [ ] Biophysics section added to database.py with clear delimiters
- [ ] All equations have:
  - [ ] Correct variable dimensions
  - [ ] LaTeX representation
  - [ ] Domain = "biophysics"
  - [ ] Relevant tags
  - [ ] Description and assumptions
  - [ ] Cross-references to related equations
- [ ] Tests pass (pytest tests/test_equations.py)
- [ ] Domain "biophysics" appears in `list_domains()`
- [ ] Equations searchable and filterable
- [ ] Documentation strings are clear and scientifically accurate
- [ ] CONTINUITY.md updated with completion status

---

## Rate Constant Dimensions Reference

For implementation accuracy:

| Reaction Order | Rate Constant Dimension | Example |
|----------------|------------------------|---------|
| 0th order | amount·length⁻³·time⁻¹ (M/s) | Zero-order enzyme kinetics at saturation |
| 1st order | time⁻¹ (s⁻¹) | Radioactive decay, k_cat in M-M |
| 2nd order | amount⁻¹·length³·time⁻¹ (M⁻¹·s⁻¹) | Bimolecular reactions |
| Pseudo-1st | time⁻¹ | 2nd order with one reactant in excess |

**Michaelis-Menten specifics:**
- v, Vmax: amount·length⁻³·time⁻¹ (reaction rate, e.g., mol/(L·s) or M/s)
- [S], Km: amount·length⁻³ (concentration, e.g., mol/L or M)
- k_cat (turnover): time⁻¹ (if added as variable)
- kcat/Km (catalytic efficiency): amount⁻¹·length³·time⁻¹ (2nd order)

---

## Equation Details (for implementer reference)

### Michaelis-Menten Equation
```
v = (Vmax * [S]) / (Km + [S])

Variables:
- v: reaction rate [M/s] = Dimension(amount=1, length=-3, time=-1)
- Vmax: maximum rate [M/s] = Dimension(amount=1, length=-3, time=-1)
- [S]: substrate concentration [M] = Dimension(amount=1, length=-3)
- Km: Michaelis constant [M] = Dimension(amount=1, length=-3)

LaTeX: v = \frac{V_{max}[S]}{K_m + [S]}
Tags: enzyme, kinetics, catalysis, biochemistry, fundamental
Assumptions:
  - Steady-state approximation
  - [S] >> [E]
  - Single substrate, irreversible
Related: Hill Equation, Lineweaver-Burk Equation
```

### Nernst Equation
```
E = (R*T / (z*F)) * ln([ion]_out / [ion]_in)

Variables:
- E: membrane potential [V] = Dimension(mass=1, length=2, time=-3, current=-1)
- R: gas constant [J/(mol·K)] = Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1)
- T: temperature [K] = Dimension(temperature=1)
- z: ion charge number [dimensionless] = Dimension()
- F: Faraday constant [C/mol] = Dimension(current=1, time=1, amount=-1)
- [ion]: concentration [M] = Dimension(amount=1, length=-3)

LaTeX: E = \frac{RT}{zF}\ln\frac{[ion]_{out}}{[ion]_{in}}
Tags: membrane, potential, ion, equilibrium, nernst, fundamental
Assumptions:
  - Equilibrium conditions
  - Single ion species
  - Ideal behavior
Related: Goldman-Hodgkin-Katz Equation
```

### Fick's First Law
```
J = -D * (dC/dx)

Variables:
- J: flux [mol/(m²·s)] = Dimension(amount=1, length=-2, time=-1)
- D: diffusion coefficient [m²/s] = Dimension(length=2, time=-1)
- C: concentration [M] = Dimension(amount=1, length=-3)
- x: position [m] = Dimension(length=1)

LaTeX: J = -D\frac{dC}{dx}
Tags: diffusion, transport, fick, fundamental
Related: Fick's Second Law, Einstein-Stokes Relation
```

### Hill Equation
```
v = Vmax * [S]^n / (K^n + [S]^n)

Variables:
- v: reaction rate [M/s] = Dimension(amount=1, length=-3, time=-1)
- Vmax: maximum rate [M/s] = Dimension(amount=1, length=-3, time=-1)
- [S]: substrate concentration [M] = Dimension(amount=1, length=-3)
- K: half-saturation constant [M] = Dimension(amount=1, length=-3)
- n: Hill coefficient [dimensionless] = Dimension()

LaTeX: v = \frac{V_{max}[S]^n}{K^n + [S]^n}
Tags: enzyme, cooperativity, allosteric, hill, biochemistry
Assumptions:
  - Cooperative binding
  - n > 1: positive cooperativity
  - n = 1: reduces to Michaelis-Menten
Related: Michaelis-Menten Equation
```

### Logistic Growth
```
dN/dt = r*N*(1 - N/K)

Variables:
- N: population size [dimensionless] = Dimension()
- t: time [s] = Dimension(time=1)
- r: intrinsic growth rate [1/s] = Dimension(time=-1)
- K: carrying capacity [dimensionless] = Dimension()

LaTeX: \frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right)
Tags: population, ecology, growth, logistic, ode
Assumptions:
  - Density-dependent growth
  - Well-mixed population
Related: Lotka-Volterra Equations
```

---

## Notes / Log

**Scientific References:**
- Michaelis-Menten: L. Michaelis & M.L. Menten (1913), standard enzyme kinetics textbooks
- Nernst: W. Nernst (1889), electrochemistry textbooks, Hille "Ion Channels of Excitable Membranes"
- Goldman equation: Goldman (1943), Hodgkin & Katz (1949)
- Hodgkin-Huxley: Hodgkin & Huxley (1952), Nobel Prize work on action potentials
- Fick's laws: A. Fick (1855), diffusion textbooks
- Cable equation: W. Rall (1960s), computational neuroscience (Dayan & Abbott)
- Hill equation: A.V. Hill (1910), cooperativity in hemoglobin oxygen binding
- Logistic: Verhulst (1838), ecology textbooks
- Lotka-Volterra: Lotka (1925), Volterra (1926), ecological modeling

**Implementation notes:**
- Use existing dimension shortcuts where possible (_VOLTAGE, _CURRENT, _TEMP, _T, _L)
- Define new shortcuts at beginning of biophysics section
- Faraday constant F available in constants.physico_chemical
- Gas constant R available in constants.universal
- Boltzmann constant kB available in constants.universal

---
