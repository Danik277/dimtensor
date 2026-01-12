# Plan: Quantum Field Theory Equations Module

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add comprehensive QFT equation database with proper natural units support (c=ℏ=1) and handling of tensor/spinor indices. Enable dimensional verification and lookup of fundamental QFT equations including field equations, Lagrangians, propagators, cross-sections, and decay rates.

---

## Background

The existing equation database (src/dimtensor/equations/database.py) contains classical mechanics, thermodynamics, electromagnetism, quantum mechanics, relativity, and optics. QFT equations are missing, which are essential for:
- Particle physics calculations
- High-energy physics research
- Validation of QFT models in physics-informed ML
- Educational tools for QFT students

QFT uses natural units (c=ℏ=1) extensively where dimensions simplify to powers of energy. The codebase already has src/dimtensor/domains/natural.py with GeV, conversion functions, and natural unit handling.

---

## Approach

### Option A: Extend database.py with QFT section
- Add QFT equations directly to existing database.py file following established patterns
- Use natural unit dimensions from domains/natural.py
- Store equations in natural units (c=ℏ=1 convention)
- Add metadata fields for tensor structure (spinor indices, Lorentz indices)
- Pros:
  - Consistent with existing equation structure
  - All equations in one place
  - Reuses existing Equation dataclass
  - Easy to search across all physics domains
- Cons:
  - File becomes very large (currently ~1100 lines, QFT could add 200+ lines)
  - No clear separation between classical and QFT physics
  - Natural units mixed with SI units

### Option B: Separate qft_equations.py module
- Create new src/dimtensor/equations/qft.py module
- Define QFT-specific equation structure with tensor metadata
- Register equations separately but integrate with main database
- Keep natural units as default representation
- Pros:
  - Clean separation of concerns
  - More manageable file size
  - Can extend Equation dataclass with QFT-specific fields
  - Easier to add QFT-specific utilities (Feynman rules, etc.)
- Cons:
  - Need to ensure integration with existing database queries
  - Risk of fragmentation if not carefully designed

### Option C: Hybrid approach with domain-specific modules
- Create equations/domains/ subdirectory
- Move existing domains into separate files (mechanics.py, em.py, etc.)
- Add qft.py as a domain module
- database.py becomes registry only
- Pros:
  - Scalable architecture for future expansion
  - Each domain can have specialized metadata
  - Clear organization
- Cons:
  - Large refactoring required
  - May break existing code/imports
  - Out of scope for this task

### Decision: Option A - Extend database.py

Rationale:
- Minimal changes to existing structure
- Maintains backward compatibility
- QFT equations use same Equation dataclass
- Natural unit dimensions already exist in Dimension system
- Can add metadata in 'tags' and 'assumptions' fields
- File size (~1300 lines total) is still manageable
- Future refactoring to Option C can be done without breaking API

For tensor/spinor structure, we'll encode this in:
1. `tags`: Add tags like "spinor", "vector", "tensor-rank-2"
2. `assumptions`: Document index structure (e.g., "ψ is a 4-component spinor")
3. `description`: Full explanation including indices

---

## Implementation Steps

1. [ ] Research and verify QFT equations with proper dimensions in natural units
   - Dirac equation: (iγ^μ ∂_μ - m)ψ = 0
   - Klein-Gordon equation: (∂^μ ∂_μ + m²)φ = 0
   - Lagrangian densities (QED, scalar field, Dirac field)
   - Energy-momentum relation: E² = p² + m² (natural units)
   - Yukawa potential
   - Feynman propagators (scalar, fermion, photon)
   - Cross-sections (Compton, Bhabha, Møller, pair production)
   - Decay rates (muon decay, pion decay)
   - Running coupling constants

2. [ ] Define natural unit dimensions for QFT quantities
   - Field dimensions in 3+1 spacetime (φ: [M], ψ: [M]^(3/2))
   - Action is dimensionless (ℏ=1)
   - Cross-section: [M]^(-2) = [E]^(-2)
   - Decay rate: [M] = [E]
   - Coupling constants: dimensionless or various powers

3. [ ] Add QFT section to database.py following existing structure
   - Use _NATURAL_ENERGY_DIM from domains/natural.py
   - Define derived dimensions (cross-section, decay width, etc.)
   - Create 15-20 core QFT equations

4. [ ] Add appropriate metadata to equations
   - Tags: "qft", "relativistic", "field theory", "spinor", "lagrangian", etc.
   - Domain: "quantum_field_theory" or "qft"
   - LaTeX: Full LaTeX with proper indices
   - Assumptions: Natural units, spinor structure, gauge groups
   - Related: Link to connected equations

5. [ ] Add helper functions for natural units (if needed)
   - Consider adding to domains/natural.py if conversions are complex
   - Document relationship between SI and natural unit representations

6. [ ] Test dimensional consistency
   - Add test cases verifying equation dimensions
   - Test that Lagrangian dimensions are correct ([M]^4 in natural units)
   - Test cross-section dimensions
   - Test conversion between natural and SI units

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/equations/database.py | Add QFT equations section (~200 lines) at end of file before closing |
| tests/test_equations.py | Add TestQFTEquations class with dimensional verification tests |
| src/dimtensor/domains/natural.py | Possibly add field dimension constants if not already present |
| docs/guide/equations.md | Add QFT section to documentation (if file exists) |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for QFT equation registration
  - Test get_equations(domain="qft") returns QFT equations
  - Test search_equations("Dirac") finds Dirac equation
  - Test get_equation("Dirac Equation") returns correct equation

- [ ] Dimensional consistency tests
  - Verify Dirac equation: (iγ^μ ∂_μ - m)ψ = 0 has consistent dimensions
    - ∂_μ has dimension [M] in natural units
    - m has dimension [M]
    - ψ has dimension [M]^(3/2)
    - Result is dimension [M]^(3/2)
  - Verify Klein-Gordon: (∂² + m²)φ = 0
    - ∂² has dimension [M]²
    - m² has dimension [M]²
    - φ has dimension [M]
  - Verify QED Lagrangian: ℒ = ψ̄(iγ^μD_μ - m)ψ - ¼F^μν F_μν
    - Should have dimension [M]^4 in natural units (3+1 spacetime)
  - Verify cross-section has dimension [E]^(-2)
  - Verify decay rate has dimension [E]

- [ ] Integration tests
  - Test that QFT equations work with existing search/filter APIs
  - Test list_domains() includes "qft"
  - Test related equations are properly linked

- [ ] Manual verification
  - Compare with standard QFT textbooks (Peskin & Schroeder, Schwartz, Srednicki)
  - Verify natural units conventions match PDG standards
  - Check LaTeX rendering is correct

---

## Risks / Edge Cases

- **Risk 1: Natural units confusion with SI units**
  - Different equations use different conventions (c=ℏ=1 vs SI)
  - Mitigation: Clearly tag all QFT equations with "natural_units" assumption
  - Add conversion examples in docstrings
  - Document that dimensions are "as measured in natural units"

- **Risk 2: Tensor/spinor indices lose dimensional information**
  - Contractions of indices aren't captured in scalar Dimension type
  - Mitigation: Document index structure in assumptions and description
  - Use tags to indicate tensor rank
  - Future: Could extend Equation dataclass with tensor_structure field

- **Risk 3: Field dimensions are fractional powers of mass**
  - Scalar field φ: [M]
  - Spinor field ψ: [M]^(3/2)
  - Vector field A_μ: [M]
  - Dimension class uses Fraction, so should support this
  - Mitigation: Verify Fraction(3, 2) works in Dimension

- **Edge case: Running coupling constants**
  - Coupling constants depend on energy scale
  - Not purely dimensional constants
  - Handling: Note in assumptions that α = α(Q²), don't store Q-dependence

- **Edge case: Regularization and renormalization**
  - Bare vs renormalized quantities have different interpretations
  - Handling: Store renormalized quantities, note regularization in assumptions

- **Edge case: Off-shell vs on-shell**
  - Propagators differ on-shell (E² = p² + m²) vs off-shell
  - Handling: Store full propagators, note energy-momentum relation in assumptions

---

## Definition of Done

- [ ] 15-20 core QFT equations added to database.py
- [ ] All equations have proper dimensions in natural units
- [ ] Equations include: Dirac, Klein-Gordon, Lagrangians, propagators, cross-sections, decay rates
- [ ] Tags and assumptions clearly indicate natural units and tensor structure
- [ ] Tests verify dimensional consistency
- [ ] Tests pass (pytest)
- [ ] Integration with existing database API confirmed
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**2026-01-12 Initial Research**

Existing patterns observed:
- Equation dataclass has: name, formula, variables (dict[str, Dimension]), domain, tags, description, assumptions, latex, related
- Dimensions defined with shorthand: _E = Dimension(mass=1, length=2, time=-2)
- Each equation registered with register_equation()
- Domains: "mechanics", "thermodynamics", "electromagnetism", "fluid_dynamics", "relativity", "quantum", "optics", "acoustics"
- Related quantum equations already exist: Planck-Einstein, de Broglie, Heisenberg, Schrödinger, Compton, Bohr, Rydberg

Natural units infrastructure:
- domains/natural.py provides: GeV, MeV, eV, conversion functions to_natural() and from_natural()
- _NATURAL_ENERGY_DIM = Dimension(mass=1, length=2, time=-2) (SI equivalent)
- _NATURAL_LENGTH_DIM = Dimension(mass=-1, length=-2, time=2) (SI: M⁻¹ L⁻² T²)
- _NATURAL_TIME_DIM = Dimension(mass=-1, length=-2, time=2) (SI: M⁻¹ L⁻² T²)
- Supports energy, mass, momentum (all [E]), length ([E]^-1), time ([E]^-1)

QFT equations to add:
1. Dirac equation (free particle)
2. Klein-Gordon equation
3. Yukawa potential
4. QED Lagrangian
5. Scalar field Lagrangian
6. Dirac field Lagrangian
7. Feynman propagator (scalar)
8. Feynman propagator (fermion)
9. Feynman propagator (photon)
10. Energy-momentum relation (relativistic)
11. Compton scattering cross-section (already exists in quantum, check if adequate)
12. Bhabha scattering cross-section (e+e- → e+e-)
13. Møller scattering cross-section (e-e- → e-e-)
14. Pair production cross-section
15. Muon decay rate
16. Pion decay constant
17. Fine structure constant (dimensionless)
18. QCD coupling constant
19. Breit-Wigner resonance
20. Fermi's golden rule (decay rate)

Dimension considerations:
- In natural units (3+1D spacetime):
  - Action S: dimensionless (∫d⁴x ℒ, d⁴x has [M]^-4, ℒ has [M]^4)
  - Lagrangian density ℒ: [M]^4
  - Scalar field φ: [M]^1
  - Spinor field ψ: [M]^(3/2)
  - Vector field A_μ: [M]^1
  - Derivative ∂_μ: [M]^1
  - Cross-section σ: [M]^(-2) (area in natural units)
  - Decay rate Γ: [M]^1 (inverse time in natural units)

---
