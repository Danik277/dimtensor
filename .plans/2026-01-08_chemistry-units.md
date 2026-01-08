# Plan: Chemistry Units Module

**Date**: 2026-01-08
**Status**: PLANNING
**Author**: agent

---

## Goal

Add chemistry-specific units (molar concentration, molality, ppm, dalton, etc.) to enable scientific computing for chemistry and biochemistry applications.

---

## Background

Chemistry uses specialized concentration and amount units that aren't commonly found in physics libraries. Adding these will make dimtensor useful for analytical chemistry, biochemistry, and materials science.

---

## Approach

### Decision: Create domains/chemistry.py
Following the same pattern as astronomy.py in the domains/ folder.

---

## Implementation Steps

1. [x] Create `src/dimtensor/domains/chemistry.py` with units:
   - dalton (Da, u) - 1.66054e-27 kg (atomic mass unit)
   - molar (M) - mol/L (amount concentration)
   - millimolar (mM) - mmol/L
   - micromolar (uM) - umol/L
   - nanomolar (nM) - nmol/L
   - molal (m) - mol/kg (molality)
   - ppm - parts per million (dimensionless, 1e-6)
   - ppb - parts per billion (dimensionless, 1e-9)
   - ppt - parts per trillion (dimensionless, 1e-12)
   - angstrom - 1e-10 m (common in crystallography)
   - debye (D) - dipole moment unit
2. [x] Update `src/dimtensor/domains/__init__.py` to expose chemistry
3. [ ] Add tests in `tests/test_domains_chemistry.py`

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/chemistry.py | CREATE - chemistry units |
| src/dimtensor/domains/__init__.py | UPDATE - expose chemistry |
| tests/test_domains_chemistry.py | CREATE - unit tests |

---

## Testing Strategy

- [ ] Test that each unit has correct dimension
- [ ] Test concentration unit conversions
- [ ] Test that ppm/ppb/ppt are dimensionless with correct scale

---

## Risks / Edge Cases

- Risk: Molarity is amount/volume, needs compound dimension. Mitigation: Use Dimension(amount=1, length=-3).
- Edge case: ppm is often context-dependent (mass/mass, volume/volume). Handling: Document as generic ratio.

---

## Definition of Done

- [x] All implementation steps complete
- [ ] Tests pass
- [ ] Module importable: `from dimtensor.domains.chemistry import molar, dalton`
- [ ] CONTINUITY.md updated

---

## Notes / Log

(Add notes as you work)

---
