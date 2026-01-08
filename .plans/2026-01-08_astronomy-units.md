# Plan: Astronomy Units Module

**Date**: 2026-01-08
**Status**: PLANNING
**Author**: agent

---

## Goal

Add astronomy-specific units (parsec, AU, solar_mass, light_year, etc.) to enable scientific computing for astrophysics applications.

---

## Background

Astronomers work with scales vastly different from everyday SI units. Current dimtensor only has basic SI and common non-SI units. Adding astronomy units will make the library more useful for astrophysics simulations and data analysis.

---

## Approach

### Option A: New domains/ folder with separate modules
- Create `src/dimtensor/domains/` with `astronomy.py`, `chemistry.py`, etc.
- Pros: Clean separation, extensible for future domains
- Cons: Adds folder hierarchy

### Option B: Add to existing units.py
- Pros: Simple, all units in one place
- Cons: File will grow large, harder to maintain

### Decision: Option A - Create domains/ folder
This keeps domain-specific units organized and makes the library extensible. Users can `from dimtensor.domains.astronomy import parsec, AU`.

---

## Implementation Steps

1. [x] Create `src/dimtensor/domains/` folder
2. [x] Create `src/dimtensor/domains/__init__.py`
3. [x] Create `src/dimtensor/domains/astronomy.py` with units:
   - parsec (pc) - 3.0857e16 m
   - astronomical_unit (AU) - 1.495978707e11 m
   - light_year (ly) - 9.4607e15 m
   - solar_mass (M_sun) - 1.98892e30 kg
   - solar_radius (R_sun) - 6.96e8 m
   - solar_luminosity (L_sun) - 3.828e26 W
   - earth_mass (M_earth) - 5.972e24 kg
   - earth_radius (R_earth) - 6.371e6 m
   - jupiter_mass (M_jup) - 1.898e27 kg
   - jupiter_radius (R_jup) - 6.9911e7 m
   - arcsecond (arcsec) - angular
   - milliarcsecond (mas) - angular
4. [x] Update `src/dimtensor/__init__.py` to expose domains module
5. [ ] Add tests in `tests/test_domains_astronomy.py`

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/__init__.py | CREATE - package init |
| src/dimtensor/domains/astronomy.py | CREATE - astronomy units |
| src/dimtensor/__init__.py | UPDATE - expose domains |
| tests/test_domains_astronomy.py | CREATE - unit tests |

---

## Testing Strategy

- [ ] Test that each unit has correct dimension
- [ ] Test that each unit has correct scale factor (within reasonable tolerance)
- [ ] Test unit conversions (e.g., parsec to light_year)
- [ ] Test unit arithmetic (e.g., solar_mass / parsec**3 for density)

---

## Risks / Edge Cases

- Risk: Scale factor precision. Mitigation: Use CODATA/IAU recommended values.
- Edge case: Very large/small scales. Handling: Python floats handle these ranges.

---

## Definition of Done

- [x] All implementation steps complete
- [ ] Tests pass
- [ ] Module importable: `from dimtensor.domains.astronomy import parsec`
- [ ] CONTINUITY.md updated

---

## Notes / Log

(Add notes as you work)

---
