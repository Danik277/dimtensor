# Plan: Engineering Units Module

**Date**: 2026-01-08
**Status**: PLANNING
**Author**: agent

---

## Goal

Add engineering-specific units (MPa, ksi, BTU, horsepower, etc.) to enable scientific computing for mechanical, thermal, and electrical engineering applications.

---

## Background

Engineers commonly use units like BTU, horsepower, ksi (kilopounds per square inch), and various imperial/US customary units. These are essential for practical engineering calculations.

---

## Approach

### Decision: Create domains/engineering.py
Following the same pattern as other domain modules in the domains/ folder.

---

## Implementation Steps

1. [x] Create `src/dimtensor/domains/engineering.py` with units:

   **Pressure:**
   - megapascal (MPa) - 1e6 Pa
   - kilopascal (kPa) - 1e3 Pa
   - ksi - kilopounds per square inch (6.894757e6 Pa)

   **Energy/Heat:**
   - BTU (British Thermal Unit) - 1055.06 J
   - therm - 1.055e8 J (100,000 BTU)
   - kilowatt_hour (kWh) - 3.6e6 J

   **Power:**
   - horsepower (hp) - 745.7 W (mechanical)
   - metric_horsepower - 735.5 W
   - ton_refrigeration - 3516.85 W

   **Flow:**
   - gallon_per_minute (gpm) - volumetric flow
   - cubic_feet_per_minute (cfm) - volumetric flow

   **Torque:**
   - foot_pound (ft_lb) - 1.3558 N*m
   - inch_pound (in_lb) - 0.1130 N*m

   **Other:**
   - rpm - revolutions per minute (angular velocity)
   - mil - 0.001 inch (25.4e-6 m)

2. [x] Update `src/dimtensor/domains/__init__.py` to expose engineering
3. [ ] Add tests in `tests/test_domains_engineering.py`

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/engineering.py | CREATE - engineering units |
| src/dimtensor/domains/__init__.py | UPDATE - expose engineering |
| tests/test_domains_engineering.py | CREATE - unit tests |

---

## Testing Strategy

- [ ] Test that each unit has correct dimension
- [ ] Test pressure conversions (MPa to psi, ksi to Pa)
- [ ] Test energy conversions (BTU to J, kWh to J)
- [ ] Test power conversions (hp to W)

---

## Risks / Edge Cases

- Risk: Multiple "horsepower" definitions exist. Mitigation: Use mechanical hp (745.7 W) as default, provide metric_horsepower separately.
- Edge case: rpm is technically angular velocity but often treated as frequency. Handling: Use Dimension(time=-1) like Hz.

---

## Definition of Done

- [x] All implementation steps complete
- [ ] Tests pass
- [ ] Module importable: `from dimtensor.domains.engineering import MPa, hp, BTU`
- [ ] CONTINUITY.md updated

---

## Notes / Log

(Add notes as you work)

---
