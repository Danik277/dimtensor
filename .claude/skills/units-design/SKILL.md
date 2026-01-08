---
name: units-design
description: Design new unit modules for dimtensor. Use when creating astronomy, chemistry, engineering, or other domain-specific units.
allowed-tools: Read, Grep, Glob, Write, Edit
---

# Units Design Skill

When designing new unit modules for dimtensor, follow these guidelines.

## Core Concepts

**Dimension**: 7-tuple of SI base dimension exponents (L, M, T, I, Θ, N, J)
- Length (L), Mass (M), Time (T), Current (I), Temperature (Θ), Amount (N), Luminosity (J)
- Example: velocity = L¹T⁻¹ → Dimension(length=1, time=-1)

**Unit**: Dimension + scale factor relative to SI base
- Example: kilometer = Dimension(length=1) with scale=1000

## Creating New Units

```python
from dimtensor.core.dimensions import Dimension
from dimtensor.core.units import Unit

# Simple unit (single dimension)
parsec = Unit("pc", Dimension(length=1), 3.0857e16)

# Compound unit (multiple dimensions)
molar = Unit("M", Dimension(amount=1, length=-3), 1000)  # mol/L in mol/m³

# Derived unit
newton = Unit("N", Dimension(length=1, mass=1, time=-2), 1.0)
```

## Scale Factor Sources

Use authoritative values:
- **CODATA 2022** for physical constants
- **IAU** for astronomical constants
- **IUPAC** for chemical constants

Document sources in code comments.

## File Structure

Create domain modules in `src/dimtensor/domains/`:

```python
# src/dimtensor/domains/astronomy.py
"""Astronomy units for dimtensor."""

from dimtensor.core.dimensions import Dimension
from dimtensor.core.units import Unit

# Distance units (IAU 2012)
parsec = Unit("pc", Dimension(length=1), 3.0857e16)  # meters
astronomical_unit = Unit("AU", Dimension(length=1), 1.495978707e11)

# Mass units
solar_mass = Unit("M☉", Dimension(mass=1), 1.98892e30)  # kg

__all__ = ["parsec", "astronomical_unit", "solar_mass"]
```

## Checklist

Before implementing:
- [ ] Create plan in `.plans/` folder
- [ ] Research authoritative scale factors
- [ ] Determine correct SI dimensions
- [ ] Check for existing similar units

During implementation:
- [ ] Follow pattern from existing domains/
- [ ] Add to domains/__init__.py
- [ ] Use scientific notation for large/small values
- [ ] Include docstrings with sources

After implementation:
- [ ] Write tests in tests/test_domains_*.py
- [ ] Test dimension correctness
- [ ] Test conversion accuracy
- [ ] Update CONTINUITY.md
