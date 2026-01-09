# Plan: SymPy Integration

**Date**: 2026-01-09
**Status**: COMPLETED
**Author**: agent

---

## Goal

Enable seamless conversion between DimArray and SymPy expressions, preserving dimensional information for symbolic computation.

---

## Background

SymPy is the de facto standard for symbolic mathematics in Python. Scientists often need to:
- Derive symbolic expressions from physical equations
- Substitute numerical values with units
- Perform symbolic differentiation/integration
- Verify dimensional correctness of symbolic formulas

DimArray tracks physical dimensions through numerical computation. Bridging to SymPy enables symbolic manipulation with dimensional safety.

---

## Approach

### Option A: Wrapper Functions
- Simple `to_sympy()` / `from_sympy()` functions
- SymPy symbols carry dimension as assumptions
- Pros: Simple, minimal API
- Cons: May lose dimension info in complex expressions

### Option B: DimSymbol Class
- Custom SymPy Symbol subclass with dimension tracking
- Full SymPy algebra with dimension checking
- Pros: Deep integration, full dimension tracking
- Cons: Complex, may conflict with SymPy internals

### Option C: Hybrid Approach (Selected)
- Simple conversion functions for basic use
- Use SymPy's `assumptions` system for dimensions
- Create `units` module in sympy/ for unit symbols
- Pros: Best of both - simple API, works with existing SymPy
- Cons: Moderate complexity

### Decision: Option C - Hybrid Approach

Use SymPy's existing `Quantity` class from `sympy.physics.units` for unit handling, with conversion functions to/from DimArray.

---

## Implementation Steps

1. [x] Research SymPy units system (sympy.physics.units)
2. [ ] Create sympy/__init__.py module
3. [ ] Implement to_sympy() - Convert DimArray to SymPy expression with units
4. [ ] Implement from_sympy() - Convert SymPy expression to DimArray
5. [ ] Implement unit_registry bridge - Map dimtensor units to SymPy units
6. [ ] Implement symbolic_diff() - Differentiate with dimension tracking
7. [ ] Implement symbolic_integrate() - Integrate with dimension tracking
8. [ ] Add tests for SymPy integration

---

## Files to Create/Modify

| File | Change |
|------|--------|
| src/dimtensor/sympy/__init__.py | New module exports |
| src/dimtensor/sympy/conversion.py | to_sympy, from_sympy functions |
| src/dimtensor/sympy/calculus.py | symbolic_diff, symbolic_integrate |
| tests/test_sympy.py | Unit tests |

---

## Testing Strategy

- [ ] Unit tests for to_sympy with various units
- [ ] Unit tests for from_sympy with symbolic expressions
- [ ] Unit tests for differentiation (position -> velocity -> acceleration)
- [ ] Unit tests for integration (acceleration -> velocity -> position)
- [ ] Integration test: full symbolic physics workflow

---

## Risks / Edge Cases

- Risk: SymPy version compatibility. Mitigation: Test with sympy>=1.12
- Edge case: Compound units not in SymPy's registry. Handling: Fall back to base units
- Edge case: Dimensionless quantities. Handling: Use sympy.core.numbers

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-09** - Plan created. Starting research on sympy.physics.units.

---
