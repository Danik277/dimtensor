# Plan: Dimensional Inference System

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Author**: agent

---

## Goal

Implement a system that can infer physical dimensions from variable names and code patterns, helping users catch dimensional errors before they occur and enabling smarter IDE integration.

---

## Background

Currently, dimtensor requires explicit unit declarations. This plan adds optional inference capabilities:

1. **Variable Name Heuristics**: Names like "velocity" → L/T, "force" → MLT⁻²
2. **Equation Pattern Matching**: Recognize F=ma, E=mc², PV=nRT
3. **IDE Integration**: Show inferred units, highlight mismatches
4. **Linting**: `dimtensor lint` command for unit checking

### Prior Art

- F# has built-in static checking and inference of units of measure
- Guo & McCamant (2005): Annotation-less type inference for C that assigns unit types from constraints
- SymPy: Symbolic dimensional analysis
- py-dimensional-analysis: Dimensional analysis from variable systems

---

## Approach

### Option A: Static Analysis (Type Checking Style)
- Analyze AST to build constraint system
- Infer minimal units needed
- Complex, requires full Python parsing
- Cons: Very complex for v2.1.0

### Option B: Heuristic + Pattern (CHOSEN)
- Dictionary of common variable names → dimensions
- Pattern matching for common equations
- Lightweight, additive value
- Pros: Simple, immediately useful
- Cons: Not complete, can have false positives

### Option C: ML-Based Inference
- Train model on physics codebases
- Complex, requires training data
- Cons: Over-engineered for v2.1.0

### Decision: Option B

Start with heuristics and patterns. This gives us:
1. Immediate value with minimal complexity
2. Can grow the pattern database over time
3. Users can add custom patterns
4. Foundation for more advanced inference later

---

## Implementation Steps

### Phase 1: Variable Name Heuristics (Task #79)
1. [ ] Create inference/heuristics.py
2. [ ] Build dictionary: common name patterns → Dimension
3. [ ] Support prefixes: "initial_velocity", "final_velocity"
4. [ ] Support suffixes: "velocity_x", "velocity_m_per_s"
5. [ ] Configurable confidence thresholds

### Phase 2: Equation Pattern Database (Tasks #80-81)
1. [ ] Create inference/equations.py
2. [ ] Build equation database (JSON or Python)
3. [ ] Mechanics: F=ma, E=½mv², W=Fd, p=mv
4. [ ] Electromagnetics: V=IR, P=IV, F=qE
5. [ ] Thermodynamics: PV=nRT, Q=mcΔT
6. [ ] Pattern matching engine

### Phase 3: IDE Plugin Architecture (Tasks #82-83)
1. [ ] Create inference/ide.py
2. [ ] Define LSP-compatible interface
3. [ ] Hover info: show inferred units
4. [ ] Diagnostics: highlight mismatches
5. [ ] VS Code extension (basic)

### Phase 4: Linting Command (Task #84)
1. [ ] Create inference/lint.py
2. [ ] `dimtensor lint <file.py>` CLI
3. [ ] Parse Python file for DimArray usage
4. [ ] Report dimensional inconsistencies
5. [ ] Configurable strictness levels

### Phase 5: Configuration (Task #85)
1. [ ] Create inference/config.py
2. [ ] User-defined patterns
3. [ ] Custom equation database
4. [ ] Strictness settings

---

## Files to Create/Modify

| File | Change |
|------|--------|
| src/dimtensor/inference/__init__.py | NEW: Inference module init |
| src/dimtensor/inference/heuristics.py | NEW: Variable name heuristics |
| src/dimtensor/inference/equations.py | NEW: Equation pattern database |
| src/dimtensor/inference/patterns.py | NEW: Pattern matching engine |
| src/dimtensor/inference/lint.py | NEW: Linting functionality |
| src/dimtensor/inference/config.py | NEW: Configuration |
| src/dimtensor/__init__.py | MOD: Add inference export |
| pyproject.toml | MOD: Add lint CLI entry point |
| tests/test_inference.py | NEW: Inference tests |

---

## Variable Name Patterns

### Direct Mappings
```python
VARIABLE_PATTERNS = {
    # Mechanics
    "velocity": Dimension(length=1, time=-1),  # m/s
    "speed": Dimension(length=1, time=-1),     # m/s
    "acceleration": Dimension(length=1, time=-2),  # m/s²
    "force": Dimension(length=1, mass=1, time=-2),  # N = kg·m/s²
    "momentum": Dimension(length=1, mass=1, time=-1),  # kg·m/s
    "energy": Dimension(length=2, mass=1, time=-2),  # J = kg·m²/s²
    "work": Dimension(length=2, mass=1, time=-2),
    "power": Dimension(length=2, mass=1, time=-3),  # W = kg·m²/s³
    "pressure": Dimension(length=-1, mass=1, time=-2),  # Pa

    # Geometry
    "distance": Dimension(length=1),
    "length": Dimension(length=1),
    "width": Dimension(length=1),
    "height": Dimension(length=1),
    "radius": Dimension(length=1),
    "area": Dimension(length=2),
    "volume": Dimension(length=3),

    # Time
    "time": Dimension(time=1),
    "duration": Dimension(time=1),
    "period": Dimension(time=1),
    "frequency": Dimension(time=-1),

    # Mass
    "mass": Dimension(mass=1),
    "weight": Dimension(length=1, mass=1, time=-2),  # Actually force!
    "density": Dimension(length=-3, mass=1),

    # Temperature
    "temperature": Dimension(temperature=1),
    "temp": Dimension(temperature=1),

    # Electromagnetics
    "current": Dimension(current=1),
    "voltage": Dimension(length=2, mass=1, time=-3, current=-1),
    "resistance": Dimension(length=2, mass=1, time=-3, current=-2),
    "charge": Dimension(current=1, time=1),
}
```

### Suffix Patterns
```python
SUFFIX_PATTERNS = {
    "_m": Dimension(length=1),           # meters
    "_kg": Dimension(mass=1),            # kilograms
    "_s": Dimension(time=1),             # seconds
    "_a": Dimension(current=1),          # amperes
    "_k": Dimension(temperature=1),      # kelvin
    "_m_per_s": Dimension(length=1, time=-1),
    "_m_s2": Dimension(length=1, time=-2),
}
```

---

## Equation Pattern Examples

```python
EQUATION_PATTERNS = [
    # F = ma
    {
        "pattern": "force = mass * acceleration",
        "check": lambda F, m, a: F.dimension == m.dimension * a.dimension
    },
    # E = ½mv²
    {
        "pattern": "energy = 0.5 * mass * velocity ** 2",
        "alias": ["kinetic_energy", "KE"]
    },
    # PV = nRT
    {
        "pattern": "pressure * volume = amount * R * temperature",
        "constant": "R"  # Gas constant
    },
]
```

---

## Testing Strategy

- [ ] Unit tests for variable name inference
- [ ] Unit tests for equation pattern matching
- [ ] Integration test: lint on sample physics code
- [ ] Test confidence thresholds
- [ ] Test custom patterns
- [ ] Test edge cases (ambiguous names)

---

## Risks / Edge Cases

- **Risk**: False positives (wrong inference)
  - Mitigation: Confidence thresholds, user can disable

- **Risk**: Variable name collisions (e.g., "time" in non-physics context)
  - Mitigation: Context-aware inference, only infer in DimArray context

- **Edge case**: "weight" commonly used but dimensionally is force
  - Handling: Document, maybe add "mass_weight" alias

- **Edge case**: Compound names "velocity_x_component"
  - Handling: Pattern matching with regex

---

## Definition of Done

- [ ] Variable name heuristics working
- [ ] At least 10 equation patterns
- [ ] `dimtensor lint` CLI command works
- [ ] Tests pass
- [ ] README documents inference feature
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-09** - Started design. Chose heuristic + pattern approach for v2.1.0.
Key insight: This is optional enhancement, not core functionality.
Users can always override with explicit units.

---
