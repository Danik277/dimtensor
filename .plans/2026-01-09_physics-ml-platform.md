# Plan: Physics ML Platform (v3.0.0)

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Task IDs**: #102-119

---

## Goal

Create a comprehensive physics ML platform with:
1. Model hub for pre-trained physics models
2. Equation database for dimensional analysis
3. Dataset registry with unit metadata
4. CLI tools for common operations

---

## Research Summary

### Model Hub Patterns (HuggingFace, PyTorch Hub)

Key components:
- **Registry**: Factory pattern for model creation
- **Hub Integration**: Download from remote, local cache
- **Model Cards**: Metadata about model, training, usage
- **Versioning**: Git-based approach for large files

For dimtensor, we need physics-specific additions:
- Input/output dimensions for each model
- Physical domains (mechanics, thermodynamics, etc.)
- Characteristic scales used in training
- Conservation laws enforced

---

## Implementation Plan

### Phase 1: Model Registry (Tasks #102-104)

```
src/dimtensor/hub/
├── __init__.py         # Exports
├── registry.py         # Model registry and factory
├── cards.py            # Model card format
└── cache.py            # Local cache management
```

#### registry.py
```python
@dataclass
class ModelInfo:
    """Metadata about a registered model."""
    name: str
    version: str
    description: str
    input_dims: dict[str, Dimension]   # {name: dimension}
    output_dims: dict[str, Dimension]
    domain: str  # "mechanics", "thermodynamics", etc.
    characteristic_scales: dict[str, float]
    tags: list[str]
    source: str  # URL or local path

_REGISTRY: dict[str, ModelInfo] = {}

def register_model(name: str, info: ModelInfo) -> None:
    """Register a model in the local registry."""

def get_model(name: str) -> nn.Module:
    """Load a model from the registry."""

def list_models(domain: str | None = None) -> list[ModelInfo]:
    """List available models, optionally filtered by domain."""
```

#### cards.py - YAML format for model cards
```yaml
name: fluid-velocity-predictor
version: "1.0.0"
description: "Predicts fluid velocity field from boundary conditions"
domain: fluid_dynamics
input_dimensions:
  boundary_velocity: {L: 1, T: -1}  # m/s
  pressure: {M: 1, L: -1, T: -2}     # Pa
output_dimensions:
  velocity_field: {L: 1, T: -1}      # m/s
characteristic_scales:
  velocity: 10.0  # m/s
  length: 1.0     # m
  pressure: 101325.0  # Pa
architecture: DimSequential
training:
  dataset: cavity-flow-v1
  epochs: 100
  physics_loss_weight: 0.1
tags: [fluid, cfd, navier-stokes]
```

### Phase 2: Equation Database (Tasks #105-108)

Expand existing inference/equations.py to a full database:

```
src/dimtensor/equations/
├── __init__.py
├── database.py         # SQLite-backed equation storage
├── mechanics.py        # F=ma, E=½mv², etc.
├── electromagnetics.py # Maxwell's equations
├── thermodynamics.py   # PV=nRT, heat transfer
└── fluid_dynamics.py   # Navier-Stokes, Bernoulli
```

Schema:
```python
@dataclass
class Equation:
    name: str
    formula: str           # LaTeX or symbolic
    variables: dict[str, Dimension]
    domain: str
    tags: list[str]
    description: str
    assumptions: list[str]
```

### Phase 3: Dataset Registry (Tasks #109-111)

```
src/dimtensor/datasets/
├── __init__.py
├── registry.py         # Dataset metadata
├── loaders.py          # Load datasets with units
└── transforms.py       # Physics-aware transforms
```

```python
@dataclass
class DatasetInfo:
    name: str
    description: str
    columns: dict[str, Dimension]  # {col_name: dimension}
    size: int
    domain: str
    source: str  # URL or citation
    license: str

def load_dataset(name: str) -> dict[str, DimArray]:
    """Load a dataset with proper units."""
```

### Phase 4: CLI Tools (Tasks #112-119)

Expand existing CLI with new commands:

```bash
# Dimensional checking
dimtensor check script.py

# Unit conversion
dimtensor convert "100 km/h" --to "m/s"

# Info about units/dimensions
dimtensor info velocity

# Model operations
dimtensor hub list --domain=mechanics
dimtensor hub download fluid-velocity-predictor
dimtensor hub info navier-stokes-solver

# Equation lookup
dimtensor equations list --domain=thermodynamics
dimtensor equations search "energy"
```

---

## Implementation Order

1. **Task #102**: Create hub/registry.py with ModelInfo and basic registry
2. **Task #103**: Add model registration and lookup functions
3. **Task #104**: Add download/cache management
4. **Task #105**: Design equation database schema
5. **Task #106**: Populate mechanics equations
6. **Task #107**: Populate E&M equations
7. **Task #108**: Populate thermodynamics equations
8. **Task #109**: Design dataset registry schema
9. **Task #110**: Implement dataset registry
10. **Task #111**: Create sample physics datasets
11. **Task #112**: Design CLI architecture
12. **Task #113**: Implement `dimtensor check` command
13. **Task #114**: Implement `dimtensor convert` command
14. **Task #115**: Implement `dimtensor info` command
15. **Task #116**: Research SymPy integration
16. **Task #117**: Implement SymPy bridge
17. **Task #118**: Add tests
18. **Task #119**: Deploy v3.0.0

---

## File Structure After v3.0.0

```
src/dimtensor/
├── hub/
│   ├── __init__.py
│   ├── registry.py
│   ├── cards.py
│   └── cache.py
├── equations/
│   ├── __init__.py
│   ├── database.py
│   ├── mechanics.py
│   ├── electromagnetics.py
│   ├── thermodynamics.py
│   └── fluid_dynamics.py
├── datasets/
│   ├── __init__.py
│   ├── registry.py
│   ├── loaders.py
│   └── transforms.py
└── cli/
    ├── __init__.py
    ├── lint.py        # Existing
    ├── check.py       # NEW
    ├── convert.py     # NEW
    └── info.py        # NEW
```

---

## Test Plan

- hub/: 15+ tests for registry, cards, cache
- equations/: 20+ tests for each domain
- datasets/: 10+ tests for loading, transforms
- cli/: 15+ tests for each command
