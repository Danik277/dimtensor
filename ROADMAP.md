# dimtensor: Long-Term Roadmap

## Vision Statement

**dimtensor** will become the standard library for dimensionally-aware computation in scientific machine learning - the "missing link" between raw tensor operations and physically meaningful calculations.

**Ultimate Goal**: Any physicist, chemist, or engineer using Python for ML should reach for dimtensor the same way they reach for numpy.

---

## Current Status (v0.9.0) - 2026-01-08

### Completed Features

| Version | Feature | Status | Notes |
|---------|---------|--------|-------|
| v0.1.x | Foundation | DONE | Basic DimArray, units, PyPI |
| v0.2.x | Usability | DONE | Unit simplification, format strings, numpy ufuncs |
| v0.3.x | NumPy Parity | DONE | Array functions, linear algebra, reshaping |
| v0.4.x | Constants | DONE | CODATA 2022 physical constants |
| v0.5.x | Uncertainty | DONE | Error propagation through calculations |
| v0.6.x | PyTorch | DONE | DimTensor with autograd, GPU support |
| v0.7.x | JAX | DONE | DimArray with pytree, JIT, vmap, grad |
| v0.8.x | Benchmarks | DONE | Performance measurement module |
| v0.9.x | Serialization | DONE | JSON, Pandas, HDF5 support |

### Deferred Features (moved to v1.x)
- Rust backend for performance (originally v0.8)
- NetCDF support (originally v0.9)
- Parquet support (originally v0.9)
- xarray integration (originally v0.9)

### Current Metrics
- **Tests**: 316 passing, 48 skipped
- **Coverage**: 72%
- **PyPI**: v0.9.0 deployed

---

## v1.0.0 - Production Release

**Theme**: Consolidation, quality, documentation

**Status**: IN PROGRESS

### v1.0.0 Checklist

#### Code Quality (REQUIRED)
- [ ] **Code review**: Review all v0.5-v0.9 code for quality issues
  - [ ] Review `src/dimtensor/torch/dimtensor.py` (591 lines)
  - [ ] Review `src/dimtensor/jax/dimarray.py` (506 lines)
  - [ ] Review `src/dimtensor/io/*.py` (JSON, Pandas, HDF5)
  - [ ] Review `src/dimtensor/benchmarks.py` (304 lines)
- [ ] **Type safety**: Clean mypy output (0 errors)
- [ ] **Test coverage**: Increase from 72% to 85%+
- [ ] **Fix any bugs found during review**

#### Documentation (REQUIRED)
- [ ] **API documentation**: All public functions documented
- [ ] **README update**: Reflect all current features
- [ ] **Examples**: Working examples for PyTorch, JAX, serialization
- [ ] **CHANGELOG**: Ensure all changes are documented

#### Deployment (REQUIRED)
- [ ] **Version bump**: Update to 1.0.0
- [ ] **PyPI release**: Deploy stable v1.0.0
- [ ] **Git tag**: Tag release

#### Nice to Have (v1.0.1+)
- [ ] Migration guide from pint/astropy.units
- [ ] Performance benchmarks documented
- [ ] Security audit

---

## v1.x - Ecosystem & Extensions

### v1.1.0 - Missing Serialization
- [ ] NetCDF support
- [ ] Parquet support
- [ ] xarray integration

### v1.2.0 - Domain Extensions
- [ ] Astronomy units (parsec, AU, solar mass)
- [ ] Chemistry units (molar, molal, ppm, pH)
- [ ] Engineering units (MPa, ksi, BTU)

### v1.3.0 - Visualization
- [ ] Matplotlib integration (auto-labeled axes)
- [ ] Plotly integration
- [ ] Unit conversion in plots

### v1.4.0 - Validation & Constraints
- [ ] Value constraints (positive, bounded)
- [ ] Conservation laws tracking
- [ ] Custom constraints

---

## v2.x - Performance & Intelligence

### v2.0.0 - Rust Backend
**Theme**: Production-ready speed

- [ ] Rust core via PyO3
- [ ] Lazy evaluation
- [ ] Operator fusion
- [ ] Memory optimization
- [ ] Target: <10% overhead vs raw numpy

### v2.1.0 - Dimensional Inference
- [ ] Variable name heuristics
- [ ] Equation pattern matching
- [ ] IDE integration for unit hints
- [ ] Linting for dimensional errors

### v2.2.0 - Physics-Aware ML
- [ ] Physics-aware layers
- [ ] Dimensional loss functions
- [ ] Unit-aware normalization
- [ ] Automatic non-dimensionalization

---

## v3.x - Platform

### v3.0.0 - Physics ML Toolkit
- [ ] Model hub (pre-trained physics models)
- [ ] Equation database
- [ ] Dataset registry with units
- [ ] CLI tools (`dimtensor check`, `dimtensor convert`)
- [ ] Symbolic computing bridge

---

## Success Metrics

| Milestone | Metric | Target | Current |
|-----------|--------|--------|---------|
| v1.0 | Test coverage | 85%+ | 72% |
| v1.0 | mypy errors | 0 | TBD |
| v1.0 | GitHub stars | 100 | - |
| v1.5 | PyPI downloads/month | 10,000 | - |
| v2.0 | Performance overhead | <10% | ~2-5x |
| v2.0 | Contributors | 10+ | 1 |

---

## Guiding Principles

1. **Correctness over performance**: Never silently produce wrong units
2. **Zero overhead for correct code**: Optimizable when units are consistent
3. **Progressive complexity**: Simple things simple, complex things possible
4. **NumPy/PyTorch idioms**: Feel familiar to existing users
5. **Explicit over implicit**: No magical unit inference without user opt-in
6. **Interoperability**: Play nice with the ecosystem

---

## File Structure

```
src/dimtensor/
├── core/              # DimArray, units, dimensions
│   ├── dimarray.py
│   ├── dimensions.py
│   └── units.py
├── constants/         # Physical constants (CODATA 2022)
│   ├── universal.py
│   ├── electromagnetic.py
│   ├── atomic.py
│   └── physico_chemical.py
├── torch/             # PyTorch integration
│   └── dimtensor.py
├── jax/               # JAX integration
│   └── dimarray.py
├── io/                # Serialization
│   ├── json.py
│   ├── pandas.py
│   └── hdf5.py
├── benchmarks.py      # Performance measurement
├── functions.py       # Array functions
├── errors.py          # Custom exceptions
└── config.py          # Display configuration
```
