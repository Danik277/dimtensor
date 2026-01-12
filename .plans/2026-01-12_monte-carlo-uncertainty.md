# Plan: Monte Carlo Uncertainty Propagation

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add Monte Carlo uncertainty propagation to dimtensor, enabling statistical estimation of output uncertainty distributions through random sampling when analytical propagation is insufficient or unavailable.

---

## Background

dimtensor currently supports analytical uncertainty propagation (v0.5.0) using first-order Taylor series approximations:
- Addition/subtraction: `σ_z = sqrt(σ_x² + σ_y²)`
- Multiplication/division: `σ_z/|z| = sqrt((σ_x/x)² + (σ_y/y)²)`
- Power: `σ_z/|z| = |n| * σ_x/|x|`

However, analytical propagation has limitations:
1. Assumes small, uncorrelated uncertainties (Gaussian approximation)
2. Does not handle correlated inputs
3. Complex operations (std, var) drop uncertainty
4. Cannot estimate non-Gaussian output distributions
5. Higher-order effects ignored

Monte Carlo methods overcome these limitations by:
- Sampling from input distributions
- Evaluating the function for each sample
- Computing statistics from the output distribution

This is essential for:
- Physics simulations with correlated parameters
- Risk analysis and error budgets
- Nonlinear operations where linear approximations fail
- Validating analytical propagation accuracy

---

## Approach

### Option A: Simple Random Sampling
- Description: Standard Monte Carlo with pseudo-random sampling
- Pros:
  - Simple to implement
  - Works with any probability distribution
  - Easily parallelizable
- Cons:
  - Slow convergence (O(1/sqrt(N)))
  - May miss important regions with sparse sampling
  - No control over sample distribution

### Option B: Latin Hypercube Sampling (LHS)
- Description: Stratified sampling ensuring coverage of parameter space
- Pros:
  - Better coverage than random sampling
  - Faster convergence for multi-dimensional problems
  - Standard in engineering uncertainty analysis
- Cons:
  - More complex implementation
  - Requires careful correlation handling

### Option C: Quasi-Monte Carlo (Sobol sequences)
- Description: Deterministic low-discrepancy sequences
- Pros:
  - Even better convergence than LHS
  - Reproducible results
  - Well-studied for high dimensions
- Cons:
  - Requires external library (scipy.stats.qmc)
  - Correlation handling is non-trivial

### Decision: Implement all three with unified interface

Provide `monte_carlo()` function with `method` parameter:
- `method="random"` - Simple random (default, no extra deps)
- `method="lhs"` - Latin hypercube (scipy.stats.qmc)
- `method="sobol"` - Quasi-Monte Carlo (scipy.stats.qmc)

This gives users flexibility: start simple, upgrade for better convergence.

---

## Implementation Steps

1. [ ] Create `src/dimtensor/uncertainty/__init__.py` module
2. [ ] Create `src/dimtensor/uncertainty/monte_carlo.py`
   - `monte_carlo()` function taking DimArray with uncertainty
   - Sample input distributions (Gaussian by default)
   - Apply callable function to samples
   - Return statistics (mean, std, percentiles)
3. [ ] Add sampling strategies in `src/dimtensor/uncertainty/sampling.py`
   - `RandomSampler` - Basic Monte Carlo
   - `LHSSampler` - Latin hypercube sampling
   - `SobolSampler` - Quasi-Monte Carlo
   - Base class `Sampler` with common interface
4. [ ] Add correlation handling in `src/dimtensor/uncertainty/correlation.py`
   - `CorrelationMatrix` class
   - Cholesky decomposition for correlated sampling
   - Copula support (optional, future)
5. [ ] Add convergence diagnostics in `src/dimtensor/uncertainty/diagnostics.py`
   - `check_convergence()` - Monitors running statistics
   - Standard error estimates
   - Effective sample size calculation
6. [ ] Add result statistics class `src/dimtensor/uncertainty/results.py`
   - `MCResult` dataclass with mean, std, percentiles, samples
   - Histogram/KDE visualization methods
   - Comparison with analytical propagation
7. [ ] Add `monte_carlo()` method to `DimArray` class
   - Wrapper around uncertainty module
   - Preserves units through calculation
8. [ ] Add support for multiple input DimArrays with correlation
9. [ ] Add vectorized evaluation for performance
10. [ ] Add parallel sampling option (multiprocessing)

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/uncertainty/__init__.py` | New module init |
| `src/dimtensor/uncertainty/monte_carlo.py` | Core Monte Carlo implementation |
| `src/dimtensor/uncertainty/sampling.py` | Sampling strategies (random, LHS, Sobol) |
| `src/dimtensor/uncertainty/correlation.py` | Correlated input handling |
| `src/dimtensor/uncertainty/diagnostics.py` | Convergence checking |
| `src/dimtensor/uncertainty/results.py` | Result statistics container |
| `src/dimtensor/core/dimarray.py` | Add `monte_carlo()` method |
| `tests/test_monte_carlo.py` | Comprehensive tests |
| `tests/test_sampling.py` | Sampling strategy tests |
| `tests/test_correlation.py` | Correlation tests |
| `docs/guide/uncertainty.md` | Documentation and examples |
| `setup.py` or `pyproject.toml` | Optional scipy dependency |

---

## Testing Strategy

### Unit Tests

- [ ] Test random sampling produces correct statistics
  - Normal distribution input -> verify output mean/std
  - Uniform distribution support
- [ ] Test LHS coverage (each dimension stratified)
- [ ] Test Sobol sequence properties (low discrepancy)
- [ ] Test correlation matrix handling
  - Independent inputs (identity correlation)
  - Perfectly correlated inputs (correlation = 1)
  - Negative correlation
- [ ] Test convergence diagnostics
  - Standard error decreases with N
  - Running mean stabilizes
- [ ] Test MCResult class methods
  - Percentile calculation
  - Histogram generation
  - Comparison operators

### Integration Tests

- [ ] Compare MC vs analytical for simple operations
  - `a + b` with uncorrelated uncertainties
  - `a * b` with small uncertainties
  - Verify agreement within MC error
- [ ] Test nonlinear functions where analytical fails
  - `a**b` where b has uncertainty (analytical can't handle)
  - `max(a, b)` (non-differentiable)
- [ ] Test physics calculations
  - Pendulum period with correlated length/gravity uncertainty
  - Orbital mechanics with measurement errors
- [ ] Test with physical constants
  - Propagate G uncertainty through gravitational force
  - Compare with Constant multiplication

### Performance Tests

- [ ] Benchmark convergence rates
  - Random vs LHS vs Sobol for 2D, 5D, 10D problems
- [ ] Memory profiling (sample array allocation)
- [ ] Parallel speedup measurement

### Manual Verification

- [ ] Visual inspection of output histograms
- [ ] Reproduce published uncertainty analysis from literature
- [ ] Cross-check with uncertainties package (https://github.com/lebigot/uncertainties)

---

## Risks / Edge Cases

**Risk 1: Large sample arrays consume memory**
- Mitigation: Add batch processing option, stream samples instead of storing all
- Default N=10000 reasonable for most use cases

**Risk 2: Slow performance for expensive functions**
- Mitigation: Provide parallel option (multiprocessing/joblib)
- Document that MC is inherently expensive, recommend analytical when possible

**Risk 3: Correlation matrix not positive definite**
- Mitigation: Validate correlation matrix, provide helpful error messages
- Use `np.linalg.cholesky()` and catch exceptions

**Risk 4: Units lost during callable evaluation**
- Mitigation: Verify callable returns DimArray, preserve units through entire pipeline
- Add unit checks in MCResult

**Risk 5: Non-Gaussian distributions**
- Current approach: Assume Gaussian inputs (most common in physics)
- Future: Add distribution parameter to specify uniform, triangular, etc.

**Edge Case: Zero uncertainty**
- Handle gracefully: return deterministic result without sampling

**Edge Case: Singular dimension (std=0 for some parameter)**
- Treat as fixed parameter, don't sample

**Edge Case: Complex-valued operations**
- Not supported initially, document limitation

**Edge Case: Very small N (<100)**
- Warn user that results may be unreliable, recommend N≥1000

**Edge Case: Convergence failure**
- Diagnostics detect non-convergence, return warning with results

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (pytest)
- [ ] Type checking passes (mypy)
- [ ] Coverage ≥85% for new module
- [ ] Documentation with examples
  - Basic usage with default sampling
  - Correlated inputs example
  - Comparison with analytical propagation
  - Performance tips
- [ ] CONTINUITY.md updated with task completion
- [ ] Example notebook demonstrating:
  - Simple MC propagation
  - LHS vs random convergence comparison
  - Correlated parameter handling
  - Physics calculation (e.g., projectile motion with wind)

---

## Notes / Log

**Research Notes:**

**Existing uncertainty infrastructure:**
- DimArray has `uncertainty` attribute (NumPy array)
- `has_uncertainty` property
- `relative_uncertainty` property
- Propagation methods: `_propagate_add_sub`, `_propagate_mul_div`, `_propagate_power`
- Constants support uncertainty (CODATA values)

**Key design patterns from codebase:**
1. Use `_from_data_and_unit()` for internal construction (no copy)
2. Operations return new instances (immutable)
3. Uncertainty is always absolute (not relative)
4. Shape must match data shape
5. None uncertainty treated as zero

**Sampling library options:**
- `numpy.random` - Built-in, sufficient for basic MC
- `scipy.stats.qmc` - Latin hypercube, Sobol (requires scipy)
- `SALib` - Sensitivity analysis library (heavy dependency, not needed)

**Reference implementations:**
- uncertainties package: https://github.com/lebigot/uncertainties
- PyMC for MCMC (different use case, too heavy)
- scipy.stats.monte_carlo_test (statistical tests, not propagation)

**Academic references:**
1. "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code" - McKay et al. (1979) - LHS original paper
2. "Monte Carlo and Quasi-Monte Carlo Sampling" - Owen (2003)
3. ISO Guide to the Expression of Uncertainty in Measurement (GUM) - Supplement 1: Monte Carlo methods

---
