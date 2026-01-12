# Plan: Automatic Scaling Law Discovery

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Build a system that automatically discovers power law and scaling relationships from dimensional data using dimensional analysis constraints to ensure physically valid results. The system will fit multi-variable power laws of the form `y = C * x1^a * x2^b * ...` while enforcing dimensional consistency.

---

## Background

Scaling laws are ubiquitous in physics and biology:
- Allometric scaling (biological metabolism ~ mass^(3/4))
- Drag force ~ velocity^2
- Gravitational attraction ~ distance^(-2)
- Kepler's third law: period^2 ~ radius^3

Traditional power law fitting ignores dimensional constraints, often producing physically invalid results. By leveraging dimtensor's dimensional tracking, we can:
1. Automatically constrain exponents to maintain dimensional consistency
2. Detect impossible relationships early
3. Suggest valid power law forms from data
4. Validate discovered laws against known equations

---

## Approach

### Option A: Non-linear Least Squares with Constraints
- Use scipy.optimize.least_squares with dimensional constraints
- Transform to log-space for linearity
- Pros: Simple, works well for 2-3 variables, well-tested scipy backend
- Cons: Gradient-based, local minima issues, doesn't explore exponent space systematically

### Option B: Symbolic Regression (Genetic Programming)
- Evolve equation trees with dimensional validation
- Use tournament selection with dimensional consistency as fitness criterion
- Pros: Can discover complex relationships, explores space systematically
- Cons: Computationally expensive, requires significant implementation, may overfit

### Option C: Buckingham Pi Theorem + Linear Regression
- Use dimensional analysis to identify dimensionless groups
- Fit relationships between dimensionless quantities
- Pros: Mathematically rigorous, guaranteed dimensional consistency
- Cons: Requires all variables upfront, limited to products of powers

### Decision: Hybrid Approach (A + C)

**Primary**: Option C with Buckingham Pi for systematic dimensionless group discovery
**Fallback**: Option A for simple power law fitting when all exponents are sought

**Rationale**:
1. Buckingham Pi gives us the "shape" of valid equations automatically
2. Linear regression on log-transformed dimensionless groups is robust
3. Falls back to constrained optimization for simpler cases
4. Can later add Option B as an advanced feature

---

## Implementation Steps

### Phase 1: Core Power Law Fitting (Simple Case)
1. [ ] Create `dimtensor/analysis/scaling.py` module
2. [ ] Implement `PowerLawFitter` class
   - [ ] `fit(x, y)` method for single-variable power law `y = C * x^a`
   - [ ] Dimensional consistency check: `dim(y) == dim(C) * dim(x)^a`
   - [ ] Solve for valid exponent `a` using dimension algebra
   - [ ] Use scipy curve_fit in log-space for coefficient estimation
3. [ ] Add `confidence_intervals()` method for uncertainty quantification
4. [ ] Add `predict(x_new)` method returning DimArray
5. [ ] Add `score()` method (R² with dimensional awareness)

### Phase 2: Multi-variable Power Laws
6. [ ] Implement `MultiPowerLawFitter` class
   - [ ] Handle `y = C * x1^a1 * x2^a2 * ... * xn^an`
   - [ ] Set up dimensional constraint system: `dim(y) = dim(C) * Π dim(xi)^ai`
   - [ ] Represent as linear system of equations for exponents (7 equations, n unknowns)
7. [ ] Add `fit_constrained()` using scipy.optimize with dimensional constraints
8. [ ] Add exponent search strategies:
   - [ ] Integer exponents only (fastest)
   - [ ] Rational exponents (fractions with bounded denominator)
   - [ ] Free optimization (continuous)
9. [ ] Implement grid search over rational exponent space when underdetermined

### Phase 3: Buckingham Pi Implementation
10. [ ] Create `BuckinghamPiAnalyzer` class
11. [ ] Implement dimensional matrix construction from variables
12. [ ] Use null space computation to find dimensionless products
13. [ ] Generate complete set of dimensionless Pi groups
14. [ ] Fit relationship between Pi groups using linear regression
15. [ ] Convert back to dimensional form

### Phase 4: Visualization & Validation
16. [ ] Create `ScalingLawVisualizer` class
   - [ ] `plot_fit()`: data vs fitted curve with residuals
   - [ ] `plot_log_log()`: log-log plot showing power law linearity
   - [ ] `plot_residuals()`: residual analysis
   - [ ] `plot_pi_groups()`: dimensionless group relationships
17. [ ] Add comparison with equation database
18. [ ] Generate LaTeX formula strings for discovered laws
19. [ ] Add export to SymPy for symbolic manipulation

### Phase 5: Advanced Features
20. [ ] Add multi-model comparison (AIC/BIC)
21. [ ] Implement bootstrapping for robust error estimation
22. [ ] Add outlier detection and robust fitting
23. [ ] Support for hierarchical/nested power laws
24. [ ] Integration with datasets module for automatic analysis

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/analysis/__init__.py` | Create new analysis submodule |
| `src/dimtensor/analysis/scaling.py` | New file: power law fitting classes |
| `src/dimtensor/analysis/buckingham.py` | New file: Buckingham Pi implementation |
| `src/dimtensor/visualization/matplotlib.py` | Add scaling law plotting utilities |
| `src/dimtensor/visualization/plotly.py` | Add interactive scaling law visualizations |
| `tests/test_scaling.py` | New file: comprehensive tests |
| `examples/scaling_laws.ipynb` | New file: tutorial notebook |
| `docs/guide/scaling-laws.md` | New file: user guide |

---

## Testing Strategy

### Unit Tests (`tests/test_scaling.py`)
- [ ] Test single-variable power law fitting
  - [ ] Known relationship: F = k * x^2 (spring force)
  - [ ] Check dimensional consistency
  - [ ] Verify coefficient and exponent recovery
- [ ] Test multi-variable fitting
  - [ ] Ideal gas law variant: P ~ T/V
  - [ ] Gravitational law: F ~ m1*m2/r^2
- [ ] Test dimensional constraint enforcement
  - [ ] Reject impossible exponent combinations
  - [ ] Accept only dimensionally valid solutions
- [ ] Test Buckingham Pi analysis
  - [ ] Pendulum period: T(L, g) → T ~ sqrt(L/g)
  - [ ] Drag force: F(ρ, v, A) → F ~ ρ * v^2 * A
- [ ] Test with noisy data
  - [ ] Add Gaussian noise, verify robustness
  - [ ] Check confidence intervals contain true values
- [ ] Test edge cases
  - [ ] Dimensionless input/output
  - [ ] Underdetermined systems (more unknowns than constraints)
  - [ ] Overdetermined systems (more constraints than unknowns)

### Integration Tests
- [ ] Test with real datasets from `dimtensor.datasets`
  - [ ] Planetary motion data → Kepler's laws
  - [ ] Climate data → temperature scaling
- [ ] Test visualization outputs (matplotlib, plotly)
- [ ] Test SymPy export and manipulation
- [ ] Test equation database matching

### Example Notebook Tests
- [ ] Allometric scaling in biology
- [ ] Atmospheric pressure vs altitude
- [ ] Electrical power relationships
- [ ] Fluid dynamics (Reynolds number)

---

## Risks / Edge Cases

### Risk 1: Underdetermined Systems
**Issue**: More unknown exponents than dimensional constraints (e.g., y[L] = x1[L]^a * x2[L]^b has infinite solutions)
**Mitigation**:
- Detect underdetermined case and enumerate rational solutions
- Use additional constraints (sparsity, simplicity preference)
- Allow user to fix certain exponents

### Risk 2: Numerical Instability in Log-Space
**Issue**: Log transform fails for negative values, amplifies errors for small values
**Mitigation**:
- Check for non-positive values before log transform
- Use robust regression techniques
- Provide warnings about data quality

### Risk 3: Non-Power-Law Relationships
**Issue**: User tries to fit exponential or other non-power-law forms
**Mitigation**:
- Add goodness-of-fit diagnostics
- Provide warnings when log-log plot is non-linear
- Suggest alternative models in future versions

### Risk 4: Computational Complexity
**Issue**: Buckingham Pi with many variables creates many dimensionless groups
**Mitigation**:
- Limit to 5-7 variables initially
- Use sparse matrix methods for dimensional matrix
- Cache null space computations

### Edge Case: Zero or Unit Exponents
**Handling**: Some variables may not appear (exponent = 0) or appear linearly (exponent = 1). Include penalty term favoring simpler models.

### Edge Case: Scale-Dependent Relationships
**Handling**: Some relationships change form at different scales (e.g., Reynolds number transitions). Document limitation and suggest piecewise fitting.

### Edge Case: Offsetted Power Laws
**Handling**: Forms like `y = C * (x - x0)^a` require preprocessing. Detect and transform data first.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass with >90% coverage for new code
- [ ] Can discover known scaling laws from synthetic data:
  - [ ] Kepler's third law
  - [ ] Drag equation
  - [ ] Ideal gas relationships
- [ ] Visualization functions produce clear, publication-quality plots
- [ ] Documentation with examples in docs/guide/
- [ ] Tutorial notebook with 3+ real-world examples
- [ ] Integration with existing dimtensor modules (scipy, visualization)
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**Research Notes**:

1. **Dimensional Analysis Math**: For power law `y = C * x1^a1 * x2^a2 * ... * xn^an`, dimensional consistency requires:
   ```
   dim(y) = dim(C) * dim(x1)^a1 * dim(x2)^a2 * ... * dim(xn)^an
   ```
   This gives 7 linear equations (one per SI base dimension) in n+1 unknowns (n exponents + 1 for C's dimension).

2. **Buckingham Pi Theorem**: If you have n variables with k independent dimensions, you get n-k dimensionless Pi groups. The relationship must be expressible as a function of these Pi groups only.

3. **Existing Patterns**:
   - `dimtensor.scipy.optimize.curve_fit` for dimension-aware fitting
   - `dimtensor.core.dimensions.Dimension` algebra for constraint solving
   - `dimtensor.inference.equations` for equation database matching
   - `dimtensor.visualization.matplotlib` for plotting conventions

4. **Similar Work**:
   - eureqa/DataRobot: Commercial symbolic regression
   - PySR: Open-source symbolic regression (but no dimension awareness)
   - PyDy: Focuses on dynamics, not discovery

---
