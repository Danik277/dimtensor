# Plan: Error Budget Calculator for Uncertainty Contribution Analysis

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Create an error budget calculator that decomposes the total uncertainty in a computed result into contributions from each input variable, following the Guide to Uncertainty in Measurement (GUM) methodology. Enable scientists and engineers to identify dominant uncertainty sources and visualize error budgets through tables and charts.

---

## Background

dimtensor v0.5.0 introduced uncertainty propagation through arithmetic operations using standard formulas (quadrature for addition/subtraction, relative uncertainty combination for multiplication/division). However, users currently cannot answer critical questions like:

- Which input contributes most to the final uncertainty?
- How much would improving a specific measurement reduce total uncertainty?
- What is the sensitivity coefficient for each input variable?

The GUM framework addresses this through sensitivity coefficients and uncertainty budgets. This is essential for:
- **Experimental design**: Prioritizing which measurements to improve
- **Quality control**: Identifying weak links in measurement chains
- **Reporting**: ISO/IEC 17025 compliance requires uncertainty budgets
- **Education**: Teaching uncertainty analysis in physics labs

---

## Approach

### Option A: Symbolic Differentiation via Tape Recording

- **Description**: Record all operations on DimArrays (like autograd tape), compute sensitivity coefficients via chain rule
- **Pros**:
  - Exact derivatives for all supported operations
  - Works with complex multi-step calculations
  - Natural integration with existing code
  - Can leverage PyTorch's autograd infrastructure where available
- **Cons**:
  - Requires tape recording infrastructure (memory overhead)
  - Complex implementation for first version
  - May have performance impact on normal operations

### Option B: Finite Difference Approximation

- **Description**: Re-evaluate the computation with perturbed inputs to estimate partial derivatives
- **Pros**:
  - Simple to implement
  - No changes to core DimArray operations
  - Works as external analysis tool
  - User controls when to compute (no runtime overhead otherwise)
- **Cons**:
  - Requires re-computation (N+1 evaluations for N inputs)
  - Numerical precision issues with step size selection
  - User must provide the computation function explicitly
  - Cannot work with pre-computed results

### Option C: Manual Sensitivity Specification

- **Description**: User provides sensitivity coefficients manually or via symbolic math
- **Pros**:
  - Zero runtime overhead
  - Works with any calculation (even external)
  - Educational value (forces understanding of error propagation)
- **Cons**:
  - Requires manual derivation (error-prone)
  - Not practical for complex multi-step computations
  - No automation benefits

### Decision: Start with Option B, prepare for Option A

**Phase 1** (this plan): Implement finite difference approach as standalone module
- Provides immediate value
- Simple, robust implementation
- Learn usage patterns from users

**Phase 2** (future): Add tape recording for automatic differentiation
- Build on autograd/JAX infrastructure where available
- Provides "zero-overhead" error budgeting
- Can co-exist with finite difference method

This phased approach delivers value quickly while keeping the door open for more sophisticated methods.

---

## Implementation Steps

### Phase 1: Core Error Budget Calculator

1. [ ] Create `src/dimtensor/uncertainty/` subpackage
   - [ ] `__init__.py` with main exports
   - [ ] `error_budget.py` for ErrorBudget class and analysis functions
   - [ ] `sensitivity.py` for sensitivity coefficient computation

2. [ ] Implement sensitivity coefficient calculation via finite differences
   - [ ] `compute_sensitivity(func, inputs, input_name, step_factor=0.01)` → DimArray
   - [ ] Handle both scalar and array inputs
   - [ ] Smart step size selection based on relative uncertainty
   - [ ] Return dimensionally correct sensitivity (∂f/∂x_i)

3. [ ] Implement ErrorBudget class
   ```python
   class ErrorBudget:
       """Container for uncertainty contribution analysis."""
       def __init__(self, result, input_contributions, sensitivities)
       def to_dataframe() -> pd.DataFrame  # Tabular report
       def to_dict() -> dict  # For serialization
       def plot_pie(ax=None, **kwargs)  # Pie chart
       def plot_bar(ax=None, **kwargs)  # Bar chart
       def plot_pareto(ax=None, **kwargs)  # Sorted contributions
   ```

4. [ ] Implement main analysis function
   ```python
   def compute_error_budget(
       func: Callable,
       inputs: dict[str, DimArray],
       **kwargs
   ) -> ErrorBudget:
       """
       Compute error budget by analyzing how each input uncertainty
       contributes to output uncertainty.

       Args:
           func: Function that computes result from inputs
           inputs: Dict mapping input names to DimArrays with uncertainty

       Returns:
           ErrorBudget with contribution breakdown
       """
   ```

5. [ ] Add GUM-compliant reporting
   - [ ] Combined standard uncertainty (u_c)
   - [ ] Effective degrees of freedom (Welch-Satterthwaite)
   - [ ] Coverage factor k and expanded uncertainty U
   - [ ] Percent contributions summing to 100%

6. [ ] Create visualization functions integrated with existing matplotlib backend
   - [ ] Pie chart with percentage labels
   - [ ] Horizontal bar chart sorted by contribution
   - [ ] Pareto chart (cumulative contributions)
   - [ ] Waterfall chart showing error accumulation

### Phase 2: Integration & Polish

7. [ ] Add convenience method to DimArray
   - [ ] `arr.error_budget_for(func, **other_inputs)` → ErrorBudget

8. [ ] Add support for correlated inputs (covariance matrix)
   - [ ] Cross-term contributions: 2*ρ*u_i*u_j*∂f/∂x_i*∂f/∂x_j

9. [ ] Create comprehensive examples
   - [ ] Physics lab example (pendulum period from length, g)
   - [ ] Engineering example (Ohm's law error budget)
   - [ ] Multi-step calculation (kinetic energy from mass, velocity)

10. [ ] Documentation
    - [ ] API reference for uncertainty module
    - [ ] Tutorial notebook on error budgeting
    - [ ] Add section to main uncertainty guide

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/uncertainty/__init__.py` | Create subpackage, export main functions |
| `src/dimtensor/uncertainty/error_budget.py` | Implement ErrorBudget class and compute_error_budget() |
| `src/dimtensor/uncertainty/sensitivity.py` | Implement sensitivity coefficient calculation |
| `src/dimtensor/__init__.py` | Add uncertainty submodule to imports (optional convenience) |
| `src/dimtensor/core/dimarray.py` | Add error_budget_for() convenience method (optional) |
| `tests/test_error_budget.py` | Comprehensive test suite |
| `tests/test_sensitivity.py` | Test sensitivity computation edge cases |
| `examples/uncertainty/error_budget_tutorial.ipynb` | Tutorial notebook |
| `docs/guide/uncertainty.md` | Add error budget section |
| `pyproject.toml` | Ensure pandas is in dependencies (for to_dataframe()) |

---

## Testing Strategy

### Unit Tests (`tests/test_sensitivity.py`)

- [ ] Sensitivity coefficient computation
  - [ ] Linear function: ∂(a*x + b)/∂x = a
  - [ ] Power function: ∂(x^n)/∂x = n*x^(n-1)
  - [ ] Product rule: ∂(x*y)/∂x = y
  - [ ] Multivariable: ∂(x*y*z)/∂x, ∂/∂y, ∂/∂z
- [ ] Step size selection
  - [ ] Adaptive step based on relative uncertainty
  - [ ] Handle zero values gracefully
  - [ ] Array-valued inputs
- [ ] Dimensional correctness of sensitivities
  - [ ] If f has units [L], x has units [T], then ∂f/∂x has units [L/T]

### Integration Tests (`tests/test_error_budget.py`)

- [ ] Complete error budget calculation
  - [ ] Single-variable function (quadratic uncertainty)
  - [ ] Two-variable function (both contribute)
  - [ ] Many-variable function (identify dominant term)
  - [ ] Verify sum of squared contributions equals total squared uncertainty
- [ ] ErrorBudget class
  - [ ] to_dataframe() produces correct columns
  - [ ] to_dict() serialization roundtrip
  - [ ] Percentage contributions sum to 100%
- [ ] Visualization (smoke tests, no visual inspection)
  - [ ] plot_pie() runs without error
  - [ ] plot_bar() runs without error
  - [ ] plot_pareto() runs without error

### Real-world Examples (Manual Verification)

- [ ] Pendulum period: T = 2π√(L/g)
  - Known sensitivities: ∂T/∂L = π/√(gL), ∂T/∂g = -πL/g^(3/2)
  - Verify computed vs analytical
- [ ] Ohm's law: P = V²/R
  - Sensitivity to V should dominate (appears squared)
- [ ] Combined gas law: PV = nRT
  - All inputs contribute equally for typical uncertainties

### Edge Cases

- [ ] Input without uncertainty (zero contribution)
- [ ] Result without uncertainty (should warn or handle gracefully)
- [ ] Function that doesn't depend on some inputs (zero sensitivity)
- [ ] Division by near-zero (numerical stability)
- [ ] Array-valued results (report contribution for each element)

---

## Risks / Edge Cases

### Risk 1: Finite difference step size selection
**Problem**: Too small → numerical errors; too large → truncation errors
**Mitigation**:
- Default to step = max(relative_uncertainty * value, 1e-8 * |value|)
- Document that users can override step size
- Test with known analytical derivatives

### Risk 2: Computational cost for many inputs
**Problem**: N inputs requires N+1 function evaluations
**Mitigation**:
- Document computational complexity upfront
- For large N, consider parallelization (future)
- Phase 2 tape recording will eliminate this

### Risk 3: Non-differentiable functions
**Problem**: Finite differences fail at discontinuities, min/max, etc.
**Mitigation**:
- Document limitations clearly
- Catch NaN/Inf in results and warn user
- For min/max, sensitivity is ill-defined anyway (as currently implemented)

### Risk 4: Correlated inputs
**Problem**: Phase 1 assumes independent inputs (no covariance terms)
**Mitigation**:
- Document independence assumption clearly
- Phase 2 will add covariance support
- For now, users can manually add cross-terms if needed

### Risk 5: Unit compatibility in visualization
**Problem**: Contributions have different units (absolute vs squared)
**Mitigation**:
- Work with squared uncertainties internally (variance contributions)
- Report contributions as percentages (dimensionless)
- Provide both absolute (with units) and relative (%) in DataFrame

### Edge Case: Zero nominal values
**Problem**: Relative step calculation fails for x=0
**Mitigation**: Use absolute step size fallback (e.g., smallest non-zero uncertainty)

### Edge Case: Exact constants
**Problem**: Constants with zero uncertainty (e.g., π, c) have zero sensitivity
**Mitigation**: Handle gracefully, report zero contribution

### Edge Case: Array-valued results
**Problem**: Error budget for each output element could be large
**Mitigation**:
- ErrorBudget stores per-element contributions
- Provide summary statistics (mean, max contribution)
- Visualization shows aggregate or selected elements

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Core functionality works:
  - [ ] compute_error_budget() produces correct results for test cases
  - [ ] ErrorBudget class can generate reports and plots
  - [ ] Sensitivity coefficients match analytical derivatives within tolerance
- [ ] Tests pass:
  - [ ] Unit tests achieve >90% coverage of new code
  - [ ] Integration tests verify real-world examples
  - [ ] Edge cases handled gracefully
- [ ] Documentation complete:
  - [ ] API docstrings follow dimtensor conventions
  - [ ] Tutorial notebook demonstrates key use cases
  - [ ] Guide page explains GUM methodology
- [ ] Code review:
  - [ ] Follows dimtensor patterns (immutability, unit safety)
  - [ ] Type hints complete
  - [ ] No performance regressions in core DimArray operations
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

### Research Findings

**Existing uncertainty infrastructure:**
- DimArray supports `uncertainty` parameter (absolute uncertainty)
- Propagation via `_propagate_add_sub()`, `_propagate_mul_div()`, `_propagate_power()`
- Follows standard formulas: quadrature for ±, relative combination for ×/÷
- `relative_uncertainty` property computes σ/|value|
- Unit conversion scales uncertainty correctly
- Reduction operations (sum, mean) propagate uncertainty
- Min/max take uncertainty from selected element

**Visualization infrastructure:**
- matplotlib integration exists (`src/dimtensor/visualization/matplotlib.py`)
- Plotly integration exists (`src/dimtensor/visualization/plotly.py`)
- Automatic axis labeling with units
- errorbar() function already exists for uncertainty visualization
- Can build on this for error budget charts

**Key patterns:**
- Internal constructors: `DimArray._from_data_and_unit()`
- Operations return new instances (immutable)
- Type hints throughout
- Comprehensive test coverage expected

**GUM Framework:**
- Combined standard uncertainty: u_c = √(Σ(c_i * u_i)²)
- Sensitivity coefficient: c_i = ∂f/∂x_i
- Contribution: u_i(y) = |c_i| * u(x_i)
- Percent contribution: 100% * [u_i(y)]² / u_c²

### Design Decisions Made

1. **Finite differences over tape recording** for Phase 1
   - Simpler implementation
   - No overhead when not used
   - Can always add autograd later

2. **Standalone module** rather than core DimArray methods
   - Keeps core lightweight
   - Clear separation of analysis vs computation
   - Optional import for users who don't need it

3. **pandas DataFrame** for tabular output
   - Industry standard for data analysis
   - Easy export to CSV, Excel, LaTeX
   - dimtensor already has pandas integration (io.pandas)

4. **matplotlib-first** visualization
   - Consistent with existing dimtensor.visualization
   - Publication-quality output
   - Plotly version can follow in Phase 2

---
