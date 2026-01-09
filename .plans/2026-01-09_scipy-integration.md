# SciPy Integration Plan

## Goal
Integrate dimtensor with SciPy to enable unit-aware scientific computing for:
- Optimization (scipy.optimize)
- Integration (scipy.integrate)
- Interpolation (scipy.interpolate)
- Signal processing (scipy.signal)

## Research Findings

### Key SciPy modules to wrap:
1. **scipy.optimize** - minimize, curve_fit, root, least_squares
2. **scipy.integrate** - odeint, solve_ivp, quad
3. **scipy.interpolate** - interp1d, UnivariateSpline

### Challenges:
1. SciPy expects raw numpy arrays
2. Need to strip units, compute, then restore units
3. Must track units through transformations (e.g., derivative changes time dimension)

### Design Pattern:
```python
# Wrapper approach
from dimtensor.scipy import minimize, curve_fit

# minimize(fun, x0) where x0 is DimArray
result = minimize(objective, initial_guess)  # returns DimArray

# curve_fit with dimensional data
popt, pcov = curve_fit(model, x_data, y_data)  # x_data, y_data are DimArrays
```

## Implementation Plan

### Phase 1: Core wrapper module
1. Create `src/dimtensor/scipy/__init__.py`
2. Create `src/dimtensor/scipy/optimize.py`
   - `minimize()` - dimension-aware minimization
   - `curve_fit()` - dimensional curve fitting
   - `least_squares()` - dimensional least squares

### Phase 2: Integration module
1. Create `src/dimtensor/scipy/integrate.py`
   - `solve_ivp()` - ODE solver with units
   - `quad()` - numerical integration
   - Track units through derivatives (d/dt changes dimension by T^-1)

### Phase 3: Interpolation module
1. Create `src/dimtensor/scipy/interpolate.py`
   - `interp1d()` - 1D interpolation
   - `UnivariateSpline()` - spline interpolation

## Key Design Decisions

### Unit handling in optimization:
- Strip units before calling scipy
- Restore units to result based on initial guess units
- For bounded optimization, bounds must have same units as x0

### Unit handling in integration:
- For dy/dt = f(t, y), result y has same dimension as y0
- For ∫f(x)dx, result has dimension of f(x) * x

### Error handling:
- Raise DimensionError if units are incompatible
- Provide clear messages about expected vs actual dimensions

## Files to Create

| File | Purpose | Lines (est) |
|------|---------|-------------|
| scipy/__init__.py | Exports | ~20 |
| scipy/optimize.py | Optimization wrappers | ~150 |
| scipy/integrate.py | Integration wrappers | ~150 |
| scipy/interpolate.py | Interpolation wrappers | ~100 |

## Tests

| Test | Description |
|------|-------------|
| test_minimize | Dimension-aware minimization |
| test_curve_fit | Fit model to dimensional data |
| test_solve_ivp | ODE with physical units |
| test_quad | Numerical integration |
| test_interp1d | 1D interpolation |
