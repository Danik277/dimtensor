# CI/CD Integration Guide

This guide shows you how to integrate dimtensor into your GitHub Actions CI/CD workflows to ensure dimensional correctness in your physics and scientific computing projects.

## Overview

dimtensor provides reusable GitHub Actions workflows and a composite action to help you:

- **Lint for dimensional consistency** - Catch dimensional errors before they make it to production
- **Test with unit safety** - Ensure your tests verify dimensional correctness
- **Benchmark performance** - Track the overhead of dimensional checking
- **Generate badges** - Show dimensional safety status in your README

## Quick Start

### 1. Setup Action

The easiest way to get started is to use the `setup-dimtensor` composite action:

```yaml
- name: Setup dimtensor
  uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
  with:
    python-version: '3.11'
    install-extras: 'dev,torch'
```

This action:
- Sets up Python with the specified version
- Caches pip dependencies for faster builds
- Installs dimtensor with your chosen extras
- Verifies NumPy compatibility (must be < 2.0)

### 2. Dimensional Linting

Add dimensional linting to your pull requests:

```yaml
name: Dimensional Lint

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
      - run: python -m dimtensor lint src/
```

The linter will:
- Analyze your Python code for dimensional inconsistencies
- Detect mismatched units in operations (e.g., adding velocity + acceleration)
- Provide suggestions for fixing issues
- Fail the build if dimensional errors are found

### 3. Unit Tests

Run your tests with dimtensor:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
        with:
          python-version: ${{ matrix.python-version }}
          install-extras: 'dev'
      - run: pytest --cov --cov-report=xml
```

### 4. Performance Benchmarks

Track the performance overhead of dimensional checking:

```yaml
name: Benchmarks

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
        with:
          install-extras: 'dev,benchmark'
      - run: python -m pytest tests/ --benchmark-only
```

## Reusable Workflows

dimtensor provides three reusable workflows you can reference in your own repository.

### Dimensional Linting Workflow

Copy `.github/workflows/dimtensor-lint.yml` to your repository to:
- Run dimensional linting on every PR
- Comment results on pull requests
- Generate a "dimensional linting" badge
- Upload lint results as artifacts

Features:
- JSON and text output formats
- Configurable strict mode
- Automatic PR comments with issue summaries
- Badge generation for README

### Testing Workflow

Copy `.github/workflows/dimtensor-test.yml` to your repository to:
- Test on Python 3.10, 3.11, 3.12
- Test on Linux, macOS, and Windows
- Verify dimensional safety
- Run type checking with mypy
- Upload coverage to Codecov

Features:
- Matrix testing across Python versions and OS
- Minimal installation testing
- Type safety verification
- Coverage reporting

### Benchmark Workflow

Copy `.github/workflows/dimtensor-benchmark.yml` to your repository to:
- Run performance benchmarks on PRs
- Compare against baseline
- Check for performance regressions
- Generate benchmark reports

Features:
- Automatic baseline comparison
- Regression detection (fails if overhead > 5x)
- Performance badge generation
- Historical benchmark tracking

## Configuration Options

### Setup Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `python-version` | Python version to install | `'3.11'` |
| `install-extras` | Comma-separated extras (e.g., `'dev,torch,jax'`) | `'dev'` |
| `cache-key-suffix` | Additional cache key suffix | `''` |

### Available Extras

- `dev` - Development tools (pytest, mypy, ruff)
- `torch` - PyTorch integration
- `jax` - JAX integration
- `benchmark` - Benchmarking tools
- `all` - All optional dependencies

## Example Workflows

See [`examples/github-workflow-example.yml`](../../examples/github-workflow-example.yml) for a complete example that combines all three workflows.

## Dimensional Linting Details

### What Does the Linter Check?

The dimensional linter performs static analysis on your Python code to detect:

1. **Dimension Mismatches in Addition/Subtraction**
   ```python
   velocity = 10  # Inferred as L·T⁻¹
   acceleration = 5  # Inferred as L·T⁻²
   result = velocity + acceleration  # ⚠️ WARNING: dimension mismatch
   ```

2. **Variable Dimension Inference**
   ```python
   distance = 100  # Inferred as L (length)
   time = 5  # Inferred as T (time)
   speed = distance / time  # Inferred as L·T⁻¹ (velocity)
   ```

3. **Suggestions for Explicit Units**
   ```python
   # Instead of:
   x = 10

   # Use:
   x = DimArray(10, units.m)
   ```

### Linter Options

```bash
# Basic usage
python -m dimtensor lint src/

# JSON output for CI
python -m dimtensor lint src/ --format json

# Strict mode (report all inferences)
python -m dimtensor lint src/ --strict

# Non-recursive
python -m dimtensor lint src/ --no-recursive
```

### Exit Codes

- `0` - No issues found
- `1` - Warnings or errors found

### Lint Result Format

JSON output structure:
```json
[
  {
    "file": "physics.py",
    "line": 42,
    "column": 4,
    "severity": "warning",
    "code": "002",
    "message": "Potential dimension mismatch: L·T⁻¹ + L·T⁻²",
    "context": "velocity + acceleration",
    "suggestion": "Cannot add/subtract L·T⁻¹ and L·T⁻². Check your units."
  }
]
```

## Benchmarking Details

### Quick Benchmark

For CI, use the quick benchmark:

```python
from dimtensor.benchmarks import quick_benchmark

results = quick_benchmark()
# Returns: {'creation': 2.1, 'addition': 1.8, ...}
```

### Full Benchmark Suite

For detailed analysis:

```python
from dimtensor.benchmarks import benchmark_suite, print_results

results = benchmark_suite(
    sizes=[1000, 10000, 100000],
    iterations=1000
)
print_results(results)
```

### Performance Thresholds

The benchmark workflow fails if:
- Max overhead > 5.0x
- Average overhead > 3.0x (warning only)

These thresholds are configurable in the workflow file.

## Badges

Add dimensional safety badges to your README:

### Linting Badge

```markdown
![Dimensional Linting](https://img.shields.io/badge/dimensional%20linting-passing-success)
```

### Test Badge

```markdown
![Tests](https://img.shields.io/badge/tests-passing-success)
```

### Performance Badge

```markdown
![Overhead](https://img.shields.io/badge/overhead-2.1x-success)
```

## Best Practices

### 1. Run Linting on Every PR

Catch dimensional errors early:

```yaml
on:
  pull_request:
    paths:
      - '**.py'
```

### 2. Test on Multiple Python Versions

dimtensor supports Python 3.10, 3.11, and 3.12:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### 3. Cache Dependencies

The setup action automatically caches pip dependencies for faster builds.

### 4. Use Minimal Installation for Core Tests

Test that your code works with just the core dimtensor:

```yaml
- uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
  with:
    install-extras: 'dev'  # No torch, jax, etc.
```

### 5. Monitor Performance Over Time

Store benchmark results and compare:

```yaml
- uses: actions/cache@v4
  with:
    path: benchmark-results.json
    key: benchmark-${{ github.sha }}
```

## Troubleshooting

### NumPy Version Issues

If you see errors about NumPy 2.x:

```yaml
- name: Fix NumPy version
  run: pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

The setup action automatically verifies NumPy < 2.0.

### Import Errors in Tests

Make sure to install with extras:

```yaml
install-extras: 'dev,torch,jax'  # Include what you need
```

### Linter False Positives

Use type hints to help the linter:

```python
from dimtensor import DimArray, units

distance: DimArray = DimArray(100, units.m)
```

Or disable linting for specific lines:

```python
result = velocity + acceleration  # noqa: DIM002
```

## Advanced Usage

### Custom Linting Rules

Create a custom linter workflow with your own rules:

```yaml
- name: Custom dimensional checks
  run: |
    python -m dimtensor lint src/ --format json > results.json
    python scripts/check_custom_rules.py results.json
```

### Parallel Testing

Test multiple configurations in parallel:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    extras: ['torch', 'jax', 'minimal']
```

### Benchmark Comparison

Compare performance across branches:

```yaml
- name: Compare with main
  run: |
    git checkout main
    python -m dimtensor.benchmarks > baseline.txt
    git checkout ${{ github.head_ref }}
    python -m dimtensor.benchmarks > pr.txt
    diff baseline.txt pr.txt
```

## Support

For issues with CI/CD integration:
- Check the [troubleshooting guide](../troubleshooting/index.md)
- Open an issue on [GitHub](https://github.com/marcoloco23/dimtensor/issues)
- Review the [example workflow](../../examples/github-workflow-example.yml)

## Next Steps

- Read the [testing guide](testing.md)
- Learn about [dimensional inference](dimensional-inference.md)
- Explore [performance optimization](../performance/optimization.md)
