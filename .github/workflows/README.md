# GitHub Actions Workflow Templates

This directory contains reusable GitHub Actions workflow templates for projects using dimtensor.

## Available Workflows

### `dimtensor-lint.yml` - Dimensional Linting

Automatically lint your Python code for dimensional consistency issues.

**Features:**
- Runs on every pull request
- Detects dimension mismatches (e.g., adding velocity + acceleration)
- Comments results on PRs
- Generates lint badges
- Uploads results as artifacts

**Usage:**
Copy this file to your `.github/workflows/` directory.

**Triggers:**
- Pull requests that modify `.py` files
- Manual workflow dispatch

### `dimtensor-test.yml` - Unit Testing

Run comprehensive tests with dimensional safety verification.

**Features:**
- Tests on Python 3.10, 3.11, 3.12
- Tests on Linux, macOS, Windows
- Verifies dimensional safety
- Type checking with mypy
- Coverage reporting to Codecov
- Generates test badges

**Usage:**
Copy this file to your `.github/workflows/` directory.

**Triggers:**
- Push to main or develop branches
- Pull requests to main or develop
- Manual workflow dispatch

### `dimtensor-benchmark.yml` - Performance Benchmarks

Track performance overhead and detect regressions.

**Features:**
- Runs benchmarks on every PR
- Compares with baseline performance
- Fails if overhead > 5x
- Generates performance reports
- Comments results on PRs
- Creates performance badges

**Usage:**
Copy this file to your `.github/workflows/` directory.

**Triggers:**
- Pull requests to main or develop
- Push to main (stores baseline)
- Manual workflow dispatch

## Composite Action

### `setup-dimtensor` - Setup Action

Located in `.github/actions/setup-dimtensor/action.yml`, this composite action simplifies dimtensor installation.

**Inputs:**
- `python-version` (default: `'3.11'`) - Python version to install
- `install-extras` (default: `'dev'`) - Extras to install (comma-separated)
- `cache-key-suffix` (default: `''`) - Additional cache key suffix

**Outputs:**
- `python-version` - Installed Python version
- `cache-hit` - Whether cache was hit

**Example:**
```yaml
- uses: ./.github/actions/setup-dimtensor
  with:
    python-version: '3.11'
    install-extras: 'dev,torch,jax'
```

## Quick Start

### For dimtensor Users

If you're using dimtensor in your project and want to add CI/CD:

1. Copy the desired workflow files to your `.github/workflows/` directory
2. Update the setup action reference to point to dimtensor's repository:
   ```yaml
   uses: marcoloco23/dimtensor/.github/actions/setup-dimtensor@main
   ```
3. Customize triggers, Python versions, and extras as needed

### For dimtensor Development

These workflows are used in the dimtensor repository itself:

1. The workflows reference the local setup action:
   ```yaml
   uses: ./.github/actions/setup-dimtensor
   ```
2. They run on the dimtensor codebase
3. Results are used to verify releases

## Customization

### Changing Python Versions

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### Adding/Removing Extras

```yaml
install-extras: 'dev,torch,jax,all'
```

### Adjusting Performance Thresholds

In `dimtensor-benchmark.yml`:
```python
if max_overhead > 5.0:  # Change this threshold
    print('ERROR: Performance regression')
```

### Customizing Triggers

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
```

## Best Practices

1. **Run linting on every PR** - Catches dimensional errors early
2. **Test on multiple Python versions** - Ensures compatibility
3. **Monitor performance** - Track overhead over time
4. **Use caching** - The setup action caches dependencies automatically
5. **Set up branch protection** - Require lint and tests to pass

## Badge Examples

Add these to your README:

### Linting Status
```markdown
![Dimensional Linting](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/dimtensor-lint.yml/badge.svg)
```

### Test Status
```markdown
![Tests](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/dimtensor-test.yml/badge.svg)
```

### Performance
```markdown
![Benchmarks](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/dimtensor-benchmark.yml/badge.svg)
```

## Troubleshooting

### NumPy Version Issues

The setup action automatically verifies NumPy < 2.0. If you encounter issues:

```yaml
- run: pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### Cache Issues

Clear the cache by changing the `cache-key-suffix`:

```yaml
cache-key-suffix: 'v2'
```

### Import Errors

Ensure you're installing the correct extras:

```yaml
install-extras: 'dev,torch,jax'  # Include what you need
```

## Documentation

For detailed documentation, see:
- [CI/CD Integration Guide](../../docs/guide/ci-cd.md)
- [Complete Example](../../examples/github-workflow-example.yml)
- [dimtensor Documentation](https://marcoloco23.github.io/dimtensor)

## Support

Issues with workflows? Open an issue:
https://github.com/marcoloco23/dimtensor/issues
