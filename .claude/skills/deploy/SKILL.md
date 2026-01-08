---
name: deploy
description: Deploy dimtensor to PyPI. Use when releasing a new version.
allowed-tools: Read, Edit, Bash
---

# Deploy Skill

Steps to deploy a new dimtensor version to PyPI.

## Pre-Deploy Checklist

- [ ] All tests pass: `pytest`
- [ ] mypy passes: `mypy src/dimtensor --ignore-missing-imports`
- [ ] CHANGELOG.md updated with version changes
- [ ] README.md updated if needed

## Version Update

Update version in BOTH locations:

1. `pyproject.toml` (line ~7):
```toml
version = "X.Y.Z"
```

2. `src/dimtensor/__init__.py` (line ~35):
```python
__version__ = "X.Y.Z"
```

## Deploy Commands

```bash
# Working directory
cd "/Users/marcsperzel/Local Documents/Projects/Packages/dimtensor"

# Clean previous builds
rm -rf dist/ build/

# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

## Post-Deploy

1. Update CONTINUITY.md:
   - Mark deploy task as DONE
   - Add session log entry with PyPI URL
   - Update CURRENT STATE version

2. Git commit:
```bash
git add -A
git commit -m "Release vX.Y.Z: Description"
git push origin main
```

3. Verify on PyPI:
   - https://pypi.org/project/dimtensor/

## Version Numbering

- **X.Y.Z** (SemVer)
- **X**: Major breaking changes
- **Y**: New features (backward compatible)
- **Z**: Bug fixes, patches

## Troubleshooting

**twine upload fails**: Check ~/.pypirc credentials
**Build fails**: Ensure build dependencies: `pip install build twine`
