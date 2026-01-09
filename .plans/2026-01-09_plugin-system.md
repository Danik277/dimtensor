# Plan: Plugin System for Custom Units

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a plugin system that allows users to create, distribute, and share custom unit definitions as PyPI packages, enabling a community ecosystem of domain-specific unit collections (e.g., dimtensor-units-nuclear, dimtensor-units-geophysics).

---

## Background

dimtensor currently has three built-in domain modules (astronomy, chemistry, engineering) in `src/dimtensor/domains/`. As the library grows, we need:

1. **Extensibility**: Users should be able to create custom unit collections without modifying dimtensor core
2. **Shareability**: Unit collections should be distributable as standalone PyPI packages
3. **Discoverability**: Installed plugins should be automatically discoverable
4. **Safety**: Plugins should not introduce security vulnerabilities or naming conflicts

This aligns with v4.0.0 theme: "Platform Maturity - Ecosystem and community"

---

## Approach

### Option A: Entry Points (setuptools/importlib.metadata)

**Description**: Use Python's standard entry points mechanism. Plugin packages declare entry points in their `pyproject.toml`:

```toml
[project.entry-points."dimtensor.plugins"]
nuclear = "dimtensor_units_nuclear:plugin"
```

**Pros**:
- Standard Python mechanism (used by pytest, Flask, pluggy)
- Automatic discovery via `importlib.metadata.entry_points()`
- No config files to manage
- Works with all modern Python build tools (hatchling, setuptools, poetry)
- Type-safe: can validate plugin structure at load time

**Cons**:
- Requires plugin to be installed (pip install)
- Cannot load plugins from arbitrary directories
- Slightly more boilerplate for plugin authors

### Option B: Config File Discovery

**Description**: Look for config files (YAML/JSON) in well-known locations:
- `~/.dimtensor/plugins/`
- `$PROJECT/.dimtensor/plugins/`
- Environment variable `DIMTENSOR_PLUGIN_PATH`

**Pros**:
- More flexible: can load from any directory
- No installation required for local development
- Can support YAML/JSON/Python formats

**Cons**:
- Non-standard: need to define custom search algorithm
- Path management complexity
- Security risk: arbitrary file system access
- Harder to version and distribute

### Option C: Namespace Packages (PEP 420)

**Description**: Use namespace package `dimtensor_plugins.*`:
- `dimtensor_plugins.nuclear`
- `dimtensor_plugins.geophysics`

**Pros**:
- Automatic merging of packages from different sources
- Simple import mechanism

**Cons**:
- Less explicit control over plugin loading
- Harder to implement lazy loading
- No central registry for validation
- Namespace collision risks

### Decision: **Option A (Entry Points)** + Optional Config File Support

**Rationale**:
1. Entry points are the Python standard for plugin systems
2. Integrates seamlessly with PyPI distribution
3. Provides type safety and validation hooks
4. Can add config file support later for local development (Option B as secondary)

**Hybrid approach**:
- Primary: Entry points for installed packages
- Secondary: Config file discovery for local development (v4.1.0+)

---

## Implementation Steps

### Phase 1: Core Plugin Infrastructure

1. [ ] Create `src/dimtensor/plugins/__init__.py` module
2. [ ] Implement `PluginMetadata` dataclass:
   - name: str
   - version: str
   - author: str
   - units: dict[str, Unit]
   - description: str
3. [ ] Implement `PluginRegistry` class:
   - `discover_plugins()` - scan entry points
   - `load_plugin(name: str)` - load specific plugin
   - `list_plugins()` - return available plugins
   - `get_unit(plugin_name: str, unit_name: str)` - get unit from plugin
4. [ ] Add global registry singleton: `_global_registry`

### Phase 2: Plugin Loading and Validation

5. [ ] Implement entry point scanner using `importlib.metadata`
6. [ ] Add plugin validation:
   - Check required attributes (name, version, units)
   - Validate Unit objects are properly formed
   - Check for dimension consistency
7. [ ] Add error handling for malformed plugins
8. [ ] Implement lazy loading (only import when requested)

### Phase 3: Conflict Resolution

9. [ ] Design naming scheme: `plugin_name.unit_name` (e.g., `nuclear.MeV`)
10. [ ] Add conflict detection:
    - Detect duplicate plugin names
    - Detect duplicate unit symbols within plugin
11. [ ] Implement resolution strategies:
    - Default: first-loaded wins, warn on collision
    - Strict mode: error on collision
    - Explicit: user chooses via `prefer_plugin(name)`

### Phase 4: API Design

12. [ ] Add convenience functions:
    - `import_plugin(name: str)` - import all units from plugin
    - `import_unit(plugin: str, unit: str)` - import specific unit
    - `plugin_info(name: str)` - show plugin metadata
13. [ ] Add CLI commands:
    - `dimtensor plugins list` - show installed plugins
    - `dimtensor plugins info <name>` - show plugin details
    - `dimtensor plugins validate <path>` - validate plugin structure

### Phase 5: Security and Sandboxing

14. [ ] Document security model:
    - Plugins execute arbitrary Python code
    - Only load from trusted sources (PyPI, known authors)
    - No automatic execution on import
15. [ ] Add optional verification:
    - Check plugin source (pypi.org vs local)
    - Warn if loading from untrusted source
    - Support requirements.txt constraints
16. [ ] Consider future: plugin signatures, checksums (v5.0+)

### Phase 6: Documentation and Examples

17. [ ] Create plugin template repository: `dimtensor-plugin-template`
18. [ ] Write plugin author guide:
    - Package structure
    - Unit definition format
    - Testing strategy
    - Publishing to PyPI
19. [ ] Create example plugin: `dimtensor-units-nuclear`
20. [ ] Update main docs with plugin usage

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/plugins/__init__.py` | Create plugin system module |
| `src/dimtensor/plugins/registry.py` | PluginRegistry class |
| `src/dimtensor/plugins/metadata.py` | PluginMetadata dataclass |
| `src/dimtensor/plugins/loader.py` | Entry point scanning and loading |
| `src/dimtensor/plugins/validation.py` | Plugin validation logic |
| `src/dimtensor/__init__.py` | Export plugin functions |
| `src/dimtensor/__main__.py` | Add plugin CLI commands |
| `pyproject.toml` | Define entry point group `dimtensor.plugins` |
| `docs/guide/plugins.md` | User guide for using plugins |
| `docs/contributing/plugin-development.md` | Plugin author guide |
| `tests/test_plugins.py` | Plugin system tests |
| `tests/fixtures/sample_plugin.py` | Mock plugin for testing |

**New files to create**:
- Template repository: `dimtensor-plugin-template/` (separate repo)
- Example plugin: `dimtensor-units-nuclear/` (separate package)

---

## Plugin Package Structure

Example: `dimtensor-units-nuclear`

```
dimtensor-units-nuclear/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── dimtensor_units_nuclear/
│       ├── __init__.py
│       └── units.py
└── tests/
    └── test_units.py
```

**pyproject.toml**:
```toml
[project]
name = "dimtensor-units-nuclear"
version = "0.1.0"
description = "Nuclear physics units for dimtensor"
dependencies = ["dimtensor>=4.0.0"]

[project.entry-points."dimtensor.plugins"]
nuclear = "dimtensor_units_nuclear:plugin"
```

**`__init__.py`**:
```python
from dimtensor import Unit, Dimension
from dimtensor.plugins import PluginMetadata

# Define units
MeV = Unit("MeV", Dimension(mass=1, length=2, time=-2), 1.602176634e-13)
barn = Unit("barn", Dimension(length=2), 1e-28)
becquerel = Unit("Bq", Dimension(time=-1), 1.0)

# Plugin metadata
plugin = PluginMetadata(
    name="nuclear",
    version="0.1.0",
    author="Community",
    description="Nuclear physics units",
    units={
        "MeV": MeV,
        "barn": barn,
        "becquerel": becquerel,
    }
)

__all__ = ["MeV", "barn", "becquerel", "plugin"]
```

---

## Unit Definition Formats

### Option 1: Python Only (Recommended for v4.0.0)

**Pros**:
- Type-safe
- Full Python expressiveness
- Easy to validate
- Can include computed units

**Cons**:
- Requires Python knowledge
- More verbose than declarative formats

### Option 2: YAML + Python Hybrid (v4.1.0+)

Allow plugins to define units in YAML for simple cases:

```yaml
# units.yaml
name: nuclear
version: 0.1.0
units:
  MeV:
    symbol: MeV
    dimension: {mass: 1, length: 2, time: -2}
    scale: 1.602176634e-13
    description: "Mega-electronvolt"
```

Then generate Python code or load directly.

**Decision**: Start with Python only (simpler, more flexible). Add YAML support in v4.1.0 if there's demand.

---

## Testing Strategy

### Unit Tests
- [ ] Test plugin discovery (mock entry points)
- [ ] Test plugin loading (valid and invalid)
- [ ] Test conflict detection and resolution
- [ ] Test lazy loading (imports not triggered until use)
- [ ] Test registry singleton behavior
- [ ] Test CLI commands (list, info, validate)

### Integration Tests
- [ ] Create sample plugin package in `tests/fixtures/`
- [ ] Install sample plugin in temp venv, verify discovery
- [ ] Test plugin usage in DimArray operations
- [ ] Test multiple plugins loaded simultaneously

### Manual Verification
- [ ] Create `dimtensor-units-nuclear` example package
- [ ] Publish to TestPyPI
- [ ] Install and test discovery
- [ ] Verify CLI commands work

---

## Risks / Edge Cases

### Risk 1: Plugin Namespace Collisions
**Scenario**: Two plugins define units with same symbol (e.g., both define "bar")

**Mitigation**:
- Use qualified names: `nuclear.MeV` vs `chemistry.molar`
- Warn on collision, first-loaded wins
- Provide explicit `prefer_plugin()` function
- Document best practices: use unique symbols

### Risk 2: Security - Arbitrary Code Execution
**Scenario**: Malicious plugin executes harmful code on import

**Mitigation**:
- Document that plugins are trusted code
- Warn when loading from non-PyPI sources
- Future: plugin signing/verification (v5.0+)
- Recommend: review plugin source before installing
- No automatic execution - explicit load required

### Risk 3: Version Incompatibility
**Scenario**: Plugin built for dimtensor 4.0.0 breaks with 5.0.0

**Mitigation**:
- Enforce version constraints in plugin dependencies
- Use semantic versioning
- Deprecation warnings for API changes
- Plugin registry can check compatibility

### Risk 4: Performance - Slow Plugin Discovery
**Scenario**: Scanning 100 plugins on every import slows startup

**Mitigation**:
- Lazy loading: only scan when `plugins.list()` called
- Cache discovered plugins in memory
- Don't auto-load all plugins, load on-demand
- Future: persistent cache (v4.1.0+)

### Edge Case 1: Empty Plugin
**Handling**: Validate plugin has at least one unit, warn if empty

### Edge Case 2: Plugin with Invalid Units
**Handling**: Skip invalid units, log warnings, load valid ones

### Edge Case 3: Plugin Import Fails
**Handling**: Catch ImportError, log error, continue discovering others

### Edge Case 4: Circular Dependencies
**Handling**: Plugin depends on another plugin - document limitation, no automatic resolution

---

## Definition of Done

- [ ] All implementation steps complete (Phase 1-6)
- [ ] PluginRegistry with discover/load/list functionality
- [ ] Entry point scanning with importlib.metadata
- [ ] Conflict detection and resolution
- [ ] CLI commands: list, info, validate
- [ ] Comprehensive tests (unit + integration)
- [ ] Plugin author guide documentation
- [ ] Example plugin: dimtensor-units-nuclear
- [ ] Plugin template repository created
- [ ] Tests pass (target: 30+ new tests)
- [ ] CONTINUITY.md updated

---

## Community Submission Workflow

### For Plugin Authors

1. **Create Plugin Package**
   - Use `dimtensor-plugin-template` as starting point
   - Follow naming: `dimtensor-units-<domain>`
   - Include comprehensive unit definitions
   - Add tests for all units

2. **Publish to PyPI**
   - Build: `python -m build`
   - Upload: `twine upload dist/*`
   - Use semantic versioning

3. **Optional: Submit to Registry**
   - Create PR to `dimtensor-plugin-registry` repo
   - Add plugin metadata to `registry.yaml`
   - Include: name, PyPI package, description, author
   - Maintainers review and approve

### For Users

1. **Install Plugin**
   ```bash
   pip install dimtensor-units-nuclear
   ```

2. **Use Plugin**
   ```python
   from dimtensor.plugins import load_plugin
   nuclear = load_plugin("nuclear")
   from dimtensor_units_nuclear import MeV
   energy = DimArray([1.0], MeV)
   ```

3. **Discover Plugins**
   ```bash
   dimtensor plugins list
   dimtensor plugins info nuclear
   ```

### Central Registry (Optional, v4.1.0+)

Create `dimtensor-plugin-registry` repository:
- `registry.yaml` with curated plugin list
- Automated testing of registered plugins
- Community moderation
- Discovery API: `dimtensor.plugins.discover_online()`

**Decision**: Start without central registry, add in v4.1.0 if community grows.

---

## API Design Summary

```python
# Plugin discovery and loading
from dimtensor import plugins

# List all available plugins
plugins.list_plugins()  # ["nuclear", "geophysics", ...]

# Load plugin (imports all units into namespace)
nuclear = plugins.load_plugin("nuclear")
from dimtensor_units_nuclear import MeV, barn

# Or get specific unit
MeV = plugins.get_unit("nuclear", "MeV")

# Plugin info
info = plugins.plugin_info("nuclear")
# PluginMetadata(name="nuclear", version="0.1.0", ...)

# Conflict resolution
plugins.set_conflict_strategy("warn")  # "warn", "error", or "silent"
plugins.prefer_plugin("nuclear")  # When conflicts, prefer nuclear

# Validation (for plugin authors)
plugins.validate_plugin("./my-plugin")  # Check structure
```

**CLI**:
```bash
# List installed plugins
dimtensor plugins list

# Show plugin details
dimtensor plugins info nuclear

# Validate plugin during development
dimtensor plugins validate ./dimtensor-units-nuclear

# Search for plugins (v4.1.0+)
dimtensor plugins search geophysics
```

---

## Notes / Log

**2026-01-09 17:00** - Plan created by planner agent

**Key Design Decisions**:
1. Entry points as primary mechanism (standard, PyPI-friendly)
2. Python-only unit definitions for v4.0.0 (simplicity)
3. Qualified names for conflict resolution (`plugin.unit`)
4. Lazy loading for performance
5. Documentation over enforcement for security
6. No central registry in v4.0.0 (wait for community growth)

**Future Enhancements** (v4.1.0+):
- Config file discovery for local development
- YAML unit definition format
- Plugin dependency resolution
- Central plugin registry
- Plugin signing/verification
- Persistent discovery cache

**Complexity**: HIGH
- Involves setuptools/importlib internals
- Security considerations
- Conflict resolution strategies
- CLI integration
- Documentation and examples for community

**Estimated Time**: 2-3 implementation sessions
- Session 1: Core plugin infrastructure (Phase 1-2)
- Session 2: Conflict resolution, API, CLI (Phase 3-4)
- Session 3: Security, docs, example plugin (Phase 5-6)

---
