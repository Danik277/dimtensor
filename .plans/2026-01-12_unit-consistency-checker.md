# Plan: Static Unit Consistency Checker

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive static analysis tool that checks Python codebases for dimensional consistency issues by analyzing AST, propagating unit information through control flow, and detecting mismatches without executing code.

---

## Background

dimtensor already has a basic dimensional linter (src/dimtensor/cli/lint.py) that:
- Uses AST parsing to analyze Python code
- Infers dimensions from variable names using heuristics
- Detects simple addition/subtraction mismatches
- Integrates with the inference module for name-based dimension guessing

However, it lacks:
- Flow-sensitive analysis (tracking units across assignments)
- Function call analysis (propagating dimensions through function boundaries)
- Type annotation support (reading unit info from type hints)
- Comprehensive reporting with fix suggestions
- Integration points for type checkers like mypy

This plan extends the existing linter into a full static checker.

---

## Approach

### Option A: Extend Existing Linter (In-Tree)
- Description: Build on src/dimtensor/cli/lint.py by adding flow analysis, annotation parsing, and enhanced reporting
- Pros:
  - Leverages existing AST infrastructure
  - Uses existing heuristics from dimtensor.inference
  - Integrated with dimtensor CLI already
  - Can reuse LintResult, severity levels
- Cons:
  - Existing linter is name-based only
  - May need significant refactoring for flow analysis

### Option B: Separate Static Analyzer Tool
- Description: Create new module (dimtensor/analysis/) with dedicated static analysis engine
- Pros:
  - Clean separation of concerns
  - Can use advanced analysis frameworks (e.g., dataflow libraries)
  - Won't disrupt existing linter
- Cons:
  - Duplication of AST parsing logic
  - More complexity in project structure
  - Need new CLI integration

### Option C: Mypy Plugin
- Description: Create a mypy plugin that hooks into type checking
- Pros:
  - Integrates with existing type checking workflow
  - Can read actual type annotations
  - IDE integration via mypy LSP
- Cons:
  - Limited to mypy users
  - Plugin API is complex
  - May not be usable standalone

### Decision: Option A (Extend Existing Linter)

Extend the existing linter with flow analysis capabilities. This leverages existing infrastructure while keeping everything in one place. We'll add:
1. A symbol table for tracking variable dimensions across scopes
2. Flow-sensitive analysis that updates dimensions on assignment
3. Function signature tracking for cross-function analysis
4. Type annotation parsing for explicit unit declarations
5. Enhanced error messages with fix suggestions

The existing heuristic-based inference remains as a fallback for unannotated code.

---

## Implementation Steps

1. [ ] **Refactor DimensionalLinter into analyzer architecture**
   - Extract symbol table management
   - Add scope tracking (global, function, class)
   - Create VariableState dataclass (name, dimension, confidence, provenance)

2. [ ] **Implement type annotation parser**
   - Parse PEP 484 annotations for DimArray[Unit]
   - Support string annotations and forward references
   - Extract dimension info from type annotations
   - Handle Union types, Optional, etc.

3. [ ] **Add flow-sensitive analysis**
   - Track dimension changes through assignments
   - Handle augmented assignments (+=, *=, etc.)
   - Support tuple unpacking
   - Implement phi nodes for conditional branches

4. [ ] **Implement function call analysis**
   - Build function signature database (name -> (params, return_type))
   - Propagate dimensions through function calls
   - Handle built-in functions (sin, exp, sqrt, etc.)
   - Support method calls on DimArray/DimTensor

5. [ ] **Add control flow analysis**
   - Track dimensions through if/else branches
   - Handle loops (for, while)
   - Detect dimension inconsistencies across branches
   - Support with statements and context managers

6. [ ] **Enhance error detection**
   - Check binary operations (already partially done)
   - Check function arguments match expected dimensions
   - Check return type consistency
   - Detect unit conversion issues (incompatible dimensions)
   - Warn on dimensionless operations (sin, exp) with dimensionful args

7. [ ] **Improve error reporting**
   - Add "expected X but got Y" style messages
   - Suggest explicit unit conversions
   - Show dimension propagation path for complex errors
   - Add machine-readable JSON output with structured info

8. [ ] **Add configuration system**
   - Config file (.dimtensor.toml) for project-level settings
   - Disable specific checks
   - Custom dimension inference patterns
   - Strictness levels (permissive, balanced, strict)

9. [ ] **Create comprehensive test suite**
   - Test annotation parsing
   - Test flow analysis (assignments, branches, loops)
   - Test function call propagation
   - Test error detection and messages
   - Test on real physics code examples

10. [ ] **Update CLI integration**
    - Add --flow-analysis flag (default: on)
    - Add --check-annotations flag
    - Add --max-confidence-threshold for inference
    - Support --config flag for custom config files

11. [ ] **Write documentation**
    - User guide for the static checker
    - Examples of annotating code for checking
    - Guide on interpreting error messages
    - Integration guide for CI/CD pipelines

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/cli/lint.py | Major refactor: Add SymbolTable, FlowAnalyzer, AnnotationParser classes |
| src/dimtensor/inference/heuristics.py | Minor: Export additional helper functions for linter |
| src/dimtensor/core/dimensions.py | Minor: Add helper for dimension equality with tolerance |
| tests/test_lint.py | Major: Add comprehensive flow analysis tests |
| tests/test_lint_annotations.py | New: Test type annotation parsing |
| tests/test_lint_flow.py | New: Test control flow analysis |
| src/dimtensor/cli/config.py | New: Configuration file handling |
| docs/guide/static-checking.md | New: User documentation |
| .dimtensor.toml.example | New: Example configuration file |

---

## Testing Strategy

### Unit Tests
- [ ] Test SymbolTable operations (add, lookup, scope push/pop)
- [ ] Test annotation parsing with various type hint formats
- [ ] Test flow analysis on simple assignments
- [ ] Test flow analysis on conditionals
- [ ] Test function call dimension propagation
- [ ] Test error message generation

### Integration Tests
- [ ] Test full file analysis with mixed annotated/unannotated code
- [ ] Test on physics simulation code (mechanics, E&M, thermodynamics)
- [ ] Test on machine learning code with dimensionful tensors
- [ ] Test CLI with various flags and configurations

### Manual Verification
- [ ] Run on dimtensor's own codebase
- [ ] Run on example projects from docs/examples/
- [ ] Verify error messages are clear and actionable
- [ ] Check performance on large files (>1000 LOC)

---

## Risks / Edge Cases

- **Risk 1: False positives from heuristic inference**
  - Mitigation: Use confidence thresholds, allow user to suppress specific warnings, prioritize explicit annotations over inference

- **Risk 2: Complex control flow (nested loops, recursion)**
  - Mitigation: Start with simple flow analysis, add complexity incrementally, set depth limits for analysis

- **Risk 3: Dynamic code (exec, eval, metaprogramming)**
  - Mitigation: Skip analysis of dynamic code, warn user that it can't be checked statically

- **Risk 4: Performance on large codebases**
  - Mitigation: Cache analysis results, parallelize file processing, add profiling to identify bottlenecks

- **Edge Case: Dimension inference ambiguity (e.g., 't' could be time or temperature)**
  - Handling: Report multiple possibilities with confidence scores, prefer explicit annotations

- **Edge Case: NumPy/PyTorch operations on DimArray that don't preserve dimensions**
  - Handling: Whitelist known safe operations, warn on unknown operations

- **Edge Case: Unit systems (SI vs CGS vs imperial)**
  - Handling: Focus on dimensional consistency only, not unit compatibility (that's runtime behavior)

- **Edge Case: Dimensionless ratios (e.g., Mach number = velocity / sound_speed)**
  - Handling: Track dimensionless values separately, ensure they result from compatible dimension divisions

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass with >90% coverage on new code
- [ ] Documentation written and reviewed
- [ ] Example configuration file created
- [ ] CLI integration working with all flags
- [ ] Manual testing on real codebases shows useful results
- [ ] Performance acceptable (<1s for typical file, <10s for large file)
- [ ] CONTINUITY.md updated with completion notes

---

## Notes / Log

**2026-01-12** - Initial plan created
- Researched existing linter infrastructure
- Found inference module with heuristics
- Decided to extend existing linter rather than create new tool
- Key insight: existing AST visitor pattern is solid foundation

---
