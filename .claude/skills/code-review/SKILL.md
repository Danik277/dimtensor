---
name: code-review
description: Review dimtensor code for correctness, dimensional safety, and adherence to patterns. Use when reviewing new code or during consolidation phases.
allowed-tools: Read, Grep, Glob
---

# Code Review Skill

Review checklist for dimtensor code.

## Review Checklist

### Dimensional Correctness (CRITICAL)
- [ ] Addition/subtraction requires same dimension
- [ ] Multiplication/division correctly combines dimensions
- [ ] Power operations handle dimension exponents correctly
- [ ] Transcendental functions (sin, exp, log) require dimensionless input
- [ ] sqrt halves dimension exponents

### Edge Cases
- [ ] Empty arrays handled
- [ ] Scalar values handled
- [ ] Zero values (division by zero)
- [ ] None/null inputs
- [ ] inf/nan values
- [ ] Very large/small scale factors

### Error Handling
- [ ] Raises DimensionError for incompatible operations
- [ ] Raises UnitConversionError for impossible conversions
- [ ] Error messages are helpful (show dimensions involved)

### Code Patterns
- [ ] Uses `DimArray._from_data_and_unit(data, unit)` internally
- [ ] Operations return new instances (immutable)
- [ ] Follows patterns from core/dimarray.py
- [ ] Type hints on public functions
- [ ] Docstrings on public functions

### Scale Factors
- [ ] Uses authoritative values (CODATA, IAU, IUPAC)
- [ ] Scientific notation for large/small values
- [ ] Source documented in comments

### Test Coverage
- [ ] Happy path tests exist
- [ ] Edge case tests exist
- [ ] Error case tests exist
- [ ] Conversion accuracy tests with tolerances

## Review Output Format

```markdown
### File: `path/to/file.py`
**Reviewed by**: agent
**Status**: APPROVED / ISSUES FOUND / NEEDS CHANGES

**Issues Found**:
1. [CRITICAL] Description...
2. [IMPORTANT] Description...
3. [MINOR] Description...

**Recommendations**:
- Suggestion 1
- Suggestion 2
```

## After Review

Update CONTINUITY.md:
- Add review findings to CODE REVIEW FINDINGS section
- Mark review task as DONE in TASK QUEUE
- Add session log entry
