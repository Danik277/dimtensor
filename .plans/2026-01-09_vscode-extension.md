# Plan: VS Code Extension Skeleton

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a VS Code extension that provides real-time dimensional linting for Python files using dimtensor's existing CLI linter. The extension should display red squiggles for unit mismatches and dimensional errors as users type.

---

## Background

dimtensor v4.0.0 focuses on "Platform Maturity - Ecosystem and community". A VS Code extension is a critical ecosystem component that brings dimensional safety directly into developers' IDEs. The extension will integrate with dimtensor's existing CLI linter (`dimtensor lint`) which already provides:
- AST-based dimensional analysis
- JSON output format with line/column information
- Three severity levels (ERROR, WARNING, INFO)
- Detection of dimension mismatches in binary operations

---

## Approach

### Option A: Direct Diagnostics Provider
- Description: TypeScript extension that spawns `dimtensor lint` as subprocess and parses JSON output
- Pros:
  - Simpler architecture (single TypeScript file)
  - No Python LSP server needed
  - Faster initial development
  - Direct control over execution
- Cons:
  - Less performant (subprocess overhead on every save/change)
  - Limited to diagnostics only (no hover, code actions, etc.)
  - Doesn't follow VS Code LSP best practices
  - Harder to add advanced features later

### Option B: Language Server Protocol (LSP)
- Description: Python-based LSP server + TypeScript extension client
- Pros:
  - Follows VS Code best practices (like ruff-vscode, pylint)
  - Better performance (persistent server process)
  - Extensible for future features (hover info, quick fixes, code actions)
  - Standard architecture that VS Code optimizes for
  - Can leverage pygls library for LSP implementation
- Cons:
  - More complex initial setup
  - Requires Python LSP server implementation
  - Two codebases to maintain (Python + TypeScript)

### Decision: **Option B (LSP)** with phased approach

**Phase 1**: Minimal LSP server with diagnostics only (this plan)
**Phase 2**: Add hover info, code actions, quick fixes (future)

This approach provides a scalable foundation while keeping initial complexity manageable. We'll use pygls (Python Generic Language Server) for LSP implementation, following the pattern used by ruff-lsp and other modern Python tooling.

---

## Implementation Steps

### Phase 1: Project Setup
1. [ ] Create `vscode-dimtensor/` directory at repo root
2. [ ] Initialize Node.js project with `npm init`
3. [ ] Set up TypeScript configuration (`tsconfig.json`)
4. [ ] Configure webpack for bundling (`webpack.config.js`)
5. [ ] Add `.vscodeignore` to exclude dev files from package
6. [ ] Create `.gitignore` for Node.js artifacts

### Phase 2: Extension Manifest
7. [ ] Create `package.json` with extension metadata:
   - Publisher, name, version, icon
   - Activation events: `onLanguage:python`
   - Extension categories: `["Linters", "Programming Languages"]`
   - Configuration schema for settings
   - Command palette contributions
8. [ ] Define configuration options:
   - `dimtensor.lint.enabled`: Enable/disable linting
   - `dimtensor.lint.strict`: Enable strict mode
   - `dimtensor.lint.executable`: Path to Python/dimtensor
   - `dimtensor.lint.args`: Additional CLI arguments
9. [ ] Add icon and README assets

### Phase 3: TypeScript Extension Client
10. [ ] Install dependencies:
    - `vscode` (VS Code API)
    - `vscode-languageclient` (LSP client)
    - `webpack`, `webpack-cli`, `ts-loader` (bundling)
11. [ ] Create `src/extension.ts`:
    - `activate()` function to start LSP client
    - `deactivate()` function to stop server
    - Configuration change handlers
12. [ ] Implement LSP client configuration:
    - Document selector: `{ language: 'python', scheme: 'file' }`
    - Server executable: Python with LSP server script
    - Server arguments and environment
13. [ ] Add error handling and logging

### Phase 4: Python LSP Server
14. [ ] Create `bundled/tool/dimtensor_lsp/` directory
15. [ ] Install pygls: `pip install pygls`
16. [ ] Create `server.py` with pygls setup:
    - Language server class extending `LanguageServer`
    - Text document sync handlers
    - Diagnostic provider registration
17. [ ] Implement `textDocument/didOpen` handler:
    - Extract file path from URI
    - Call `dimtensor.cli.lint.lint_file()`
    - Convert `LintResult` to LSP `Diagnostic`
18. [ ] Implement `textDocument/didChange` handler:
    - Debounce to avoid excessive linting
    - Re-run linter on document changes
19. [ ] Implement `textDocument/didSave` handler:
    - Immediate linting on save
20. [ ] Map dimtensor severity to LSP severity:
    - `ERROR` → `DiagnosticSeverity.Error`
    - `WARNING` → `DiagnosticSeverity.Warning`
    - `INFO` → `DiagnosticSeverity.Information`

### Phase 5: Integration and Testing
21. [ ] Create test workspace in `test-workspace/`
22. [ ] Add sample Python files with dimensional errors
23. [ ] Test extension in VS Code Extension Development Host:
    - Open Python file with dimension mismatch
    - Verify red squiggles appear
    - Check PROBLEMS panel for diagnostics
24. [ ] Test configuration changes:
    - Disable linting via settings
    - Enable strict mode
    - Verify custom executable path
25. [ ] Test edge cases:
    - Files with syntax errors
    - Large files (performance)
    - Rapid typing (debouncing)

### Phase 6: Build and Package
26. [ ] Create build script in `package.json`:
    - `"compile"`: TypeScript compilation
    - `"watch"`: Watch mode for development
    - `"package"`: Create `.vsix` with vsce
27. [ ] Install vsce: `npm install -g @vscode/vsce`
28. [ ] Test packaging: `vsce package`
29. [ ] Verify `.vsix` file size and contents
30. [ ] Document installation instructions

### Phase 7: Documentation
31. [ ] Create `vscode-dimtensor/README.md`:
    - Feature overview
    - Installation instructions
    - Configuration options
    - Screenshots
32. [ ] Create `CHANGELOG.md`
33. [ ] Add usage examples
34. [ ] Document marketplace publishing process

---

## Files to Modify

| File | Change |
|------|--------|
| `vscode-dimtensor/package.json` | Extension manifest with metadata, activation events, settings schema |
| `vscode-dimtensor/tsconfig.json` | TypeScript compiler configuration |
| `vscode-dimtensor/webpack.config.js` | Webpack bundling configuration |
| `vscode-dimtensor/src/extension.ts` | Extension entry point and LSP client setup |
| `vscode-dimtensor/bundled/tool/dimtensor_lsp/server.py` | Python LSP server implementation |
| `vscode-dimtensor/bundled/tool/dimtensor_lsp/requirements.txt` | Python dependencies (pygls, dimtensor) |
| `vscode-dimtensor/.vscodeignore` | Files to exclude from package |
| `vscode-dimtensor/.gitignore` | Node.js and build artifacts |
| `vscode-dimtensor/README.md` | User documentation |
| `vscode-dimtensor/CHANGELOG.md` | Version history |
| `vscode-dimtensor/test-workspace/*.py` | Sample files for testing |

---

## Testing Strategy

### Development Testing
- [ ] Use VS Code Extension Development Host (`F5` in VS Code)
- [ ] Test with sample files containing known dimension errors
- [ ] Verify diagnostics appear in real-time
- [ ] Check PROBLEMS panel for correct file paths and line numbers
- [ ] Test configuration changes without restarting

### Integration Testing
- [ ] Test with real dimtensor projects
- [ ] Verify compatibility with Python extension
- [ ] Test Jupyter notebook support (future)
- [ ] Performance test with large files (1000+ lines)

### Manual Verification
- [ ] Install packaged `.vsix` in clean VS Code instance
- [ ] Verify icon and description in Extensions panel
- [ ] Check that settings appear in Settings UI
- [ ] Test on different operating systems (Windows, macOS, Linux)

### Automated Testing (Future)
- [ ] Unit tests for TypeScript extension client
- [ ] Unit tests for Python LSP server
- [ ] Integration tests with VS Code test runner
- [ ] CI/CD with GitHub Actions

---

## Risks / Edge Cases

### Risk 1: Python Environment Discovery
**Problem**: Extension needs to find Python interpreter with dimtensor installed
**Mitigation**:
- Use VS Code Python extension's API to get active interpreter
- Fallback to `python3`, `python` in PATH
- Allow user to specify custom path in settings
- Show clear error message if dimtensor not found

### Risk 2: Performance on Large Files
**Problem**: Linting large files may cause UI lag
**Mitigation**:
- Run LSP server in separate process (prevents blocking)
- Implement debouncing (500ms delay after typing stops)
- Only lint on save for files > 1000 lines
- Add setting to disable real-time linting

### Risk 3: Subprocess Management
**Problem**: LSP server process may not terminate cleanly
**Mitigation**:
- Use VS Code's `LanguageClient` which handles lifecycle
- Implement proper cleanup in `deactivate()`
- Add timeout for server startup
- Log server stdout/stderr for debugging

### Risk 4: Version Compatibility
**Problem**: Extension may break with old dimtensor versions
**Mitigation**:
- Check dimtensor version on server startup
- Require minimum version (3.6.0+) in `package.json`
- Show warning if version mismatch detected
- Document required dimtensor version in README

### Risk 5: Windows Path Handling
**Problem**: Windows paths with backslashes may break URI conversion
**Mitigation**:
- Use VS Code's `Uri.file()` for path conversion
- Test on Windows in CI
- Follow LSP spec for file URIs (`file:///C:/...`)

### Edge Case 1: Syntax Errors
**Handling**: Linter already returns ERROR diagnostic for syntax errors; extension will display them

### Edge Case 2: Non-Python Files
**Handling**: Document selector limits to `language: 'python'`; extension won't activate for other files

### Edge Case 3: Unsaved Files
**Handling**: Write buffer content to temp file, lint it, clean up (follow ruff-vscode pattern)

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Extension activates on Python files
- [ ] Diagnostics appear for dimension errors
- [ ] Configuration options work correctly
- [ ] README and documentation complete
- [ ] Can build `.vsix` package with `vsce package`
- [ ] Tested on sample files with known errors
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-09 - Initial Planning**

Key architecture decisions:
- LSP-based approach for scalability
- pygls library for Python LSP server
- Follow ruff-vscode patterns for extension structure
- Phase 1 focuses on diagnostics only
- Future phases will add hover, code actions, quick fixes

Dependencies:
- TypeScript: vscode, vscode-languageclient, webpack
- Python: pygls, dimtensor
- Build tools: npm, webpack, vsce

Estimated complexity: **Medium**
- LSP adds complexity but provides better foundation
- dimtensor linter already works; just need integration
- Reference implementations available (ruff-vscode, pylint)

Next steps:
- Spawn implementer agent to execute Phase 1-3 (project setup + manifest)
- Create basic TypeScript extension skeleton
- Implement minimal LSP server for diagnostics

---

## Future Enhancements (Out of Scope)

These features are deferred to future versions:

### Phase 2: Enhanced Diagnostics
- Code actions for quick fixes (e.g., "Convert to compatible units")
- Hover info showing variable dimensions
- Signature help for dimtensor functions

### Phase 3: Advanced Features
- "Fix all" command to auto-correct dimension errors where possible
- Inline unit suggestions based on variable names
- Integration with dimtensor inference engine
- Visualization of dimension flow through code

### Phase 4: Marketplace Publishing
- Publisher account setup
- Marketplace listing with screenshots
- CI/CD for automated releases
- Telemetry and error reporting

---
