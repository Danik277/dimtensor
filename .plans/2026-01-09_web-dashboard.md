# Plan: Web Dashboard for Model Hub

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Build an interactive web dashboard to browse dimtensor's models, datasets, and equations with search/filter capabilities and code generation for easy integration into user projects.

---

## Background

dimtensor v4.0.0 focuses on "Platform Maturity - Ecosystem and community". The library has:
- Model hub (hub/registry.py) with 0+ registered models (extensible)
- Dataset registry (datasets/registry.py) with 10+ built-in datasets (pendulum, burgers, navier_stokes, etc.)
- Equation database (equations/database.py) with 67+ physics equations across 9 domains

Currently, users must programmatically explore these via Python APIs. A web dashboard will:
1. Lower barrier to entry for discovery
2. Provide visual overview of available resources
3. Generate copy-paste code snippets
4. Enable sharing via URL
5. Build community engagement

---

## Approach

### Option A: Streamlit

**Description**: Pure Python web framework designed for data apps

**Pros**:
- Minimal code (~200-300 lines for MVP)
- Native support for dataframes, code blocks, search
- Free hosting on Streamlit Cloud or HF Spaces
- Easy to iterate and deploy
- No frontend skills needed
- Good for scientific/data apps

**Cons**:
- Limited customization (fixed layout patterns)
- No built-in API endpoint support
- Reloads entire page on interaction (can be slow)
- Session state can be tricky

### Option B: Gradio

**Description**: ML-focused web UI framework

**Pros**:
- Similar simplicity to Streamlit
- Native Hugging Face Spaces integration
- Good for model demos
- Can export as API

**Cons**:
- More focused on model I/O than data exploration
- Less flexible for multi-page apps
- Smaller ecosystem than Streamlit

### Option C: FastAPI + Static Frontend

**Description**: API-first approach with Vue/React/Svelte frontend

**Pros**:
- Full control over design
- Native API for programmatic access
- Best performance
- Modern web patterns

**Cons**:
- Requires frontend expertise (~1000+ lines)
- Longer development time (2-3x)
- More complex deployment
- Harder to maintain

### Option D: Dash (Plotly)

**Description**: Python framework for analytical web apps

**Pros**:
- Good for data-heavy visualizations
- Callback-based interactivity
- Plotly integration

**Cons**:
- More verbose than Streamlit
- Steeper learning curve
- Less intuitive for text/table-heavy apps

### Decision: Streamlit (Option A)

**Rationale**:
1. **Speed**: Can ship MVP in 1-2 days vs weeks for FastAPI
2. **Maintainability**: Pure Python, no frontend dependencies
3. **Deployment**: Free Streamlit Cloud hosting with GitHub integration
4. **User base**: Popular in scientific Python community
5. **Iteration**: Easy to add pages/features incrementally
6. **Future-proof**: Can add FastAPI backend later if needed for API access

**Migration path**: Start with Streamlit for v4.0.0, optionally add FastAPI endpoints in v4.0.1+ if community requests programmatic access.

---

## Implementation Steps

### Phase 1: Core Dashboard (MVP)

1. [ ] Create `src/dimtensor/web/` directory structure
2. [ ] Create `src/dimtensor/web/app.py` - main Streamlit app with multi-page navigation
3. [ ] Create `src/dimtensor/web/pages/1_Models.py` - model browser page
   - [ ] Display table of all registered models with ModelInfo fields
   - [ ] Filter by domain (dropdown)
   - [ ] Filter by tags (multiselect)
   - [ ] Search by name/description
   - [ ] Click to expand: show full ModelCard with input/output dimensions
   - [ ] Generate code snippet for `load_model()`
4. [ ] Create `src/dimtensor/web/pages/2_Datasets.py` - dataset browser page
   - [ ] Display table of all registered datasets
   - [ ] Filter by domain
   - [ ] Filter by tags
   - [ ] Search by name/description
   - [ ] Show features/targets with dimensions
   - [ ] Generate code snippet for `load_dataset()`
5. [ ] Create `src/dimtensor/web/pages/3_Equations.py` - equation browser page
   - [ ] Display searchable equation list
   - [ ] Filter by domain (mechanics, thermodynamics, etc.)
   - [ ] Filter by tags
   - [ ] Search by name/formula/variables
   - [ ] Show LaTeX rendering (using st.latex)
   - [ ] Display variables with dimensions
   - [ ] Show related equations as links
   - [ ] Generate code snippet for dimensional validation
6. [ ] Create `src/dimtensor/web/utils.py` - helper functions (code generation, formatting)
7. [ ] Create `src/dimtensor/web/README.md` - deployment instructions
8. [ ] Add `streamlit` to `pyproject.toml` optional dependencies as `[web]`

### Phase 2: Enhanced Features

9. [ ] Add "Home" page with:
   - [ ] Project overview and links
   - [ ] Quick stats (# models, datasets, equations)
   - [ ] Quick start guide
   - [ ] Link to documentation
10. [ ] Add export functionality:
    - [ ] Export filtered results to JSON
    - [ ] Generate Jupyter notebook with selected items
11. [ ] Add comparison view:
    - [ ] Select multiple models/datasets/equations
    - [ ] Side-by-side comparison table
12. [ ] Add visualization:
    - [ ] Domain distribution pie chart
    - [ ] Tag word cloud
    - [ ] Dimensional analysis graph

### Phase 3: Deployment

13. [ ] Create `streamlit_app.py` in repo root (Streamlit Cloud entry point)
14. [ ] Create `.streamlit/config.toml` for theme customization
15. [ ] Test locally with `streamlit run streamlit_app.py`
16. [ ] Deploy to Streamlit Cloud (connect GitHub repo)
17. [ ] Add dashboard URL to README.md
18. [ ] Create short demo video/GIF

### Phase 4: Documentation & Community

19. [ ] Add web dashboard section to CLAUDE.md
20. [ ] Create tutorial: "Adding your model to the hub"
21. [ ] Create tutorial: "Browsing the dashboard"
22. [ ] Add dashboard badge to README

---

## Files to Modify

| File | Change |
|------|--------|
| **NEW** `src/dimtensor/web/` | Create directory |
| **NEW** `src/dimtensor/web/__init__.py` | Empty module init |
| **NEW** `src/dimtensor/web/app.py` | Main Streamlit app (Home page) |
| **NEW** `src/dimtensor/web/pages/1_Models.py` | Model browser page |
| **NEW** `src/dimtensor/web/pages/2_Datasets.py` | Dataset browser page |
| **NEW** `src/dimtensor/web/pages/3_Equations.py` | Equation browser page |
| **NEW** `src/dimtensor/web/utils.py` | Helper functions (code generation, formatting) |
| **NEW** `src/dimtensor/web/README.md` | Web app documentation |
| **NEW** `streamlit_app.py` | Entry point for Streamlit Cloud deployment |
| **NEW** `.streamlit/config.toml` | Streamlit theme/settings |
| `pyproject.toml` | Add `streamlit>=1.30.0` to `[project.optional-dependencies]` under `web` |
| `README.md` | Add link to web dashboard |
| `CONTINUITY.md` | Mark task #181 as DONE |

---

## Testing Strategy

### Manual Testing

- [ ] Run locally: `streamlit run streamlit_app.py`
- [ ] Verify all pages load without errors
- [ ] Test search functionality on each page
- [ ] Test filtering (domain, tags)
- [ ] Verify code generation produces valid Python
- [ ] Test with no models registered (graceful handling)
- [ ] Test with empty search results
- [ ] Test LaTeX rendering for equations
- [ ] Test responsive design (mobile, tablet, desktop)

### Integration Testing

- [ ] Create test script that:
  - Registers dummy model/dataset/equation
  - Launches Streamlit in test mode
  - Verifies data appears in dashboard
  - Cleans up test data

### Deployment Testing

- [ ] Test Streamlit Cloud deployment
- [ ] Verify URL is accessible
- [ ] Test sharing URL with others
- [ ] Monitor for errors in Streamlit Cloud logs

---

## Risks / Edge Cases

### Risk 1: Empty Registries
**Issue**: Dashboard crashes if no models/datasets registered
**Mitigation**:
- Add defensive checks for empty registries
- Show helpful message: "No models registered yet. Here's how to add one..."
- Link to registration tutorial

### Risk 2: LaTeX Rendering Issues
**Issue**: Complex LaTeX equations may not render in Streamlit
**Mitigation**:
- Test all 67 equations during development
- Fallback to plain text formula if LaTeX fails
- Use `st.latex()` which uses KaTeX

### Risk 3: Performance with Large Registry
**Issue**: Loading 1000+ models may be slow
**Mitigation**:
- Use `st.cache_data` for registry loading
- Implement pagination if needed
- Profile with 100+ entries

### Risk 4: Code Generation Edge Cases
**Issue**: Generated code may not work for all models
**Mitigation**:
- Template approach with clear placeholders
- Show both minimal and full examples
- Add comments explaining parameters

### Risk 5: Deployment Dependencies
**Issue**: Streamlit Cloud may have version conflicts
**Mitigation**:
- Pin streamlit version in pyproject.toml
- Test with minimal environment locally
- Use requirements.txt for Streamlit Cloud

### Edge Case: Special Characters in Search
**Handling**: Sanitize search input, use regex-safe matching

### Edge Case: Malformed Model Cards
**Handling**: Defensive dict access with `.get()`, show partial info

### Edge Case: Missing Optional Fields
**Handling**: Use empty strings/lists as defaults, conditional display

---

## Definition of Done

- [ ] All implementation steps complete (Phase 1-4)
- [ ] Manual testing complete
- [ ] Dashboard deployed to Streamlit Cloud
- [ ] URL added to README.md
- [ ] CONTINUITY.md updated with completion status
- [ ] Basic documentation exists (web/README.md)
- [ ] At least one demo screenshot or GIF captured

---

## Notes / Log

### Architecture Details

**Directory structure**:
```
src/dimtensor/web/
├── __init__.py
├── app.py              # Home page (Streamlit entry)
├── utils.py            # Code generation, formatters
├── README.md           # Deployment guide
└── pages/
    ├── 1_Models.py     # Auto-discovered by Streamlit
    ├── 2_Datasets.py
    └── 3_Equations.py

streamlit_app.py        # Root entry point: imports src.dimtensor.web.app
.streamlit/
└── config.toml         # Theme colors, settings
```

**Page naming**: Streamlit auto-discovers pages/ with numeric prefixes for ordering.

**Caching strategy**: Use `@st.cache_data` for:
- Loading registries (list_models, list_datasets, get_equations)
- Expensive computations

**Code generation template**:
```python
# For models:
from dimtensor.hub import load_model
model = load_model("{name}")
# Input dimensions: {input_dims}
# Output dimensions: {output_dims}

# For datasets:
from dimtensor.datasets import load_dataset
data = load_dataset("{name}")
# Features: {features}
# Targets: {targets}

# For equations:
from dimtensor.equations import get_equation
eq = get_equation("{name}")
# Formula: {formula}
# Variables: {variables}
```

**Streamlit Cloud deployment**:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repo
4. Point to `streamlit_app.py`
5. Auto-deploys on push

**Alternative hosting**:
- Hugging Face Spaces (requires Dockerfile)
- Self-hosted (Docker + nginx)
- Railway, Render, Heroku

**Future enhancements** (post-v4.0.0):
- User authentication for model uploads
- API endpoints via FastAPI integration
- Live model inference demos
- Dataset preview/plotting
- Equation solver/calculator
- Integration with GitHub Actions for auto-registry updates
- Analytics (most viewed models/equations)
- Community ratings/comments

---
