# dimtensor Web Dashboard

Interactive web dashboard for browsing the dimtensor model hub, datasets, and equation database.

## Features

- **Models Browser**: Explore pre-trained physics-aware neural networks with dimensional metadata
- **Datasets Browser**: Discover physics datasets for training ML models
- **Equations Browser**: Search 67+ physics equations with LaTeX rendering and dimensional validation

## Installation

Install dimtensor with web dashboard support:

```bash
pip install dimtensor[web]
```

This installs Streamlit and other required dependencies.

## Running Locally

### Option 1: From repository root

```bash
streamlit run streamlit_app.py
```

### Option 2: Direct module run

```bash
python -m streamlit run src/dimtensor/web/app.py
```

The dashboard will open in your browser at http://localhost:8501

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Point to `streamlit_app.py` as the entry point
5. Deploy (auto-deploys on push)

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose Streamlit as the SDK
3. Upload the repository files
4. Set entry point to `streamlit_app.py`

### Self-hosted (Docker)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[web]"

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t dimtensor-web .
docker run -p 8501:8501 dimtensor-web
```

## Usage

### Browsing Models

1. Navigate to **Models** page
2. Use filters to narrow by domain or tags
3. Search by name or description
4. Expand models to see:
   - Input/output dimensions
   - Architecture details
   - Code snippets for loading

### Browsing Datasets

1. Navigate to **Datasets** page
2. Filter by domain (mechanics, fluid_dynamics, etc.)
3. View feature and target dimensions
4. Get code snippets for loading datasets

### Browsing Equations

1. Navigate to **Equations** page
2. Search across 67+ physics equations
3. View LaTeX-rendered formulas
4. See dimensional metadata for each variable
5. Get code snippets for dimensional validation

## Configuration

Customize the dashboard theme by editing `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4A90E2"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F7FA"
textColor = "#262730"
font = "sans serif"
```

## Adding Your Own Content

### Register a Model

```python
from dimtensor.hub import register_model, ModelInfo
from dimtensor import Dimension

info = ModelInfo(
    name="my-model",
    version="1.0.0",
    description="My physics model",
    domain="mechanics",
    input_dims={"x": Dimension(length=1)},
    output_dims={"y": Dimension(length=1, time=-1)},
    tags=["velocity", "predictor"],
)

@register_model("my-model", info=info)
def create_my_model():
    return MyModel()
```

### Register a Dataset

```python
from dimtensor.datasets import register_dataset, DatasetInfo
from dimtensor import Dimension

info = DatasetInfo(
    name="my-dataset",
    description="My physics dataset",
    domain="mechanics",
    features={"t": Dimension(time=1)},
    targets={"x": Dimension(length=1)},
    tags=["trajectory"],
)

@register_dataset("my-dataset", info=info)
def load_my_dataset():
    return load_data()
```

### Register an Equation

```python
from dimtensor.equations import register_equation, Equation
from dimtensor import Dimension

eq = Equation(
    name="My Equation",
    formula="y = mx + b",
    variables={
        "y": Dimension(length=1),
        "m": Dimension(length=1, time=-1),
        "x": Dimension(time=1),
        "b": Dimension(length=1),
    },
    domain="kinematics",
    tags=["linear", "motion"],
    latex=r"y = mx + b",
)

register_equation(eq)
```

Then restart the Streamlit app to see your new content!

## Architecture

```
src/dimtensor/web/
├── __init__.py         # Package init
├── app.py              # Home page (entry point)
├── utils.py            # Shared utilities (code generation, filtering)
├── README.md           # This file
└── pages/
    ├── 1_Models.py     # Models browser (auto-discovered by Streamlit)
    ├── 2_Datasets.py   # Datasets browser
    └── 3_Equations.py  # Equations browser

streamlit_app.py        # Root entry point for Streamlit Cloud
.streamlit/
└── config.toml         # Streamlit theme and settings
```

Streamlit automatically discovers pages in the `pages/` directory and creates navigation links.

## Troubleshooting

### Port already in use

If port 8501 is already in use:

```bash
streamlit run streamlit_app.py --server.port=8502
```

### Caching issues

Clear Streamlit cache:

```bash
streamlit cache clear
```

### Import errors

Make sure dimtensor is installed in editable mode:

```bash
pip install -e ".[web]"
```

## Contributing

Contributions welcome! To add features:

1. Fork the repository
2. Create a feature branch
3. Make your changes to files in `src/dimtensor/web/`
4. Test locally with `streamlit run streamlit_app.py`
5. Submit a pull request

## License

MIT License - same as dimtensor
