"""Streamlit web dashboard for dimtensor model hub.

Home page showing overview and quick stats.
"""

import streamlit as st


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="dimtensor Hub",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📐 dimtensor Model Hub")
    st.markdown("### Unit-aware tensors for physics and scientific machine learning")

    # Overview
    st.markdown("""
    Welcome to the **dimtensor hub**! This dashboard lets you explore:
    - **Models**: Pre-trained physics-aware neural networks
    - **Datasets**: Physics datasets with dimensional metadata
    - **Equations**: Database of physics equations for dimensional validation

    Use the sidebar to navigate between pages.
    """)

    # Quick stats
    st.markdown("---")
    st.subheader("Quick Stats")

    col1, col2, col3 = st.columns(3)

    try:
        from dimtensor.hub import list_models
        models = list_models()
        col1.metric("Models", len(models))
    except Exception:
        col1.metric("Models", "N/A")

    try:
        from dimtensor.datasets import list_datasets
        datasets = list_datasets()
        col2.metric("Datasets", len(datasets))
    except Exception:
        col2.metric("Datasets", "N/A")

    try:
        from dimtensor.equations import get_equations
        equations = get_equations()
        col3.metric("Equations", len(equations))
    except Exception:
        col3.metric("Equations", "N/A")

    # Getting started
    st.markdown("---")
    st.subheader("Getting Started")

    st.markdown("""
    #### Installation

    ```bash
    pip install dimtensor
    ```

    #### Quick Example

    ```python
    from dimtensor import DimArray
    import dimtensor.units as u

    # Create unit-aware arrays
    distance = DimArray([1, 2, 3], u.meter)
    time = DimArray([1, 2, 3], u.second)

    # Operations respect units
    velocity = distance / time  # Result has units of m/s
    print(velocity)  # [1. 1. 1.] m/s

    # Dimensional errors are caught automatically
    # distance + time  # Raises DimensionError!
    ```

    #### Browse the Hub

    Use the sidebar to explore:
    - **Models**: Browse and load pre-trained physics models
    - **Datasets**: Discover datasets for training physics-informed ML
    - **Equations**: Search physics equations with dimensional metadata
    """)

    # Resources
    st.markdown("---")
    st.subheader("Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Documentation**
        - [User Guide](https://github.com/marcoloco23/dimtensor)
        - [API Reference](https://github.com/marcoloco23/dimtensor)
        - [Examples](https://github.com/marcoloco23/dimtensor/tree/main/examples)
        """)

    with col2:
        st.markdown("""
        **Community**
        - [GitHub](https://github.com/marcoloco23/dimtensor)
        - [Issues](https://github.com/marcoloco23/dimtensor/issues)
        - [Discussions](https://github.com/marcoloco23/dimtensor/discussions)
        """)

    with col3:
        st.markdown("""
        **Contributing**
        - [Contributing Guide](https://github.com/marcoloco23/dimtensor/blob/main/CONTRIBUTING.md)
        - [Code of Conduct](https://github.com/marcoloco23/dimtensor/blob/main/CODE_OF_CONDUCT.md)
        - [Development Setup](https://github.com/marcoloco23/dimtensor#development)
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Made with ❤️ by the dimtensor community |
        <a href='https://github.com/marcoloco23/dimtensor'>GitHub</a> |
        MIT License
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
