"""Models browser page for dimtensor hub."""

import streamlit as st

st.set_page_config(
    page_title="Models - dimtensor Hub",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data
def load_models():
    """Load all models from the registry."""
    from dimtensor.hub import list_models
    return list_models()


@st.cache_data
def get_domains(models):
    """Get all unique domains from models."""
    from dimtensor.web.utils import get_all_domains
    return get_all_domains(models)


@st.cache_data
def get_tags(models):
    """Get all unique tags from models."""
    from dimtensor.web.utils import get_all_tags
    return get_all_tags(models)


def main():
    """Main function for models page."""
    st.title("🤖 Model Browser")
    st.markdown("Browse pre-trained physics-aware neural networks with dimensional metadata.")

    # Load models
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("No models are currently registered in the hub. Register models using `dimtensor.hub.register_model`.")
        return

    if not models:
        st.info("No models are currently registered in the hub.")
        st.markdown("""
        ### How to register a model

        ```python
        from dimtensor.hub import register_model, ModelInfo
        from dimtensor import Dimension

        # Define model metadata
        info = ModelInfo(
            name="my-model",
            version="1.0.0",
            description="My physics model",
            domain="mechanics",
            input_dims={"x": Dimension(length=1)},
            output_dims={"y": Dimension(length=1, time=-1)},
            tags=["velocity", "predictor"],
        )

        # Register model
        @register_model("my-model", info=info)
        def create_my_model():
            return MyModel()
        ```
        """)
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Domain filter
    domains = get_domains(models)
    selected_domain = st.sidebar.selectbox(
        "Domain",
        ["All"] + domains,
        index=0,
    )

    # Tag filter
    all_tags = get_tags(models)
    selected_tags = st.sidebar.multiselect(
        "Tags (must have all)",
        all_tags,
    )

    # Search
    search_query = st.sidebar.text_input("Search", placeholder="Search by name or description")

    # Apply filters
    from dimtensor.web.utils import filter_by_domain, filter_by_tags, search_items

    filtered = models
    filtered = filter_by_domain(filtered, selected_domain)
    filtered = filter_by_tags(filtered, selected_tags)
    filtered = search_items(filtered, search_query, ["name", "description", "domain", "tags"])

    # Display results
    st.markdown(f"### Found {len(filtered)} model(s)")

    if not filtered:
        st.warning("No models match your filters. Try adjusting your search criteria.")
        return

    # Display models as expandable cards
    for model in filtered:
        with st.expander(f"**{model.name}** v{model.version} - {model.domain}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {model.description or 'N/A'}")
                st.markdown(f"**Domain:** {model.domain}")
                st.markdown(f"**Architecture:** {model.architecture or 'N/A'}")
                st.markdown(f"**Author:** {model.author or 'N/A'}")
                st.markdown(f"**License:** {model.license}")

                if model.tags:
                    tags_str = ", ".join(f"`{tag}`" for tag in model.tags)
                    st.markdown(f"**Tags:** {tags_str}")

            with col2:
                st.markdown("**Dimensions**")
                if model.input_dims:
                    st.markdown("*Inputs:*")
                    for name, dim in model.input_dims.items():
                        st.markdown(f"- `{name}`: {dim}")
                if model.output_dims:
                    st.markdown("*Outputs:*")
                    for name, dim in model.output_dims.items():
                        st.markdown(f"- `{name}`: {dim}")

            # Characteristic scales
            if model.characteristic_scales:
                st.markdown("**Characteristic Scales**")
                for name, value in model.characteristic_scales.items():
                    st.markdown(f"- `{name}`: {value}")

            # Source
            if model.source:
                st.markdown(f"**Source:** {model.source}")

            # Code generation
            st.markdown("**Code to load this model:**")
            from dimtensor.web.utils import generate_model_code
            code = generate_model_code(model.name, model)
            st.code(code, language="python")


if __name__ == "__main__":
    main()
