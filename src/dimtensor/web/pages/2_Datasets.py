"""Datasets browser page for dimtensor hub."""

import streamlit as st

st.set_page_config(
    page_title="Datasets - dimtensor Hub",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_datasets():
    """Load all datasets from the registry."""
    from dimtensor.datasets import list_datasets
    return list_datasets()


@st.cache_data
def get_domains(datasets):
    """Get all unique domains from datasets."""
    from dimtensor.web.utils import get_all_domains
    return get_all_domains(datasets)


@st.cache_data
def get_tags(datasets):
    """Get all unique tags from datasets."""
    from dimtensor.web.utils import get_all_tags
    return get_all_tags(datasets)


def main():
    """Main function for datasets page."""
    st.title("📊 Dataset Browser")
    st.markdown("Explore physics datasets with dimensional metadata for training physics-informed ML models.")

    # Load datasets
    try:
        datasets = load_datasets()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return

    if not datasets:
        st.info("No datasets are currently registered.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Domain filter
    domains = get_domains(datasets)
    selected_domain = st.sidebar.selectbox(
        "Domain",
        ["All"] + domains,
        index=0,
    )

    # Tag filter
    all_tags = get_tags(datasets)
    selected_tags = st.sidebar.multiselect(
        "Tags (must have all)",
        all_tags,
    )

    # Search
    search_query = st.sidebar.text_input("Search", placeholder="Search by name or description")

    # Apply filters
    from dimtensor.web.utils import filter_by_domain, filter_by_tags, search_items

    filtered = datasets
    filtered = filter_by_domain(filtered, selected_domain)
    filtered = filter_by_tags(filtered, selected_tags)
    filtered = search_items(filtered, search_query, ["name", "description", "domain", "tags"])

    # Display results
    st.markdown(f"### Found {len(filtered)} dataset(s)")

    if not filtered:
        st.warning("No datasets match your filters. Try adjusting your search criteria.")
        return

    # Display datasets as expandable cards
    for dataset in filtered:
        with st.expander(f"**{dataset.name}** - {dataset.domain}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {dataset.description or 'N/A'}")
                st.markdown(f"**Domain:** {dataset.domain}")

                if dataset.size:
                    st.markdown(f"**Size:** {dataset.size:,} samples")
                else:
                    st.markdown("**Size:** Variable")

                if dataset.tags:
                    tags_str = ", ".join(f"`{tag}`" for tag in dataset.tags)
                    st.markdown(f"**Tags:** {tags_str}")

                if dataset.license:
                    st.markdown(f"**License:** {dataset.license}")

                if dataset.source:
                    st.markdown(f"**Source:** {dataset.source}")

            with col2:
                st.markdown("**Dimensions**")
                if dataset.features:
                    st.markdown("*Features:*")
                    for name, dim in dataset.features.items():
                        st.markdown(f"- `{name}`: {dim}")
                if dataset.targets:
                    st.markdown("*Targets:*")
                    for name, dim in dataset.targets.items():
                        st.markdown(f"- `{name}`: {dim}")

            # Citation
            if dataset.citation:
                st.markdown("**Citation:**")
                st.info(dataset.citation)

            # Code generation
            st.markdown("**Code to load this dataset:**")
            from dimtensor.web.utils import generate_dataset_code
            code = generate_dataset_code(dataset.name, dataset)
            st.code(code, language="python")


if __name__ == "__main__":
    main()
