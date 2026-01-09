"""Equations browser page for dimtensor hub."""

import streamlit as st

st.set_page_config(
    page_title="Equations - dimtensor Hub",
    page_icon="🧮",
    layout="wide",
)


@st.cache_data
def load_equations():
    """Load all equations from the database."""
    from dimtensor.equations import get_equations
    return get_equations()


@st.cache_data
def get_domains():
    """Get all unique domains from equations."""
    from dimtensor.equations import list_domains
    return list_domains()


@st.cache_data
def get_tags(equations):
    """Get all unique tags from equations."""
    from dimtensor.web.utils import get_all_tags
    return get_all_tags(equations)


def main():
    """Main function for equations page."""
    st.title("🧮 Equation Browser")
    st.markdown("Search and explore physics equations with dimensional metadata.")

    # Load equations
    try:
        equations = load_equations()
    except Exception as e:
        st.error(f"Error loading equations: {e}")
        return

    if not equations:
        st.info("No equations are currently registered.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Domain filter
    domains = get_domains()
    selected_domain = st.sidebar.selectbox(
        "Domain",
        ["All"] + domains,
        index=0,
    )

    # Tag filter
    all_tags = get_tags(equations)
    selected_tags = st.sidebar.multiselect(
        "Tags (must have all)",
        all_tags,
    )

    # Search
    search_query = st.sidebar.text_input("Search", placeholder="Search equations, variables, formulas")

    # Apply filters
    from dimtensor.web.utils import filter_by_domain, filter_by_tags, search_items

    filtered = equations
    filtered = filter_by_domain(filtered, selected_domain)
    filtered = filter_by_tags(filtered, selected_tags)
    filtered = search_items(
        filtered,
        search_query,
        ["name", "description", "formula", "domain", "tags"]
    )

    # Display results
    st.markdown(f"### Found {len(filtered)} equation(s)")

    if not filtered:
        st.warning("No equations match your filters. Try adjusting your search criteria.")
        return

    # Display equations as expandable cards
    for eq in filtered:
        with st.expander(f"**{eq.name}** - {eq.domain}"):
            # Two-column layout
            col1, col2 = st.columns([3, 2])

            with col1:
                # LaTeX rendering
                if eq.latex:
                    st.markdown("**Equation:**")
                    try:
                        st.latex(eq.latex)
                    except Exception:
                        # Fallback to text formula
                        st.code(eq.formula)
                else:
                    st.markdown("**Formula:**")
                    st.code(eq.formula)

                # Description
                if eq.description:
                    st.markdown(f"**Description:** {eq.description}")

                st.markdown(f"**Domain:** {eq.domain}")

                # Tags
                if eq.tags:
                    tags_str = ", ".join(f"`{tag}`" for tag in eq.tags)
                    st.markdown(f"**Tags:** {tags_str}")

            with col2:
                # Variables with dimensions
                st.markdown("**Variables:**")
                if eq.variables:
                    for var_name, dim in eq.variables.items():
                        st.markdown(f"- `{var_name}`: {dim}")
                else:
                    st.markdown("*No variables defined*")

            # Assumptions
            if eq.assumptions:
                st.markdown("**Assumptions:**")
                for assumption in eq.assumptions:
                    st.markdown(f"- {assumption}")

            # Related equations
            if eq.related:
                related_str = ", ".join(f"`{r}`" for r in eq.related)
                st.markdown(f"**Related equations:** {related_str}")

            # Code generation
            st.markdown("**Example code for dimensional validation:**")
            from dimtensor.web.utils import generate_equation_code
            code = generate_equation_code(eq.name, eq)
            st.code(code, language="python")


if __name__ == "__main__":
    main()
