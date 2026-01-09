"""Tests for the physics equation database."""

import pytest

from dimtensor import Dimension
from dimtensor.equations import (
    Equation,
    get_equation,
    get_equations,
    list_domains,
    register_equation,
    search_equations,
)
from dimtensor.equations.database import clear_equations, _EQUATIONS


@pytest.fixture(autouse=True)
def preserve_registry():
    """Preserve and restore registry for tests that modify it."""
    original = _EQUATIONS.copy()
    yield
    _EQUATIONS.clear()
    _EQUATIONS.update(original)


class TestEquationCreation:
    """Tests for Equation dataclass."""

    def test_basic_creation(self):
        """Test creating an equation."""
        eq = Equation(
            name="test_equation",
            formula="F = m * a",
            variables={"F": Dimension(mass=1, length=1, time=-2)},
            domain="mechanics",
        )
        assert eq.name == "test_equation"
        assert eq.formula == "F = m * a"
        assert eq.domain == "mechanics"

    def test_with_variables(self):
        """Test equation with variables."""
        eq = Equation(
            name="kinetic_energy",
            formula="E = 0.5 * m * v^2",
            variables={
                "E": Dimension(mass=1, length=2, time=-2),
                "m": Dimension(mass=1),
                "v": Dimension(length=1, time=-1),
            },
            domain="mechanics",
        )
        assert len(eq.variables) == 3
        assert eq.variables["m"] == Dimension(mass=1)
        assert eq.variables["v"] == Dimension(length=1, time=-1)

    def test_to_dict(self):
        """Test serialization to dict."""
        eq = Equation(
            name="test",
            formula="x = y",
            variables={"x": Dimension(length=1)},
            domain="test",
            tags=["test", "example"],
        )
        data = eq.to_dict()
        assert data["name"] == "test"
        assert data["formula"] == "x = y"
        assert data["tags"] == ["test", "example"]


class TestEquationRegistry:
    """Tests for equation registry functions."""

    def test_list_domains(self):
        """Test listing all domains."""
        domains = list_domains()
        assert "mechanics" in domains
        assert "thermodynamics" in domains
        assert "electromagnetism" in domains
        assert len(domains) >= 5

    def test_get_all_equations(self):
        """Test getting all equations."""
        equations = get_equations()
        assert len(equations) >= 20
        assert all(isinstance(eq, Equation) for eq in equations)

    def test_get_by_domain(self):
        """Test filtering by domain."""
        mechanics = get_equations(domain="mechanics")
        assert len(mechanics) >= 8
        assert all(eq.domain == "mechanics" for eq in mechanics)

    def test_get_by_tags(self):
        """Test filtering by tags."""
        energy_eqs = get_equations(tags=["energy"])
        assert len(energy_eqs) >= 3
        assert all("energy" in eq.tags for eq in energy_eqs)

    def test_get_by_domain_and_tags(self):
        """Test combined filtering."""
        mech_energy = get_equations(domain="mechanics", tags=["energy"])
        assert len(mech_energy) >= 1
        assert all(eq.domain == "mechanics" for eq in mech_energy)
        assert all("energy" in eq.tags for eq in mech_energy)

    def test_get_equation_by_name(self):
        """Test getting specific equation."""
        eq = get_equation("Newton's Second Law")
        assert eq.name == "Newton's Second Law"
        assert eq.formula == "F = ma"
        # Check force dimension in variables
        assert eq.variables["F"] == Dimension(mass=1, length=1, time=-2)

    def test_get_equation_not_found(self):
        """Test error for missing equation."""
        with pytest.raises(KeyError, match="not found"):
            get_equation("nonexistent_equation")

    def test_search_equations(self):
        """Test searching equations."""
        results = search_equations("energy")
        assert len(results) >= 5
        # Should find kinetic, potential, mass-energy, etc.

    def test_search_by_formula(self):
        """Test search matches formula."""
        results = search_equations("ma")
        assert any("Newton" in eq.name for eq in results)

    def test_search_by_description(self):
        """Test search matches description."""
        results = search_equations("Newton")
        assert len(results) >= 1

    def test_search_case_insensitive(self):
        """Test case-insensitive search."""
        upper = search_equations("ENERGY")
        lower = search_equations("energy")
        assert len(upper) == len(lower)


class TestRegistration:
    """Tests for registering custom equations."""

    def test_register_direct(self):
        """Test direct registration."""
        eq = Equation(
            name="custom_equation_test",
            formula="x = a + b",
            variables={"x": Dimension(length=1)},
            domain="test",
        )
        register_equation(eq)

        result = get_equation("custom_equation_test")
        assert result.name == "custom_equation_test"


class TestBuiltinEquations:
    """Tests for built-in physics equations."""

    def test_newton_second_law(self):
        """Test Newton's second law dimensions."""
        eq = get_equation("Newton's Second Law")
        # Force dimension
        assert eq.variables["F"] == Dimension(mass=1, length=1, time=-2)

    def test_kinetic_energy(self):
        """Test kinetic energy dimensions."""
        eq = get_equation("Kinetic Energy")
        # Energy dimension
        assert eq.variables["KE"] == Dimension(mass=1, length=2, time=-2)

    def test_ideal_gas_law(self):
        """Test ideal gas law."""
        eq = get_equation("Ideal Gas Law")
        assert "PV" in eq.formula or "nRT" in eq.formula
        # Check pressure dimension
        assert eq.variables["P"] == Dimension(mass=1, length=-1, time=-2)

    def test_coulombs_law(self):
        """Test Coulomb's law."""
        eq = get_equation("Coulomb's Law")
        # Force dimension
        assert eq.variables["F"] == Dimension(mass=1, length=1, time=-2)

    def test_mass_energy_equivalence(self):
        """Test E = mc²."""
        eq = get_equation("Mass-Energy Equivalence")
        assert "c" in eq.formula
        # Energy dimension
        assert eq.variables["E"] == Dimension(mass=1, length=2, time=-2)

    def test_planck_einstein(self):
        """Test Planck-Einstein relation."""
        eq = get_equation("Planck-Einstein Relation")
        # Energy dimension
        assert eq.variables["E"] == Dimension(mass=1, length=2, time=-2)


class TestDimensionalVerification:
    """Tests for dimensional verification of equations."""

    def test_force_equation_consistency(self):
        """Verify F = ma has consistent dimensions."""
        eq = get_equation("Newton's Second Law")
        # m [kg] * a [m/s²] = [kg·m/s²] = [N]
        m_dim = eq.variables["m"]
        a_dim = eq.variables["a"]
        f_dim = eq.variables["F"]
        assert m_dim * a_dim == f_dim

    def test_energy_equation_consistency(self):
        """Verify E = ½mv² has consistent dimensions."""
        eq = get_equation("Kinetic Energy")
        # m [kg] * v² [m²/s²] = [kg·m²/s²] = [J]
        m_dim = eq.variables["m"]
        v_dim = eq.variables["v"]
        ke_dim = eq.variables["KE"]
        # v² has dimensions v_dim * v_dim
        assert m_dim * v_dim * v_dim == ke_dim

    def test_ohms_law_consistency(self):
        """Verify V = IR has consistent dimensions."""
        eq = get_equation("Ohm's Law")
        i_dim = eq.variables["I"]
        r_dim = eq.variables["R"]
        v_dim = eq.variables["V"]
        assert i_dim * r_dim == v_dim

    def test_bernoulli_pressure_consistency(self):
        """Verify pressure terms in Bernoulli's equation."""
        eq = get_equation("Bernoulli's Equation")
        rho_dim = eq.variables["rho"]
        v_dim = eq.variables["v"]
        p_dim = eq.variables["P"]
        # ½ρv² should have pressure dimensions [kg/(m·s²)]
        dynamic_pressure = rho_dim * v_dim * v_dim
        assert dynamic_pressure == p_dim
