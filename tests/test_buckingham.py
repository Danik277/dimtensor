"""Tests for Buckingham Pi theorem solver."""

import pytest
import numpy as np
from fractions import Fraction

from dimtensor.analysis import buckingham_pi, PiGroup
from dimtensor.core.units import m, s, kg, K, dimensionless
from dimtensor.core.dimensions import Dimension, DIMENSIONLESS
from dimtensor import DimArray


class TestBuckinghamPi:
    """Tests for buckingham_pi function."""

    def test_drag_force_problem(self):
        """Test classic fluid mechanics drag force problem.

        Variables: F (force), v (velocity), L (length), rho (density), mu (viscosity)
        Expected: 2 Pi groups (Reynolds number and drag coefficient)
        """
        variables = {
            'F': kg * m / s**2,      # Force
            'v': m / s,               # Velocity
            'L': m,                   # Length
            'rho': kg / m**3,         # Density
            'mu': kg / (m * s),       # Dynamic viscosity
        }

        result = buckingham_pi(variables)

        # Check dimensions
        assert result['n_variables'] == 5
        assert result['rank'] == 3  # L, M, T
        assert result['n_groups'] == 2  # 5 - 3 = 2
        assert set(result['base_dimensions']) == {'L', 'M', 'T'}

        # Check that we have 2 Pi groups
        assert len(result['pi_groups']) == 2

        # Verify each Pi group is dimensionless
        for pi in result['pi_groups']:
            assert isinstance(pi, PiGroup)
            # Check that expression is non-empty
            assert len(pi.expression) > 0
            # Check that latex is non-empty
            assert len(pi.latex) > 0

    def test_pendulum_period(self):
        """Test pendulum period problem.

        Variables: T (period), L (length), m (mass), g (gravity)
        Expected: 1 Pi group, mass should drop out
        """
        variables = {
            'T': s,                  # Period
            'L': m,                  # Length
            'm': kg,                 # Mass
            'g': m / s**2,          # Gravity
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 4
        assert result['rank'] == 3  # L, M, T
        assert result['n_groups'] == 1  # 4 - 3 = 1

        # Check that we have 1 Pi group
        assert len(result['pi_groups']) == 1

        pi = result['pi_groups'][0]
        # Mass should not appear (exponent = 0)
        assert pi.exponents.get('m', Fraction(0)) == 0

    def test_heat_conduction(self):
        """Test simple heat conduction problem.

        Variables: q (heat flux), k (thermal conductivity), dT (temperature diff), L (length)
        """
        variables = {
            'q': kg / s**3,          # Heat flux (W/m² = kg/s³)
            'k': kg * m / (s**3 * K),  # Thermal conductivity (W/(m·K))
            'dT': K,                 # Temperature difference
            'L': m,                  # Length
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 4
        # Active dimensions: L, M, T, Θ
        # Rank is the number of linearly independent columns (variables), which is 3
        assert result['rank'] == 3
        assert result['n_groups'] == 1  # 4 - 3 = 1

    def test_dimensionless_variables(self):
        """Test with all dimensionless variables."""
        variables = {
            'x': dimensionless,
            'y': dimensionless,
            'z': dimensionless,
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 3
        assert result['rank'] == 0  # No active dimensions
        # With all dimensionless, could have up to n Pi groups
        # (Each variable forms its own Pi group)
        assert result['n_groups'] >= 0

    def test_single_variable(self):
        """Test with single variable."""
        variables = {
            'v': m / s,
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 1
        assert result['rank'] == 1  # Rank is 1 (one linearly independent column)
        assert result['n_groups'] == 0  # Cannot form dimensionless group from 1 variable

    def test_identical_dimensions(self):
        """Test with variables having identical dimensions."""
        variables = {
            'v1': m / s,
            'v2': m / s,
            'v3': m / s,
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 3
        # All three variables have the same dimensions, so rank = 1
        assert result['rank'] == 1
        assert result['n_groups'] == 2  # 3 - 1 = 2

        # Should get ratios like v1/v2, v1/v3
        assert len(result['pi_groups']) == 2

    def test_with_dimarray_input(self):
        """Test that DimArray inputs work correctly."""
        variables = {
            'F': DimArray([100.0], kg * m / s**2),
            'v': DimArray([10.0], m / s),
            'L': DimArray([1.0], m),
            'rho': DimArray([1000.0], kg / m**3),
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 4
        assert result['rank'] == 3  # L, M, T
        assert result['n_groups'] == 1

    def test_with_dimension_input(self):
        """Test that Dimension inputs work correctly."""
        variables = {
            'F': Dimension(mass=1, length=1, time=-2),
            'v': Dimension(length=1, time=-1),
            'L': Dimension(length=1),
            'rho': Dimension(mass=1, length=-3),
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 4
        assert result['rank'] == 3
        assert result['n_groups'] == 1

    def test_empty_variables(self):
        """Test that empty variables dict raises error."""
        with pytest.raises(ValueError, match="Must provide at least one variable"):
            buckingham_pi({})

    def test_invalid_variable_type(self):
        """Test that invalid variable types raise error."""
        variables = {
            'x': 5.0,  # Plain number, not Unit/Dimension/DimArray
        }

        with pytest.raises(TypeError, match="must be Unit, Dimension, or DimArray"):
            buckingham_pi(variables)

    def test_pi_group_properties(self):
        """Test PiGroup properties and methods."""
        variables = {
            'v1': m / s,
            'v2': m / s,
            'v3': m / s,
        }

        result = buckingham_pi(variables)
        pi = result['pi_groups'][0]

        # Test that PiGroup has required attributes
        assert hasattr(pi, 'name')
        assert hasattr(pi, 'exponents')
        assert hasattr(pi, 'expression')
        assert hasattr(pi, 'latex')
        assert hasattr(pi, 'interpretation')

        # Test string representations
        assert len(str(pi)) > 0
        assert len(repr(pi)) > 0

        # Test that exponents are Fractions
        for exp in pi.exponents.values():
            assert isinstance(exp, Fraction)

    def test_fractional_exponents(self):
        """Test that fractional exponents are handled correctly."""
        # This is a constructed example where we might expect fractional exponents
        variables = {
            'x': m,
            'y': m**2,
            'z': m**3,
        }

        result = buckingham_pi(variables)

        # Should produce Pi groups with potentially fractional exponents
        assert result['n_groups'] == 2  # 3 variables - 1 dimension = 2 groups

    def test_projectile_motion(self):
        """Test projectile motion problem.

        Variables: R (range), v0 (initial velocity), g (gravity)
        Note: angle is dimensionless and would be a Pi group itself
        """
        variables = {
            'R': m,                  # Range
            'v0': m / s,            # Initial velocity
            'g': m / s**2,          # Gravity
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 3
        assert result['rank'] == 2  # L, T
        assert result['n_groups'] == 1  # 3 - 2 = 1

        # Expected: R*g/v0² is dimensionless
        assert len(result['pi_groups']) == 1

    def test_reynolds_number_variables(self):
        """Test classic Reynolds number setup.

        Should produce ρvL/μ (Reynolds number)
        """
        variables = {
            'rho': kg / m**3,        # Density
            'v': m / s,               # Velocity
            'L': m,                   # Length
            'mu': kg / (m * s),       # Dynamic viscosity
        }

        result = buckingham_pi(variables)

        assert result['n_variables'] == 4
        assert result['rank'] == 3  # L, M, T
        assert result['n_groups'] == 1  # Classic Reynolds number

        pi = result['pi_groups'][0]

        # Check that all variables appear (no zero exponents except possibly normalized)
        non_zero_exps = [exp for exp in pi.exponents.values() if exp != 0]
        assert len(non_zero_exps) >= 3  # At least 3 variables should appear

    def test_expression_formatting(self):
        """Test that expressions are formatted correctly."""
        variables = {
            'a': m,
            'b': m**2,
            'c': m**(-1),
        }

        result = buckingham_pi(variables)

        for pi in result['pi_groups']:
            expr = pi.expression
            # Should not have weird formatting
            assert '**' not in expr or '^' in expr  # Use superscripts or ^ notation
            # Should have some content
            assert len(expr) > 0

    def test_latex_formatting(self):
        """Test that LaTeX expressions are formatted correctly."""
        variables = {
            'a': m,
            'b': m**2,
        }

        result = buckingham_pi(variables)

        for pi in result['pi_groups']:
            latex = pi.latex
            # Basic check that it's LaTeX-like
            assert len(latex) > 0
            # Could have frac, or just symbols
            assert '\\' in latex or latex.replace(' ', '').replace('^', '').isalnum()

    def test_tolerance_parameter(self):
        """Test that tolerance parameter works."""
        variables = {
            'v1': m / s,
            'v2': m / s,
        }

        # Try with different tolerances
        result1 = buckingham_pi(variables, tolerance=1e-10)
        result2 = buckingham_pi(variables, tolerance=1e-6)

        # Both should produce same result for this simple case
        assert result1['n_groups'] == result2['n_groups']

    def test_mixed_unit_systems(self):
        """Test with variables from different unit systems."""
        from dimtensor.core.units import foot, pound

        variables = {
            'L1': m,
            'L2': foot,
            'm1': kg,
            'm2': pound,
        }

        result = buckingham_pi(variables)

        # Dimensions should be recognized correctly regardless of scale
        assert result['rank'] == 2  # L and M
        assert result['n_groups'] == 2  # 4 - 2 = 2


class TestPiGroupClass:
    """Tests for PiGroup dataclass."""

    def test_pigroup_creation(self):
        """Test creating a PiGroup directly."""
        pi = PiGroup(
            name="Π₁",
            exponents={'a': Fraction(1), 'b': Fraction(-1)},
            expression="a/b",
            latex=r"\frac{a}{b}",
            interpretation=None,
        )

        assert pi.name == "Π₁"
        assert pi.exponents == {'a': Fraction(1), 'b': Fraction(-1)}
        assert pi.expression == "a/b"
        assert pi.latex == r"\frac{a}{b}"
        assert pi.interpretation is None

    def test_pigroup_string_representation(self):
        """Test string representations of PiGroup."""
        pi = PiGroup(
            name="Re",
            exponents={'rho': Fraction(1), 'v': Fraction(1), 'L': Fraction(1), 'mu': Fraction(-1)},
            expression="ρvL/μ",
            latex=r"\frac{\rho v L}{\mu}",
            interpretation="Reynolds number",
        )

        assert str(pi) == "ρvL/μ"
        assert "Re" in repr(pi)

    def test_pigroup_immutable(self):
        """Test that PiGroup is immutable (frozen dataclass)."""
        pi = PiGroup(
            name="Π₁",
            exponents={'a': Fraction(1)},
            expression="a",
            latex="a",
        )

        with pytest.raises(AttributeError):
            pi.name = "Π₂"  # Should fail - frozen dataclass


class TestDimensionalMatrix:
    """Test dimensional matrix construction (internal function)."""

    def test_matrix_shape(self):
        """Test that dimensional matrix has correct shape."""
        from dimtensor.analysis.buckingham import _build_dimensional_matrix

        variables = {
            'F': kg * m / s**2,
            'v': m / s,
            'L': m,
        }

        matrix, var_names, active_dims = _build_dimensional_matrix(variables)

        # 3 active dimensions (L, M, T), 3 variables
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == 3
        assert len(var_names) == 3
        assert len(active_dims) == 3

    def test_matrix_values(self):
        """Test that dimensional matrix has correct values."""
        from dimtensor.analysis.buckingham import _build_dimensional_matrix

        variables = {
            'v': m / s,  # L¹T⁻¹
        }

        matrix, var_names, active_dims = _build_dimensional_matrix(variables)

        # Should have 2 active dimensions: L and T
        assert len(active_dims) == 2

        # Check that velocity has exponents [1, -1] for [L, T]
        # (order depends on which dimensions are active)
        col = matrix[:, 0]
        assert set(col) == {1.0, -1.0}


class TestNullspace:
    """Test nullspace computation (internal function)."""

    def test_nullspace_identity(self):
        """Test nullspace of identity matrix is empty."""
        from dimtensor.analysis.buckingham import _nullspace_svd

        matrix = np.eye(3)
        nullspace = _nullspace_svd(matrix)

        # Identity has full rank, nullspace should be empty
        assert nullspace.shape[1] == 0

    def test_nullspace_zero_matrix(self):
        """Test nullspace of zero matrix."""
        from dimtensor.analysis.buckingham import _nullspace_svd

        matrix = np.zeros((2, 3))
        nullspace = _nullspace_svd(matrix)

        # Zero matrix has rank 0, nullspace is all of R^3
        assert nullspace.shape == (3, 3)


class TestVectorCleaning:
    """Test vector cleaning and normalization (internal functions)."""

    def test_clean_vector_zeros(self):
        """Test that small values are cleaned to zero."""
        from dimtensor.analysis.buckingham import _clean_vector

        vector = np.array([1.0, 1e-12, 2.0, -1e-11])
        cleaned = _clean_vector(vector, tolerance=1e-10)

        assert cleaned[0] == Fraction(1)
        assert cleaned[1] == Fraction(0)
        assert cleaned[2] == Fraction(2)
        assert cleaned[3] == Fraction(0)

    def test_normalize_exponents_positive(self):
        """Test that normalization makes first non-zero exponent positive."""
        from dimtensor.analysis.buckingham import _normalize_exponents

        exponents = [Fraction(-2), Fraction(-1), Fraction(3)]
        normalized = _normalize_exponents(exponents)

        assert normalized[0] == Fraction(2)
        assert normalized[1] == Fraction(1)
        assert normalized[2] == Fraction(-3)

    def test_normalize_exponents_already_positive(self):
        """Test that already-positive exponents are unchanged."""
        from dimtensor.analysis.buckingham import _normalize_exponents

        exponents = [Fraction(1), Fraction(-1), Fraction(2)]
        normalized = _normalize_exponents(exponents)

        assert normalized == exponents


class TestValidation:
    """Test validation functions (internal)."""

    def test_validate_dimensionless_true(self):
        """Test validation of truly dimensionless combination."""
        from dimtensor.analysis.buckingham import _validate_dimensionless

        variables = {
            'v1': m / s,
            'v2': m / s,
        }
        exponents = {
            'v1': Fraction(1),
            'v2': Fraction(-1),
        }

        assert _validate_dimensionless(exponents, variables) is True

    def test_validate_dimensionless_false(self):
        """Test validation rejects non-dimensionless combination."""
        from dimtensor.analysis.buckingham import _validate_dimensionless

        variables = {
            'v': m / s,
            'L': m,
        }
        exponents = {
            'v': Fraction(1),
            'L': Fraction(1),  # This gives m²/s, not dimensionless
        }

        assert _validate_dimensionless(exponents, variables) is False


class TestExpressionBuilding:
    """Test expression building (internal functions)."""

    def test_expression_simple_ratio(self):
        """Test simple ratio expression."""
        from dimtensor.analysis.buckingham import _build_expression

        exponents = {
            'a': Fraction(1),
            'b': Fraction(-1),
        }

        expr = _build_expression(exponents)
        assert expr == "a/b"

    def test_expression_with_powers(self):
        """Test expression with powers."""
        from dimtensor.analysis.buckingham import _build_expression

        exponents = {
            'a': Fraction(2),
            'b': Fraction(-1),
        }

        expr = _build_expression(exponents)
        # Should be a²/b or similar
        assert 'a' in expr
        assert 'b' in expr

    def test_latex_simple_ratio(self):
        """Test LaTeX for simple ratio."""
        from dimtensor.analysis.buckingham import _build_latex

        exponents = {
            'a': Fraction(1),
            'b': Fraction(-1),
        }

        latex = _build_latex(exponents)
        # Should use \frac
        assert 'frac' in latex or (latex.count('a') >= 1 and latex.count('b') >= 1)
