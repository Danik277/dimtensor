"""Tests for dimensional inference."""

import pytest

from dimtensor.core.dimensions import Dimension
from dimtensor.inference import (
    infer_dimension,
    get_matching_patterns,
    InferenceResult,
    VARIABLE_PATTERNS,
)
from dimtensor.inference.heuristics import suggest_dimension


class TestInferDimension:
    """Test dimension inference from variable names."""

    def test_exact_match_velocity(self):
        """Velocity should be inferred as L/T."""
        result = infer_dimension("velocity")
        assert result is not None
        assert result.dimension == Dimension(length=1, time=-1)
        assert result.confidence >= 0.9
        assert result.source == "exact"

    def test_exact_match_force(self):
        """Force should be inferred as MLT⁻²."""
        result = infer_dimension("force")
        assert result is not None
        assert result.dimension == Dimension(length=1, mass=1, time=-2)
        assert result.confidence >= 0.9

    def test_exact_match_energy(self):
        """Energy should be inferred as ML²T⁻²."""
        result = infer_dimension("energy")
        assert result is not None
        assert result.dimension == Dimension(length=2, mass=1, time=-2)

    def test_exact_match_pressure(self):
        """Pressure should be inferred as M/(LT²)."""
        result = infer_dimension("pressure")
        assert result is not None
        assert result.dimension == Dimension(length=-1, mass=1, time=-2)

    def test_case_insensitive(self):
        """Inference should be case insensitive."""
        result1 = infer_dimension("Velocity")
        result2 = infer_dimension("VELOCITY")
        result3 = infer_dimension("velocity")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1.dimension == result2.dimension == result3.dimension

    def test_no_match_unknown(self):
        """Unknown variable names should return None."""
        result = infer_dimension("foo_bar_baz")
        assert result is None

    def test_no_match_too_short(self):
        """Very short names might not match."""
        result = infer_dimension("xy")
        assert result is None

    def test_prefixed_variable(self):
        """Prefixed variables should still be recognized."""
        result = infer_dimension("initial_velocity")
        assert result is not None
        assert result.dimension == Dimension(length=1, time=-1)
        assert result.confidence < 0.9  # Lower confidence for prefixed

    def test_final_prefix(self):
        """Final prefix should work."""
        result = infer_dimension("final_position")
        assert result is not None
        assert result.dimension == Dimension(length=1)

    def test_component_suffix(self):
        """Component suffixes should be handled."""
        result = infer_dimension("velocity_x")
        assert result is not None
        assert result.dimension == Dimension(length=1, time=-1)

    def test_unit_suffix(self):
        """Unit suffixes should provide dimension."""
        result = infer_dimension("distance_m")
        assert result is not None
        assert result.dimension == Dimension(length=1)
        assert result.source == "suffix"

    def test_compound_unit_suffix(self):
        """Compound unit suffixes should work."""
        result = infer_dimension("speed_m_per_s")
        assert result is not None
        assert result.dimension == Dimension(length=1, time=-1)

    def test_minimum_confidence_filter(self):
        """Should filter by minimum confidence."""
        # This should match with low confidence
        result = infer_dimension("the_velocity_data", min_confidence=0.8)
        assert result is None  # Filtered out

        result = infer_dimension("the_velocity_data", min_confidence=0.4)
        assert result is not None  # Not filtered


class TestGetMatchingPatterns:
    """Test getting all matching patterns."""

    def test_multiple_matches(self):
        """Should return multiple matching patterns."""
        results = get_matching_patterns("initial_velocity_x")
        assert len(results) >= 1

    def test_sorted_by_confidence(self):
        """Results should be sorted by confidence."""
        results = get_matching_patterns("velocity")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_empty_for_unknown(self):
        """Should return empty list for unknown names."""
        results = get_matching_patterns("completely_unknown_name_xyz")
        assert results == []


class TestInferenceResult:
    """Test InferenceResult named tuple."""

    def test_named_tuple_unpacking(self):
        """Should be unpackable as named tuple."""
        result = InferenceResult(
            dimension=Dimension(length=1),
            confidence=0.9,
            pattern="distance",
            source="exact",
        )

        dim, conf, pat, src = result
        assert dim == Dimension(length=1)
        assert conf == 0.9
        assert pat == "distance"
        assert src == "exact"


class TestSuggestDimension:
    """Test human-readable suggestions."""

    def test_known_variable(self):
        """Should provide meaningful suggestion for known variable."""
        suggestion = suggest_dimension("velocity")
        assert "velocity" in suggestion
        assert "L" in suggestion or "length" in suggestion.lower()
        assert "confidence" in suggestion

    def test_unknown_variable(self):
        """Should indicate no inference for unknown variable."""
        suggestion = suggest_dimension("xyzzy_unknown")
        assert "no dimension" in suggestion.lower() or "no inference" in suggestion.lower()


class TestVariablePatterns:
    """Test the variable patterns dictionary."""

    def test_mechanics_variables(self):
        """Mechanics variables should be present."""
        assert "velocity" in VARIABLE_PATTERNS
        assert "acceleration" in VARIABLE_PATTERNS
        assert "force" in VARIABLE_PATTERNS
        assert "momentum" in VARIABLE_PATTERNS
        assert "energy" in VARIABLE_PATTERNS

    def test_geometry_variables(self):
        """Geometry variables should be present."""
        assert "distance" in VARIABLE_PATTERNS
        assert "area" in VARIABLE_PATTERNS
        assert "volume" in VARIABLE_PATTERNS

    def test_time_variables(self):
        """Time variables should be present."""
        assert "time" in VARIABLE_PATTERNS
        assert "frequency" in VARIABLE_PATTERNS

    def test_temperature_variables(self):
        """Temperature variables should be present."""
        assert "temperature" in VARIABLE_PATTERNS

    def test_electromagnetics_variables(self):
        """Electromagnetics variables should be present."""
        assert "current" in VARIABLE_PATTERNS
        assert "voltage" in VARIABLE_PATTERNS
        assert "resistance" in VARIABLE_PATTERNS


class TestPhysicsIntegration:
    """Test integration with physics calculations."""

    def test_kinetic_energy_formula(self):
        """Test inference for kinetic energy formula variables."""
        # KE = 0.5 * m * v^2
        m_result = infer_dimension("mass")
        v_result = infer_dimension("velocity")
        ke_result = infer_dimension("kinetic_energy")

        assert m_result is not None
        assert v_result is not None
        assert ke_result is not None

        # Calculate expected dimension for KE from m*v^2
        m_dim = m_result.dimension
        v_dim = v_result.dimension
        expected_ke = m_dim * v_dim * v_dim

        assert ke_result.dimension == expected_ke

    def test_force_formula(self):
        """Test inference for F = ma variables."""
        m_result = infer_dimension("mass")
        a_result = infer_dimension("acceleration")
        f_result = infer_dimension("force")

        assert m_result is not None
        assert a_result is not None
        assert f_result is not None

        # F = m * a
        expected_f = m_result.dimension * a_result.dimension
        assert f_result.dimension == expected_f

    def test_pressure_formula(self):
        """Test inference for P = F/A variables."""
        f_result = infer_dimension("force")
        a_result = infer_dimension("area")
        p_result = infer_dimension("pressure")

        assert f_result is not None
        assert a_result is not None
        assert p_result is not None

        # P = F / A
        expected_p = f_result.dimension / a_result.dimension
        assert p_result.dimension == expected_p
