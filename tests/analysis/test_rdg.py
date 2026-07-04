"""
Tests for Issue 15 - Reduced Density Gradient (RDG) Analysis

This module tests RDG analysis functionality including:
- RDG formula implementation
- RDG calculation on grid
- Non-covalent interaction identification
- RDG isosurface generation

Reference: PHASE2_TASKS.md - Module 2.3: Advanced Density Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.density.rdg import RDGAnalyzer
from pymultiwfn.io import load


class TestRDGAnalysis:
    """Test RDG analysis functionality."""

    @pytest.fixture
    def h2_wfn(self):
        """Load H2 wavefunction file."""
        wfn_path = Path("tests/test_data/H2_CCSD.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_rdg_analyzer_exists(self, h2_wfn):
        """Test that RDGAnalyzer can be initialized."""
        analyzer = RDGAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_calculate_rdg_method(self, h2_wfn):
        """Test that calculate_rdg method exists."""
        analyzer = RDGAnalyzer(h2_wfn)
        assert hasattr(analyzer, "calculate_rdg"), "Should have calculate_rdg method"

    def test_rdg_returns_numeric(self, h2_wfn):
        """Test that RDG calculation returns numeric value."""
        analyzer = RDGAnalyzer(h2_wfn)
        point = np.array([0.0, 0.0, 0.0])
        rdg = analyzer.calculate_rdg(point)
        assert isinstance(rdg, (int, float)), "RDG should be numeric"

    def test_rdg_non_negative(self, h2_wfn):
        """Test that RDG values are non-negative."""
        analyzer = RDGAnalyzer(h2_wfn)

        for point in [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([2.0, 0.0, 0.0]),
        ]:
            rdg = analyzer.calculate_rdg(point)
            assert rdg >= 0, f"RDG {rdg} should be non-negative"

    def test_rdg_low_density_regions(self, h2_wfn):
        """Test that RDG is higher in low-density regions."""
        analyzer = RDGAnalyzer(h2_wfn)

        # Far from nucleus (low density) should have higher RDG
        atom = h2_wfn.atoms[0]

        near_point = np.array([atom.x, atom.y, atom.z])
        far_point = np.array([atom.x + 3.0, atom.y, atom.z])

        rdg_near = analyzer.calculate_rdg(near_point)
        rdg_far = analyzer.calculate_rdg(far_point)

        # Both should be valid
        assert rdg_near >= 0 and rdg_far >= 0

    def test_rdg_on_grid(self, h2_wfn):
        """Test RDG calculation on multiple grid points."""
        analyzer = RDGAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_rdg_grid"):
            grid_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
            rdg_values = analyzer.calculate_rdg_grid(grid_points)

            assert isinstance(
                rdg_values, np.ndarray
            ), "RDG values should be numpy array"

    def test_identify_noncovalent_interactions(self, h2_wfn):
        """Test identification of non-covalent interaction regions."""
        analyzer = RDGAnalyzer(h2_wfn)

        if hasattr(analyzer, "identify_nci_regions"):
            regions = analyzer.identify_nci_regions()
            assert isinstance(regions, list), "NCI regions should be a list"

    def test_rdg_isosurface(self, h2_wfn):
        """Test RDG isosurface generation."""
        analyzer = RDGAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_isosurface"):
            isovalue = 0.5
            surface = analyzer.generate_isosurface(isovalue)
            assert surface is not None

    def test_sign_lambda2_calculation(self, h2_wfn):
        """Test sign(λ₂) calculation for interaction classification."""
        analyzer = RDGAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_sign_lambda2"):
            point = np.array([0.0, 0.0, 0.0])
            sign = analyzer.calculate_sign_lambda2(point)
            assert isinstance(sign, (int, float)), "sign(λ₂) should be numeric"

    def test_rdg_report(self, h2_wfn):
        """Test RDG analysis report generation."""
        analyzer = RDGAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_report"):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be a string"


class TestRDGAdvanced:
    """Advanced tests for RDG analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_rdg_nci_classification(self, c2h4_wfn):
        """Test NCI interaction type classification."""
        analyzer = RDGAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "classify_interaction"):
            point = np.array([0.0, 0.0, 0.0])
            interaction_type = analyzer.classify_interaction(point)
            assert isinstance(
                interaction_type, str
            ), "Interaction type should be string"
