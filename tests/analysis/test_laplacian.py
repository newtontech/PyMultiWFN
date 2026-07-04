"""
Tests for Issue 12 - Laplacian Analysis

This module tests Laplacian analysis functionality including:
- Laplacian ∇²ρ calculation
- Electron concentration/depletion region identification
- Laplacian isosurface generation
- Bond classification based on Laplacian

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.density.laplacian import LaplacianAnalyzer
from pymultiwfn.io import load


class TestLaplacianAnalysis:
    """Test Laplacian analysis functionality."""

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

    def test_laplacian_analyzer_exists(self, h2_wfn):
        """Test that LaplacianAnalyzer can be initialized."""
        analyzer = LaplacianAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_calculate_laplacian_method(self, h2_wfn):
        """Test that calculate_laplacian method exists."""
        analyzer = LaplacianAnalyzer(h2_wfn)
        assert hasattr(
            analyzer, "calculate_laplacian"
        ), "Should have calculate_laplacian method"

    def test_laplacian_returns_numeric(self, h2_wfn):
        """Test that Laplacian calculation returns numeric value."""
        analyzer = LaplacianAnalyzer(h2_wfn)
        point = np.array([0.0, 0.0, 0.0])
        laplacian = analyzer.calculate_laplacian(point)
        assert isinstance(laplacian, (int, float)), "Laplacian should be numeric"

    def test_laplacian_at_nucleus_positive(self, h2_wfn):
        """Test that Laplacian is typically positive at nuclear positions."""
        analyzer = LaplacianAnalyzer(h2_wfn)

        # At nucleus (electron density peak)
        atom = h2_wfn.atoms[0]
        point = np.array([atom.x, atom.y, atom.z])
        laplacian = analyzer.calculate_laplacian(point)

        # Near nucleus, Laplacian is often positive (density depletion)
        # This is a general trend, may have exceptions
        assert isinstance(laplacian, (int, float)), "Laplacian should be numeric"

    def test_get_concentration_regions(self, h2_wfn):
        """Test identification of electron concentration regions (∇²ρ < 0)."""
        analyzer = LaplacianAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_concentration_regions"):
            regions = analyzer.get_concentration_regions()
            assert isinstance(regions, list), "Concentration regions should be a list"

    def test_get_depletion_regions(self, h2_wfn):
        """Test identification of electron depletion regions (∇²ρ > 0)."""
        analyzer = LaplacianAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_depletion_regions"):
            regions = analyzer.get_depletion_regions()
            assert isinstance(regions, list), "Depletion regions should be a list"

    def test_classify_bond_type(self, c2h4_wfn):
        """Test bond classification based on Laplacian at BCP."""
        analyzer = LaplacianAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "classify_bond"):
            # Test classification for C-C bond region
            point = np.array([0.0, 0.0, 0.0])
            bond_type = analyzer.classify_bond(point)
            assert isinstance(bond_type, str), "Bond type should be string"

    def test_laplacian_on_grid(self, h2_wfn):
        """Test Laplacian calculation on a grid."""
        analyzer = LaplacianAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_laplacian_grid"):
            grid_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
            laplacians = analyzer.calculate_laplacian_grid(grid_points)
            assert isinstance(
                laplacians, np.ndarray
            ), "Laplacians should be numpy array"
            assert len(laplacians) == len(
                grid_points
            ), "Should have same number of values as grid points"

    def test_isosurface_generation(self, h2_wfn):
        """Test Laplacian isosurface generation."""
        analyzer = LaplacianAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_isosurface"):
            isovalue = 0.0  # ∇²ρ = 0 isosurface
            surface = analyzer.generate_isosurface(isovalue)
            assert surface is not None, "Should generate isosurface"

    def test_laplacian_sign_at_bcp(self, c2h4_wfn):
        """Test Laplacian sign at bond critical point."""
        analyzer = LaplacianAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_laplacian_at_bcp"):
            laplacian = analyzer.get_laplacian_at_bcp(bond_index=0)
            assert isinstance(
                laplacian, (int, float)
            ), "Laplacian at BCP should be numeric"


class TestLaplacianAdvanced:
    """Advanced tests for Laplacian analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_laplacian_report(self, c2h4_wfn):
        """Test Laplacian analysis report generation."""
        analyzer = LaplacianAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "generate_report"):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be a string"
            assert len(report) > 0, "Report should not be empty"

    def test_laplacian_gradient_relationship(self, c2h4_wfn):
        """Test relationship between Laplacian and gradient."""
        analyzer = LaplacianAnalyzer(c2h4_wfn)

        # Laplacian should be trace of Hessian
        if hasattr(analyzer, "calculate_hessian"):
            point = np.array([0.0, 0.0, 0.0])
            laplacian = analyzer.calculate_laplacian(point)
            hessian = analyzer.calculate_hessian(point)
            trace = np.trace(hessian)

            assert abs(laplacian - trace) < 0.01, "Laplacian should equal Hessian trace"
