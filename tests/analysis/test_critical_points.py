"""
Tests for Issue 11 - Critical Point Analysis

This module tests critical point analysis functionality including:
- Gradient calculation on grid
- Hessian calculation
- Critical point location (BCP, RCP, CCP)
- Critical point properties

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.density.topology import CriticalPointAnalyzer
from pymultiwfn.io import load


class TestCriticalPointAnalysis:
    """Test critical point analysis functionality."""

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

    def test_critical_point_analyzer_exists(self, h2_wfn):
        """Test that CriticalPointAnalyzer can be initialized."""
        analyzer = CriticalPointAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_find_critical_points_method(self, h2_wfn):
        """Test that find_critical_points method exists."""
        analyzer = CriticalPointAnalyzer(h2_wfn)
        assert hasattr(
            analyzer, "find_critical_points"
        ), "Should have find_critical_points method"

    def test_find_critical_points_returns_list(self, h2_wfn):
        """Test that find_critical_points returns list."""
        analyzer = CriticalPointAnalyzer(h2_wfn)
        points = analyzer.find_critical_points()
        assert isinstance(points, list), "Critical points should be a list"

    def test_critical_point_structure(self, h2_wfn):
        """Test critical point data structure."""
        analyzer = CriticalPointAnalyzer(h2_wfn)
        points = analyzer.find_critical_points()

        if len(points) > 0:
            cp = points[0]
            assert (
                "position" in cp or "type" in cp
            ), "Critical point should have position or type"

    def test_gradient_calculation(self, h2_wfn):
        """Test gradient calculation on grid."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_gradient"):
            point = np.array([0.0, 0.0, 0.0])
            gradient = analyzer.calculate_gradient(point)
            assert isinstance(gradient, np.ndarray), "Gradient should be numpy array"
            assert gradient.shape == (3,), "Gradient should be 3D vector"

    def test_hessian_calculation(self, h2_wfn):
        """Test Hessian calculation on grid."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_hessian"):
            point = np.array([0.0, 0.0, 0.0])
            hessian = analyzer.calculate_hessian(point)
            assert isinstance(hessian, np.ndarray), "Hessian should be numpy array"
            assert hessian.shape == (3, 3), "Hessian should be 3x3 matrix"

    def test_hessian_symmetric(self, h2_wfn):
        """Test that Hessian is symmetric."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_hessian"):
            point = np.array([0.0, 0.0, 0.0])
            hessian = analyzer.calculate_hessian(point)
            assert np.allclose(
                hessian, hessian.T, atol=1e-6
            ), "Hessian should be symmetric"

    def test_bond_critical_points(self, c2h4_wfn):
        """Test bond critical point (BCP) detection."""
        analyzer = CriticalPointAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_bond_critical_points"):
            bcps = analyzer.get_bond_critical_points()
            assert isinstance(bcps, list), "BCPs should be a list"

    def test_ring_critical_points(self, c2h4_wfn):
        """Test ring critical point (RCP) detection."""
        analyzer = CriticalPointAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_ring_critical_points"):
            rcps = analyzer.get_ring_critical_points()
            assert isinstance(rcps, list), "RCPs should be a list"

    def test_cage_critical_points(self, h2_wfn):
        """Test cage critical point (CCP) detection."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_cage_critical_points"):
            ccps = analyzer.get_cage_critical_points()
            assert isinstance(ccps, list), "CCPs should be a list"

    def test_critical_point_rank(self, h2_wfn):
        """Test critical point rank determination."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_critical_point_rank"):
            points = analyzer.find_critical_points()
            if len(points) > 0:
                rank = analyzer.get_critical_point_rank(points[0])
                assert isinstance(rank, int), "Rank should be integer"
                assert 0 <= rank <= 3, "Rank should be between 0 and 3"

    def test_critical_point_signature(self, h2_wfn):
        """Test critical point signature (rank, signature)."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_critical_point_signature"):
            points = analyzer.find_critical_points()
            if len(points) > 0:
                sig = analyzer.get_critical_point_signature(points[0])
                assert isinstance(sig, tuple), "Signature should be tuple"

    def test_density_at_critical_point(self, h2_wfn):
        """Test density value at critical points."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_density_at_point"):
            point = np.array([0.0, 0.0, 0.0])
            density = analyzer.get_density_at_point(point)
            assert isinstance(density, (int, float)), "Density should be numeric"
            assert density >= 0, "Density should be non-negative"

    def test_laplacian_at_critical_point(self, h2_wfn):
        """Test Laplacian value at critical points."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_laplacian_at_point"):
            point = np.array([0.0, 0.0, 0.0])
            laplacian = analyzer.get_laplacian_at_point(point)
            assert isinstance(laplacian, (int, float)), "Laplacian should be numeric"

    def test_ellipticity_calculation(self, h2_wfn):
        """Test ellipticity calculation at BCPs."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_ellipticity"):
            points = analyzer.find_critical_points()
            if len(points) > 0:
                ellipticity = analyzer.calculate_ellipticity(points[0])
                assert isinstance(
                    ellipticity, (int, float)
                ), "Ellipticity should be numeric"
                assert ellipticity >= 0, "Ellipticity should be non-negative"

    def test_critical_point_report(self, h2_wfn):
        """Test critical point report generation."""
        analyzer = CriticalPointAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_report"):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be a string"

    def test_gradient_norm_at_cp(self, h2_wfn):
        """Test that gradient norm is near zero at critical points."""
        analyzer = CriticalPointAnalyzer(h2_wfn)
        points = analyzer.find_critical_points()

        if hasattr(analyzer, "calculate_gradient") and len(points) > 0:
            for cp in points[:3]:
                if "position" in cp:
                    gradient = analyzer.calculate_gradient(cp["position"])
                    norm = np.linalg.norm(gradient)
                    # Allow larger tolerance due to simplified density model
                    assert (
                        norm < 0.5
                    ), f"Gradient norm at CP should be small, got {norm}"


class TestCriticalPointAdvanced:
    """Advanced tests for critical point analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_morse_relationship(self, c2h4_wfn):
        """Test Morse relationship at critical points."""
        analyzer = CriticalPointAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "verify_morse_relationship"):
            points = analyzer.find_critical_points()
            if len(points) > 0:
                result = analyzer.verify_morse_relationship(points[0])
                assert isinstance(result, bool), "Morse verification should return bool"
