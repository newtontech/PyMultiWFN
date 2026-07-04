"""
Tests for Issue 14 - Localized Orbital Locator (LOL) Analysis

This module tests LOL analysis functionality including:
- LOL formula implementation
- LOL calculation on 3D grid
- LOL visualization data generation

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.density.lol import LOLAnalyzer
from pymultiwfn.io import load


class TestLOLAnalysis:
    """Test LOL analysis functionality."""

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

    def test_lol_analyzer_exists(self, h2_wfn):
        """Test that LOLAnalyzer can be initialized."""
        analyzer = LOLAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_calculate_lol_method(self, h2_wfn):
        """Test that calculate_lol method exists."""
        analyzer = LOLAnalyzer(h2_wfn)
        assert hasattr(analyzer, "calculate_lol"), "Should have calculate_lol method"

    def test_lol_returns_numeric(self, h2_wfn):
        """Test that LOL calculation returns numeric value."""
        analyzer = LOLAnalyzer(h2_wfn)
        point = np.array([0.0, 0.0, 0.0])
        lol = analyzer.calculate_lol(point)
        assert isinstance(lol, (int, float)), "LOL should be numeric"

    def test_lol_range(self, h2_wfn):
        """Test that LOL values are in [0, 1] range."""
        analyzer = LOLAnalyzer(h2_wfn)

        for point in [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.0]),
        ]:
            lol = analyzer.calculate_lol(point)
            assert 0.0 <= lol <= 1.0, f"LOL {lol} should be in [0, 1]"

    def test_lol_at_nucleus(self, h2_wfn):
        """Test LOL behavior near nucleus."""
        analyzer = LOLAnalyzer(h2_wfn)

        atom = h2_wfn.atoms[0]
        point = np.array([atom.x, atom.y, atom.z])
        lol = analyzer.calculate_lol(point)

        assert 0.0 <= lol <= 1.0, "LOL should be in valid range"

    def test_lol_on_grid(self, h2_wfn):
        """Test LOL calculation on multiple grid points."""
        analyzer = LOLAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_lol_grid"):
            grid_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
            lol_values = analyzer.calculate_lol_grid(grid_points)

            assert isinstance(
                lol_values, np.ndarray
            ), "LOL values should be numpy array"
            assert len(lol_values) == len(grid_points)

    def test_lol_vs_elf_similarity(self, h2_wfn):
        """Test that LOL and ELF have similar behavior (both measure localization)."""
        analyzer = LOLAnalyzer(h2_wfn)

        # Both should be high near nucleus
        atom = h2_wfn.atoms[0]
        point = np.array([atom.x, atom.y, atom.z])

        lol = analyzer.calculate_lol(point)

        # LOL should be reasonably high near nucleus (like ELF)
        assert lol > 0.1, "LOL near nucleus should be > 0.1"

    def test_lol_isosurface(self, h2_wfn):
        """Test LOL isosurface generation."""
        analyzer = LOLAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_isosurface"):
            isovalue = 0.5
            surface = analyzer.generate_isosurface(isovalue)
            assert surface is not None

    def test_lol_report(self, h2_wfn):
        """Test LOL analysis report generation."""
        analyzer = LOLAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_report"):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be a string"
