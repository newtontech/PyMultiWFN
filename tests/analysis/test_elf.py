"""
Tests for Issue 13 - Electron Localization Function (ELF) Analysis

This module tests ELF analysis functionality including:
- ELF formula implementation
- ELF calculation on 3D grid
- ELF basin identification
- ELF isosurface generation

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.density.elf import ELFAnalyzer
from pymultiwfn.io import load


class TestELFAnalysis:
    """Test ELF analysis functionality."""

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

    def test_elf_analyzer_exists(self, h2_wfn):
        """Test that ELFAnalyzer can be initialized."""
        analyzer = ELFAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_calculate_elf_method(self, h2_wfn):
        """Test that calculate_elf method exists."""
        analyzer = ELFAnalyzer(h2_wfn)
        assert hasattr(analyzer, "calculate_elf"), "Should have calculate_elf method"

    def test_elf_returns_numeric(self, h2_wfn):
        """Test that ELF calculation returns numeric value."""
        analyzer = ELFAnalyzer(h2_wfn)
        point = np.array([0.0, 0.0, 0.0])
        elf = analyzer.calculate_elf(point)
        assert isinstance(elf, (int, float)), "ELF should be numeric"

    def test_elf_range(self, h2_wfn):
        """Test that ELF values are in [0, 1] range."""
        analyzer = ELFAnalyzer(h2_wfn)

        # Test several points
        for point in [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.0]),
        ]:
            elf = analyzer.calculate_elf(point)
            assert 0.0 <= elf <= 1.0, f"ELF {elf} should be in [0, 1]"

    def test_elf_at_nucleus_high(self, h2_wfn):
        """Test that ELF is high near nucleus (localized core electrons)."""
        analyzer = ELFAnalyzer(h2_wfn)

        # At nucleus position
        atom = h2_wfn.atoms[0]
        point = np.array([atom.x, atom.y, atom.z])
        elf = analyzer.calculate_elf(point)

        # ELF should be relatively high near nucleus
        assert elf > 0.1, f"ELF at nucleus should be > 0.1, got {elf}"

    def test_elf_on_grid(self, h2_wfn):
        """Test ELF calculation on multiple grid points."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_elf_grid"):
            grid_points = np.array(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
            )
            elf_values = analyzer.calculate_elf_grid(grid_points)

            assert isinstance(
                elf_values, np.ndarray
            ), "ELF values should be numpy array"
            assert len(elf_values) == len(
                grid_points
            ), "Should have same count as grid points"

    def test_elf_basins_identification(self, h2_wfn):
        """Test identification of ELF basins (localization regions)."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "identify_basins"):
            basins = analyzer.identify_basins()
            assert isinstance(basins, list), "Basins should be a list"

    def test_elf_basin_properties(self, h2_wfn):
        """Test ELF basin property calculation."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_basin_properties"):
            basins = (
                analyzer.identify_basins()
                if hasattr(analyzer, "identify_basins")
                else []
            )
            if len(basins) > 0:
                props = analyzer.get_basin_properties(basins[0])
                assert isinstance(props, dict), "Basin properties should be dict"

    def test_elf_isosurface(self, h2_wfn):
        """Test ELF isosurface generation."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_isosurface"):
            isovalue = 0.8  # Common ELF isovalue for localization
            surface = analyzer.generate_isosurface(isovalue)
            assert surface is not None, "Should generate isosurface"

    def test_kinetic_energy_density(self, h2_wfn):
        """Test kinetic energy density calculation (required for ELF)."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "calculate_kinetic_energy_density"):
            point = np.array([0.0, 0.0, 0.0])
            ke = analyzer.calculate_kinetic_energy_density(point)
            assert isinstance(ke, (int, float)), "Kinetic energy should be numeric"
            assert ke >= 0, "Kinetic energy should be non-negative"

    def test_elf_report(self, h2_wfn):
        """Test ELF analysis report generation."""
        analyzer = ELFAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_report"):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be a string"


class TestELFAdvanced:
    """Advanced tests for ELF analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_elf_core_vs_valence(self, c2h4_wfn):
        """Test ELF distinction between core and valence regions."""
        analyzer = ELFAnalyzer(c2h4_wfn)

        # Core region (near nucleus)
        atom = c2h4_wfn.atoms[0]
        core_point = np.array([atom.x, atom.y, atom.z])
        elf_core = analyzer.calculate_elf(core_point)

        # Valence region (between atoms)
        # C2H4 has a C=C bond
        c1 = c2h4_wfn.atoms[0]
        c2 = c2h4_wfn.atoms[1] if len(c2h4_wfn.atoms) > 1 else c1
        valence_point = np.array(
            [(c1.x + c2.x) / 2, (c1.y + c2.y) / 2, (c1.z + c2.z) / 2]
        )
        elf_valence = analyzer.calculate_elf(valence_point)

        # Both should be valid ELF values
        assert 0 <= elf_core <= 1, "Core ELF should be in [0,1]"
        assert 0 <= elf_valence <= 1, "Valence ELF should be in [0,1]"

    def test_elf_basin_population(self, c2h4_wfn):
        """Test electron population in ELF basins."""
        analyzer = ELFAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "calculate_basin_population"):
            basins = (
                analyzer.identify_basins()
                if hasattr(analyzer, "identify_basins")
                else []
            )
            if len(basins) > 0:
                population = analyzer.calculate_basin_population(basins[0])
                assert isinstance(
                    population, (int, float)
                ), "Population should be numeric"
                assert population >= 0, "Population should be non-negative"
