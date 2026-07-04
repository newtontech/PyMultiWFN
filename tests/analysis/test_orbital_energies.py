"""
Tests for Issue 6 - Orbital Energy Analysis

This module tests the orbital energy analysis functionality including:
- MO energy extraction from wavefunction files
- HOMO-LUMO gap calculation
- Fermi level calculation
- Orbital energy diagrams

Reference: PHASE2_TASKS.md - Task 2.1.1: MO Energy Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.io import load
from pymultiwfn.orbitals import OrbitalsAnalyzer


class TestOrbitalEnergyExtraction:
    """Test extraction of MO energies from wavefunction files."""

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

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_load_orbital_analyzer(self, h2_wfn):
        """Test that OrbitalsAnalyzer can be initialized."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_extract_alpha_energies(self, h2_wfn):
        """Test extraction of alpha orbital energies."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        energies = analyzer.alpha_energies

        assert energies is not None
        assert isinstance(energies, np.ndarray)
        assert len(energies) > 0
        # Energies should be in Hartree (typically -2 to +2 for valence orbitals)
        assert np.all(np.abs(energies) < 100)

    def test_energy_sorting(self, h2_wfn):
        """Test that orbital energies are sorted (ascending)."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        energies = analyzer.alpha_energies

        # Energies should be sorted in ascending order
        assert np.all(energies[:-1] <= energies[1:])

    def test_extract_beta_energies_restricted(self, h2_wfn):
        """Test beta energies for restricted calculations (should be None or same as alpha)."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        # For restricted calculations, beta energies might be None or identical to alpha
        if analyzer.beta_energies is not None:
            np.testing.assert_array_almost_equal(
                analyzer.alpha_energies, analyzer.beta_energies, decimal=6
            )


class TestHOMOLUMOAnalysis:
    """Test HOMO-LUMO gap calculation."""

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

    def test_homo_identification(self, h2_wfn):
        """Test identification of HOMO (highest occupied molecular orbital)."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        # HOMO index should be a positive integer
        assert analyzer.homo_index >= 0

        # HOMO energy should be defined
        assert analyzer.homo_energy is not None
        assert isinstance(analyzer.homo_energy, float)

    def test_lumo_identification(self, h2_wfn):
        """Test identification of LUMO (lowest unoccupied molecular orbital)."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        # LUMO index should be HOMO + 1
        assert analyzer.lumo_index == analyzer.homo_index + 1

        # LUMO energy should be defined
        assert analyzer.lumo_energy is not None
        assert isinstance(analyzer.lumo_energy, float)

    def test_homo_lumo_gap(self, h2_wfn):
        """Test HOMO-LUMO gap calculation."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        # Gap should be calculated (may be 0 if LUMO doesn't exist in file)
        gap = analyzer.gap
        assert isinstance(gap, float)
        assert gap >= 0

        # If LUMO exists, gap should equal LUMO - HOMO
        if analyzer.lumo_index < len(analyzer.alpha_energies):
            expected_gap = analyzer.lumo_energy - analyzer.homo_energy
            assert abs(gap - expected_gap) < 1e-10

    def test_homo_lumo_gap_with_c2h4(self, c2h4_wfn):
        """Test HOMO-LUMO gap with C2H4 molecule."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # C2H4 has well-defined HOMO
        homo_energy = analyzer.homo_energy
        assert homo_energy < 0, "HOMO should be negative for bound molecule"

        # Check if LUMO exists in the orbital range
        if analyzer.lumo_index < len(analyzer.alpha_energies):
            gap = analyzer.gap
            assert gap > 0, "Gap should be positive when LUMO exists"


class TestFermiLevel:
    """Test Fermi level calculation."""

    @pytest.fixture
    def h2_wfn(self):
        """Load H2 wavefunction file."""
        wfn_path = Path("tests/test_data/H2_CCSD.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_fermi_level_calculation(self, h2_wfn):
        """Test Fermi level calculation."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        # Fermi level should be defined
        fermi = analyzer.fermi_level
        assert fermi is not None
        assert isinstance(fermi, float)

    def test_fermi_level_position(self, h2_wfn):
        """Test that Fermi level is between HOMO and LUMO."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        fermi = analyzer.fermi_level
        homo = analyzer.homo_energy
        lumo = analyzer.lumo_energy

        # Fermi level should be between HOMO and LUMO
        assert (
            homo <= fermi <= lumo
        ), f"Fermi level {fermi} should be between HOMO {homo} and LUMO {lumo}"


class TestOrbitalEnergyDiagram:
    """Test orbital energy diagram generation."""

    @pytest.fixture
    def h2_wfn(self):
        """Load H2 wavefunction file."""
        wfn_path = Path("tests/test_data/H2_CCSD.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_energy_diagram_data(self, h2_wfn):
        """Test generation of energy diagram data."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        # Get energy diagram data
        diagram_data = analyzer.get_energy_diagram(n_orbitals=5)

        assert diagram_data is not None
        assert isinstance(diagram_data, dict)

        # Should contain energies and occupations
        assert "energies" in diagram_data
        assert "occupations" in diagram_data

        # Should have requested number of orbitals around Fermi level
        assert len(diagram_data["energies"]) <= 5

    def test_occupations_in_diagram(self, h2_wfn):
        """Test that occupations are included in energy diagram."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        diagram_data = analyzer.get_energy_diagram(n_orbitals=5)
        occupations = diagram_data["occupations"]

        # Occupations should be between 0 and 2 (for spin-restricted)
        assert np.all(occupations >= 0)
        assert np.all(occupations <= 2)


class TestMultiwfnConsistency:
    """Test consistency with Multiwfn reference data."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_homo_energy_reasonable(self, c2h4_wfn):
        """Test that HOMO energy is in reasonable range."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # HOMO for C2H4 should be negative (bound state)
        homo = analyzer.homo_energy
        assert homo < 0, f"HOMO energy {homo} should be negative for bound molecule"
        assert homo > -2.0, f"HOMO energy {homo} seems too low for C2H4"

    def test_orbital_energy_sorting(self, c2h4_wfn):
        """Test that orbital energies are properly sorted."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        energies = analyzer.alpha_energies

        # Energies should be in ascending order
        for i in range(len(energies) - 1):
            assert (
                energies[i] <= energies[i + 1]
            ), f"Orbital energies not sorted: MO{i}={energies[i]}, MO{i+1}={energies[i+1]}"
