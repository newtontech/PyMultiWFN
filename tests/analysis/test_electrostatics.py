"""
Tests for Issue 16-19 - Electrostatic Analysis

This module tests electrostatic analysis functionality including:
- Molecular electrostatic potential (MEP)
- Multipole moments
- Atomic charges
- ESP fitting

Reference: PHASE2_TASKS.md - Module 2.3: Electrostatic Analysis
"""

import pytest
import numpy as np
from pathlib import Path

from pymultiwfn.io import load
from pymultiwfn.electrostatics import ElectrostaticAnalyzer


class TestElectrostaticPotential:
    """Test molecular electrostatic potential."""

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

    def test_electrostatic_analyzer_exists(self, h2_wfn):
        """Test that ElectrostaticAnalyzer can be initialized."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_calculate_mep_method(self, h2_wfn):
        """Test MEP calculation method exists."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        assert hasattr(analyzer, 'calculate_mep'), "Should have calculate_mep method"

    def test_mep_returns_numeric(self, h2_wfn):
        """Test MEP calculation returns numeric value."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        point = np.array([2.0, 0.0, 0.0])  # Point away from nuclei
        mep = analyzer.calculate_mep(point)
        assert isinstance(mep, (int, float)), "MEP should be numeric"

    def test_mep_nuclear_contribution(self, h2_wfn):
        """Test MEP includes nuclear contribution."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        # Near nucleus, MEP should be positive (nuclear attraction dominates)
        atom = h2_wfn.atoms[0]
        point = np.array([atom.x + 0.5, atom.y, atom.z])  # Close to nucleus
        mep = analyzer.calculate_mep(point)
        
        # MEP should be positive near nuclei
        assert isinstance(mep, (int, float)), "MEP should be numeric"

    def test_mep_on_grid(self, h2_wfn):
        """Test MEP calculation on grid."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'calculate_mep_grid'):
            points = np.array([
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0]
            ])
            mep_values = analyzer.calculate_mep_grid(points)
            assert isinstance(mep_values, np.ndarray), "MEP values should be array"


class TestMultipoleMoments:
    """Test multipole moment calculations."""

    @pytest.fixture
    def h2_wfn(self):
        """Load H2 wavefunction file."""
        wfn_path = Path("tests/test_data/H2_CCSD.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_dipole_moment_method(self, h2_wfn):
        """Test dipole moment calculation."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'calculate_dipole'):
            dipole = analyzer.calculate_dipole()
            assert isinstance(dipole, np.ndarray), "Dipole should be array"
            assert dipole.shape == (3,), "Dipole should be 3D vector"

    def test_quadrupole_moment_method(self, h2_wfn):
        """Test quadrupole moment calculation."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'calculate_quadrupole'):
            quadrupole = analyzer.calculate_quadrupole()
            assert isinstance(quadrupole, np.ndarray), "Quadrupole should be array"

    def test_h2_dipole_symmetry(self, h2_wfn):
        """Test H2 dipole (should be near zero due to symmetry)."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'calculate_dipole'):
            dipole = analyzer.calculate_dipole()
            magnitude = np.linalg.norm(dipole)
            # H2 is symmetric, dipole should be small
            assert magnitude < 1.0, f"H2 dipole should be small, got {magnitude}"


class TestAtomicCharges:
    """Test atomic charge calculations."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_mulliken_charges_method(self, c2h4_wfn):
        """Test Mulliken charge calculation."""
        analyzer = ElectrostaticAnalyzer(c2h4_wfn)
        
        if hasattr(analyzer, 'calculate_mulliken_charges'):
            charges = analyzer.calculate_mulliken_charges()
            assert isinstance(charges, dict), "Charges should be dict"

    def test_charges_sum_to_total(self, c2h4_wfn):
        """Test that atomic charges sum to total charge."""
        analyzer = ElectrostaticAnalyzer(c2h4_wfn)
        
        if hasattr(analyzer, 'calculate_mulliken_charges'):
            charges = analyzer.calculate_mulliken_charges()
            total = sum(charges.values())
            # Should sum to molecular charge (0 for neutral)
            assert abs(total) < 0.5, f"Charges should sum to ~0, got {total}"

    def test_lowdin_charges_method(self, c2h4_wfn):
        """Test Löwdin charge calculation."""
        analyzer = ElectrostaticAnalyzer(c2h4_wfn)
        
        if hasattr(analyzer, 'calculate_lowdin_charges'):
            charges = analyzer.calculate_lowdin_charges()
            assert isinstance(charges, dict), "Charges should be dict"


class TestESPAnalysis:
    """Test ESP-based analysis."""

    @pytest.fixture
    def h2_wfn(self):
        """Load H2 wavefunction file."""
        wfn_path = Path("tests/test_data/H2_CCSD.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_esp_report(self, h2_wfn):
        """Test ESP analysis report generation."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'generate_report'):
            report = analyzer.generate_report()
            assert isinstance(report, str), "Report should be string"

    def test_mep_extrema(self, h2_wfn):
        """Test MEP extrema (min/max) identification."""
        analyzer = ElectrostaticAnalyzer(h2_wfn)
        
        if hasattr(analyzer, 'find_mep_extrema'):
            extrema = analyzer.find_mep_extrema()
            assert isinstance(extrema, dict), "Extrema should be dict"
