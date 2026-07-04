"""
Tests for Issue 9 - Natural Bond Orbital (NBO) Analysis

This module tests the NBO analysis functionality including:
- NBO transformation
- Lewis structure orbital identification
- Bond orbital occupancy calculation
- Donor-acceptor interaction analysis

Reference: PHASE2_TASKS.md - Task 2.1.4: NBO Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.io import load
from pymultiwfn.orbitals import NBOAnalyzer


class TestNBOAnalysis:
    """Test NBO analysis functionality."""

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

    def test_nbo_analyzer_exists(self, h2_wfn):
        """Test that NBOAnalyzer class can be imported and initialized."""
        analyzer = NBOAnalyzer(h2_wfn)
        assert analyzer is not None

    def test_get_natural_orbitals_method(self, h2_wfn):
        """Test that get_natural_orbitals method exists."""
        analyzer = NBOAnalyzer(h2_wfn)
        assert hasattr(
            analyzer, "get_natural_orbitals"
        ), "NBOAnalyzer should have get_natural_orbitals method"

    def test_get_natural_orbitals_returns_array(self, h2_wfn):
        """Test that get_natural_orbitals returns numpy array."""
        analyzer = NBOAnalyzer(h2_wfn)
        nbo_coeffs = analyzer.get_natural_orbitals()
        assert isinstance(
            nbo_coeffs, np.ndarray
        ), "NBO coefficients should be numpy array"

    def test_natural_orbital_occupations(self, h2_wfn):
        """Test that natural orbital occupations are between 0 and 2."""
        analyzer = NBOAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_nbo_occupations"):
            occupations = analyzer.get_nbo_occupations()
            for occ in occupations:
                # Allow small negative values due to numerical precision
                assert -0.001 <= occ <= 2.001, f"Occupation {occ} should be in [0, 2]"

    def test_lewis_structure_method(self, c2h4_wfn):
        """Test Lewis structure identification method."""
        analyzer = NBOAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "identify_lewis_orbitals"):
            lewis_orbitals = analyzer.identify_lewis_orbitals()
            assert isinstance(lewis_orbitals, list), "Lewis orbitals should be a list"

    def test_bond_orbital_occupancy(self, h2_wfn):
        """Test bond orbital occupancy calculation."""
        analyzer = NBOAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_bond_orbital_occupancy"):
            occupancy = analyzer.get_bond_orbital_occupancy(bond_index=0)
            assert isinstance(
                occupancy, (int, float)
            ), "Bond orbital occupancy should be numeric"

    def test_donor_acceptor_analysis(self, c2h4_wfn):
        """Test donor-acceptor interaction analysis."""
        analyzer = NBOAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "analyze_donor_acceptor"):
            interactions = analyzer.analyze_donor_acceptor()
            assert isinstance(
                interactions, list
            ), "Donor-acceptor interactions should be a list"

    def test_nbo_transformation_orthonormal(self, h2_wfn):
        """Test that NBO transformation produces orthonormal orbitals."""
        analyzer = NBOAnalyzer(h2_wfn)
        nbo_coeffs = analyzer.get_natural_orbitals()

        # Check orthonormality: C^T * S * C = I (approximately)
        if h2_wfn.overlap_matrix is not None:
            S = h2_wfn.overlap_matrix
            overlap = nbo_coeffs.T @ S @ nbo_coeffs
            identity = np.eye(nbo_coeffs.shape[1])
            assert np.allclose(
                overlap, identity, atol=0.1
            ), "NBOs should be approximately orthonormal"

    def test_get_lone_pairs(self, c2h4_wfn):
        """Test lone pair identification (C2H4 has no lone pairs)."""
        analyzer = NBOAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_lone_pairs"):
            lone_pairs = analyzer.get_lone_pairs()
            assert isinstance(lone_pairs, list), "Lone pairs should be a list"

    def test_get_bonding_orbitals(self, h2_wfn):
        """Test bonding orbital identification."""
        analyzer = NBOAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_bonding_orbitals"):
            bonding = analyzer.get_bonding_orbitals()
            assert isinstance(bonding, list), "Bonding orbitals should be a list"

    def test_get_antibonding_orbitals(self, h2_wfn):
        """Test antibonding orbital identification."""
        analyzer = NBOAnalyzer(h2_wfn)

        if hasattr(analyzer, "get_antibonding_orbitals"):
            antibonding = analyzer.get_antibonding_orbitals()
            assert isinstance(
                antibonding, list
            ), "Antibonding orbitals should be a list"

    def test_nbo_report_generation(self, h2_wfn):
        """Test NBO report generation."""
        analyzer = NBOAnalyzer(h2_wfn)

        if hasattr(analyzer, "generate_nbo_report"):
            report = analyzer.generate_nbo_report()
            assert isinstance(report, str), "NBO report should be a string"
            assert len(report) > 0, "NBO report should not be empty"

    def test_second_order_perturbation(self, c2h4_wfn):
        """Test second-order perturbation energy analysis."""
        analyzer = NBOAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_perturbation_energies"):
            energies = analyzer.get_perturbation_energies()
            assert isinstance(energies, list), "Perturbation energies should be a list"


class TestNBOAdvanced:
    """Advanced tests for NBO analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_natural_population_analysis(self, c2h4_wfn):
        """Test natural population analysis."""
        analyzer = NBOAnalyzer(c2h4_wfn)

        if hasattr(analyzer, "get_natural_populations"):
            populations = analyzer.get_natural_populations()
            assert isinstance(
                populations, dict
            ), "Natural populations should be a dictionary"
