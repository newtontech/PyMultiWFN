"""
Tests for Issue 8 - Orbital Overlap Analysis

This module tests the orbital overlap analysis functionality including:
- Overlap between specific MOs
- Orbital interaction strength analysis
- Overlap matrix generation
- Bonding/antibonding character identification

Reference: PHASE2_TASKS.md - Task 2.1.3: Orbital Overlap Analysis
"""

import pytest
import numpy as np
from pathlib import Path

from pymultiwfn.io import load
from pymultiwfn.orbitals import OrbitalsAnalyzer


class TestOrbitalOverlap:
    """Test orbital overlap analysis."""

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

    def test_overlap_method_exists(self, h2_wfn):
        """Test that get_orbital_overlap method exists."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        assert hasattr(analyzer, 'get_orbital_overlap'), "OrbitalsAnalyzer should have get_orbital_overlap method"

    def test_overlap_returns_float(self, h2_wfn):
        """Test that get_orbital_overlap returns a float."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        overlap = analyzer.get_orbital_overlap(mo_i=0, mo_j=1)
        assert isinstance(overlap, (int, float)), "Overlap should be numeric"

    def test_overlap_symmetric(self, h2_wfn):
        """Test that orbital overlap is symmetric: S(i,j) = S(j,i)."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        s_01 = analyzer.get_orbital_overlap(mo_i=0, mo_j=1)
        s_10 = analyzer.get_orbital_overlap(mo_i=1, mo_j=0)
        assert abs(s_01 - s_10) < 1e-10, "Overlap should be symmetric"

    def test_self_overlap_is_one(self, h2_wfn):
        """Test that self-overlap S(i,i) = 1.0 for normalized orbitals."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        n_mo = len(analyzer.alpha_energies)
        
        for i in range(min(5, n_mo)):  # Test first 5 orbitals
            self_overlap = analyzer.get_orbital_overlap(mo_i=i, mo_j=i)
            assert abs(self_overlap - 1.0) < 0.01, f"Self-overlap for MO {i} should be 1.0, got {self_overlap}"

    def test_overlap_range(self, h2_wfn):
        """Test that orbital overlaps are in valid range [-1, 1]."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        n_mo = len(analyzer.alpha_energies)
        
        for i in range(min(5, n_mo)):
            for j in range(min(5, n_mo)):
                overlap = analyzer.get_orbital_overlap(mo_i=i, mo_j=j)
                assert -1.01 <= overlap <= 1.01, f"Overlap S({i},{j})={overlap} out of range [-1,1]"

    def test_overlap_matrix_method(self, c2h4_wfn):
        """Test get_overlap_matrix method."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        
        # Get overlap matrix for a subset of orbitals
        overlap_matrix = analyzer.get_overlap_matrix(mo_indices=[0, 1, 2])
        
        assert isinstance(overlap_matrix, np.ndarray), "Overlap matrix should be numpy array"
        assert overlap_matrix.shape == (3, 3), f"Overlap matrix should be 3x3, got {overlap_matrix.shape}"

    def test_overlap_matrix_diagonal_one(self, c2h4_wfn):
        """Test that overlap matrix diagonal elements are 1.0."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        overlap_matrix = analyzer.get_overlap_matrix(mo_indices=[0, 1, 2])
        
        for i in range(3):
            assert abs(overlap_matrix[i, i] - 1.0) < 0.01, f"Diagonal element [{i},{i}] should be 1.0"

    def test_overlap_matrix_symmetric(self, c2h4_wfn):
        """Test that overlap matrix is symmetric."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        overlap_matrix = analyzer.get_overlap_matrix(mo_indices=[0, 1, 2, 3])
        
        # Check symmetry
        assert np.allclose(overlap_matrix, overlap_matrix.T, atol=1e-10), "Overlap matrix should be symmetric"

    def test_negative_mo_index_error(self, h2_wfn):
        """Test that negative MO index raises appropriate error."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        
        with pytest.raises((ValueError, IndexError)):
            analyzer.get_orbital_overlap(mo_i=-1, mo_j=0)

    def test_bonding_character_method(self, c2h4_wfn):
        """Test get_bonding_character method if available."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        
        if hasattr(analyzer, 'get_bonding_character'):
            # For HOMO-LUMO interaction
            homo_idx = analyzer.homo_index
            lumo_idx = analyzer.lumo_index
            
            if lumo_idx < len(analyzer.alpha_energies):
                character = analyzer.get_bonding_character(mo_i=homo_idx, mo_j=lumo_idx)
                assert character in ['bonding', 'antibonding', 'non-bonding', 'mixed'], \
                    f"Invalid bonding character: {character}"


class TestOrbitalOverlapAdvanced:
    """Advanced tests for orbital overlap analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_orbital_interaction_strength(self, c2h4_wfn):
        """Test orbital interaction strength analysis."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        
        if hasattr(analyzer, 'get_interaction_strength'):
            # Test interaction strength for HOMO-LUMO
            homo_idx = analyzer.homo_index
            lumo_idx = analyzer.lumo_index
            
            if lumo_idx < len(analyzer.alpha_energies):
                strength = analyzer.get_interaction_strength(mo_i=homo_idx, mo_j=lumo_idx)
                assert isinstance(strength, (int, float)), "Interaction strength should be numeric"
                assert 0.0 <= strength <= 1.0, f"Interaction strength should be in [0,1], got {strength}"
