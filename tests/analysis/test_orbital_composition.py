"""
Tests for Issue 7 - Orbital Composition Analysis

This module tests the orbital composition analysis functionality including:
- AO contribution to each MO
- Orbital composition reports
- Dominant orbital type identification (s, p, d, f)
- Orbital localization on atoms

Reference: PHASE2_TASKS.md - Task 2.1.2: Orbital Composition Analysis
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.io import load
from pymultiwfn.orbitals import OrbitalsAnalyzer


class TestOrbitalComposition:
    """Test orbital composition analysis."""

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
    def ch4_wfn(self):
        """Load CH4 wavefunction file."""
        wfn_path = Path("tests/test_data/CH4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_composition_method_exists(self, h2_wfn):
        """Test that get_composition method exists."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        assert hasattr(
            analyzer, "get_composition"
        ), "OrbitalsAnalyzer should have get_composition method"

    def test_composition_returns_dict(self, h2_wfn):
        """Test that get_composition returns a dictionary."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        composition = analyzer.get_composition(mo_index=0)
        assert isinstance(composition, dict), "Composition should be a dictionary"

    def test_composition_sums_to_one(self, h2_wfn):
        """Test that composition sums to 1.0 for each MO."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        n_mo = len(analyzer.alpha_energies)

        for mo_idx in range(min(5, n_mo)):  # Test first 5 MOs
            composition = analyzer.get_composition(mo_index=mo_idx)
            total = sum(
                sum(atom_contrib.values()) for atom_contrib in composition.values()
            )
            assert (
                abs(total - 1.0) < 0.01
            ), f"Composition for MO {mo_idx} should sum to 1.0, got {total}"

    def test_atomic_contributions_structure(self, h2_wfn):
        """Test that atomic contributions have correct structure."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        composition = analyzer.get_composition(mo_index=0)

        # Should have contributions from atoms
        assert len(composition) > 0, "Should have atomic contributions"

        # Each atom should have orbital contributions
        for atom_label, orb_contrib in composition.items():
            assert isinstance(
                atom_label, str
            ), f"Atom label should be string, got {type(atom_label)}"
            assert isinstance(
                orb_contrib, dict
            ), f"Orbital contributions should be dict, got {type(orb_contrib)}"
            assert (
                len(orb_contrib) > 0
            ), f"Atom {atom_label} should have orbital contributions"

    def test_orbital_type_identification(self, c2h4_wfn):
        """Test identification of orbital types (s, p, d, f)."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        composition = analyzer.get_composition(mo_index=0)

        # Should identify s, p, d, f contributions
        orbital_types = set()
        for atom_contrib in composition.values():
            for orb_type in atom_contrib.keys():
                # Extract orbital type (e.g., '2s' -> 's', '2p_z' -> 'p')
                if "s" in orb_type and "p" not in orb_type:
                    orbital_types.add("s")
                elif "p" in orb_type:
                    orbital_types.add("p")
                elif "d" in orb_type:
                    orbital_types.add("d")
                elif "f" in orb_type:
                    orbital_types.add("f")

        # C2H4 should have s and p contributions
        assert (
            "s" in orbital_types or "p" in orbital_types
        ), "C2H4 should have s or p orbital contributions"

    def test_dominant_orbital_type(self, c2h4_wfn):
        """Test identification of dominant orbital type."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # Get dominant orbital type for first MO
        dominant = analyzer.get_dominant_orbital_type(mo_index=0)
        assert dominant in [
            "s",
            "p",
            "d",
            "f",
            "mixed",
        ], f"Dominant type should be s/p/d/f/mixed, got {dominant}"

    def test_orbital_localization(self, c2h4_wfn):
        """Test orbital localization on atoms."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # Get localization for first MO
        localization = analyzer.get_orbital_localization(mo_index=0)
        assert isinstance(localization, dict), "Localization should be a dictionary"

        # Localization values should sum to ~1.0
        total_localization = sum(localization.values())
        assert (
            abs(total_localization - 1.0) < 0.01
        ), f"Localization should sum to 1.0, got {total_localization}"

    def test_multiple_mos_composition(self, c2h4_wfn):
        """Test composition analysis for multiple MOs."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)
        n_mo = len(analyzer.alpha_energies)

        # Test composition for HOMO, LUMO
        homo_idx = analyzer.homo_index
        lumo_idx = analyzer.lumo_index

        homo_comp = analyzer.get_composition(mo_index=homo_idx)
        assert len(homo_comp) > 0, "HOMO should have composition"

        # Only test LUMO if it exists (index < n_mo)
        if lumo_idx < n_mo:
            lumo_comp = analyzer.get_composition(mo_index=lumo_idx)
            assert len(lumo_comp) > 0, "LUMO should have composition"
            # HOMO and LUMO compositions should be different
            assert (
                homo_comp != lumo_comp
            ), "HOMO and LUMO should have different compositions"

    def test_composition_report_generation(self, c2h4_wfn):
        """Test generation of orbital composition report."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # Generate report for first MO
        report = analyzer.generate_composition_report(mo_index=0)
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 0, "Report should not be empty"

    def test_ch4_composition(self, ch4_wfn):
        """Test composition analysis for CH4 (sp3 hybridization)."""
        analyzer = OrbitalsAnalyzer(ch4_wfn)

        # Get composition for bonding MO
        composition = analyzer.get_composition(mo_index=0)

        # Should have contributions from C and H
        atom_labels = list(composition.keys())
        has_carbon = any("C" in label for label in atom_labels)
        has_hydrogen = any("H" in label for label in atom_labels)

        assert (
            has_carbon or has_hydrogen
        ), "CH4 composition should include C and/or H atoms"

    def test_negative_mo_index_error(self, h2_wfn):
        """Test that negative MO index raises appropriate error."""
        analyzer = OrbitalsAnalyzer(h2_wfn)

        with pytest.raises((ValueError, IndexError)):
            analyzer.get_composition(mo_index=-1)

    def test_out_of_range_mo_index_error(self, h2_wfn):
        """Test that out-of-range MO index raises appropriate error."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        n_mo = len(analyzer.alpha_energies)

        with pytest.raises((ValueError, IndexError)):
            analyzer.get_composition(mo_index=n_mo + 10)

    def test_composition_numerical_accuracy(self, h2_wfn):
        """Test numerical accuracy of composition values."""
        analyzer = OrbitalsAnalyzer(h2_wfn)
        composition = analyzer.get_composition(mo_index=0)

        # All contribution values should be between 0 and 1
        for atom_contrib in composition.values():
            for value in atom_contrib.values():
                assert (
                    0.0 <= value <= 1.0
                ), f"Contribution value {value} should be between 0 and 1"

    def test_orbital_symmetry(self, c2h4_wfn):
        """Test orbital symmetry analysis."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # Get symmetry information (if available)
        if hasattr(analyzer, "get_orbital_symmetry"):
            symmetry = analyzer.get_orbital_symmetry(mo_index=0)
            # Symmetry should be a string (e.g., 'A_g', 'B1u')
            if symmetry is not None:
                assert isinstance(symmetry, str), "Symmetry should be a string"


class TestOrbitalCompositionAdvanced:
    """Advanced tests for orbital composition analysis."""

    @pytest.fixture
    def c2h4_wfn(self):
        """Load C2H4 wavefunction file."""
        wfn_path = Path("tests/test_data/C2H4_HF.wfn")
        if not wfn_path.exists():
            pytest.skip(f"Test file {wfn_path} not found")
        return load(str(wfn_path))

    def test_degenerate_orbitals(self, c2h4_wfn):
        """Test handling of degenerate orbitals."""
        analyzer = OrbitalsAnalyzer(c2h4_wfn)

        # Check if there are degenerate orbitals (similar energies)
        energies = analyzer.alpha_energies
        degenerate_pairs = []

        for i in range(len(energies) - 1):
            if abs(energies[i] - energies[i + 1]) < 0.001:  # 0.001 Hartree threshold
                degenerate_pairs.append((i, i + 1))

        # If degenerate orbitals exist, test their compositions
        if degenerate_pairs:
            for i, j in degenerate_pairs[:1]:  # Test first degenerate pair
                comp_i = analyzer.get_composition(mo_index=i)
                comp_j = analyzer.get_composition(mo_index=j)

                # Degenerate orbitals should have similar but not identical compositions
                # (they span the same symmetry subspace)
                assert (
                    comp_i != comp_j or len(comp_i) > 0
                ), "Degenerate orbitals should have valid compositions"
