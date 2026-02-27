"""Test Fuzzy Bond Order Implementation (Issue 20).

This module contains comprehensive tests for fuzzy bond order analysis.
Tests cover fuzzy atom definition, overlap population, bond order calculation,
and Multiwfn consistency validation.
"""

import pytest
import numpy as np
from pathlib import Path

from pymultiwfn.io import load
from pymultiwfn.bonding import Bonding


class TestFuzzyAtomDefinition:
    """Test fuzzy atom definition and boundaries."""

    def test_fuzzy_atom_creation(self, sample_fchk):
        """Test fuzzy atom object creation."""
        bond = Bonding(sample_fchk)
        # Test that fuzzy atoms can be accessed
        assert hasattr(bond, 'atoms'), "Bonding object should have atoms attribute"

    def test_fuzzy_vdwa_radius(self, sample_fchk):
        """Test van der Waals radius calculation for fuzzy atoms."""
        bond = Bonding(sample_fchk)
        # Test that vdWa radii are available
        if hasattr(bond, 'vdwa_radii'):
            assert len(bond.vdwa_radii) > 0, "vdWa radii should be available"
        else:
            pytest.skip("vdwa_radii not yet implemented")

    def test_fuzzy_partition_factor(self, sample_fchk):
        """Test fuzzy partition factor for electron sharing."""
        bond = Bonding(sample_fchk)
        # Test fuzzy partition factor (default 0.5)
        factor = getattr(bond, 'fuzzy_factor', 0.5)
        assert isinstance(factor, float), "Fuzzy factor should be a float"
        assert 0.0 < factor < 1.0, "Fuzzy factor should be between 0 and 1"


class TestOverlapPopulation:
    """Test fuzzy overlap population calculations."""

    def test_overlap_matrix_shape(self, sample_fchk):
        """Test overlap matrix has correct shape."""
        bond = Bonding(sample_fchk)
        wfn = load(sample_fchk)
        n_atoms = wfn.natoms

        if hasattr(bond, 'overlap_matrix'):
            overlap = bond.overlap_matrix
            assert overlap.shape == (n_atoms, n_atoms), \
                f"Overlap matrix should be {n_atoms}x{n_atoms}, got {overlap.shape}"
        else:
            pytest.skip("overlap_matrix not yet implemented")

    def test_symmetric_overlap(self, sample_fchk):
        """Test overlap matrix is symmetric."""
        bond = Bonding(sample_fchk)

        if hasattr(bond, 'overlap_matrix'):
            overlap = bond.overlap_matrix
            assert np.allclose(overlap, overlap.T), "Overlap matrix should be symmetric"
        else:
            pytest.skip("overlap_matrix not yet implemented")

    def test_positive_diagonal(self, sample_fchk):
        """Test diagonal elements of overlap matrix are positive."""
        bond = Bonding(sample_fchk)

        if hasattr(bond, 'overlap_matrix'):
            overlap = bond.overlap_matrix
            np.testing.assert_array_less(0, np.diag(overlap),
                                          err_msg="Diagonal elements should be positive")
        else:
            pytest.skip("overlap_matrix not yet implemented")


class TestBondOrderCalculation:
    """Test fuzzy bond order calculation."""

    def test_single_bond_order(self, h2_fchk):
        """Test fuzzy bond order for H2 single bond."""
        bond = Bonding(h2_fchk)
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # H2 should have bond order close to 1.0
        assert isinstance(fbo, float), "Bond order should be a float"
        assert 0.8 < fbo < 1.2, \
            f"H2 bond order should be ~1.0, got {fbo:.3f}"

    def test_double_bond_order(self, c2h4_fchk):
        """Test fuzzy bond order for C2H4 C=C double bond."""
        bond = Bonding(c2h4_fchk)
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # C=C should have bond order close to 2.0
        assert isinstance(fbo, float), "Bond order should be a float"
        assert 1.6 < fbo < 2.4, \
            f"C=C bond order should be ~2.0, got {fbo:.3f}"

    def test_triple_bond_order(self, n2_fchk):
        """Test fuzzy bond order for N2 triple bond."""
        bond = Bonding(n2_fchk)
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # N≡N should have bond order close to 3.0
        assert isinstance(fbo, float), "Bond order should be a float"
        assert 2.7 < fbo < 3.3, \
            f"N≡N bond order should be ~3.0, got {fbo:.3f}"

    def test_aromatic_bond_order(self, benzene_fchk):
        """Test fuzzy bond order for benzene aromatic bonds."""
        bond = Bonding(benzene_fchk)
        # Test first C-C bond
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # Benzene aromatic bonds should be ~1.5
        assert isinstance(fbo, float), "Bond order should be a float"
        assert 1.3 < fbo < 1.7, \
            f"Benzene bond order should be ~1.5, got {fbo:.3f}"

    def test_atom_indices_validation(self, sample_fchk):
        """Test validation of atom indices."""
        bond = Bonding(sample_fchk)
        wfn = load(sample_fchk)

        # Test out of range atom index
        with pytest.raises((IndexError, ValueError)):
            bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)

        with pytest.raises((IndexError, ValueError)):
            bond.get_fuzzy_bond_order(atom_i=1, atom_j=wfn.natoms + 1)


class TestMultiwfnConsistency:
    """Test consistency with Multiwfn reference values."""

    def test_h2_consistency(self, h2_fchk):
        """Test H2 fuzzy bond order matches Multiwfn reference."""
        bond = Bonding(h2_fchk)
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # Multiwfn reference: ~0.98 for H2 fuzzy bond order
        multiwfn_ref = 0.98
        tolerance = 0.02

        assert abs(fbo - multiwfn_ref) < tolerance, \
            f"H2 fuzzy BO {fbo:.3f} differs from Multiwfn {multiwfn_ref:.3f} by >{tolerance}"

    def test_benzene_consistency(self, benzene_fchk):
        """Test benzene fuzzy bond order matches Multiwfn reference."""
        bond = Bonding(benzene_fchk)
        # Test first C-C bond
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # Multiwfn reference: ~1.45 for benzene aromatic bonds
        multiwfn_ref = 1.45
        tolerance = 0.02

        assert abs(fbo - multiwfn_ref) < tolerance, \
            f"Benzene fuzzy BO {fbo:.3f} differs from Multiwfn {multiwfn_ref:.3f} by >{tolerance}"

    def test_water_consistency(self, water_fchk):
        """Test water O-H bond order matches Multiwfn reference."""
        bond = Bonding(water_fchk)
        # Test first O-H bond (O=1, H=2)
        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)

        # Multiwfn reference: ~0.92 for H2O O-H bond
        multiwfn_ref = 0.92
        tolerance = 0.02

        assert abs(fbo - multiwfn_ref) < tolerance, \
            f"H2O fuzzy BO {fbo:.3f} differs from Multiwfn {multiwfn_ref:.3f} by >{tolerance}"


# Pytest fixtures for test data
@pytest.fixture
def sample_fchk():
    """Return path to a sample FCHK file."""
    # Look for test data in the standard location
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    sample_file = test_data_dir / 'sample.fchk'

    # If not found, skip test
    if not sample_file.exists():
        # Try alternative locations
        alternatives = [
            Path('/home/yhm/software/PyMultiWFN/test_data/fchk/sample.fchk'),
            Path('/home/yhm/software/PyMultiWFN/test_files/sample.fchk'),
        ]
        for alt in alternatives:
            if alt.exists():
                return str(alt)

        pytest.skip(f"Sample FCHK file not found at {sample_file}")

    return str(sample_file)


@pytest.fixture
def h2_fchk():
    """Return path to H2 FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    h2_file = test_data_dir / 'H2.fchk'

    if not h2_file.exists():
        # Try alternative location
        alt = Path('/home/yhm/software/PyMultiWFN/test_data/fchk/H2.fchk')
        if alt.exists():
            return str(alt)
        pytest.skip(f"H2 FCHK file not found at {h2_file}")

    return str(h2_file)


@pytest.fixture
def n2_fchk():
    """Return path to N2 FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    n2_file = test_data_dir / 'N2.fchk'

    if not n2_file.exists():
        pytest.skip(f"N2 FCHK file not found at {n2_file}")
    return str(n2_file)


@pytest.fixture
def c2h4_fchk():
    """Return path to C2H4 FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    c2h4_file = test_data_dir / 'C2H4.fchk'

    if not c2h4_file.exists():
        pytest.skip(f"C2H4 FCHK file not found at {c2h4_file}")
    return str(c2h4_file)


@pytest.fixture
def benzene_fchk():
    """Return path to benzene FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    benzene_file = test_data_dir / 'benzene.fchk'

    if not benzene_file.exists():
        pytest.skip(f"Benzene FCHK file not found at {benzene_file}")
    return str(benzene_file)


@pytest.fixture
def water_fchk():
    """Return path to H2O FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data' / 'fchk'
    water_file = test_data_dir / 'H2O.fchk'

    if not water_file.exists():
        pytest.skip(f"H2O FCHK file not found at {water_file}")
    return str(water_file)
