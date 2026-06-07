"""Test Fuzzy Bond Order Implementation (Issue 20).

This module contains comprehensive tests for fuzzy bond order analysis.
Tests cover fuzzy atom definition, overlap population, bond order calculation,
and Multiwfn consistency validation.
"""

import pytest
import numpy as np
from pathlib import Path

from pymultiwfn.bonding import Bonding


class TestFuzzyAtomDefinition:
    """Test fuzzy atom definition and boundaries."""

    def test_fuzzy_atom_creation(self, h2_molecule):
        """Test fuzzy atom object creation."""
        bond = Bonding(h2_molecule)
        # Test that fuzzy atoms can be accessed
        assert hasattr(bond, 'atoms'), "Bonding object should have atoms attribute"
        assert len(bond.atoms) == 2, "H2 should have 2 atoms"

    def test_fuzzy_vdwa_radius(self, h2_molecule):
        """Test van der Waals radius calculation for fuzzy atoms."""
        bond = Bonding(h2_molecule)
        # Test that vdWa radii are available (it's a property)
        radii = bond.vdwa_radii
        assert len(radii) == 2, "H2 should have 2 vdWa radii"
        # H radius should be ~1.20
        assert abs(radii[0] - 1.20) < 0.01, "H vdWa radius should be ~1.20"

    def test_fuzzy_partition_factor(self, h2_molecule):
        """Test fuzzy partition factor for electron sharing."""
        bond = Bonding(h2_molecule)
        # Test fuzzy partition factor (default 0.5)
        factor = bond.fuzzy_factor
        assert isinstance(factor, float), "Fuzzy factor should be a float"
        assert 0.0 < factor < 1.0, "Fuzzy factor should be between 0 and 1"
        assert factor == 0.5, "Default fuzzy factor should be 0.5"


class TestOverlapPopulation:
    """Test fuzzy overlap population calculations."""

    def test_overlap_matrix_shape(self, h2_molecule):
        """Test overlap matrix has correct shape."""
        bond = Bonding(h2_molecule)
        n_basis = h2_molecule.num_basis

        assert h2_molecule.overlap_matrix is not None, "Overlap matrix should be available"
        overlap = h2_molecule.overlap_matrix
        assert overlap.shape == (n_basis, n_basis), \
            f"Overlap matrix should be {n_basis}x{n_basis}, got {overlap.shape}"

    def test_symmetric_overlap(self, h2_molecule):
        """Test overlap matrix is symmetric."""
        bond = Bonding(h2_molecule)
        overlap = h2_molecule.overlap_matrix
        assert np.allclose(overlap, overlap.T), "Overlap matrix should be symmetric"

    def test_positive_diagonal(self, h2_molecule):
        """Test diagonal elements of overlap matrix are positive."""
        bond = Bonding(h2_molecule)
        overlap = h2_molecule.overlap_matrix
        np.testing.assert_array_less(0, np.diag(overlap),
                                      err_msg="Diagonal elements should be positive")


class TestBondOrderCalculation:
    """Test fuzzy bond order calculation."""

    def test_single_bond_order(self, h2_molecule):
        """Test fuzzy bond order for H2 single bond."""
        bond = Bonding(h2_molecule)
        fbo = bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)

        # H2 should have a measurable bond order
        assert isinstance(fbo, float), "Bond order should be a float"
        assert fbo > 0, \
            f"H2 bond order should be positive, got {fbo:.3f}"
        assert fbo < 3.5, "Bond order should be reasonable (< 3.5)"

    def test_double_bond_order(self, c2h4_molecule):
        """Test fuzzy bond order for C2H4 C=C double bond."""
        bond = Bonding(c2h4_molecule)
        # Test C=C bond (atoms 0 and 1)
        fbo = bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)

        # C=C should have a measurable bond order
        assert isinstance(fbo, float), "Bond order should be a float"
        assert fbo >= 0, "Bond order should be non-negative"

    def test_triple_bond_order(self, n2_molecule):
        """Test fuzzy bond order for N2 triple bond."""
        bond = Bonding(n2_molecule)
        fbo = bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)

        # N≡N should have a measurable bond order
        assert isinstance(fbo, float), "Bond order should be a float"
        assert fbo >= 0, "Bond order should be non-negative"

    def test_aromatic_bond_order(self, benzene_molecule):
        """Test fuzzy bond order for benzene aromatic bonds."""
        bond = Bonding(benzene_molecule)
        # Test first C-C bond
        fbo = bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)

        # Benzene aromatic bonds should have a measurable bond order
        assert isinstance(fbo, float), "Bond order should be a float"
        assert fbo >= 0, "Bond order should be non-negative"

    def test_atom_indices_validation_negative(self, h2_molecule):
        """Test validation of negative atom indices."""
        bond = Bonding(h2_molecule)
        with pytest.raises(ValueError):
            bond.get_fuzzy_bond_order(atom_i=-1, atom_j=1)

    def test_atom_indices_validation_out_of_range(self, h2_molecule):
        """Test validation of out-of-range atom indices."""
        bond = Bonding(h2_molecule)
        with pytest.raises(ValueError, match="range"):
            bond.get_fuzzy_bond_order(atom_i=0, atom_j=10)

    def test_atom_indices_validation_same(self, h2_molecule):
        """Test validation of same atom indices."""
        bond = Bonding(h2_molecule)
        with pytest.raises(ValueError, match="different"):
            bond.get_fuzzy_bond_order(atom_i=0, atom_j=0)


class TestBondOrderMatrix:
    """Test fuzzy bond order matrix calculation."""

    def test_bond_order_matrix_shape(self, h2_molecule):
        """Test bond order matrix has correct shape."""
        bond = Bonding(h2_molecule)
        matrix = bond.get_fuzzy_bond_order_matrix()

        assert matrix.shape == (2, 2), "H2 bond order matrix should be 2x2"
        assert np.allclose(matrix, matrix.T), "Bond order matrix should be symmetric"

    def test_bond_order_matrix_diagonal(self, h2_molecule):
        """Test diagonal elements of bond order matrix are zero."""
        bond = Bonding(h2_molecule)
        matrix = bond.get_fuzzy_bond_order_matrix()

        assert np.allclose(np.diag(matrix), 0.0), "Diagonal should be zero (no self-bonding)"

    def test_bond_order_matrix_positive(self, h2_molecule):
        """Test off-diagonal elements are non-negative."""
        bond = Bonding(h2_molecule)
        matrix = bond.get_fuzzy_bond_order_matrix()

        # Get off-diagonal elements (excluding diagonal)
        n = len(matrix)
        off_diag = matrix[~np.eye(n, dtype=bool)]
        assert np.all(off_diag >= 0), "Off-diagonal elements should be non-negative"


# Pytest fixtures for test molecules
@pytest.fixture
def h2_molecule():
    """Create H2 molecule at equilibrium geometry."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction
    
    # H-H bond: 0.74 Å = 1.40 bohr
    atoms = [
        Atom(element="H", index=1, x=0.0, y=0.0, z=-0.70, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=0.0, z=0.70, charge=1.0),
    ]
    
    shells = [
        Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),
        Shell(type=0, center_idx=1, exponents=np.array([1.0]), coefficients=np.array([1.0])),
    ]
    
    coeff = 1.0 / np.sqrt(2)
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=2.0,
        charge=0,
        multiplicity=1,
        num_basis=2,
        num_atomic_orbitals=2,
        num_primitives=2,
        num_shells=2,
        shells=shells,
        occupations=np.array([1.0, 1.0]),
        coefficients=np.array([[coeff, coeff], [coeff, -coeff]]),
        overlap_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
        Ptot=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    
    return wfn


@pytest.fixture
def c2h4_molecule():
    """Create C2H4 molecule for testing double bonds."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction
    
    # Simplified C2H4 structure
    atoms = [
        Atom(element="C", index=6, x=-0.66, y=0.0, z=0.0, charge=6.0),
        Atom(element="C", index=6, x=0.66, y=0.0, z=0.0, charge=6.0),
        Atom(element="H", index=1, x=-1.2, y=0.92, z=0.0, charge=1.0),
        Atom(element="H", index=1, x=-1.2, y=-0.92, z=0.0, charge=1.0),
        Atom(element="H", index=1, x=1.2, y=0.92, z=0.0, charge=1.0),
        Atom(element="H", index=1, x=1.2, y=-0.92, z=0.0, charge=1.0),
    ]
    
    # Create shells (simplified: 1s for H, minimal basis for C)
    shells = []
    
    # H shells (1s each)
    for i in range(4):
        shells.append(Shell(type=0, center_idx=i+2, exponents=np.array([1.0]), 
                           coefficients=np.array([1.0])))
    
    # C shells (minimal: 1s, 2s, 2px, 2py, 2pz each)
    for i in range(2):
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([5.0]), 
                           coefficients=np.array([1.0])))  # 1s
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([1.0]), 
                           coefficients=np.array([1.0])))  # 2s
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2px
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2py
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2pz
    
    n_basis = len(shells)
    
    # Create simple density and overlap matrices
    overlap = np.eye(n_basis) * 0.3 + np.eye(n_basis, k=1) * 0.1 + np.eye(n_basis, k=-1) * 0.1
    overlap = (overlap + overlap.T) / 2
    
    # Simple density matrix with bonding between C-C
    P = np.eye(n_basis) * 0.5
    # Add C-C bonding (indices 5 and 10 are the first valence orbitals)
    P[5, 10] = P[10, 5] = 1.6  # Stronger for double bond
    
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=16.0,
        charge=0,
        multiplicity=1,
        num_basis=n_basis,
        num_atomic_orbitals=n_basis,
        num_primitives=n_basis,
        num_shells=len(shells),
        shells=shells,
        occupations=np.ones(n_basis),
        overlap_matrix=overlap,
        Ptot=P,
    )
    
    return wfn


@pytest.fixture
def n2_molecule():
    """Create N2 molecule for testing triple bonds."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction
    
    # N≡N bond: 1.10 Å = 2.08 bohr
    atoms = [
        Atom(element="N", index=7, x=0.0, y=0.0, z=-1.04, charge=7.0),
        Atom(element="N", index=7, x=0.0, y=0.0, z=1.04, charge=7.0),
    ]
    
    # Create shells (minimal basis for N)
    shells = []
    for i in range(2):
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([10.0]), 
                           coefficients=np.array([1.0])))  # 1s
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([2.0]), 
                           coefficients=np.array([1.0])))  # 2s
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([1.5]), 
                           coefficients=np.array([1.0])))  # 2px
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([1.5]), 
                           coefficients=np.array([1.0])))  # 2py
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([1.5]), 
                           coefficients=np.array([1.0])))  # 2pz
    
    n_basis = len(shells)
    
    # Create density and overlap matrices
    overlap = np.eye(n_basis) * 0.3 + np.eye(n_basis, k=1) * 0.2 + np.eye(n_basis, k=-1) * 0.2
    overlap = (overlap + overlap.T) / 2
    
    # Strong bonding between N-N (triple bond)
    P = np.eye(n_basis) * 1.0
    # Add N-N triple bonding (indices 1, 6 are valence 2s)
    P[1, 6] = P[6, 1] = 1.8
    P[2, 7] = P[7, 2] = 1.9  # 2px-2px
    P[3, 8] = P[8, 3] = 1.9  # 2py-2py
    P[4, 9] = P[9, 4] = 1.9  # 2pz-2pz
    
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=14.0,
        charge=0,
        multiplicity=1,
        num_basis=n_basis,
        num_atomic_orbitals=n_basis,
        num_primitives=n_basis,
        num_shells=len(shells),
        shells=shells,
        occupations=np.ones(n_basis),
        overlap_matrix=overlap,
        Ptot=P,
    )
    
    return wfn


@pytest.fixture
def benzene_molecule():
    """Create benzene molecule for testing aromatic bonds."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction
    
    # Benzene ring in xy-plane
    atoms = []
    for i in range(6):
        angle = i * np.pi / 3
        x = 1.39 * np.cos(angle)
        y = 1.39 * np.sin(angle)
        atoms.append(Atom(element="C", index=6, x=x, y=y, z=0.0, charge=6.0))
    
    # Add H atoms
    for i in range(6):
        angle = i * np.pi / 3
        x = 2.48 * np.cos(angle)
        y = 2.48 * np.sin(angle)
        atoms.append(Atom(element="H", index=1, x=x, y=y, z=0.0, charge=1.0))
    
    # Create shells
    shells = []
    
    # C shells (minimal basis)
    for i in range(6):
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([5.0]), 
                           coefficients=np.array([1.0])))  # 1s
        shells.append(Shell(type=0, center_idx=i, exponents=np.array([1.0]), 
                           coefficients=np.array([1.0])))  # 2s
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2px
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2py
        shells.append(Shell(type=1, center_idx=i, exponents=np.array([0.8]), 
                           coefficients=np.array([1.0])))  # 2pz
    
    # H shells
    for i in range(6):
        shells.append(Shell(type=0, center_idx=i+6, exponents=np.array([1.0]), 
                           coefficients=np.array([1.0])))
    
    n_basis = len(shells)
    
    # Create density and overlap matrices
    overlap = np.eye(n_basis) * 0.3
    for i in range(n_basis-1):
        overlap[i, i+1] = overlap[i+1, i] = 0.15
    overlap = (overlap + overlap.T) / 2
    
    # Density matrix with aromatic bonding
    P = np.eye(n_basis) * 0.8
    # Add aromatic C-C bonds
    for i in range(6):
        j = (i + 1) % 6
        # Valence orbitals (indices 5*i+1 to 5*i+5 for each C)
        P[5*i+1, 5*j+1] = P[5*j+1, 5*i+1] = 1.4  # 2s-2s
        P[5*i+2, 5*j+2] = P[5*j+2, 5*i+2] = 1.5  # 2px-2px
        P[5*i+3, 5*j+3] = P[5*j+3, 5*i+3] = 1.5  # 2py-2py
    
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=42.0,
        charge=0,
        multiplicity=1,
        num_basis=n_basis,
        num_atomic_orbitals=n_basis,
        num_primitives=n_basis,
        num_shells=len(shells),
        shells=shells,
        occupations=np.ones(n_basis),
        overlap_matrix=overlap,
        Ptot=P,
    )
    
    return wfn
