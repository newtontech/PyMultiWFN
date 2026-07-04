"""Test Delocalization Index Implementation (Issue 4).

This module contains comprehensive tests for delocalization index analysis.
Tests cover 2-center DI, 3-center DI, aromaticity indices, and bond
classification.
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.bonding.delocalization import (
    DelocalizationIndex,
    DelocalizationResult,
    calculate_aromaticity_index,
    calculate_di_matrix,
    calculate_flu,
    calculate_pdi,
    classify_bond_from_di,
    delocalization_index,
    three_center_delocalization_index,
)
from pymultiwfn.io import load


class TestDelocalizationIndexFunction:
    """Test the core delocalization_index function."""

    def test_basic_di_calculation(self):
        """Test basic DI calculation with simple matrices."""
        # Simple 2x2 case: two atoms with 1 basis function each
        P = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = np.array([[1.0, 0.75], [0.75, 1.0]])

        di = delocalization_index(P, S, [0], [1])

        # DI = 2 * |P_01 * S_01| = 2 * |0.5 * 0.75| = 0.75
        assert isinstance(di, float), "DI should be a float"
        assert di > 0, "DI should be positive for bonded atoms"
        assert abs(di - 0.75) < 0.01, f"Expected DI ~0.75, got {di:.4f}"

    def test_di_with_multiple_basis_functions(self):
        """Test DI with multiple basis functions per atom."""
        # 4 basis functions: 2 for atom A, 2 for atom B
        P = np.array(
            [
                [1.0, 0.3, 0.5, 0.2],
                [0.3, 1.0, 0.2, 0.5],
                [0.5, 0.2, 1.0, 0.3],
                [0.2, 0.5, 0.3, 1.0],
            ]
        )
        S = np.eye(4)  # Orthogonal basis
        S[0, 2] = S[2, 0] = 0.7
        S[1, 3] = S[3, 1] = 0.6

        di = delocalization_index(P, S, [0, 1], [2, 3])

        assert isinstance(di, float), "DI should be a float"
        assert di >= 0, "DI should be non-negative"

    def test_di_zero_for_non_bonded(self):
        """Test DI is zero when there's no overlap."""
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        S = np.eye(2)  # No off-diagonal overlap

        di = delocalization_index(P, S, [0], [1])

        assert di == 0.0, "DI should be zero for non-interacting atoms"

    def test_di_empty_indices(self):
        """Test DI with empty basis function indices."""
        P = np.eye(2)
        S = np.eye(2)

        # Empty list for one atom
        di = delocalization_index(P, S, [], [0])
        assert di == 0.0, "DI should be zero with empty indices"

        di = delocalization_index(P, S, [0], [])
        assert di == 0.0, "DI should be zero with empty indices"

    def test_di_matrix_shape_validation(self):
        """Test that mismatched matrix shapes raise error."""
        P = np.eye(3)
        S = np.eye(2)

        with pytest.raises(ValueError, match="shape"):
            delocalization_index(P, S, [0], [1])

    def test_di_index_out_of_range(self):
        """Test that out-of-range indices raise error."""
        P = np.eye(2)
        S = np.eye(2)

        with pytest.raises(ValueError, match="out of range"):
            delocalization_index(P, S, [0], [5])


class TestThreeCenterDI:
    """Test 3-center delocalization index."""

    def test_three_center_di_basic(self):
        """Test basic 3-center DI calculation."""
        # Create 3x3 matrices for 3 atoms
        P = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.5],
                [0.3, 0.5, 1.0],
            ]
        )
        S = np.array(
            [
                [1.0, 0.7, 0.5],
                [0.7, 1.0, 0.7],
                [0.5, 0.7, 1.0],
            ]
        )

        di_3c = three_center_delocalization_index(P, S, [0], [1], [2])

        assert isinstance(di_3c, float), "3-center DI should be a float"
        assert di_3c >= 0, "3-center DI should be non-negative"

    def test_three_center_di_symmetric(self):
        """Test 3-center DI is symmetric in atom ordering."""
        P = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.5],
                [0.3, 0.5, 1.0],
            ]
        )
        S = np.array(
            [
                [1.0, 0.7, 0.5],
                [0.7, 1.0, 0.7],
                [0.5, 0.7, 1.0],
            ]
        )

        di_3c_abc = three_center_delocalization_index(P, S, [0], [1], [2])
        di_3c_bca = three_center_delocalization_index(P, S, [1], [2], [0])
        di_3c_cab = three_center_delocalization_index(P, S, [2], [0], [1])

        # All permutations should give same result
        assert abs(di_3c_abc - di_3c_bca) < 1e-10, "3C DI should be symmetric"
        assert abs(di_3c_abc - di_3c_cab) < 1e-10, "3C DI should be symmetric"


class TestDIMatrix:
    """Test DI matrix calculation."""

    def test_di_matrix_shape(self):
        """Test DI matrix has correct shape."""
        # 4 basis functions for 2 atoms
        P = np.eye(4)
        S = np.eye(4)
        for i in range(2):
            for j in range(2, 4):
                P[i, j] = P[j, i] = 0.5
                S[i, j] = S[j, i] = 0.7

        atom_indices = {0: [0, 1], 1: [2, 3]}

        di_matrix = calculate_di_matrix(P, S, atom_indices, 2)

        assert di_matrix.shape == (2, 2), f"Expected (2,2), got {di_matrix.shape}"

    def test_di_matrix_symmetric(self):
        """Test DI matrix is symmetric."""
        P = np.array(
            [
                [1.0, 0.5, 0.3, 0.2],
                [0.5, 1.0, 0.2, 0.3],
                [0.3, 0.2, 1.0, 0.5],
                [0.2, 0.3, 0.5, 1.0],
            ]
        )
        S = np.array(
            [
                [1.0, 0.8, 0.6, 0.4],
                [0.8, 1.0, 0.4, 0.6],
                [0.6, 0.4, 1.0, 0.8],
                [0.4, 0.6, 0.8, 1.0],
            ]
        )
        atom_indices = {0: [0, 1], 1: [2, 3]}

        di_matrix = calculate_di_matrix(P, S, atom_indices, 2)

        assert np.allclose(di_matrix, di_matrix.T), "DI matrix should be symmetric"

    def test_di_matrix_zero_diagonal(self):
        """Test DI matrix has zero diagonal (no self-delocalization)."""
        P = np.eye(4)
        S = np.eye(4)
        atom_indices = {0: [0, 1], 1: [2, 3]}

        di_matrix = calculate_di_matrix(P, S, atom_indices, 2)

        assert np.allclose(np.diag(di_matrix), 0), "Diagonal should be zero"


class TestBondClassification:
    """Test bond type classification from DI values."""

    def test_weak_bond(self):
        """Test classification of weak bonds."""
        assert classify_bond_from_di(0.1) == "weak"
        assert classify_bond_from_di(0.2) == "weak"

    def test_single_bond(self):
        """Test classification of single bonds."""
        assert classify_bond_from_di(0.5) == "single"
        assert classify_bond_from_di(0.7) == "single"

    def test_aromatic_bond(self):
        """Test classification of aromatic bonds."""
        assert classify_bond_from_di(1.0) == "aromatic"
        assert classify_bond_from_di(1.2) == "aromatic"

    def test_double_bond(self):
        """Test classification of double bonds."""
        assert classify_bond_from_di(1.5) == "double"
        assert classify_bond_from_di(1.7) == "double"

    def test_triple_bond(self):
        """Test classification of triple bonds."""
        assert classify_bond_from_di(2.0) == "triple"
        assert classify_bond_from_di(2.3) == "triple"

    def test_very_strong_bond(self):
        """Test classification of very strong bonds."""
        assert classify_bond_from_di(3.0) == "very_strong"
        assert classify_bond_from_di(5.0) == "very_strong"


class TestAromaticityIndices:
    """Test aromaticity index calculations."""

    def test_aromaticity_index_perfect_aromatic(self):
        """Test aromaticity index for perfect aromatic system."""
        # Create ideal aromatic DI matrix (benzene-like)
        di_matrix = np.zeros((6, 6))
        # Adjacent bonds have DI = 1.4 (perfect aromatic)
        for i in range(6):
            j = (i + 1) % 6
            di_matrix[i, j] = di_matrix[j, i] = 1.4

        ring = [0, 1, 2, 3, 4, 5]
        ai = calculate_aromaticity_index(di_matrix, ring)

        # Should be close to 1.0 for perfect aromatic
        assert abs(ai - 1.0) < 0.01, f"Expected AI ~1.0, got {ai:.4f}"

    def test_aromaticity_index_non_aromatic(self):
        """Test aromaticity index for non-aromatic system."""
        # Create non-aromatic DI matrix (localized bonds)
        di_matrix = np.zeros((6, 6))
        # Alternating single/double bonds
        for i in range(6):
            j = (i + 1) % 6
            di_matrix[i, j] = di_matrix[j, i] = 1.0 if i % 2 == 0 else 1.8

        ring = [0, 1, 2, 3, 4, 5]
        ai = calculate_aromaticity_index(di_matrix, ring)

        # Should be different from 1.0
        assert ai != 1.0, "Non-aromatic should have AI != 1.0"

    def test_pdi_benzene_like(self):
        """Test PDI for benzene-like system."""
        di_matrix = np.zeros((6, 6))
        # Adjacent: DI = 1.4, para: DI = 0.1
        for i in range(6):
            j_adj = (i + 1) % 6
            di_matrix[i, j_adj] = di_matrix[j_adj, i] = 1.4

            j_para = (i + 3) % 6
            di_matrix[i, j_para] = di_matrix[j_para, i] = 0.1

        ring = [0, 1, 2, 3, 4, 5]
        pdi = calculate_pdi(di_matrix, ring)

        # PDI should be around 0.1 for this setup
        assert 0 < pdi < 0.5, f"Expected small PDI, got {pdi:.4f}"

    def test_pdi_requires_six_atoms(self):
        """Test PDI raises error for non-6-membered rings."""
        di_matrix = np.zeros((5, 5))
        ring = [0, 1, 2, 3, 4]

        with pytest.raises(ValueError, match="6"):
            calculate_pdi(di_matrix, ring)

    def test_flu_perfect_aromatic(self):
        """Test FLU for perfect aromatic system."""
        di_matrix = np.zeros((6, 6))
        # Each atom has DI = 1.4 with each neighbor
        for i in range(6):
            j_prev = (i - 1) % 6
            j_next = (i + 1) % 6
            di_matrix[i, j_prev] = di_matrix[j_prev, i] = 1.4
            di_matrix[i, j_next] = di_matrix[j_next, i] = 1.4

        ring = [0, 1, 2, 3, 4, 5]
        flu = calculate_flu(di_matrix, ring)

        # FLU should be close to 0 for perfect aromatic
        assert flu < 0.1, f"Expected FLU ~0, got {flu:.4f}"

    def test_flu_non_aromatic(self):
        """Test FLU for non-aromatic system."""
        di_matrix = np.zeros((6, 6))
        # Localized bonds: alternating strong/weak
        for i in range(6):
            j_prev = (i - 1) % 6
            j_next = (i + 1) % 6
            # Alternating bond strengths
            strength = 1.8 if i % 2 == 0 else 1.0
            di_matrix[i, j_prev] = di_matrix[j_prev, i] = strength
            di_matrix[i, j_next] = di_matrix[j_next, i] = strength

        ring = [0, 1, 2, 3, 4, 5]
        flu = calculate_flu(di_matrix, ring)

        # FLU should be positive for non-aromatic
        assert flu > 0, "FLU should be positive for non-aromatic"


class TestDelocalizationIndexClass:
    """Test the DelocalizationIndex class."""

    def test_initialization_with_wavefunction(self, simple_h2_wfn):
        """Test initialization with a wavefunction object."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        assert di_analyzer.natoms == 2, "Should have 2 atoms"
        assert di_analyzer.density_matrix is not None, "Should have density matrix"
        assert di_analyzer.overlap_matrix is not None, "Should have overlap matrix"

    def test_get_delocalization_index(self, simple_h2_wfn):
        """Test getting DI between two atoms."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        di_val = di_analyzer.get_delocalization_index(0, 1)

        assert isinstance(di_val, float), "DI should be a float"
        assert di_val >= 0, "DI should be non-negative"

    def test_get_di_matrix(self, simple_h2_wfn):
        """Test getting full DI matrix."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        di_matrix = di_analyzer.get_di_matrix()

        assert di_matrix.shape == (2, 2), f"Expected (2,2), got {di_matrix.shape}"
        assert np.allclose(di_matrix, di_matrix.T), "DI matrix should be symmetric"

    def test_get_three_center_di(self, simple_h3_wfn):
        """Test 3-center DI calculation."""
        di_analyzer = DelocalizationIndex(simple_h3_wfn)

        di_3c = di_analyzer.get_three_center_di(0, 1, 2)

        assert isinstance(di_3c, float), "3-center DI should be a float"
        assert di_3c >= 0, "3-center DI should be non-negative"

    def test_get_bond_type(self, simple_h2_wfn):
        """Test bond type classification."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        bond_type = di_analyzer.get_bond_type(0, 1)

        assert isinstance(bond_type, str), "Bond type should be a string"
        assert bond_type in [
            "weak",
            "single",
            "aromatic",
            "double",
            "triple",
            "very_strong",
        ]

    def test_invalid_atom_indices(self, simple_h2_wfn):
        """Test error handling for invalid atom indices."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        with pytest.raises(ValueError):
            di_analyzer.get_delocalization_index(0, 5)

    def test_same_atom_di(self, simple_h2_wfn):
        """Test DI for same atom is zero."""
        di_analyzer = DelocalizationIndex(simple_h2_wfn)

        di_val = di_analyzer.get_delocalization_index(0, 0)

        assert di_val == 0.0, "DI for same atom should be zero"

    def test_is_aromatic_ring(self, benzene_wfn):
        """Test aromatic ring detection."""
        di_analyzer = DelocalizationIndex(benzene_wfn)
        ring = [0, 1, 2, 3, 4, 5]

        is_aromatic = di_analyzer.is_aromatic_ring(ring)

        assert isinstance(is_aromatic, bool), "Should return boolean"

    def test_get_all_ring_dis(self, benzene_wfn):
        """Test getting all DIs in a ring."""
        di_analyzer = DelocalizationIndex(benzene_wfn)
        ring = [0, 1, 2, 3, 4, 5]

        all_dis = di_analyzer.get_all_dihedral_dis(ring)

        assert isinstance(all_dis, dict), "Should return dictionary"
        assert len(all_dis) == 15, "6-membered ring has 15 unique pairs"


class TestDelocalizationResult:
    """Test DelocalizationResult dataclass."""

    def test_result_creation(self):
        """Test creating a DelocalizationResult."""
        result = DelocalizationResult(atom_i=0, atom_j=1, di_value=1.4)

        assert result.atom_i == 0
        assert result.atom_j == 1
        assert result.di_value == 1.4
        assert result.bond_type == "unknown"

    def test_result_with_bond_type(self):
        """Test creating a DelocalizationResult with bond type."""
        result = DelocalizationResult(
            atom_i=0, atom_j=1, di_value=1.4, bond_type="aromatic"
        )

        assert result.bond_type == "aromatic"

    def test_result_repr(self):
        """Test string representation of result."""
        result = DelocalizationResult(atom_i=0, atom_j=1, di_value=1.4)
        repr_str = repr(result)

        assert "DI" in repr_str
        assert "0" in repr_str
        assert "1" in repr_str


# Pytest fixtures for test data
@pytest.fixture
def simple_h2_wfn():
    """Create a simple H2 wavefunction for testing."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction

    atoms = [
        Atom(element="H", index=1, x=0.0, y=0.0, z=-0.7, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=0.0, z=0.7, charge=1.0),
    ]

    shells = [
        Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        ),
        Shell(
            type=0,
            center_idx=1,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        ),
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
        occupations=np.array([2.0, 0.0]),
        coefficients=np.array([[coeff, coeff], [coeff, -coeff]]),
        overlap_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
    )
    wfn.calculate_density_matrices()

    return wfn


@pytest.fixture
def simple_h3_wfn():
    """Create a simple H3 wavefunction for 3-center testing."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction

    atoms = [
        Atom(element="H", index=1, x=0.0, y=0.0, z=-1.0, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=0.0, z=1.0, charge=1.0),
    ]

    shells = [
        Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        ),
        Shell(
            type=0,
            center_idx=1,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        ),
        Shell(
            type=0,
            center_idx=2,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        ),
    ]

    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=3.0,
        charge=0,
        multiplicity=2,
        num_basis=3,
        num_atomic_orbitals=3,
        num_primitives=3,
        num_shells=3,
        shells=shells,
        occupations=np.array([2.0, 1.0, 0.0]),
        coefficients=np.eye(3),
        overlap_matrix=np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.6], [0.3, 0.6, 1.0]]),
    )
    wfn.calculate_density_matrices()

    return wfn


@pytest.fixture
def benzene_wfn():
    """Create a simplified benzene wavefunction for aromaticity testing."""
    from pymultiwfn.core.data import Atom, Shell, Wavefunction

    # Create 6 carbon atoms in a hexagon
    atoms = []
    for i in range(6):
        angle = i * np.pi / 3
        x = 1.4 * np.cos(angle)
        y = 1.4 * np.sin(angle)
        z = 0.0
        atoms.append(Atom(element="C", index=6, x=x, y=y, z=z, charge=6.0))

    # Simplified basis: 1 function per atom
    shells = []
    for i in range(6):
        shells.append(
            Shell(
                type=0,
                center_idx=i,
                exponents=np.array([1.0]),
                coefficients=np.array([1.0]),
            )
        )

    # Create density and overlap matrices for aromatic system
    n_basis = 6
    P = np.zeros((n_basis, n_basis))
    S = np.eye(n_basis)

    # Set up aromatic-like bonding pattern
    for i in range(6):
        j = (i + 1) % 6
        P[i, j] = P[j, i] = 0.5  # Bond order ~1.5
        S[i, j] = S[j, i] = 0.7  # Overlap between neighbors

    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=30.0,  # 6 carbons * 4 valence electrons (simplified)
        charge=0,
        multiplicity=1,
        num_basis=n_basis,
        num_atomic_orbitals=n_basis,
        num_primitives=n_basis,
        num_shells=n_basis,
        shells=shells,
        occupations=np.ones(n_basis),
        coefficients=np.eye(n_basis),
        overlap_matrix=S,
        Ptot=P,
    )

    return wfn
