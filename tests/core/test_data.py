"""
Comprehensive pytest tests for core data classes in PyMultiWFN.

Tests cover:
- Atom class: initialization, properties, validation
- Shell class: initialization, properties, validation
- Wavefunction class: initialization, methods, edge cases
"""

from typing import Dict, List

import numpy as np
import pytest

from pymultiwfn.core.data import Atom, Shell, Wavefunction

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_atom():
    """Create a sample Atom instance for testing."""
    return Atom(element="C", index=6, x=0.0, y=0.0, z=0.0, charge=6.0)


@pytest.fixture
def sample_atoms():
    """Create a list of sample Atom instances for testing."""
    return [
        Atom("H", 1, 0.0, 0.0, 0.0, 1.0),
        Atom("O", 8, 0.0, 0.0, 1.0, 8.0),
        Atom("C", 6, 1.0, 0.0, 0.0, 6.0),
    ]


@pytest.fixture
def sample_shell():
    """Create a sample Shell instance for testing."""
    return Shell(
        type=0,  # S shell
        center_idx=0,
        exponents=np.array([0.5, 1.0, 2.0]),
        coefficients=np.array([0.1, 0.2, 0.3]),
    )


@pytest.fixture
def sample_shells():
    """Create a list of sample Shell instances for testing."""
    return [
        Shell(0, 0, np.array([0.5, 1.0]), np.array([0.1, 0.2])),  # S shell
        Shell(1, 0, np.array([0.3, 0.6]), np.array([0.15, 0.25])),  # P shell
        Shell(
            -1, 1, np.array([0.4, 0.8]), np.array([[0.12, 0.22], [0.13, 0.23]])
        ),  # SP shell
    ]


@pytest.fixture
def sample_wavefunction():
    """Create a sample Wavefunction instance for testing."""
    wf = Wavefunction()
    wf.atoms = [
        Atom("H", 1, 0.0, 0.0, 0.0, 1.0),
        Atom("H", 1, 0.0, 0.0, 0.74, 1.0),
    ]
    wf.num_electrons = 2.0
    wf.charge = 0
    wf.multiplicity = 1
    wf.num_basis = 2
    wf.shells = [
        Shell(0, 0, np.array([0.5]), np.array([0.1])),
        Shell(0, 1, np.array([0.5]), np.array([0.1])),
    ]
    return wf


@pytest.fixture
def restricted_wavefunction():
    """Create a restricted closed-shell wavefunction."""
    wf = Wavefunction()
    wf.atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0)]
    wf.num_electrons = 2.0
    wf.multiplicity = 1
    wf.is_unrestricted = False
    wf.num_basis = 2
    wf.coefficients = np.array([[0.7, 0.7], [0.7, -0.7]])
    wf.energies = np.array([-0.5, 0.3])
    return wf


@pytest.fixture
def unrestricted_wavefunction():
    """Create an unrestricted open-shell wavefunction."""
    wf = Wavefunction()
    wf.atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0)]
    wf.num_electrons = 1.0
    wf.multiplicity = 2
    wf.is_unrestricted = True
    wf.num_basis = 1
    wf.coefficients = np.array([[1.0]])
    wf.energies = np.array([-0.5])
    wf.coefficients_beta = np.array([[1.0]])
    wf.energies_beta = np.array([-0.45])
    return wf


# =============================================================================
# Atom Class Tests
# =============================================================================


class TestAtom:
    """Test suite for Atom class."""

    def test_atom_initialization(self, sample_atom):
        """Test that Atom initializes correctly with valid parameters."""
        assert sample_atom.element == "C"
        assert sample_atom.index == 6
        assert sample_atom.x == 0.0
        assert sample_atom.y == 0.0
        assert sample_atom.z == 0.0
        assert sample_atom.charge == 6.0

    @pytest.mark.parametrize(
        "element, index, x, y, z, charge",
        [
            ("H", 1, 0.0, 0.0, 0.0, 1.0),
            ("He", 2, 1.0, 2.0, 3.0, 2.0),
            ("Li", 3, -1.5, 0.5, 2.3, 3.0),
            ("C", 6, 0.0, 0.0, 0.0, 4.0),  # ECP case
        ],
    )
    def test_atom_various_elements(self, element, index, x, y, z, charge):
        """Test Atom initialization with various elements and coordinates."""
        atom = Atom(element, index, x, y, z, charge)
        assert atom.element == element
        assert atom.index == index
        assert atom.x == x
        assert atom.y == y
        assert atom.z == z
        assert atom.charge == charge

    def test_atom_coord_property(self, sample_atom):
        """Test that coord property returns correct numpy array."""
        coord = sample_atom.coord
        assert isinstance(coord, np.ndarray)
        assert coord.shape == (3,)
        assert coord[0] == 0.0
        assert coord[1] == 0.0
        assert coord[2] == 0.0

    def test_atom_coord_mutation(self):
        """Test that coord property returns a new array each time."""
        atom = Atom("C", 6, 1.0, 2.0, 3.0, 6.0)
        coord1 = atom.coord
        coord2 = atom.coord
        # Should be different array objects
        assert coord1 is not coord2
        # But with same values
        assert np.array_equal(coord1, coord2)

    @pytest.mark.parametrize(
        "x, y, z",
        [
            (0.0, 0.0, 0.0),
            (1.5, 2.3, -0.7),
            (-1.0, -2.0, -3.0),
            (1e-10, 1e-10, 1e-10),  # Very small values
        ],
    )
    def test_atom_various_coordinates(self, x, y, z):
        """Test Atom with various coordinate values."""
        atom = Atom("H", 1, x, y, z, 1.0)
        assert atom.x == x
        assert atom.y == y
        assert atom.z == z
        np.testing.assert_array_equal(atom.coord, np.array([x, y, z]))

    def test_atom_with_ecp_charge(self):
        """Test Atom with effective core potential (charge != index)."""
        # Transition metal with ECP
        atom = Atom("Fe", 26, 0.0, 0.0, 0.0, 16.0)
        assert atom.index == 26
        assert atom.charge == 16.0


# =============================================================================
# Shell Class Tests
# =============================================================================


class TestShell:
    """Test suite for Shell class."""

    def test_shell_initialization_s(self, sample_shell):
        """Test Shell initialization with S shell."""
        assert sample_shell.type == 0
        assert sample_shell.center_idx == 0
        assert len(sample_shell.exponents) == 3
        assert len(sample_shell.coefficients) == 3

    @pytest.mark.parametrize(
        "shell_type, name",
        [
            (0, "S"),
            (1, "P"),
            (2, "D"),
            (3, "F"),
            (-1, "SP"),
        ],
    )
    def test_shell_various_types(self, shell_type, name):
        """Test Shell initialization with various shell types."""
        shell = Shell(
            type=shell_type,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([0.5]),
        )
        assert shell.type == shell_type

    def test_shell_sp_shape(self):
        """Test that SP shell has correct coefficient shape (2, N)."""
        exponents = np.array([0.5, 1.0, 2.0])
        coefficients = np.array(
            [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]  # S coefficients  # P coefficients
        )
        shell = Shell(-1, 0, exponents, coefficients)
        assert shell.type == -1
        assert shell.coefficients.shape == (2, 3)

    def test_shell_exponents_coefficients_length_match(self):
        """Test that exponents and coefficients arrays have matching lengths."""
        exponents = np.array([0.5, 1.0, 2.0, 3.0])
        coefficients = np.array([0.1, 0.2, 0.3, 0.4])
        shell = Shell(0, 0, exponents, coefficients)
        assert len(shell.exponents) == len(shell.coefficients)

    def test_shell_center_idx(self):
        """Test shell center index assignment."""
        shell = Shell(0, 5, np.array([1.0]), np.array([0.5]))
        assert shell.center_idx == 5

    def test_shell_multiple_primitives(self):
        """Test shell with multiple primitive Gaussians."""
        n_primitives = 10
        exponents = np.linspace(0.1, 10.0, n_primitives)
        coefficients = np.random.rand(n_primitives)
        shell = Shell(2, 0, exponents, coefficients)
        assert len(shell.exponents) == n_primitives
        assert len(shell.coefficients) == n_primitives


# =============================================================================
# Wavefunction Class Tests - Initialization
# =============================================================================


class TestWavefunctionInitialization:
    """Test suite for Wavefunction class initialization."""

    def test_wavefunction_default_initialization(self):
        """Test Wavefunction with default parameters."""
        wf = Wavefunction()
        assert wf.atoms == []
        assert wf.num_electrons == 0.0
        assert wf.charge == 0
        assert wf.multiplicity == 1
        assert wf.shells == []
        assert wf.num_basis == 0
        assert wf.num_primitives == 0
        assert wf.num_shells == 0
        assert wf.is_unrestricted is False
        assert wf.coefficients is None
        assert wf.energies is None
        assert wf.occupations is None

    def test_wavefunction_custom_initialization(self):
        """Test Wavefunction with custom parameters."""
        atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0)]
        wf = Wavefunction(
            atoms=atoms,
            num_electrons=1.0,
            charge=0,
            multiplicity=2,
            num_basis=1,
            title="Hydrogen",
            method="HF",
            basis_set_name="STO-3G",
        )
        assert len(wf.atoms) == 1
        assert wf.num_electrons == 1.0
        assert wf.charge == 0
        assert wf.multiplicity == 2
        assert wf.num_basis == 1
        assert wf.title == "Hydrogen"
        assert wf.method == "HF"
        assert wf.basis_set_name == "STO-3G"


# =============================================================================
# Wavefunction Class Tests - Atoms
# =============================================================================


class TestWavefunctionAtoms:
    """Test suite for Wavefunction atom management."""

    def test_num_atoms_property(self, sample_wavefunction):
        """Test num_atoms property returns correct count."""
        assert sample_wavefunction.num_atoms == 2

    def test_num_atoms_empty(self):
        """Test num_atoms with empty atom list."""
        wf = Wavefunction()
        assert wf.num_atoms == 0

    def test_add_atom_basic(self):
        """Test adding a single atom."""
        wf = Wavefunction()
        wf.add_atom("H", 1, 0.0, 0.0, 0.0)
        assert len(wf.atoms) == 1
        assert wf.atoms[0].element == "H"
        assert wf.atoms[0].index == 1
        assert wf.atoms[0].charge == 1.0

    def test_add_atom_with_charge(self):
        """Test adding atom with explicit charge."""
        wf = Wavefunction()
        wf.add_atom("Fe", 26, 0.0, 0.0, 0.0, charge=16.0)
        assert wf.atoms[0].charge == 16.0
        assert wf.atoms[0].index == 26

    def test_add_multiple_atoms(self):
        """Test adding multiple atoms."""
        wf = Wavefunction()
        wf.add_atom("H", 1, 0.0, 0.0, 0.0)
        wf.add_atom("O", 8, 0.0, 0.0, 1.0)
        wf.add_atom("C", 6, 1.0, 0.0, 0.0)
        assert wf.num_atoms == 3
        assert wf.atoms[0].element == "H"
        assert wf.atoms[1].element == "O"
        assert wf.atoms[2].element == "C"


# =============================================================================
# Wavefunction Class Tests - Property Aliases
# =============================================================================


class TestWavefunctionPropertyAliases:
    """Test suite for Wavefunction property aliases."""

    def test_mo_energies_alias(self, sample_wavefunction):
        """Test mo_energies property alias."""
        energies = np.array([-1.0, -0.5, 0.0])
        sample_wavefunction.energies = energies
        assert np.array_equal(sample_wavefunction.mo_energies, energies)

    def test_mo_energies_none(self, sample_wavefunction):
        """Test mo_energies when energies is None."""
        assert sample_wavefunction.energies is None
        assert sample_wavefunction.mo_energies is None

    def test_mo_coefficients_alias(self, sample_wavefunction):
        """Test mo_coefficients property alias."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sample_wavefunction.coefficients = coeffs
        assert np.array_equal(sample_wavefunction.mo_coefficients, coeffs)

    def test_mo_coefficients_none(self, sample_wavefunction):
        """Test mo_coefficients when coefficients is None."""
        assert sample_wavefunction.coefficients is None
        assert sample_wavefunction.mo_coefficients is None

    def test_mo_occupations_alias(self, sample_wavefunction):
        """Test mo_occupations property alias."""
        occs = np.array([2.0, 0.0])
        sample_wavefunction.occupations = occs
        assert np.array_equal(sample_wavefunction.mo_occupations, occs)

    def test_mo_occupations_none(self, sample_wavefunction):
        """Test mo_occupations when occupations is None."""
        assert sample_wavefunction.occupations is None
        assert sample_wavefunction.mo_occupations is None


# =============================================================================
# Wavefunction Class Tests - Occupation Inference
# =============================================================================


class TestWavefunctionOccupationInference:
    """Test suite for orbital occupation inference."""

    def test_infer_occupations_restricted_closed_shell(self, restricted_wavefunction):
        """Test occupation inference for restricted closed-shell system."""
        restricted_wavefunction._infer_occupations()
        assert restricted_wavefunction.occupations is not None
        # For 2 electrons in restricted, first MO should be occupied
        assert restricted_wavefunction.occupations[0] == 2.0

    def test_infer_occupations_unrestricted_open_shell(self, unrestricted_wavefunction):
        """Test occupation inference for unrestricted open-shell system."""
        unrestricted_wavefunction._infer_occupations()
        assert unrestricted_wavefunction.occupations is not None
        assert unrestricted_wavefunction.occupations_beta is not None
        # For 1 electron, alpha should be occupied
        assert unrestricted_wavefunction.occupations[0] == 1.0

    def test_infer_occupations_no_change_if_set(self):
        """Test that occupations are recalculated if only alpha is set (need both for unrestricted)."""
        wf = Wavefunction()
        wf.num_electrons = 2.0
        wf.multiplicity = 1
        wf.is_unrestricted = False
        wf.num_basis = 2
        wf.coefficients = np.array([[0.7, 0.7], [0.7, -0.7]])
        wf.energies = np.array([-0.5, 0.3])
        wf.occupations = np.array([1.5, 0.5])  # Custom occupation

        # The code only skips if both occupations AND occupations_beta are set
        # Since this is restricted (no beta occupations), it will recalculate
        wf._infer_occupations()
        # After recalculation, should have proper occupations for 2 electrons
        assert wf.occupations[0] == 2.0


# =============================================================================
# Wavefunction Class Tests - Density Matrices
# =============================================================================


class TestWavefunctionDensityMatrices:
    """Test suite for density matrix calculations."""

    def test_calculate_density_matrices_restricted(self, restricted_wavefunction):
        """Test density matrix calculation for restricted case."""
        restricted_wavefunction._infer_occupations()
        restricted_wavefunction.calculate_density_matrices()

        assert restricted_wavefunction.Palpha is not None
        assert restricted_wavefunction.Pbeta is not None
        assert restricted_wavefunction.Ptot is not None

        # Check shapes
        n_basis = restricted_wavefunction.num_basis
        assert restricted_wavefunction.Palpha.shape == (n_basis, n_basis)
        assert restricted_wavefunction.Pbeta.shape == (n_basis, n_basis)
        assert restricted_wavefunction.Ptot.shape == (n_basis, n_basis)

        # For restricted, Pbeta should be all zeros
        np.testing.assert_array_equal(
            restricted_wavefunction.Pbeta, np.zeros((n_basis, n_basis))
        )

    def test_calculate_density_matrices_unrestricted(self, unrestricted_wavefunction):
        """Test density matrix calculation for unrestricted case."""
        unrestricted_wavefunction._infer_occupations()
        unrestricted_wavefunction.calculate_density_matrices()

        assert unrestricted_wavefunction.Palpha is not None
        assert unrestricted_wavefunction.Pbeta is not None
        assert unrestricted_wavefunction.Ptot is not None

        # Check that Ptot = Palpha + Pbeta
        expected_Ptot = (
            unrestricted_wavefunction.Palpha + unrestricted_wavefunction.Pbeta
        )
        np.testing.assert_array_almost_equal(
            unrestricted_wavefunction.Ptot, expected_Ptot
        )

    def test_calculate_density_matrices_no_coefficients(self):
        """Test density matrix calculation when coefficients are None."""
        wf = Wavefunction()
        wf.num_basis = 2
        wf.calculate_density_matrices()

        assert wf.Palpha is not None
        assert wf.Pbeta is not None
        assert wf.Ptot is not None
        np.testing.assert_array_equal(wf.Palpha, np.zeros((2, 2)))
        np.testing.assert_array_equal(wf.Pbeta, np.zeros((2, 2)))
        np.testing.assert_array_equal(wf.Ptot, np.zeros((2, 2)))

    def test_calculate_density_matrices_zero_basis(self):
        """Test density matrix calculation with zero basis functions."""
        wf = Wavefunction()
        wf.num_basis = 0
        wf.calculate_density_matrices()

        assert wf.Palpha is not None
        assert wf.Pbeta is not None
        assert wf.Ptot is not None
        np.testing.assert_array_equal(wf.Palpha, np.zeros((0, 0)))
        np.testing.assert_array_equal(wf.Pbeta, np.zeros((0, 0)))


# =============================================================================
# Wavefunction Class Tests - Overlap Matrix
# =============================================================================


class TestWavefunctionOverlapMatrix:
    """Test suite for overlap matrix calculation."""

    def test_calculate_overlap_matrix_basic(self, sample_wavefunction):
        """Test basic overlap matrix calculation."""
        sample_wavefunction.calculate_overlap_matrix()

        assert sample_wavefunction.overlap_matrix is not None
        assert sample_wavefunction.overlap_matrix.shape == (2, 2)
        assert np.allclose(
            sample_wavefunction.overlap_matrix, sample_wavefunction.overlap_matrix.T
        )
        assert sample_wavefunction.overlap_matrix[0, 1] > 0.0

    def test_calculate_overlap_matrix_zero_basis(self):
        """Test overlap matrix with zero basis functions."""
        wf = Wavefunction()
        wf.num_basis = 0
        wf.calculate_overlap_matrix()

        assert wf.overlap_matrix is not None
        np.testing.assert_array_equal(wf.overlap_matrix, np.array([]))

    def test_calculate_overlap_matrix_requires_shells_without_fallback(self):
        """Test that missing shell data does not silently produce identity."""
        wf = Wavefunction()
        wf.num_basis = 5

        with pytest.raises(ValueError, match="without basis shell information"):
            wf.calculate_overlap_matrix()

    def test_calculate_overlap_matrix_identity_fallback_is_explicit(self):
        """Test explicit identity fallback for synthetic orthonormal basis tests."""
        wf = Wavefunction()
        wf.num_basis = 5
        with pytest.warns(RuntimeWarning, match="identity overlap matrix fallback"):
            overlap = wf.calculate_overlap_matrix(allow_identity_fallback=True)

        np.testing.assert_array_equal(overlap, np.eye(5))


# =============================================================================
# Wavefunction Class Tests - Atomic Basis Mapping
# =============================================================================


class TestWavefunctionAtomicBasisMapping:
    """Test suite for atomic basis function mapping."""

    def test_get_atomic_basis_indices_empty(self):
        """Test atomic basis mapping with no shells."""
        wf = Wavefunction()
        wf.atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0)]
        mapping = wf.get_atomic_basis_indices()

        assert isinstance(mapping, dict)
        assert len(mapping) == 1
        assert mapping[0] == []

    def test_get_atomic_basis_indices_s_shell(self):
        """Test atomic basis mapping with S shell (1 function)."""
        wf = Wavefunction()
        wf.atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0), Atom("H", 1, 0.0, 0.0, 0.74, 1.0)]
        wf.num_basis = 2
        wf.shells = [
            Shell(0, 0, np.array([0.5]), np.array([0.1])),  # S on atom 0
            Shell(0, 1, np.array([0.5]), np.array([0.1])),  # S on atom 1
        ]
        mapping = wf.get_atomic_basis_indices()

        assert mapping[0] == [0]
        assert mapping[1] == [1]

    def test_get_atomic_basis_indices_p_shell(self):
        """Test atomic basis mapping with P shell (3 functions)."""
        wf = Wavefunction()
        wf.atoms = [Atom("C", 6, 0.0, 0.0, 0.0, 6.0)]
        wf.num_basis = 3
        wf.shells = [
            Shell(1, 0, np.array([0.5]), np.array([0.1])),  # P shell on atom 0
        ]
        mapping = wf.get_atomic_basis_indices()

        assert mapping[0] == [0, 1, 2]

    def test_get_atomic_basis_indices_d_shell(self):
        """Test atomic basis mapping with D shell (5 functions)."""
        wf = Wavefunction()
        wf.atoms = [Atom("Fe", 26, 0.0, 0.0, 0.0, 26.0)]
        wf.num_basis = 5
        wf.shells = [
            Shell(2, 0, np.array([0.5]), np.array([0.1])),  # D shell on atom 0
        ]
        mapping = wf.get_atomic_basis_indices()

        assert mapping[0] == [0, 1, 2, 3, 4]

    def test_get_atomic_basis_indices_sp_shell(self):
        """Test atomic basis mapping with SP shell (4 functions: 1S + 3P)."""
        wf = Wavefunction()
        wf.atoms = [Atom("C", 6, 0.0, 0.0, 0.0, 6.0)]
        wf.num_basis = 4
        wf.shells = [
            Shell(
                -1, 0, np.array([0.5]), np.array([[0.1, 0.15]])
            ),  # SP shell on atom 0
        ]
        mapping = wf.get_atomic_basis_indices()

        assert mapping[0] == [0, 1, 2, 3]

    def test_get_atomic_basis_indices_mixed_shells(self):
        """Test atomic basis mapping with mixed shell types."""
        wf = Wavefunction()
        wf.atoms = [Atom("C", 6, 0.0, 0.0, 0.0, 6.0), Atom("H", 1, 0.0, 0.0, 1.0, 1.0)]
        wf.num_basis = 8  # 4 from C (SP) + 1 from H (S) + 3 from C (P)
        wf.shells = [
            Shell(-1, 0, np.array([0.5]), np.array([[0.1, 0.15]])),  # SP on C
            Shell(0, 1, np.array([0.3]), np.array([0.2])),  # S on H
            Shell(1, 0, np.array([0.4]), np.array([0.25])),  # P on C
        ]
        mapping = wf.get_atomic_basis_indices()

        # SP shell: indices 0, 1, 2, 3
        # S shell on H: index 4
        # P shell on C: indices 5, 6, 7
        assert sorted(mapping[0]) == [0, 1, 2, 3, 5, 6, 7]
        assert mapping[1] == [4]

    def test_get_atomic_basis_indices_invalid_shell_type(self):
        """Test that invalid shell type raises ValueError."""
        wf = Wavefunction()
        wf.atoms = [Atom("H", 1, 0.0, 0.0, 0.0, 1.0)]
        wf.num_basis = 1
        wf.shells = [
            Shell(-99, 0, np.array([0.5]), np.array([0.1])),  # Invalid type
        ]

        with pytest.raises(ValueError, match="Unknown shell type"):
            wf.get_atomic_basis_indices()


# =============================================================================
# Edge Cases and Type Checking
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and type checking."""

    def test_wavefunction_negative_electrons(self):
        """Test wavefunction with negative electron count (unphysical but should not crash)."""
        wf = Wavefunction()
        wf.num_electrons = -1.0
        # Should not raise error, though unphysical
        assert wf.num_electrons == -1.0

    def test_wavefunction_high_multiplicity(self):
        """Test wavefunction with high multiplicity."""
        wf = Wavefunction()
        wf.num_electrons = 3.0
        wf.multiplicity = 4  # Triplet state
        wf.is_unrestricted = True
        # Should handle correctly
        assert wf.multiplicity == 4

    def test_atom_zero_coordinates(self):
        """Test atom at origin."""
        atom = Atom("O", 8, 0.0, 0.0, 0.0, 8.0)
        np.testing.assert_array_equal(atom.coord, np.zeros(3))

    def test_atom_negative_coordinates(self):
        """Test atom with negative coordinates."""
        atom = Atom("C", 6, -1.5, -2.3, -0.7, 6.0)
        assert atom.x == -1.5
        assert atom.y == -2.3
        assert atom.z == -0.7

    def test_shell_empty_arrays(self):
        """Test shell with empty exponent/coefficient arrays."""
        shell = Shell(0, 0, np.array([]), np.array([]))
        assert len(shell.exponents) == 0
        assert len(shell.coefficients) == 0

    def test_wavefunction_large_system(self):
        """Test wavefunction representing a large system."""
        wf = Wavefunction()
        # Add 100 atoms
        for i in range(100):
            wf.add_atom("H", 1, float(i), 0.0, 0.0)
        assert wf.num_atoms == 100

    def test_wavefunction_float_electrons(self):
        """Test wavefunction with fractional electron count."""
        wf = Wavefunction()
        wf.num_electrons = 2.5  # Can occur in certain calculations
        assert wf.num_electrons == 2.5

    @pytest.mark.parametrize("invalid_type", ["string", None, [], {}])
    def test_atom_element_type(self, invalid_type):
        """Test that Atom element can be any string-like type."""
        # Python doesn't enforce types at runtime
        # This test documents current behavior
        atom = Atom(invalid_type, 1, 0.0, 0.0, 0.0, 1.0)
        assert atom.element == invalid_type


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for multiple components working together."""

    def test_water_molecule_workflow(self):
        """Test creating a complete water molecule wavefunction."""
        wf = Wavefunction()
        wf.title = "Water"
        wf.method = "DFT"
        wf.basis_set_name = "6-31G"
        wf.num_electrons = 10.0
        wf.charge = 0
        wf.multiplicity = 1

        # Add atoms
        wf.add_atom("O", 8, 0.0, 0.0, 0.0)
        wf.add_atom("H", 1, 0.0, 0.758, 0.504)
        wf.add_atom("H", 1, 0.0, -0.758, 0.504)

        assert wf.num_atoms == 3
        assert wf.atoms[0].element == "O"
        assert wf.atoms[1].element == "H"
        assert wf.atoms[2].element == "H"

    def test_hydrogen_molecule_restricted(self):
        """Test complete H2 molecule with restricted calculation."""
        wf = Wavefunction()
        wf.title = "Hydrogen"
        wf.method = "HF"
        wf.basis_set_name = "STO-3G"
        wf.num_electrons = 2.0
        wf.charge = 0
        wf.multiplicity = 1
        wf.is_unrestricted = False
        wf.num_basis = 2

        # Add atoms
        wf.add_atom("H", 1, 0.0, 0.0, 0.0)
        wf.add_atom("H", 1, 0.0, 0.0, 0.74)

        # Set MO coefficients and energies
        wf.coefficients = np.array([[0.7, 0.7], [0.7, -0.7]])
        wf.energies = np.array([-0.5, 0.3])

        # Infer occupations
        wf._infer_occupations()
        assert wf.occupations[0] == 2.0
        assert wf.occupations[1] == 0.0

        # Calculate density matrices
        wf.calculate_density_matrices()
        assert wf.Palpha is not None
        assert wf.Ptot is not None

        # Synthetic fixture has coefficients but no shell basis data, so it
        # must opt into the orthonormal-basis fallback explicitly.
        with pytest.warns(RuntimeWarning, match="identity overlap matrix fallback"):
            wf.calculate_overlap_matrix(allow_identity_fallback=True)
        assert wf.overlap_matrix is not None

    def test_hydrogen_atom_unrestricted(self):
        """Test complete H atom with unrestricted calculation."""
        wf = Wavefunction()
        wf.title = "Hydrogen Atom"
        wf.method = "UHF"
        wf.basis_set_name = "STO-3G"
        wf.num_electrons = 1.0
        wf.charge = 0
        wf.multiplicity = 2
        wf.is_unrestricted = True
        wf.num_basis = 1

        wf.add_atom("H", 1, 0.0, 0.0, 0.0)

        wf.coefficients = np.array([[1.0]])
        wf.energies = np.array([-0.5])
        wf.coefficients_beta = np.array([[1.0]])
        wf.energies_beta = np.array([-0.45])

        wf._infer_occupations()
        assert wf.occupations[0] == 1.0

        wf.calculate_density_matrices()
        assert wf.Palpha is not None
        assert wf.Pbeta is not None
        assert wf.Ptot is not None
