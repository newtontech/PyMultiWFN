"""Tests for fuzzy bond order helpers and the Bonding public API."""

import numpy as np
import pytest

from pymultiwfn import Bonding as TopLevelBonding
from pymultiwfn.bonding import Bonding
from pymultiwfn.bonding.fuzzy import (
    FuzzyAtom,
    calculate_fuzzy_bond_order_matrix,
    fuzzy_bond_order,
    get_default_vdwa_radius,
)


def test_bonding_is_exported_from_top_level_package():
    assert TopLevelBonding is Bonding


@pytest.fixture
def h2_density_matrix():
    return np.array([[1.0, 1.0], [1.0, 1.0]])


@pytest.fixture
def h2_overlap_matrix():
    return np.eye(2)


@pytest.mark.unit
class TestFuzzyAtomDefinition:
    def test_fuzzy_atom_creation(self):
        atom = FuzzyAtom(
            atom_index=0,
            element="H",
            coordinates=[0.0, 0.0, 0.0],
            vdwa_radius=get_default_vdwa_radius("H"),
        )

        assert atom.symbol == "H"
        assert atom.coordinates.shape == (3,)
        assert atom.fuzzy_factor == 0.5

    def test_fuzzy_atom_rejects_invalid_coordinates(self):
        with pytest.raises(ValueError, match="Coordinates must be 3D"):
            FuzzyAtom(
                atom_index=0,
                element="H",
                coordinates=[0.0, 0.0],
                vdwa_radius=1.2,
            )

    def test_fuzzy_atom_rejects_invalid_factor(self):
        with pytest.raises(ValueError, match="Fuzzy factor"):
            FuzzyAtom(
                atom_index=0,
                element="H",
                coordinates=[0.0, 0.0, 0.0],
                vdwa_radius=1.2,
                fuzzy_factor=0.0,
            )

    def test_fuzzy_atom_accepts_full_partition_factor(self):
        atom = FuzzyAtom(
            atom_index=0,
            element="H",
            coordinates=[0.0, 0.0, 0.0],
            vdwa_radius=1.2,
            fuzzy_factor=1.0,
        )

        assert atom.fuzzy_factor == 1.0

    def test_vdw_radius_lookup_is_case_insensitive(self):
        assert get_default_vdwa_radius("cl") == get_default_vdwa_radius("Cl")


@pytest.mark.unit
class TestFuzzyBondOrderMatrix:
    def test_matrix_is_symmetric_with_zero_diagonal(
        self, h2_density_matrix, h2_overlap_matrix
    ):
        matrix = calculate_fuzzy_bond_order_matrix(
            h2_density_matrix,
            h2_overlap_matrix,
            fuzzy_factor=1.0,
        )

        assert matrix.shape == (2, 2)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)
        assert matrix[0, 1] == pytest.approx(1.0)

    def test_matrix_supports_atom_to_basis_mapping(self):
        density = np.array(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        overlap = np.eye(3)

        matrix = calculate_fuzzy_bond_order_matrix(
            density,
            overlap,
            fuzzy_factor=1.0,
            atomic_basis_indices={0: [0, 1], 1: [2]},
        )

        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == pytest.approx(0.25)
        assert matrix[1, 0] == pytest.approx(0.25)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_matrix_scales_with_fuzzy_factor(
        self, h2_density_matrix, h2_overlap_matrix
    ):
        matrix = calculate_fuzzy_bond_order_matrix(
            h2_density_matrix,
            h2_overlap_matrix,
            fuzzy_factor=0.5,
        )

        assert matrix[0, 1] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "density, overlap, error",
        [
            (np.ones((2, 3)), np.eye(2), "density_matrix must be square"),
            (np.eye(2), np.ones((2, 3)), "overlap_matrix must be square"),
            (np.eye(2), np.eye(3), "same shape"),
        ],
    )
    def test_matrix_rejects_invalid_shapes(self, density, overlap, error):
        with pytest.raises(ValueError, match=error):
            calculate_fuzzy_bond_order_matrix(density, overlap)

    def test_matrix_rejects_invalid_fuzzy_factor(self):
        with pytest.raises(ValueError, match="fuzzy_factor"):
            calculate_fuzzy_bond_order_matrix(np.eye(2), np.eye(2), fuzzy_factor=0.0)


@pytest.mark.unit
class TestFuzzyBondOrder:
    def test_uses_one_based_atom_indices(self, h2_density_matrix, h2_overlap_matrix):
        value = fuzzy_bond_order(
            h2_density_matrix,
            h2_overlap_matrix,
            atom_i=1,
            atom_j=2,
            fuzzy_factor=1.0,
        )

        assert value == pytest.approx(1.0)

    def test_rejects_zero_based_atom_index(self, h2_density_matrix, h2_overlap_matrix):
        with pytest.raises(IndexError, match="out of range"):
            fuzzy_bond_order(h2_density_matrix, h2_overlap_matrix, atom_i=0, atom_j=2)

    def test_rejects_same_atom(self, h2_density_matrix, h2_overlap_matrix):
        with pytest.raises(ValueError, match="different"):
            fuzzy_bond_order(h2_density_matrix, h2_overlap_matrix, atom_i=1, atom_j=1)


@pytest.mark.reference
@pytest.mark.requires_data
class TestBondingReferenceData:
    def test_h2_file_matches_reference_bond_order(self, h2_wavefunction_file):
        bond = Bonding(h2_wavefunction_file)

        fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)
        matrix = bond.get_fuzzy_bond_order_matrix()

        assert fbo == pytest.approx(matrix[0, 1])
        assert fbo == pytest.approx(1.0, abs=0.15)
        assert matrix.shape == (bond.natoms, bond.natoms)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_c2h2_file_has_triple_cc_bond(self, c2h2_wavefunction_file):
        bond = Bonding(c2h2_wavefunction_file)
        matrix = bond.get_fuzzy_bond_order_matrix()

        carbon_indices = [
            idx for idx, atom in enumerate(bond.atoms) if atom.element.upper() == "C"
        ]
        hydrogen_indices = [
            idx for idx, atom in enumerate(bond.atoms) if atom.element.upper() == "H"
        ]
        assert len(carbon_indices) == 2
        assert len(hydrogen_indices) == 2

        cc_bond = matrix[np.ix_(carbon_indices, carbon_indices)].max()
        ch_bonds = matrix[np.ix_(carbon_indices, hydrogen_indices)]

        assert cc_bond == pytest.approx(3.93, abs=0.25)
        assert cc_bond > ch_bonds.max()

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "water_wavefunction_file",
            "n2_wavefunction_file",
            "benzene_wavefunction_file",
            "c2h4_wavefunction_file",
        ],
    )
    def test_reference_files_load_or_skip_cleanly(self, request, fixture_name):
        wavefunction_file = request.getfixturevalue(fixture_name)
        bond = Bonding(wavefunction_file)

        try:
            matrix = bond.get_fuzzy_bond_order_matrix()
        except NotImplementedError as exc:
            pytest.skip(f"Reference overlap representation is unsupported: {exc}")

        assert matrix.shape == (bond.natoms, bond.natoms)
        assert np.allclose(matrix, matrix.T)
