"""Test Intrinsic Bond Order Implementation (Issue 3).

This module contains comprehensive tests for intrinsic bond order analysis.
Tests cover intrinsic bond strength, bond polarity correction, bond order
matrix generation, and comparison with Wiberg bond orders.
"""

from pathlib import Path

import numpy as np
import pytest

from pymultiwfn.bonding import Bonding
from pymultiwfn.bonding.intrinsic import (
    ELECTRONEGATIVITY,
    calculate_bond_polarity,
    calculate_intrinsic_bond_order,
    calculate_intrinsic_bond_order_matrix,
    calculate_wiberg_bond_order,
    get_electronegativity,
    intrinsic_bond_order,
)
from pymultiwfn.io import load


class TestElectronegativity:
    """Test electronegativity values and lookups."""

    def test_common_elements(self):
        """Test electronegativity for common elements."""
        # Test a few key elements
        assert abs(get_electronegativity("H") - 2.20) < 0.01
        assert abs(get_electronegativity("C") - 2.55) < 0.01
        assert abs(get_electronegativity("N") - 3.04) < 0.01
        assert abs(get_electronegativity("O") - 3.44) < 0.01
        assert abs(get_electronegativity("F") - 3.98) < 0.01

    def test_case_insensitive(self):
        """Test that element lookup is case-insensitive."""
        assert get_electronegativity("c") == get_electronegativity("C")
        assert get_electronegativity("o") == get_electronegativity("O")

    def test_invalid_element(self):
        """Test that invalid elements raise ValueError."""
        with pytest.raises(ValueError, match="No electronegativity defined"):
            get_electronegativity("Xx")


class TestBondPolarity:
    """Test bond polarity correction calculations."""

    def test_homodiatomic_polarity(self, sample_wfn):
        """Test that homodiatomic bonds have low polarity."""
        wfn = sample_wfn
        # For H2, polarity should be very low (same element)
        atom_to_bfs = wfn.get_atomic_basis_indices()

        if wfn.num_atoms >= 2:
            polarity = calculate_bond_polarity(
                "H", "H", wfn.Ptot, wfn.overlap_matrix, atom_to_bfs[0], atom_to_bfs[1]
            )
            assert (
                0.0 <= polarity < 0.1
            ), f"H-H polarity should be low, got {polarity:.3f}"

    def test_heterodiatomic_polarity(self, sample_wfn):
        """Test that heterodiatomic bonds have higher polarity."""
        wfn = sample_wfn
        # For bonds between different elements, polarity should be > 0
        atom_to_bfs = wfn.get_atomic_basis_indices()

        if wfn.num_atoms >= 2:
            # Find first pair of different elements
            for i in range(wfn.num_atoms):
                for j in range(i + 1, wfn.num_atoms):
                    elem_i = wfn.atoms[i].element
                    elem_j = wfn.atoms[j].element
                    if elem_i != elem_j:
                        polarity = calculate_bond_polarity(
                            elem_i,
                            elem_j,
                            wfn.Ptot,
                            wfn.overlap_matrix,
                            atom_to_bfs[i],
                            atom_to_bfs[j],
                        )
                        assert (
                            0.0 <= polarity <= 1.0
                        ), f"Polarity should be in [0, 1], got {polarity:.3f}"
                        return

    def test_polarity_range(self, sample_wfn):
        """Test that polarity is always in [0, 1]."""
        wfn = sample_wfn
        atom_to_bfs = wfn.get_atomic_basis_indices()

        for i in range(wfn.num_atoms):
            for j in range(i + 1, wfn.num_atoms):
                polarity = calculate_bond_polarity(
                    wfn.atoms[i].element,
                    wfn.atoms[j].element,
                    wfn.Ptot,
                    wfn.overlap_matrix,
                    atom_to_bfs[i],
                    atom_to_bfs[j],
                )
                assert (
                    0.0 <= polarity <= 1.0
                ), f"Polarity {polarity:.3f} not in [0, 1] for {i}-{j}"


class TestWibergBondOrder:
    """Test Wiberg bond order calculations."""

    def test_wiberg_bond_order_positive(self, sample_wfn):
        """Test that Wiberg bond orders are non-negative."""
        wfn = sample_wfn
        atom_to_bfs = wfn.get_atomic_basis_indices()

        for i in range(wfn.num_atoms):
            for j in range(i + 1, wfn.num_atoms):
                wbo = calculate_wiberg_bond_order(
                    wfn.Ptot, wfn.overlap_matrix, atom_to_bfs[i], atom_to_bfs[j]
                )
                assert (
                    wbo >= 0.0
                ), f"Wiberg BO should be non-negative, got {wbo:.3f} for {i}-{j}"

    def test_wiberg_symmetric(self, sample_wfn):
        """Test that Wiberg bond order is symmetric."""
        wfn = sample_wfn
        atom_to_bfs = wfn.get_atomic_basis_indices()

        for i in range(wfn.num_atoms):
            for j in range(i + 1, wfn.num_atoms):
                wbo_ij = calculate_wiberg_bond_order(
                    wfn.Ptot, wfn.overlap_matrix, atom_to_bfs[i], atom_to_bfs[j]
                )
                wbo_ji = calculate_wiberg_bond_order(
                    wfn.Ptot, wfn.overlap_matrix, atom_to_bfs[j], atom_to_bfs[i]
                )
                assert (
                    abs(wbo_ij - wbo_ji) < 1e-10
                ), f"Wiberg BO not symmetric: {wbo_ij:.6f} vs {wbo_ji:.6f}"


class TestIntrinsicBondOrder:
    """Test intrinsic bond order calculations."""

    def test_ibo_single_bond(self, h2_fchk):
        """Test intrinsic bond order for H2 single bond."""
        bond = Bonding(h2_fchk)
        ibo = bond.get_intrinsic_bond_order(atom_i=0, atom_j=1)

        # H2 should have bond order close to 1.0
        assert isinstance(ibo, float), "Bond order should be a float"
        # IBO should be <= Wiberg BO due to polarity correction
        assert 0.0 <= ibo <= 1.5, f"H2 IBO should be in [0, 1.5], got {ibo:.3f}"

    def test_ibo_vs_wiberg(self, sample_wfn):
        """Test that IBO <= Wiberg BO due to polarity correction."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        ibo_matrix = result.bond_order_matrix
        wiberg_matrix = result.wiberg_matrix

        # IBO should be <= Wiberg BO for all bonds
        for i in range(bond.natoms):
            for j in range(i + 1, bond.natoms):
                assert (
                    ibo_matrix[i, j] <= wiberg_matrix[i, j] + 1e-6
                ), f"IBO {ibo_matrix[i, j]:.3f} > Wiberg {wiberg_matrix[i, j]:.3f} for {i}-{j}"

    def test_ibo_matrix_shape(self, sample_wfn):
        """Test that IBO matrix has correct shape."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        assert result.bond_order_matrix.shape == (
            bond.natoms,
            bond.natoms,
        ), f"Matrix shape {result.bond_order_matrix.shape} != ({bond.natoms}, {bond.natoms})"
        assert result.polarity_matrix.shape == (
            bond.natoms,
            bond.natoms,
        ), f"Polarity matrix shape mismatch"
        assert result.wiberg_matrix.shape == (
            bond.natoms,
            bond.natoms,
        ), f"Wiberg matrix shape mismatch"

    def test_ibo_matrix_symmetric(self, sample_wfn):
        """Test that IBO matrix is symmetric."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        ibo_matrix = result.bond_order_matrix
        polarity_matrix = result.polarity_matrix
        wiberg_matrix = result.wiberg_matrix

        assert np.allclose(ibo_matrix, ibo_matrix.T), "IBO matrix not symmetric"
        assert np.allclose(
            polarity_matrix, polarity_matrix.T
        ), "Polarity matrix not symmetric"
        assert np.allclose(
            wiberg_matrix, wiberg_matrix.T
        ), "Wiberg matrix not symmetric"

    def test_ibo_matrix_diagonal_zero(self, sample_wfn):
        """Test that diagonal elements of IBO matrix are zero."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        assert np.allclose(
            np.diag(result.bond_order_matrix), 0.0
        ), "IBO matrix diagonal should be zero"
        assert np.allclose(
            np.diag(result.wiberg_matrix), 0.0
        ), "Wiberg matrix diagonal should be zero"

    def test_atom_indices_validation(self, sample_fchk):
        """Test validation of atom indices."""
        bond = Bonding(sample_fchk)
        wfn = load(sample_fchk)

        # Test out of range atom index
        with pytest.raises((IndexError, ValueError)):
            bond.get_intrinsic_bond_order(atom_i=-1, atom_j=0)

        with pytest.raises((IndexError, ValueError)):
            bond.get_intrinsic_bond_order(atom_i=0, atom_j=wfn.natoms)


class TestBondingClassIntegration:
    """Test integration with Bonding class."""

    def test_bonding_class_has_ibo_methods(self, sample_fchk):
        """Test that Bonding class has IBO methods."""
        bond = Bonding(sample_fchk)

        assert hasattr(
            bond, "get_intrinsic_bond_order"
        ), "Bonding should have get_intrinsic_bond_order method"
        assert hasattr(
            bond, "get_intrinsic_bond_order_matrix"
        ), "Bonding should have get_intrinsic_bond_order_matrix method"

    def test_bonding_ibo_api(self, sample_fchk):
        """Test Bonding class IBO API as specified in issue."""
        bond = Bonding(sample_fchk)

        # Test the API from the issue description
        # Note: The issue uses 1-based indexing, but our implementation uses 0-based
        if bond.natoms >= 2:
            ibo = bond.get_intrinsic_bond_order(atom_i=0, atom_j=1)
            assert isinstance(ibo, float), "IBO should be a float"
            assert ibo >= 0.0, "IBO should be non-negative"

    def test_intrinsic_bond_result_class(self, sample_wfn):
        """Test IntrinsicBondResult dataclass."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        # Test that result has all required attributes
        assert hasattr(result, "bond_order_matrix")
        assert hasattr(result, "polarity_matrix")
        assert hasattr(result, "wiberg_matrix")
        assert hasattr(result, "n_atoms")

        # Test get_bond_order method
        if bond.natoms >= 2:
            ibo = result.get_bond_order(0, 1)
            assert isinstance(ibo, float)

        # Test get_polarity method
        if bond.natoms >= 2:
            polarity = result.get_polarity(0, 1)
            assert isinstance(polarity, float)
            assert 0.0 <= polarity <= 1.0


class TestComparisonWithWiberg:
    """Test comparison between IBO and Wiberg bond orders."""

    def test_ibo_lower_for_polar_bonds(self, water_fchk):
        """Test that IBO is lower than Wiberg for polar bonds."""
        if water_fchk is None:
            pytest.skip("Water test file not available")

        bond = Bonding(water_fchk)
        result = bond.get_intrinsic_bond_order_matrix()

        # For O-H bonds (polar), IBO should be significantly lower than Wiberg
        # Find O-H bonds (assuming O is atom 0, H are atoms 1 and 2)
        if bond.natoms >= 3:
            # Check O-H bonds
            for h_idx in [1, 2]:
                ibo = result.bond_order_matrix[0, h_idx]
                wiberg = result.wiberg_matrix[0, h_idx]

                # O-H is polar, so IBO should be noticeably lower
                # (unless the bond is already very weak)
                if wiberg > 0.5:
                    assert (
                        ibo < wiberg
                    ), f"O-H IBO {ibo:.3f} should be < Wiberg {wiberg:.3f}"

    def test_ibo_similar_for_nonpolar_bonds(self, sample_wfn):
        """Test that IBO is similar to Wiberg for nonpolar bonds."""
        bond = Bonding(sample_wfn)
        result = bond.get_intrinsic_bond_order_matrix()

        # For homodiatomic bonds, IBO should be very close to Wiberg
        # (polarity correction should be minimal)
        for i in range(bond.natoms):
            for j in range(i + 1, bond.natoms):
                elem_i = bond.wfn.atoms[i].element
                elem_j = bond.wfn.atoms[j].element

                if elem_i == elem_j:  # Same element
                    ibo = result.bond_order_matrix[i, j]
                    wiberg = result.wiberg_matrix[i, j]
                    polarity = result.polarity_matrix[i, j]

                    # Polarity should be low for same element
                    assert (
                        polarity < 0.1
                    ), f"Polarity {polarity:.3f} too high for {elem_i}-{elem_j}"

                    # IBO should be close to Wiberg
                    if wiberg > 0.1:  # Only check if bond exists
                        relative_diff = abs(ibo - wiberg) / wiberg
                        assert (
                            relative_diff < 0.1
                        ), f"IBO {ibo:.3f} differs too much from Wiberg {wiberg:.3f}"


# Pytest fixtures for test data
@pytest.fixture
def sample_fchk():
    """Return path to a sample FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / "test_data" / "fchk"
    sample_file = test_data_dir / "sample.fchk"

    if not sample_file.exists():
        # Try alternative locations
        alternatives = [
            Path("/home/yhm/software/PyMultiWFN/test_data/fchk/sample.fchk"),
            Path("/home/yhm/software/PyMultiWFN/test_files/sample.fchk"),
        ]
        for alt in alternatives:
            if alt.exists():
                return str(alt)

        pytest.skip(f"Sample FCHK file not found at {sample_file}")

    return str(sample_file)


@pytest.fixture
def sample_wfn(sample_fchk):
    """Return a Wavefunction object for testing."""
    from pymultiwfn.io import load

    wfn = load(sample_fchk)

    # Ensure density matrices are calculated
    if wfn.Ptot is None:
        wfn.calculate_density_matrices()

    return wfn


@pytest.fixture
def h2_fchk():
    """Return path to H2 FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / "test_data" / "fchk"
    h2_file = test_data_dir / "H2.fchk"

    if not h2_file.exists():
        # Try alternative location
        alt = Path("/home/yhm/software/PyMultiWFN/test_data/fchk/H2.fchk")
        if alt.exists():
            return str(alt)
        pytest.skip(f"H2 FCHK file not found at {h2_file}")

    return str(h2_file)


@pytest.fixture
def water_fchk():
    """Return path to H2O FCHK file."""
    test_data_dir = Path(__file__).parent.parent.parent / "test_data" / "fchk"
    water_file = test_data_dir / "H2O.fchk"

    if not water_file.exists():
        pytest.skip(f"H2O FCHK file not found at {water_file}")
    return str(water_file)
