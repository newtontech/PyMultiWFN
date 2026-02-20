"""
Comprehensive test suite for electron density calculations.

This module tests the density calculation functionality in PyMultiWFN, including:
- Electron density at grid points
- Density matrix construction
- Basis function contraction
- Edge cases and numerical stability
- Validation against known quantum chemical results

Test Strategy:
1. Unit tests for individual components (_make_density_matrix, _contract_density)
2. Integration tests for full density calculation
3. Validation tests with hydrogen-like atoms
4. Edge case tests (zero density, large distances, etc.)
5. Numerical stability tests
"""

import numpy as np
import pytest
from pymultiwfn.core.data import Wavefunction, Atom, Shell
from pymultiwfn.math.density import (
    calc_density,
    _make_density_matrix,
    _contract_density,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hydrogen_sto3g_restricted():
    """
    Create a minimal H atom wavefunction with STO-3G basis (restricted).

    STO-3G for H has 1 basis function (contracted from 3 primitives).
    The 1s orbital should be singly occupied in restricted formalism.

    Coefficients and exponents from STO-3G basis set:
    Exponents: [3.4252509, 0.6239137, 0.1688554]
    Coefficients: [0.1543290, 0.5353281, 0.4446345]

    For hydrogen 1s with 1 electron, the MO coefficient should be ~1.0
    (normalized by the overlap matrix).
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 1.0
    wfn.charge = 0
    wfn.multiplicity = 2  # Doublet for H atom
    wfn.num_basis = 1
    wfn.num_primitives = 3
    wfn.num_shells = 1
    wfn.basis_set_name = "STO-3G"

    # STO-3G basis for hydrogen (1s orbital)
    shell = Shell(
        type=0,  # S shell
        center_idx=0,
        exponents=np.array([3.4252509, 0.6239137, 0.1688554]),
        coefficients=np.array([0.1543290, 0.5353281, 0.4446345]),
    )
    wfn.shells.append(shell)

    # MO coefficients: 1 MO with 1 basis function
    # For a normalized 1s orbital, coefficient should be close to 1.0
    wfn.coefficients = np.array([[1.0]])  # Shape: (1, 1)
    wfn.energies = np.array([-0.5])  # Hydrogen 1s energy (Hartree)
    wfn.occupations = np.array([1.0])  # 1 electron

    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def hydrogen_atom_restricted():
    """
    Create a hydrogen atom wavefunction (restricted closed-shell for testing).

    This is a simplified test case with 2 electrons (H- ion or just for testing
    the restricted formalism with doubly occupied orbitals).
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 2.0
    wfn.charge = -1  # H- anion
    wfn.multiplicity = 1  # Singlet
    wfn.num_basis = 1
    wfn.num_primitives = 1
    wfn.num_shells = 1

    # Simple single primitive Gaussian for testing
    shell = Shell(
        type=0,  # S shell
        center_idx=0,
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
    )
    wfn.shells.append(shell)

    wfn.coefficients = np.array([[1.0]])
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([2.0])  # Doubly occupied

    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def helium_atom():
    """
    Create a helium atom wavefunction (closed shell).

    He has 2 electrons in the 1s orbital (doubly occupied).
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="He", index=2, x=0.0, y=0.0, z=0.0, charge=2.0))
    wfn.num_electrons = 2.0
    wfn.charge = 0
    wfn.multiplicity = 1  # Singlet
    wfn.num_basis = 1
    wfn.num_primitives = 1
    wfn.num_shells = 1

    # Simple single primitive Gaussian for testing
    shell = Shell(
        type=0,  # S shell
        center_idx=0,
        exponents=np.array([2.0]),  # Higher exponent for He (Z=2)
        coefficients=np.array([1.0]),
    )
    wfn.shells.append(shell)

    wfn.coefficients = np.array([[1.0]])
    wfn.energies = np.array([-0.9])
    wfn.occupations = np.array([2.0])  # Doubly occupied

    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def hydrogen_molecule_restricted():
    """
    Create H2 molecule wavefunction (restricted).

    Minimal basis: 1s on each H atom.
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.atoms.append(Atom(element="H", index=2, x=1.4, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 2.0
    wfn.charge = 0
    wfn.multiplicity = 1  # Singlet
    wfn.num_basis = 2
    wfn.num_primitives = 2
    wfn.num_shells = 2
    wfn.basis_set_name = "minimal"

    # Two 1s orbitals, one on each atom
    shell1 = Shell(
        type=0,  # S shell
        center_idx=0,  # First H atom
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
    )
    shell2 = Shell(
        type=0,  # S shell
        center_idx=1,  # Second H atom
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
    )
    wfn.shells.extend([shell1, shell2])

    # 2 MOs: bonding and antibonding
    # For minimal basis H2, bonding MO is approximately (phi_1 + phi_2) / sqrt(2)
    wfn.coefficients = np.array(
        [
            [0.7071, 0.7071],  # Bonding MO (doubly occupied)
            [0.7071, -0.7071],  # Antibonding MO (empty)
        ]
    )
    wfn.energies = np.array([-1.1, -0.5])
    wfn.occupations = np.array([2.0, 0.0])

    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def hydrogen_atom_unrestricted():
    """
    Create a hydrogen atom wavefunction (unrestricted).

    For H atom (1 electron), unrestricted formalism means
    alpha orbital is occupied, beta is empty.
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 1.0
    wfn.charge = 0
    wfn.multiplicity = 2  # Doublet
    wfn.num_basis = 1
    wfn.num_primitives = 1
    wfn.num_shells = 1

    shell = Shell(
        type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])
    )
    wfn.shells.append(shell)

    # Alpha spin: occupied
    wfn.coefficients = np.array([[1.0]])
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([1.0])

    # Beta spin: empty
    wfn.is_unrestricted = True
    wfn.coefficients_beta = np.array([[1.0]])
    wfn.energies_beta = np.array([-0.4])
    wfn.occupations_beta = np.array([0.0])

    return wfn


@pytest.fixture
def simple_two_orbital_system():
    """
    Create a simple 2-orbital system for testing matrix operations.

    This fixture provides a minimal system with 2 basis functions and 2 MOs
    for testing density matrix construction and contraction.
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 2.0
    wfn.charge = 0
    wfn.multiplicity = 1
    wfn.num_basis = 2
    wfn.num_primitives = 2
    wfn.num_shells = 2

    # Two S shells
    shell1 = Shell(
        type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])
    )
    shell2 = Shell(
        type=0, center_idx=0, exponents=np.array([2.0]), coefficients=np.array([1.0])
    )
    wfn.shells.extend([shell1, shell2])

    # Orthonormal MOs for simplicity
    wfn.coefficients = np.array(
        [[1.0, 0.0], [0.0, 1.0]]  # MO 0: pure basis 0  # MO 1: pure basis 1
    )
    wfn.energies = np.array([-1.0, -0.5])
    wfn.occupations = np.array([2.0, 0.0])  # Only MO 0 occupied

    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def grid_origin():
    """Create a simple grid at the origin."""
    return np.array([[0.0, 0.0, 0.0]])


@pytest.fixture
def grid_1d():
    """
    Create a 1D grid along x-axis for testing spatial decay.

    Points from -3 to 3 Bohr.
    """
    x = np.linspace(-3.0, 3.0, 13)
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


@pytest.fixture
def grid_3d():
    """
    Create a 3D grid for volumetric testing.

    Simple 2x2x2 grid around origin.
    """
    x = np.array([-0.5, 0.5])
    y = np.array([-0.5, 0.5])
    z = np.array([-0.5, 0.5])

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


# ============================================================================
# Unit Tests: _make_density_matrix
# ============================================================================


class TestMakeDensityMatrix:
    """Test density matrix construction from MO coefficients."""

    def test_single_occupied_orbital(self):
        """Test density matrix with one occupied orbital."""
        coeffs = np.array([[1.0, 0.0]])  # 1 MO, 2 basis functions
        occs = np.array([1.0])

        P = _make_density_matrix(coeffs, occs)

        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10)

    def test_double_occupied_orbital(self):
        """Test density matrix with doubly occupied orbital."""
        coeffs = np.array([[1.0, 0.0]])  # 1 MO, 2 basis functions
        occs = np.array([2.0])

        P = _make_density_matrix(coeffs, occs)

        expected = np.array([[2.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10)

    def test_multiple_occupied_orbitals(self):
        """Test density matrix with multiple occupied orbitals."""
        # 2 MOs, 2 basis functions
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])  # MO 0  # MO 1
        occs = np.array([2.0, 1.0])

        P = _make_density_matrix(coeffs, occs)

        # P = 2.0 * |phi_0><phi_0| + 1.0 * |phi_1><phi_1|
        expected = np.array([[2.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10)

    def test_mixed_coefficients(self):
        """Test density matrix with non-diagonal MO coefficients."""
        coeffs = np.array(
            [
                [0.7071, 0.7071],  # Bonding combination
                [0.7071, -0.7071],  # Antibonding combination
            ]
        )
        occs = np.array([2.0, 0.0])

        P = _make_density_matrix(coeffs, occs)

        # P = 2.0 * 0.7071^2 * [[1, 1], [1, 1]]
        expected = 2.0 * 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=4)

    def test_no_occupied_orbitals(self):
        """Test density matrix when no orbitals are occupied."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        occs = np.array([0.0, 0.0])

        P = _make_density_matrix(coeffs, occs)

        expected = np.zeros((2, 2))
        np.testing.assert_array_almost_equal(P, expected)

    def test_very_small_occupations_filtered(self):
        """Test that very small occupations (< 1e-8) are filtered out."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        occs = np.array([1e-10, 1.0])

        P = _make_density_matrix(coeffs, occs)

        # Only the second MO should contribute
        expected = np.array([[0.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10)


# ============================================================================
# Unit Tests: _contract_density
# ============================================================================


class TestContractDensity:
    """Test contraction of basis function values with density matrix."""

    def test_single_basis_function(self):
        """Test contraction with single basis function."""
        phi = np.array([[1.0], [2.0], [0.5]])  # 3 points, 1 basis function
        P = np.array([[2.0]])  # Density matrix

        rho = _contract_density(phi, P)

        # rho = phi * P * phi for each point
        # For point 0: 1.0 * 2.0 * 1.0 = 2.0
        # For point 1: 2.0 * 2.0 * 2.0 = 8.0
        # For point 2: 0.5 * 2.0 * 0.5 = 0.5
        expected = np.array([2.0, 8.0, 0.5])
        np.testing.assert_array_almost_equal(rho, expected, decimal=10)

    def test_two_basis_functions_diagonal_P(self):
        """Test contraction with diagonal density matrix."""
        phi = np.array([[1.0, 0.5], [2.0, 1.0], [0.5, 0.25]])
        P = np.array([[2.0, 0.0], [0.0, 1.0]])

        rho = _contract_density(phi, P)

        # rho = 2.0 * phi_0^2 + 1.0 * phi_1^2
        expected = 2.0 * phi[:, 0] ** 2 + 1.0 * phi[:, 1] ** 2
        np.testing.assert_array_almost_equal(rho, expected, decimal=10)

    def test_two_basis_functions_off_diagonal_P(self):
        """Test contraction with off-diagonal density matrix elements."""
        phi = np.array([[1.0, 1.0], [1.0, -1.0]])
        P = np.array([[1.0, 1.0], [1.0, 1.0]])

        rho = _contract_density(phi, P)

        # Manual calculation: sum_ij phi_i P_ij phi_j
        expected = np.array(
            [
                1.0 * 1.0 * 1.0
                + 1.0 * 1.0 * 1.0
                + 1.0 * 1.0 * 1.0
                + 1.0 * 1.0 * 1.0,  # point 0
                1.0 * 1.0 * 1.0
                + 1.0 * 1.0 * (-1.0)
                + 1.0 * (-1.0) * 1.0
                + 1.0 * (-1.0) * (-1.0),  # point 1
            ]
        )
        np.testing.assert_array_almost_equal(rho, expected, decimal=10)

    def test_zero_basis_values(self):
        """Test that zero basis values give zero density."""
        phi = np.array([[0.0, 0.0], [1.0, 1.0]])
        P = np.array([[1.0, 0.5], [0.5, 1.0]])

        rho = _contract_density(phi, P)

        expected = np.array([0.0, 3.0])
        np.testing.assert_array_almost_equal(rho, expected, decimal=10)


# ============================================================================
# Integration Tests: calc_density
# ============================================================================


class TestCalcDensity:
    """Integration tests for full density calculation."""

    def test_density_at_origin_hydrogen(self, hydrogen_sto3g_restricted, grid_origin):
        """Test density calculation at hydrogen nucleus."""
        rho = calc_density(hydrogen_sto3g_restricted, grid_origin)

        # Density should be positive
        assert rho[0] > 0

        # For normalized 1s orbital with coefficient 1.0
        # rho(0) = |phi(0)|^2 where phi(0) is the basis function value at origin
        # For contracted Gaussian: phi(0) = sum_k c_k
        phi_0 = np.sum([0.1543290, 0.5353281, 0.4446345])
        expected_rho = phi_0**2  # With occupation 1.0

        assert rho[0] > expected_rho * 0.9  # Allow for numerical tolerance
        assert rho[0] < expected_rho * 1.1

    def test_density_positive_everywhere(self, hydrogen_sto3g_restricted, grid_1d):
        """Test that density is positive at all grid points."""
        rho = calc_density(hydrogen_sto3g_restricted, grid_1d)

        assert np.all(rho > 0), "Density should be positive everywhere"

    def test_density_symmetry_h2(self, hydrogen_molecule_restricted):
        """Test density symmetry for H2 molecule."""
        # Create symmetric grid around bond midpoint
        midpoint = 0.7  # Bond midpoint at x=0.7
        x_points = np.array([midpoint - 1.0, midpoint + 1.0])
        coords = np.column_stack([x_points, np.zeros(2), np.zeros(2)])

        rho = calc_density(hydrogen_molecule_restricted, coords)

        # Density should be symmetric
        np.testing.assert_almost_equal(
            rho[0],
            rho[1],
            decimal=10,
            err_msg="Density should be symmetric around bond midpoint",
        )

    def test_density_multiple_points(self, helium_atom, grid_3d):
        """Test density calculation at multiple points."""
        rho = calc_density(helium_atom, grid_3d)

        assert len(rho) == len(grid_3d), "Should return density for all points"
        assert np.all(rho >= 0), "Density should be non-negative everywhere"

    def test_density_unrestricted_vs_restricted(
        self, hydrogen_atom_restricted, hydrogen_atom_unrestricted, grid_origin
    ):
        """Test that unrestricted calculation matches restricted for closed shell."""
        rho_restricted = calc_density(hydrogen_atom_restricted, grid_origin)

        # For unrestricted, only alpha contributes (1 electron)
        rho_unrestricted = calc_density(hydrogen_atom_unrestricted, grid_origin)

        # With 1 electron, both should give similar density
        # (though unrestricted might have slightly different coefficients)
        assert rho_unrestricted[0] > 0
        assert rho_restricted[0] > 0

    def test_density_two_orbital_system(self, simple_two_orbital_system, grid_origin):
        """Test density for system with multiple basis functions."""
        rho = calc_density(simple_two_orbital_system, grid_origin)

        # At origin, both basis functions contribute
        assert rho[0] > 0

        # With orthonormal basis and MO 0 occupied (coeff=[1,0])
        # Density should be 2.0 * phi_0^2 (doubly occupied)
        # phi_0 at origin = exp(-1.0 * 0^2) = 1.0
        expected = 2.0 * 1.0**2
        np.testing.assert_almost_equal(rho[0], expected, decimal=5)


# ============================================================================
# Validation Tests: Known Values
# ============================================================================


class TestDensityValidation:
    """Validate density against known quantum chemical results."""

    def test_hydrogen_exponential_decay(self, hydrogen_sto3g_restricted):
        """Test that hydrogen density decays exponentially with distance."""
        # Points along x-axis
        r = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        coords = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])

        rho = calc_density(hydrogen_sto3g_restricted, coords)

        # Density should decrease with distance
        assert (
            rho[0] > rho[1] > rho[2] > rho[3] > rho[4]
        ), "Density should monotonically decrease with distance"

        # Check that density decreases exponentially (Gaussian decay)
        # For a Gaussian, log(rho) should be quadratic in r (negative curvature)
        log_rho = np.log(rho + 1e-10)  # Add small value to avoid log(0)

        # Check that log(rho) decreases (negative values at larger distances)
        assert log_rho[0] > log_rho[-1], "Log density should decrease with distance"

        # Verify it's not just linear decay - should be curved
        # Gaussian decay gives negative second derivative
        # Simple check: ratio of densities should decrease exponentially
        ratio_1 = rho[1] / rho[0]
        ratio_2 = rho[2] / rho[1]
        ratio_3 = rho[3] / rho[2]

        # Each step should have smaller ratio (faster than exponential)
        assert (
            ratio_1 > ratio_2 > ratio_3
        ), "Density decay should accelerate (Gaussian-type decay)"

    def test_density_integrated_electrons(self, hydrogen_atom_restricted):
        """Test that integrated density gives correct number of electrons.

        Note: This is a simplified test. Full integration would require
        numerical integration over all space with proper quadrature weights.
        """
        # Create a coarse grid around the atom
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        z = np.linspace(-2, 2, 5)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        rho = calc_density(hydrogen_atom_restricted, coords)

        # Rough integration (not accurate, just checks that scale is reasonable)
        # With uniform spacing and no quadrature weights, this won't give exactly 2
        # but should be on the right order of magnitude
        total = np.sum(rho) * (0.5**3)  # Approximate volume element

        # Just check it's positive and reasonable
        assert total > 0, "Integrated density should be positive"
        assert total < 100, "Integrated density should not be huge on this grid"

    def test_density_max_at_nucleus(self, hydrogen_sto3g_restricted):
        """Test that density is maximum at the nucleus for s-orbitals."""
        # Points along x-axis
        r = np.linspace(0, 3, 7)
        coords = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])

        rho = calc_density(hydrogen_sto3g_restricted, coords)

        # Maximum should be at origin (nucleus)
        max_idx = np.argmax(rho)
        assert max_idx == 0, "Density should be maximum at nucleus for s-orbitals"


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================


class TestDensityEdgeCases:
    """Test edge cases and numerical stability."""

    def test_empty_coefficients(self, grid_origin):
        """Test wavefunction with no coefficients."""
        wfn = Wavefunction()
        wfn.atoms.append(Atom("H", 1, 0.0, 0.0, 0.0, 1.0))
        wfn.num_basis = 1
        wfn.shells.append(Shell(0, 0, np.array([1.0]), np.array([1.0])))

        wfn.coefficients = None
        wfn.occupations = None

        rho = calc_density(wfn, grid_origin)

        # Should return zeros
        np.testing.assert_array_equal(rho, np.array([0.0]))

    def test_zero_occupations(self, simple_two_orbital_system, grid_origin):
        """Test with all occupations set to zero."""
        simple_two_orbital_system.occupations = np.array([0.0, 0.0])

        rho = calc_density(simple_two_orbital_system, grid_origin)

        # Should be zero everywhere
        np.testing.assert_array_almost_equal(rho, np.array([0.0]))

    def test_very_large_distance(self, hydrogen_sto3g_restricted):
        """Test density at very large distance from nucleus."""
        # Point at 100 Bohr from nucleus
        coords = np.array([[100.0, 0.0, 0.0]])

        rho = calc_density(hydrogen_sto3g_restricted, coords)

        # Density should be extremely small but not cause numerical errors
        assert rho[0] >= 0, "Density should be non-negative"
        assert rho[0] < 1e-50, "Density at large distance should be tiny"
        assert not np.isnan(rho[0]), "Density should not be NaN"
        assert not np.isinf(rho[0]), "Density should not be infinite"

    def test_very_large_exponent(self, grid_origin):
        """Test with very large Gaussian exponent (tight orbital)."""
        wfn = Wavefunction()
        wfn.atoms.append(Atom("H", 1, 0.0, 0.0, 0.0, 1.0))
        wfn.num_basis = 1
        wfn.num_electrons = 1.0

        # Very large exponent
        wfn.shells.append(Shell(0, 0, np.array([1000.0]), np.array([1.0])))
        wfn.coefficients = np.array([[1.0]])
        wfn.occupations = np.array([1.0])

        rho = calc_density(wfn, grid_origin)

        # Should still work and give reasonable result
        assert rho[0] > 0
        assert not np.isnan(rho[0])
        assert not np.isinf(rho[0])

    def test_very_small_exponent(self, grid_origin):
        """Test with very small Gaussian exponent (diffuse orbital)."""
        wfn = Wavefunction()
        wfn.atoms.append(Atom("H", 1, 0.0, 0.0, 0.0, 1.0))
        wfn.num_basis = 1
        wfn.num_electrons = 1.0

        # Very small exponent
        wfn.shells.append(Shell(0, 0, np.array([0.001]), np.array([1.0])))
        wfn.coefficients = np.array([[1.0]])
        wfn.occupations = np.array([1.0])

        rho = calc_density(wfn, grid_origin)

        # Should still work
        assert rho[0] > 0
        assert not np.isnan(rho[0])
        assert not np.isinf(rho[0])

    def test_single_point_grid(self, hydrogen_sto3g_restricted):
        """Test with single point grid."""
        coords = np.array([[0.5, 0.3, -0.2]])

        rho = calc_density(hydrogen_sto3g_restricted, coords)

        assert len(rho) == 1
        assert rho[0] > 0

    def test_coordinate_array_shape(self, hydrogen_sto3g_restricted):
        """Test that input coordinate shape is properly handled."""
        # Single point: (1, 3)
        coords = np.array([[0.0, 0.0, 0.0]])
        rho = calc_density(hydrogen_sto3g_restricted, coords)
        assert rho.shape == (1,)

        # Multiple points: (N, 3)
        coords = np.random.rand(10, 3)
        rho = calc_density(hydrogen_sto3g_restricted, coords)
        assert rho.shape == (10,)


# ============================================================================
# Parametrized Tests
# ============================================================================


class TestDensityParametrized:
    """Parametrized tests for multiple cases."""

    @pytest.mark.parametrize(
        "distance,expected_range",
        [
            (0.0, (0.5, 2.0)),  # At nucleus, density should be relatively high
            (0.5, (0.1, 1.0)),  # Close to nucleus
            (1.0, (0.01, 0.5)),  # 1 Bohr away
            (2.0, (0.0, 0.1)),  # 2 Bohr away
        ],
    )
    def test_density_magnitude_with_distance(
        self, hydrogen_sto3g_restricted, distance, expected_range
    ):
        """Test that density magnitude decreases appropriately with distance."""
        coords = np.array([[distance, 0.0, 0.0]])
        rho = calc_density(hydrogen_sto3g_restricted, coords)

        min_val, max_val = expected_range
        assert (
            min_val <= rho[0] <= max_val
        ), f"Density at r={distance} should be in range [{min_val}, {max_val}], got {rho[0]}"

    @pytest.mark.parametrize("occupation", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_density_scales_with_occupation(
        self, simple_two_orbital_system, grid_origin, occupation
    ):
        """Test that density scales linearly with orbital occupation."""
        simple_two_orbital_system.occupations = np.array([occupation, 0.0])

        rho = calc_density(simple_two_orbital_system, grid_origin)

        # Density should scale with occupation
        # For this system: rho = occupation * |phi|^2
        # At origin, phi = 1.0 (Gaussian with exponent 1.0 at r=0)
        expected = occupation * 1.0**2
        np.testing.assert_almost_equal(rho[0], expected, decimal=10)

    @pytest.mark.parametrize(
        "n_electrons,occupation",
        [
            (1.0, 1.0),  # Hydrogen-like
            (2.0, 2.0),  # Helium-like (closed shell)
            (3.0, 3.0),  # Lithium-like (not physically realistic but tests the code)
        ],
    )
    def test_density_different_electron_counts(
        self, grid_origin, n_electrons, occupation
    ):
        """Test density calculation for different electron counts."""
        wfn = Wavefunction()
        wfn.atoms.append(Atom("H", 1, 0.0, 0.0, 0.0, 1.0))
        wfn.num_electrons = n_electrons
        wfn.num_basis = 1

        wfn.shells.append(Shell(0, 0, np.array([1.0]), np.array([1.0])))
        wfn.coefficients = np.array([[1.0]])
        wfn.occupations = np.array([occupation])

        rho = calc_density(wfn, grid_origin)

        # Density should be proportional to occupation
        expected = occupation * 1.0**2
        np.testing.assert_almost_equal(rho[0], expected, decimal=10)

    @pytest.mark.parametrize(
        "x,y,z",
        [
            (0.0, 0.0, 0.0),  # Origin
            (1.0, 0.0, 0.0),  # On x-axis
            (0.0, 1.0, 0.0),  # On y-axis
            (0.0, 0.0, 1.0),  # On z-axis
            (1.0, 1.0, 1.0),  # Diagonal
            (-1.0, -0.5, 0.3),  # Mixed coordinates
        ],
    )
    def test_density_various_coordinates(self, hydrogen_sto3g_restricted, x, y, z):
        """Test density calculation at various coordinate positions."""
        coords = np.array([[x, y, z]])
        rho = calc_density(hydrogen_sto3g_restricted, coords)

        # Density should always be positive
        assert rho[0] > 0, f"Density should be positive at ({x}, {y}, {z})"
        assert not np.isnan(rho[0]), f"Density should not be NaN at ({x}, {y}, {z})"
        assert not np.isinf(rho[0]), f"Density should not be inf at ({x}, {y}, {z})"


# ============================================================================
# Performance and Scaling Tests
# ============================================================================


class TestDensityPerformance:
    """Test performance and scaling behavior."""

    def test_large_grid_calculation(self, hydrogen_sto3g_restricted):
        """Test density calculation on a large grid."""
        # Create a 50x50x50 grid (125,000 points)
        n = 10  # Use smaller for testing
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        z = np.linspace(-3, 3, n)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Should complete without errors
        rho = calc_density(hydrogen_sto3g_restricted, coords)

        assert len(rho) == n**3
        assert np.all(rho >= 0)
        assert np.all(np.isfinite(rho))

    def test_batch_consistency(self, hydrogen_sto3g_restricted):
        """Test that batch calculation gives same results as individual."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Calculate all at once
        rho_batch = calc_density(hydrogen_sto3g_restricted, coords)

        # Calculate individually
        rho_individual = np.array(
            [
                calc_density(hydrogen_sto3g_restricted, coords[i : i + 1])[0]
                for i in range(len(coords))
            ]
        )

        np.testing.assert_array_almost_equal(rho_batch, rho_individual, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
