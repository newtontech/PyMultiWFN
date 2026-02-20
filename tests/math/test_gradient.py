"""
Comprehensive test suite for electron density gradient calculations.

This module tests the gradient calculation functionality in PyMultiWFN, including:
- Electron density gradient at grid points
- Gradient matrix construction
- Basis function gradient evaluation
- Edge cases and numerical stability
- Validation against numerical differentiation

Test Strategy:
1. Unit tests for individual components (gradient evaluation, contraction)
2. Integration tests for full gradient calculation
3. Validation tests with numerical differentiation
4. Edge case tests (zero gradient, symmetry, etc.)
5. Numerical stability tests
"""

import numpy as np
import pytest
from pymultiwfn.core.data import Wavefunction, Atom, Shell
from pymultiwfn.math.gradient import (
    calc_density_gradient,
    calc_density_laplacian,
    _eval_contraction_gradient,
    _eval_contraction_laplacian,
    _contract_gradient,
    _make_density_matrix,
)
from pymultiwfn.math.density import calc_density

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hydrogen_sto3g_restricted():
    """
    Create a minimal H atom wavefunction with STO-3G basis (restricted).

    STO-3G for H has 1 basis function (contracted from 3 primitives).
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 1.0
    wfn.charge = 0
    wfn.multiplicity = 2
    wfn.num_basis = 1
    wfn.num_primitives = 3
    wfn.num_shells = 1

    # STO-3G basis for hydrogen
    shell = Shell(
        type=0,
        center_idx=0,
        exponents=np.array([3.4252509, 0.6239137, 0.1688554]),
        coefficients=np.array([0.1543290, 0.5353281, 0.4446345]),
    )
    wfn.shells.append(shell)

    wfn.coefficients = np.array([[1.0]])
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([1.0])
    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def hydrogen_sto3g_p_shell():
    """
    Create H atom with a P shell for testing angular derivatives.
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 1.0
    wfn.charge = 0
    wfn.num_basis = 3
    wfn.num_primitives = 2
    wfn.num_shells = 1

    # Simple P shell
    shell = Shell(
        type=1,  # P shell
        center_idx=0,
        exponents=np.array([1.0, 0.5]),
        coefficients=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    wfn.shells.append(shell)

    wfn.coefficients = np.array([[1.0, 0.0, 0.0]])  # Px orbital
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([1.0])
    wfn.is_unrestricted = False

    return wfn


@pytest.fixture
def h2_sto3g():
    """
    Create H2 molecule wavefunction with STO-3G basis.
    """
    wfn = Wavefunction()
    wfn.atoms.append(Atom(element="H", index=1, x=-0.7, y=0.0, z=0.0, charge=1.0))
    wfn.atoms.append(Atom(element="H", index=2, x=0.7, y=0.0, z=0.0, charge=1.0))
    wfn.num_electrons = 2.0
    wfn.charge = 0
    wfn.multiplicity = 1
    wfn.num_basis = 2
    wfn.num_primitives = 6
    wfn.num_shells = 2

    # STO-3G for H atom 1
    shell1 = Shell(
        type=0,
        center_idx=0,
        exponents=np.array([3.4252509, 0.6239137, 0.1688554]),
        coefficients=np.array([0.1543290, 0.5353281, 0.4446345]),
    )
    wfn.shells.append(shell1)

    # STO-3G for H atom 2
    shell2 = Shell(
        type=0,
        center_idx=1,
        exponents=np.array([3.4252509, 0.6239137, 0.1688554]),
        coefficients=np.array([0.1543290, 0.5353281, 0.4446345]),
    )
    wfn.shells.append(shell2)

    # Simple H2 bonding orbital (symmetric combination)
    wfn.coefficients = np.array([[0.7071, 0.7071]])
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([2.0])
    wfn.is_unrestricted = False

    return wfn


# ============================================================================
# Unit Tests: Contraction Gradient
# ============================================================================


class TestContractionGradient:
    """Tests for radial contraction gradient evaluation."""

    def test_single_primitive_gradient(self):
        """Test gradient of single primitive Gaussian."""
        exps = np.array([1.0])
        coeffs = np.array([1.0])

        # Test at (1, 0, 0)
        r_vec = np.array([[1.0, 0.0, 0.0]])
        r2 = np.array([1.0])

        grad = _eval_contraction_gradient(exps, coeffs, r2, r_vec)

        # Analytical: d/dx[exp(-x²)] = -2x*exp(-x²)
        # At x=1: -2*1*exp(-1) = -2/e
        expected_dx = -2.0 * np.exp(-1.0)
        np.testing.assert_allclose(grad[0, 0], expected_dx, rtol=1e-6)
        np.testing.assert_allclose(grad[0, 1], 0.0, rtol=1e-6)
        np.testing.assert_allclose(grad[0, 2], 0.0, rtol=1e-6)

    def test_single_primitive_gradient_yz(self):
        """Test gradient in y and z directions."""
        exps = np.array([1.0])
        coeffs = np.array([1.0])

        r_vec = np.array([[0.0, 2.0, 3.0]])
        r2 = np.array([13.0])  # 0 + 4 + 9 = 13

        grad = _eval_contraction_gradient(exps, coeffs, r2, r_vec)

        # d/dy: -2*y*exp(-r²) = -4*exp(-13)
        expected_dy = -4.0 * np.exp(-13.0)
        # d/dz: -2*z*exp(-r²) = -6*exp(-13)
        expected_dz = -6.0 * np.exp(-13.0)

        np.testing.assert_allclose(grad[0, 0], 0.0, rtol=1e-6)
        np.testing.assert_allclose(grad[0, 1], expected_dy, rtol=1e-6)
        np.testing.assert_allclose(grad[0, 2], expected_dz, rtol=1e-6)

    def test_contracted_gradient(self):
        """Test gradient of contracted Gaussian."""
        exps = np.array([1.0, 2.0])
        coeffs = np.array([0.5, 0.5])

        r_vec = np.array([[1.0, 0.0, 0.0]])
        r2 = np.array([1.0])

        grad = _eval_contraction_gradient(exps, coeffs, r2, r_vec)

        # Sum of contributions from both primitives
        expected = -2.0 * 1.0 * 0.5 * np.exp(-1.0) + -2.0 * 2.0 * 0.5 * np.exp(-2.0)
        np.testing.assert_allclose(grad[0, 0], expected, rtol=1e-6)

    def test_gradient_at_origin(self):
        """Test gradient at origin should be zero for s-type functions."""
        exps = np.array([1.0])
        coeffs = np.array([1.0])

        r_vec = np.array([[0.0, 0.0, 0.0]])
        r2 = np.array([0.0])

        grad = _eval_contraction_gradient(exps, coeffs, r2, r_vec)

        # At origin, gradient of s-type Gaussian is zero
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)


# ============================================================================
# Unit Tests: Contraction Laplacian
# ============================================================================


class TestContractionLaplacian:
    """Tests for radial contraction Laplacian evaluation."""

    def test_single_primitive_laplacian(self):
        """Test Laplacian of single primitive Gaussian."""
        exps = np.array([1.0])
        coeffs = np.array([1.0])

        r2 = np.array([1.0])
        lap = _eval_contraction_laplacian(exps, coeffs, r2)

        # Analytical: ∇²[exp(-αr²)] = (4α²r² - 6α) * exp(-αr²)
        # At r²=1, α=1: (4*1*1 - 6*1)*exp(-1) = -2*exp(-1)
        expected = -2.0 * np.exp(-1.0)
        np.testing.assert_allclose(lap, expected, rtol=1e-6)

    def test_laplacian_at_origin(self):
        """Test Laplacian at origin."""
        exps = np.array([2.0])
        coeffs = np.array([1.0])

        r2 = np.array([0.0])
        lap = _eval_contraction_laplacian(exps, coeffs, r2)

        # At origin (r²=0): (0 - 6α)*1 = -6α = -12
        expected = -12.0
        np.testing.assert_allclose(lap, expected, rtol=1e-6)

    def test_contracted_laplacian(self):
        """Test Laplacian of contracted Gaussian."""
        exps = np.array([1.0, 2.0])
        coeffs = np.array([0.5, 0.5])

        r2 = np.array([1.0])
        lap = _eval_contraction_laplacian(exps, coeffs, r2)

        # Sum of contributions
        term1 = (4 * 1.0 * 1.0 - 6 * 1.0) * 0.5 * np.exp(-1.0)
        term2 = (4 * 2.0 * 2.0 - 6 * 2.0) * 0.5 * np.exp(-2.0)
        expected = term1 + term2

        np.testing.assert_allclose(lap, expected, rtol=1e-6)


# ============================================================================
# Unit Tests: Gradient Contraction
# ============================================================================


class TestGradientContraction:
    """Tests for gradient contraction with density matrix."""

    def test_simple_gradient_contraction(self):
        """Test gradient contraction with simple 1-basis case."""
        phi = np.array([[1.0]])  # 1 point, 1 basis
        # grad_phi shape: (N_points, 3, N_basis) = (1, 3, 1)
        # ∇φ = [1, 0, 0]
        grad_phi = np.array([[[1.0], [0.0], [0.0]]])
        P = np.array([[1.0]])

        grad = _contract_gradient(phi, grad_phi, P)

        # ∇ρ = 2 * P * φ * ∇φ = 2 * 1 * 1 * [1, 0, 0] = [2, 0, 0]
        expected = np.array([[2.0, 0.0, 0.0]])
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_two_basis_gradient(self):
        """Test gradient with two basis functions."""
        phi = np.array([[1.0, 1.0]])
        grad_phi = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]  # ∇φ₁ = [1, 0, 0]  # ∇φ₂ = [0, 1, 0]
        )  # (1, 2, 3) but we need (1, 3, 2)
        grad_phi = grad_phi.transpose(0, 2, 1)  # Now (1, 3, 2)
        P = np.array([[1.0, 0.0], [0.0, 1.0]])

        grad = _contract_gradient(phi, grad_phi, P)

        # For diagonal P: ∇ρ = 2 * Σ_i P_ii φ_i ∇φ_i
        expected = 2 * (
            1.0 * 1.0 * np.array([1.0, 0.0, 0.0])
            + 1.0 * 1.0 * np.array([0.0, 1.0, 0.0])
        )
        expected = expected.reshape(1, 3)
        np.testing.assert_allclose(grad, expected, rtol=1e-6)


# ============================================================================
# Integration Tests: Full Gradient Calculation
# ============================================================================


class TestGradientCalculation:
    """Integration tests for full gradient calculation."""

    def test_hydrogen_gradient_symmetry(self, hydrogen_sto3g_restricted):
        """Test that hydrogen gradient has correct symmetry."""
        wfn = hydrogen_sto3g_restricted

        # Test at several points symmetric about origin
        coords = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        grad = calc_density_gradient(wfn, coords)

        # Gradient should point toward the nucleus
        # At (1, 0, 0): gradient should be negative in x
        assert grad[0, 0] < 0
        # At (-1, 0, 0): gradient should be positive in x
        assert grad[1, 0] > 0

        # Magnitudes should be equal due to symmetry
        np.testing.assert_allclose(
            np.linalg.norm(grad[0]), np.linalg.norm(grad[1]), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.linalg.norm(grad[2]), np.linalg.norm(grad[3]), rtol=1e-5
        )

    def test_hydrogen_gradient_at_nucleus(self, hydrogen_sto3g_restricted):
        """Test that gradient at nucleus is zero."""
        wfn = hydrogen_sto3g_restricted

        # At nucleus (0, 0, 0)
        coords = np.array([[0.0, 0.0, 0.0]])
        grad = calc_density_gradient(wfn, coords)

        # Gradient should be zero at nucleus (maximum of density)
        np.testing.assert_allclose(grad, 0.0, atol=1e-5)

    def test_h2_gradient_bond_midpoint(self, h2_sto3g):
        """Test gradient at H2 bond midpoint."""
        wfn = h2_sto3g

        # At bond midpoint (0, 0, 0)
        coords = np.array([[0.0, 0.0, 0.0]])
        grad = calc_density_gradient(wfn, coords)

        # Due to symmetry, gradient should be close to zero
        np.testing.assert_allclose(grad, 0.0, atol=1e-4)

    def test_h2_gradient_bond_axis(self, h2_sto3g):
        """Test gradient along H2 bond axis."""
        wfn = h2_sto3g

        # Points along bond axis
        coords = np.array(
            [
                [-1.0, 0.0, 0.0],  # Left of left H
                [0.0, 0.0, 0.0],  # Between atoms
                [1.0, 0.0, 0.0],  # Right of right H
            ]
        )

        grad = calc_density_gradient(wfn, coords)

        # All gradients should be along x-axis (bond direction)
        np.testing.assert_allclose(grad[:, 1], 0.0, atol=1e-5)  # y component
        np.testing.assert_allclose(grad[:, 2], 0.0, atol=1e-5)  # z component

        # Gradient direction should point toward density maximum
        # At (-1, 0, 0): should point right (positive x)
        assert grad[0, 0] > 0
        # At (1, 0, 0): should point left (negative x)
        assert grad[2, 0] < 0


# ============================================================================
# Validation Tests: Numerical Differentiation
# ============================================================================


class TestNumericalValidation:
    """Validate gradient against numerical differentiation."""

    def test_gradient_vs_numerical(self, hydrogen_sto3g_restricted):
        """Test analytical gradient against numerical differentiation."""
        wfn = hydrogen_sto3g_restricted

        # Test point
        coord = np.array([0.5, 0.3, 0.2])

        # Analytical gradient
        grad_analytical = calc_density_gradient(wfn, coord.reshape(1, 3))[0]

        # Numerical gradient using central difference
        h = 1e-5
        grad_numerical = np.zeros(3)

        for d in range(3):
            coord_plus = coord.copy()
            coord_minus = coord.copy()
            coord_plus[d] += h
            coord_minus[d] -= h

            rho_plus = calc_density(wfn, coord_plus.reshape(1, 3))[0]
            rho_minus = calc_density(wfn, coord_minus.reshape(1, 3))[0]

            grad_numerical[d] = (rho_plus - rho_minus) / (2 * h)

        # Compare analytical and numerical gradients
        np.testing.assert_allclose(
            grad_analytical, grad_numerical, rtol=1e-3, atol=1e-5
        )

    def test_gradient_vs_numerical_h2(self, h2_sto3g):
        """Test gradient vs numerical for H2 molecule."""
        wfn = h2_sto3g

        # Test at several points
        test_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        )

        for i, coord in enumerate(test_points):
            grad_analytical = calc_density_gradient(wfn, coord.reshape(1, 3))[0]

            # Numerical gradient
            h = 1e-5
            grad_numerical = np.zeros(3)

            for d in range(3):
                coord_plus = coord.copy()
                coord_minus = coord.copy()
                coord_plus[d] += h
                coord_minus[d] -= h

                rho_plus = calc_density(wfn, coord_plus.reshape(1, 3))[0]
                rho_minus = calc_density(wfn, coord_minus.reshape(1, 3))[0]

                grad_numerical[d] = (rho_plus - rho_minus) / (2 * h)

            np.testing.assert_allclose(
                grad_analytical,
                grad_numerical,
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"Failed at point {i}: {coord}",
            )


# ============================================================================
# Laplacian Tests
# ============================================================================


class TestLaplacianCalculation:
    """Tests for Laplacian calculation."""

    def test_hydrogen_laplacian_at_nucleus(self, hydrogen_sto3g_restricted):
        """Test Laplacian at hydrogen nucleus."""
        wfn = hydrogen_sto3g_restricted

        coords = np.array([[0.0, 0.0, 0.0]])
        lap = calc_density_laplacian(wfn, coords)

        # Laplacian at nucleus should be negative (density maximum)
        assert lap[0] < 0

    def test_hydrogen_laplacian_far_field(self, hydrogen_sto3g_restricted):
        """Test Laplacian far from nucleus."""
        wfn = hydrogen_sto3g_restricted

        coords = np.array([[10.0, 0.0, 0.0]])
        lap = calc_density_laplacian(wfn, coords)

        # Far from nucleus, density and its derivatives should be near zero
        np.testing.assert_allclose(lap, 0.0, atol=1e-10)

    def test_h2_laplacian_symmetry(self, h2_sto3g):
        """Test Laplacian symmetry for H2."""
        wfn = h2_sto3g

        coords = np.array(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        )

        lap = calc_density_laplacian(wfn, coords)

        # Due to symmetry, Laplacians should be equal
        np.testing.assert_allclose(lap[0], lap[1], rtol=1e-5)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_density_gradient(self, hydrogen_sto3g_restricted):
        """Test gradient where density is zero."""
        wfn = hydrogen_sto3g_restricted

        # Very far from nucleus
        coords = np.array([[100.0, 0.0, 0.0]])
        grad = calc_density_gradient(wfn, coords)

        # Gradient should be zero (density is zero)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_large_exponent_stability(self):
        """Test numerical stability with large exponents."""
        wfn = Wavefunction()
        wfn.atoms.append(Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0))
        wfn.num_electrons = 1.0
        wfn.charge = 0
        wfn.multiplicity = 2
        wfn.num_basis = 1
        wfn.num_primitives = 1
        wfn.num_shells = 1

        # Very tight Gaussian (large exponent)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([100.0]),
            coefficients=np.array([1.0]),
        )
        wfn.shells.append(shell)

        wfn.coefficients = np.array([[1.0]])
        wfn.energies = np.array([-0.5])
        wfn.occupations = np.array([1.0])
        wfn.is_unrestricted = False

        # At a point where density is very small but not zero
        coords = np.array([[1.0, 0.0, 0.0]])

        # Should not overflow or produce NaN
        grad = calc_density_gradient(wfn, coords)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance and vectorization tests."""

    def test_vectorized_gradient_calculation(self, h2_sto3g):
        """Test that gradient calculation is properly vectorized."""
        wfn = h2_sto3g

        # Generate many points
        n_points = 1000
        coords = np.random.randn(n_points, 3)

        # Should complete quickly
        import time

        start = time.time()
        grad = calc_density_gradient(wfn, coords)
        elapsed = time.time() - start

        # Should take less than 1 second for 1000 points
        assert elapsed < 1.0

        # Check output shape
        assert grad.shape == (n_points, 3)

    def test_gradient_density_consistency(self, h2_sto3g):
        """Test that gradient is consistent with density values."""
        wfn = h2_sto3g

        # Generate points on a sphere
        theta = np.linspace(0, 2 * np.pi, 10)
        phi = np.linspace(0, np.pi, 10)
        coords = []

        for t in theta:
            for p in phi:
                r = 1.0
                x = r * np.sin(p) * np.cos(t)
                y = r * np.sin(p) * np.sin(t)
                z = r * np.cos(p)
                coords.append([x, y, z])

        coords = np.array(coords)
        grad = calc_density_gradient(wfn, coords)
        rho = calc_density(wfn, coords)

        # Gradient magnitude should correlate with density changes
        # (This is a qualitative check)
        assert grad.shape[0] == rho.shape[0]
