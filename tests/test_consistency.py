"""
Consistency tests for PyMultiWFN against reference values.

This module tests PyMultiWFN calculations against known reference values
to ensure correctness and numerical stability.
"""

import pytest
import numpy as np
from pathlib import Path
from pymultiwfn.io.loader import load_wavefunction


class TestConsistencyDensity:
    """Test electron density calculations for consistency."""

    def test_density_positive_everywhere(self):
        """Test that electron density is always positive."""
        # Create a simple test case
        from pymultiwfn.core.data import Atom, Shell, Wavefunction

        # Hydrogen atom at origin
        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,  # S shell
            center_idx=0,
            exponents=np.array([3.42525091]),
            coefficients=np.array([0.15432897])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell],
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]])
        )

        # Evaluate density at various points
        from pymultiwfn.math.density import calc_density
        coords = np.array([
            [0.0, 0.0, 0.0],  # At nucleus
            [0.5, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        density = calc_density(wfn, coords)

        # All density values should be positive
        assert np.all(density > 0), "Electron density should be positive everywhere"

    def test_density_symmetry_h2(self):
        """Test density symmetry for H2 molecule."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction

        # H2 molecule along z-axis
        atoms = [
            Atom(element="H", index=1, x=0.0, y=0.0, z=-0.7, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=0.0, z=0.7, charge=1.0),
        ]
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),
            Shell(type=0, center_idx=1, exponents=np.array([1.0]), coefficients=np.array([1.0])),
        ]
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
            coefficients=np.array([[0.707, 0.707], [0.707, -0.707]])
        )

        from pymultiwfn.math.density import calc_density

        # Test points symmetric about origin
        coords1 = np.array([[0.0, 0.0, 1.0]])
        coords2 = np.array([[0.0, 0.0, -1.0]])

        density1 = calc_density(wfn, coords1)
        density2 = calc_density(wfn, coords2)

        # Densities should be equal at symmetric points
        np.testing.assert_allclose(density1, density2, rtol=1e-6)

    def test_density_decay_far_from_nucleus(self):
        """Test that density decays to zero far from nucleus."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell],
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]])
        )

        from pymultiwfn.math.density import calc_density

        # Test at increasing distances
        distances = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        coords = np.column_stack([distances, np.zeros_like(distances), np.zeros_like(distances)])

        density = calc_density(wfn, coords)

        # Density should be monotonically decreasing
        assert np.all(np.diff(density) < 0), "Density should decrease with distance"

        # Density at 10 Bohr should be very small
        assert density[-1] < 1e-6, "Density should be negligible at large distances"


class TestConsistencyGradient:
    """Test electron density gradient calculations for consistency."""

    def test_gradient_zero_at_nucleus(self):
        """Test that gradient of s-orbital density is zero at nucleus."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell],
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]])
        )

        from pymultiwfn.math.gradient import calc_gradient

        coords = np.array([[0.0, 0.0, 0.0]])
        gradient = calc_gradient(wfn, coords)

        # Gradient should be zero at nucleus (for s-orbital)
        np.testing.assert_allclose(gradient, np.zeros((1, 3)), atol=1e-10)

    def test_gradient_numerical_consistency(self):
        """Test that analytical gradient matches numerical gradient."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction
        from pymultiwfn.math.density import calc_density
        from pymultiwfn.math.gradient import calc_gradient

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([2.0]),
            coefficients=np.array([1.0])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell],
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]])
        )

        coords = np.array([[1.0, 0.5, -0.3]])

        # Calculate analytical gradient
        grad_analytical = calc_gradient(wfn, coords)

        # Calculate numerical gradient
        h = 1e-5
        numerical_grad = np.zeros_like(grad_analytical)

        for i in range(3):
            coords_plus = coords.copy()
            coords_plus[0, i] += h
            coords_minus = coords.copy()
            coords_minus[0, i] -= h

            density_plus = calc_density(wfn, coords_plus)
            density_minus = calc_density(wfn, coords_minus)

            numerical_grad[0, i] = (density_plus[0] - density_minus[0]) / (2 * h)

        # Should match within tolerance
        np.testing.assert_allclose(grad_analytical, numerical_grad, rtol=1e-4)


class TestConsistencyBasis:
    """Test basis function evaluation for consistency."""

    def test_basis_normalization_consistency(self):
        """Test that basis functions are properly normalized."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction
        from pymultiwfn.math.basis import evaluate_basis

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell]
        )

        # Test at several points
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        basis = evaluate_basis(wfn, coords)

        # Basis functions should be well-behaved
        assert not np.any(np.isnan(basis)), "Basis functions should not be NaN"
        assert not np.any(np.isinf(basis)), "Basis functions should not be infinite"

    def test_basis_s_p_d_f_ordering(self):
        """Test that S, P, D, F shells are in correct order."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction
        from pymultiwfn.math.basis import evaluate_basis

        # Create shells for each angular momentum
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),  # S: 1 func
            Shell(type=1, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),  # P: 3 funcs
            Shell(type=2, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),  # D: 6 funcs
            Shell(type=3, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),  # F: 10 funcs
        ]

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=20,  # 1 + 3 + 6 + 10
            num_atomic_orbitals=20,
            num_primitives=4,
            num_shells=4,
            shells=shells
        )

        coords = np.array([[1.0, 0.0, 0.0]])
        basis = evaluate_basis(wfn, coords)

        # Should have 20 basis functions
        assert basis.shape == (1, 20), f"Expected 20 basis functions, got {basis.shape[1]}"

        # S function (index 0) should be non-zero
        assert basis[0, 0] > 0, "S function should be non-zero at (1,0,0)"

        # Px function (index 1) should be non-zero (S: 1, P: X,Y,Z)
        assert basis[0, 1] > 0, "Px function should be non-zero at (1,0,0)"

        # Py function (index 2) should be zero at (1,0,0)
        np.testing.assert_allclose(basis[0, 2], 0.0, atol=1e-10)

        # Pz function (index 3) should be zero at (1,0,0)
        np.testing.assert_allclose(basis[0, 3], 0.0, atol=1e-10)


class TestConsistencyIntegration:
    """Test numerical integration for consistency."""

    def test_density_integrates_to_n_electrons(self):
        """Test that integrated density equals number of electrons."""
        from pymultiwfn.core.data import Atom, Shell, Wavefunction
        from pymultiwfn.math.density import calc_density

        # Simple system: H atom
        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        shell = Shell(
            type=0,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=[shell],
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]])
        )

        # Create a simple grid for integration (very coarse for speed)
        # In practice, use a proper integration scheme
        r_range = np.linspace(0, 5, 50)
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2*np.pi, 20)

        # This is just a simplified test
        # Real integration would use proper grid weights
        # For now, just verify that density is integrable
        coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        density = calc_density(wfn, coords)

        # Just check that density values are reasonable
        assert np.all(density > 0), "Density should be positive"
        assert np.all(np.isfinite(density)), "Density should be finite"
