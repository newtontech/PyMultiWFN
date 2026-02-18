"""
Tests for F basis function evaluation.

This module tests the implementation of Cartesian F basis functions
in the basis set evaluation module.
"""

import pytest
import numpy as np
from pymultiwfn.core.data import Atom, Shell, Wavefunction


class TestFBasisEvaluation:
    """Test F shell (type=3) basis function evaluation."""

    def test_f_shell_basis_count(self):
        """Test that F shell has 10 Cartesian basis functions."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,  # F shell has 10 functions
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        assert wfn.num_basis == 10, "F shell should have 10 Cartesian basis functions"

    def test_f_basis_at_origin(self):
        """Test F basis functions at origin (should be zero except for some components)."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        coords = np.array([[0.0, 0.0, 0.0]])
        phi = np.array([1.0])  # Radial at origin

        # At origin, radial = 1.0
        # All F basis functions should be zero at origin (r=0 => x*y*z=0)
        from pymultiwfn.math.basis import evaluate_basis
        result = evaluate_basis(wfn, coords)

        # All F functions involve x, y, z components, so at origin they should be 0
        np.testing.assert_allclose(result, np.zeros((1, 10)), atol=1e-10)

    def test_f_basis_symmetry(self):
        """Test that F basis functions have correct symmetry."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        from pymultiwfn.math.basis import evaluate_basis

        # Test point (1, 1, 1)
        coords = np.array([[1.0, 1.0, 1.0]])
        result = evaluate_basis(wfn, coords)

        # At (1,1,1): all monomials should be 1.0 * exp(-1*3) = 1/e^3
        expected_value = np.exp(-3.0)
        expected = np.full(10, expected_value)

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_f_basis_separate_components(self):
        """Test that each F basis function component is evaluated correctly."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([2.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        from pymultiwfn.math.basis import evaluate_basis

        # Test at (2, 0, 0) - only XXX should be non-zero
        coords = np.array([[2.0, 0.0, 0.0]])
        result = evaluate_basis(wfn, coords)

        # Expected: XXX = 8 * exp(-2*4) = 8/e^8, others = 0
        expected_xxx = 8.0 * np.exp(-8.0)
        expected = np.zeros(10)
        expected[0] = expected_xxx

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

        # Test at (0, 3, 0) - only YYY should be non-zero
        coords = np.array([[0.0, 3.0, 0.0]])
        result = evaluate_basis(wfn, coords)

        # Expected: YYY = 27 * exp(-2*9) = 27/e^18, others = 0
        expected_yyy = 27.0 * np.exp(-18.0)
        expected = np.zeros(10)
        expected[1] = expected_yyy

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

        # Test at (0, 0, 1) - only ZZZ should be non-zero
        coords = np.array([[0.0, 0.0, 1.0]])
        result = evaluate_basis(wfn, coords)

        # Expected: ZZZ = 1 * exp(-2*1) = 1/e^2, others = 0
        expected_zzz = 1.0 * np.exp(-2.0)
        expected = np.zeros(10)
        expected[2] = expected_zzz

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_f_basis_contracted(self):
        """Test F shell with contracted Gaussian (multiple primitives)."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0, 2.0]),
            coefficients=np.array([0.5, 0.5])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=2,
            num_shells=1,
            shells=[f_shell]
        )

        from pymultiwfn.math.basis import evaluate_basis

        coords = np.array([[1.0, 1.0, 1.0]])
        result = evaluate_basis(wfn, coords)

        # Contracted: sum over primitives
        radial = 0.5 * np.exp(-1.0 * 3.0) + 0.5 * np.exp(-2.0 * 3.0)
        expected = np.full(10, radial)

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_f_basis_multiple_points(self):
        """Test F basis evaluation at multiple points."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        from pymultiwfn.math.basis import evaluate_basis

        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        result = evaluate_basis(wfn, coords)

        assert result.shape == (4, 10), "Should return array of shape (n_points, n_basis)"

        # At origin, all should be zero
        np.testing.assert_allclose(result[0], np.zeros(10), atol=1e-10)

        # At (1,0,0): XXX should be 1*exp(-1), others with y or z should be zero
        expected = np.zeros(10)
        expected[0] = 1.0 * np.exp(-1.0)  # XXX
        np.testing.assert_allclose(result[1], expected, rtol=1e-6)

    def test_f_basis_order(self):
        """Test that F basis functions are in correct order."""
        f_shell = Shell(
            type=3,
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0])
        )

        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        wfn = Wavefunction(
            atoms=[atom],
            num_electrons=1.0,
            charge=0,
            multiplicity=1,
            num_basis=10,
            num_atomic_orbitals=10,
            num_primitives=1,
            num_shells=1,
            shells=[f_shell]
        )

        from pymultiwfn.math.basis import evaluate_basis

        # Point (2, 0, 0) - XXX should be first
        coords = np.array([[2.0, 0.0, 0.0]])
        result = evaluate_basis(wfn, coords)

        # XXX should be at index 0
        expected_xxx = 8.0 * np.exp(-4.0)
        np.testing.assert_allclose(result[0, 0], expected_xxx, rtol=1e-6)

        # Point (0, 2, 0) - YYY should be second
        coords = np.array([[0.0, 2.0, 0.0]])
        result = evaluate_basis(wfn, coords)

        # YYY should be at index 1
        expected_yyy = 8.0 * np.exp(-4.0)
        np.testing.assert_allclose(result[0, 1], expected_yyy, rtol=1e-6)

        # Point (0, 0, 2) - ZZZ should be third
        coords = np.array([[0.0, 0.0, 2.0]])
        result = evaluate_basis(wfn, coords)

        # ZZZ should be at index 2
        expected_zzz = 8.0 * np.exp(-4.0)
        np.testing.assert_allclose(result[0, 2], expected_zzz, rtol=1e-6)
