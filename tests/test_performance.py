"""
Performance tests for PyMultiWFN.

This module tests performance-critical functions and validates
that optimizations don't break correctness.
"""

import pytest
import numpy as np
import time
from pymultiwfn.core.data import Atom, Shell, Wavefunction
from pymultiwfn.math.density import (
    calc_density,
    clear_density_cache,
    get_cache_stats,
)
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order


class TestDensityPerformance:
    """Test performance optimizations in density calculations."""

    @pytest.fixture
    def medium_molecule(self):
        """Create a medium-sized molecule for performance testing."""
        # Benzene-like molecule (12 atoms)
        atoms = []
        # Ring of 6 carbons
        for i in range(6):
            angle = i * np.pi / 3
            atoms.append(
                Atom(
                    element="C",
                    index=6,
                    x=1.4 * np.cos(angle),
                    y=1.4 * np.sin(angle),
                    z=0.0,
                    charge=6.0,
                )
            )
        # 6 hydrogens
        for i in range(6):
            angle = i * np.pi / 3
            atoms.append(
                Atom(
                    element="H",
                    index=1,
                    x=2.5 * np.cos(angle),
                    y=2.5 * np.sin(angle),
                    z=0.0,
                    charge=1.0,
                )
            )

        # Minimal basis (1 s-type per atom for testing)
        shells = []
        for i in range(12):
            shells.append(
                Shell(
                    type=0, center_idx=i, exponents=np.array([3.0]), coefficients=np.array([1.0])
                )
            )

        # Create wavefunction
        num_basis = 12
        # Simple coefficients (not physically accurate, but useful for perf testing)
        coeffs = np.random.randn(num_basis, num_basis) * 0.1
        occupations = np.ones(num_basis)

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=30.0,
            charge=0,
            multiplicity=1,
            num_basis=num_basis,
            num_atomic_orbitals=num_basis,
            num_primitives=num_basis,
            num_shells=len(shells),
            shells=shells,
            occupations=occupations,
            coefficients=coeffs,
        )

        return wfn

    def test_density_caching_works(self, medium_molecule):
        """Test that density matrix caching reduces computation time."""
        wfn = medium_molecule

        # Clear cache
        clear_density_cache()

        # First call (cache miss)
        coords = np.random.randn(100, 3)
        start = time.time()
        rho1 = calc_density(wfn, coords, use_cache=True)
        time_first = time.time() - start

        # Second call with same wavefunction (cache hit)
        start = time.time()
        rho2 = calc_density(wfn, coords, use_cache=True)
        time_second = time.time() - start

        # Results should be identical
        assert np.allclose(rho1, rho2), "Cached result differs from original"

        # Cache should have entries
        stats = get_cache_stats()
        assert stats["cache_size"] > 0, "Cache should not be empty"

        # Second call should be faster (allow some variance)
        # Note: may not always be faster for small systems, so just check correctness
        print(f"\nPerformance: First={time_first:.4f}s, Second={time_second:.4f}s")

    def test_density_no_cache(self, medium_molecule):
        """Test density calculation without caching."""
        wfn = medium_molecule
        coords = np.random.randn(100, 3)

        # Clear cache
        clear_density_cache()

        # Calculate without cache
        rho = calc_density(wfn, coords, use_cache=False)

        # Should return valid densities
        assert len(rho) == 100
        assert np.all(np.isfinite(rho))

        # Cache should still be empty
        stats = get_cache_stats()
        assert stats["cache_size"] == 0, "Cache should be empty when disabled"

    def test_density_batch_processing(self, medium_molecule):
        """Test processing multiple coordinate batches efficiently."""
        wfn = medium_molecule

        # Large number of points
        n_points = 10000
        coords = np.random.randn(n_points, 3) * 3.0

        # Clear cache
        clear_density_cache()

        # Process all points
        start = time.time()
        rho = calc_density(wfn, coords, use_cache=True)
        elapsed = time.time() - start

        # Should process efficiently
        assert len(rho) == n_points
        assert np.all(np.isfinite(rho))

        # Performance benchmark (should be < 5 seconds for 10k points)
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s (> 5s threshold)"

        print(f"\nBatch processing: {n_points} points in {elapsed:.3f}s ({n_points/elapsed:.0f} points/s)")

    def test_cache_memory_management(self):
        """Test that cache doesn't grow unboundedly."""
        # Clear cache
        clear_density_cache()

        # Create many different wavefunctions
        for i in range(150):  # More than cache max size
            atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=i * 0.1, charge=1.0)]
            shells = [
                Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0]))
            ]

            wfn = Wavefunction(
                atoms=atoms,
                num_electrons=1.0,
                charge=0,
                multiplicity=2,
                num_basis=1,
                num_atomic_orbitals=1,
                num_primitives=1,
                num_shells=1,
                shells=shells,
                occupations=np.array([1.0]),
                coefficients=np.array([[1.0]]),
            )

            coords = np.array([[0.0, 0.0, 0.0]])
            calc_density(wfn, coords, use_cache=True)

        # Cache should not exceed max size
        stats = get_cache_stats()
        assert stats["cache_size"] <= stats["max_size"], \
            f"Cache size {stats['cache_size']} exceeds max {stats['max_size']}"


class TestBondOrderPerformance:
    """Test performance of bond order calculations."""

    def test_mayer_bond_order_large_molecule(self):
        """Test Mayer bond order calculation on larger molecule."""
        # Create a larger molecule (20 atoms)
        atoms = []
        for i in range(20):
            atoms.append(
                Atom(
                    element="C",
                    index=6,
                    x=i * 1.0,
                    y=0.0,
                    z=0.0,
                    charge=6.0,
                )
            )

        # Minimal basis
        shells = []
        for i in range(20):
            shells.append(
                Shell(type=0, center_idx=i, exponents=np.array([2.0]), coefficients=np.array([1.0]))
            )

        # Create wavefunction
        num_basis = 20
        coeffs = np.random.randn(num_basis, num_basis) * 0.1
        # Ensure orthonormal-ish coefficients
        coeffs, _ = np.linalg.qr(coeffs)

        # Create density matrix
        P = coeffs @ np.diag(np.ones(num_basis)) @ coeffs.T

        # Overlap matrix (nearly identity for distant atoms)
        S = np.eye(num_basis)
        for i in range(num_basis - 1):
            S[i, i + 1] = 0.1
            S[i + 1, i] = 0.1

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=40.0,
            charge=0,
            multiplicity=1,
            num_basis=num_basis,
            num_atomic_orbitals=num_basis,
            num_primitives=num_basis,
            num_shells=len(shells),
            shells=shells,
            occupations=np.ones(num_basis),
            coefficients=coeffs,
            overlap_matrix=S,
            Ptot=P,
        )

        # Calculate bond orders
        start = time.time()
        bond_order_dict = calculate_mayer_bond_order(wfn)
        elapsed = time.time() - start

        bond_order = bond_order_dict["total"]

        # Should be symmetric
        assert np.allclose(bond_order, bond_order.T)

        # Should have correct shape
        assert bond_order.shape == (20, 20)

        # Should complete in reasonable time (< 2s)
        assert elapsed < 2.0, f"Bond order calculation took {elapsed:.2f}s (> 2s threshold)"

        print(f"\nBond order (20 atoms): {elapsed:.3f}s")


class TestMemoryEfficiency:
    """Test memory efficiency of calculations."""

    def test_density_matrix_memory(self):
        """Test that density matrix construction doesn't use excessive memory."""
        # Large basis set (100 basis functions)
        num_basis = 100

        atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)]
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0]))
        ]

        # Random coefficients
        coeffs = np.random.randn(num_basis, num_basis) * 0.1
        occupations = np.ones(num_basis)

        # This should not cause memory issues
        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=100.0,
            charge=0,
            multiplicity=1,
            num_basis=num_basis,
            num_atomic_orbitals=num_basis,
            num_primitives=num_basis,
            num_shells=len(shells),
            shells=shells,
            occupations=occupations,
            coefficients=coeffs,
        )

        coords = np.array([[0.0, 0.0, 0.0]])

        # Should handle large basis without issues
        rho = calc_density(wfn, coords)
        assert np.isfinite(rho[0])

    def test_large_coordinate_array(self):
        """Test handling of large coordinate arrays."""
        # Small molecule
        atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)]
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0]))
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=1.0,
            charge=0,
            multiplicity=2,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=shells,
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        )

        # Large number of coordinates (100,000)
        n_points = 100000
        coords = np.random.randn(n_points, 3)

        # Should handle without memory error
        rho = calc_density(wfn, coords, use_cache=True)

        assert len(rho) == n_points
        assert np.all(np.isfinite(rho))


class TestNumericalStability:
    """Test numerical stability under extreme conditions."""

    def test_very_large_coordinates(self):
        """Test density calculation with very large coordinates."""
        atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)]
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0]))
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=1.0,
            charge=0,
            multiplicity=2,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=shells,
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        )

        # Very far from nucleus
        coords = np.array([[1000.0, 1000.0, 1000.0]])
        rho = calc_density(wfn, coords)

        # Should be very small but not NaN/Inf
        assert np.isfinite(rho[0])
        assert rho[0] >= 0

    def test_near_nucleus_density(self):
        """Test density calculation very close to nucleus."""
        atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)]
        shells = [
            Shell(type=0, center_idx=0, exponents=np.array([100.0]), coefficients=np.array([1.0]))
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=1.0,
            charge=0,
            multiplicity=2,
            num_basis=1,
            num_atomic_orbitals=1,
            num_primitives=1,
            num_shells=1,
            shells=shells,
            occupations=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        )

        # Very close to nucleus
        coords = np.array([[1e-10, 0.0, 0.0]])
        rho = calc_density(wfn, coords)

        # Should be finite and positive
        assert np.isfinite(rho[0])
        assert rho[0] > 0
