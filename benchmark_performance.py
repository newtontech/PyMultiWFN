#!/usr/bin/env python3
"""
Performance benchmark script for PyMultiWFN.

Benchmarks key operations and reports performance metrics.
Run with: python benchmark_performance.py
"""

import time
import numpy as np
from pymultiwfn.core.data import Atom, Shell, Wavefunction
from pymultiwfn.math.density import calc_density, clear_density_cache, get_cache_stats
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order


def create_test_molecule(n_atoms: int) -> Wavefunction:
    """Create a test molecule with n_atoms."""
    atoms = []
    for i in range(n_atoms):
        atoms.append(
            Atom(
                element="C",
                index=6,
                x=i * 1.5,
                y=0.0,
                z=0.0,
                charge=6.0,
            )
        )

    shells = []
    for i in range(n_atoms):
        shells.append(
            Shell(type=0, center_idx=i, exponents=np.array([2.0]), coefficients=np.array([1.0]))
        )

    num_basis = n_atoms
    coeffs = np.random.randn(num_basis, num_basis) * 0.1
    coeffs, _ = np.linalg.qr(coeffs)  # Orthonormalize

    return Wavefunction(
        atoms=atoms,
        num_electrons=float(n_atoms * 4),
        charge=0,
        multiplicity=1,
        num_basis=num_basis,
        num_atomic_orbitals=num_basis,
        num_primitives=num_basis,
        num_shells=len(shells),
        shells=shells,
        occupations=np.ones(num_basis),
        coefficients=coeffs,
    )


def benchmark_density_calculation():
    """Benchmark density calculation performance."""
    print("\n" + "=" * 60)
    print("DENSITY CALCULATION BENCHMARK")
    print("=" * 60)

    # Test different system sizes
    sizes = [5, 10, 20, 50]
    n_points = 1000

    for n_atoms in sizes:
        wfn = create_test_molecule(n_atoms)
        coords = np.random.randn(n_points, 3) * 3.0

        # Clear cache
        clear_density_cache()

        # Time first call
        start = time.time()
        rho = calc_density(wfn, coords, use_cache=True)
        time_first = time.time() - start

        # Time second call (cached)
        start = time.time()
        rho = calc_density(wfn, coords, use_cache=True)
        time_cached = time.time() - start

        # Calculate speedup
        speedup = time_first / time_cached if time_cached > 0 else float('inf')

        print(f"\n{n_atoms} atoms, {n_points} points:")
        print(f"  First call:   {time_first:.4f}s")
        print(f"  Cached call:  {time_cached:.4f}s")
        print(f"  Speedup:      {speedup:.2f}x")
        print(f"  Rate:         {n_points/time_first:.0f} points/s")


def benchmark_bond_order():
    """Benchmark bond order calculation performance."""
    print("\n" + "=" * 60)
    print("BOND ORDER CALCULATION BENCHMARK")
    print("=" * 60)

    sizes = [5, 10, 20, 50]

    for n_atoms in sizes:
        wfn = create_test_molecule(n_atoms)

        # Add overlap matrix
        S = np.eye(n_atoms)
        for i in range(n_atoms - 1):
            S[i, i + 1] = 0.2
            S[i + 1, i] = 0.2

        # Add density matrix
        P = wfn.coefficients @ np.diag(wfn.occupations) @ wfn.coefficients.T

        wfn.overlap_matrix = S
        wfn.Ptot = P

        # Time calculation
        start = time.time()
        bond_order = calculate_mayer_bond_order(wfn)
        elapsed = time.time() - start

        print(f"\n{n_atoms} atoms:")
        print(f"  Time:     {elapsed:.4f}s")
        print(f"  Rate:     {n_atoms/elapsed:.0f} atoms/s")


def benchmark_cache_efficiency():
    """Benchmark cache efficiency."""
    print("\n" + "=" * 60)
    print("CACHE EFFICIENCY BENCHMARK")
    print("=" * 60)

    # Create molecule
    wfn = create_test_molecule(10)
    coords = np.random.randn(100, 3)

    # Clear cache
    clear_density_cache()

    # Make multiple calls with same wavefunction
    times = []
    for i in range(10):
        start = time.time()
        calc_density(wfn, coords, use_cache=True)
        times.append(time.time() - start)

    # Get cache stats
    stats = get_cache_stats()

    print(f"\n10 consecutive calls with same wavefunction:")
    print(f"  First call:    {times[0]:.4f}s")
    print(f"  Average rest:  {np.mean(times[1:]):.4f}s")
    print(f"  Cache size:    {stats['cache_size']}")
    print(f"  Cache keys:    {len(stats['cache_keys'])}")


def benchmark_memory_usage():
    """Benchmark memory usage for large systems."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE BENCHMARK")
    print("=" * 60)

    import sys

    sizes = [10, 50, 100, 200]

    print(f"\n{'Atoms':<10} {'Basis':<10} {'Mem (MB)':<12} {'Time (s)':<10}")
    print("-" * 45)

    for n_atoms in sizes:
        # Measure memory before
        mem_before = sys.getsizeof([])

        start = time.time()
        wfn = create_test_molecule(n_atoms)
        creation_time = time.time() - start

        # Estimate memory usage
        mem_atoms = sys.getsizeof(wfn.atoms)
        mem_shells = sys.getsizeof(wfn.shells)
        mem_coeffs = sys.getsizeof(wfn.coefficients) if wfn.coefficients is not None else 0
        total_mem = (mem_atoms + mem_shells + mem_coeffs) / (1024 * 1024)  # MB

        print(f"{n_atoms:<10} {n_atoms:<10} {total_mem:<12.2f} {creation_time:<10.4f}")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("PyMultiWFN Performance Benchmarks")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")

    try:
        benchmark_density_calculation()
        benchmark_bond_order()
        benchmark_cache_efficiency()
        benchmark_memory_usage()

        print("\n" + "=" * 60)
        print("BENCHMARKS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
