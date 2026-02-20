#!/usr/bin/env python3
"""
Debug and fix Mulliken population calculation
The issue: expected total population = 10.0, actual = 60.28
"""

import numpy as np
import sys

sys.path.insert(0, "/home/yhm/software/PyMultiWFN")

from pymultiwfn.core.data import Wavefunction, Atom, Shell


def create_water_with_normalized_mo():
    """Create water molecule with properly normalized MO coefficients"""
    wf = Wavefunction()
    wf.charge = 0
    wf.multiplicity = 1
    wf.num_electrons = 10.0
    wf.is_unrestricted = False

    # Add atoms (coordinates in Bohr)
    # O at origin, H atoms at ~0.96 Angstrom, 104.5 degree angle
    r_oh = 0.96 * 1.889726
    angle_half = 104.5 * np.pi / 360.0
    wf.add_atom("O", 8, 0.0, 0.0, 0.0)
    wf.add_atom("H", 1, r_oh * np.sin(angle_half), r_oh * np.cos(angle_half), 0.0)
    wf.add_atom("H", 1, -r_oh * np.sin(angle_half), r_oh * np.cos(angle_half), 0.0)

    # Simplified basis: 7 basis functions (O: 1s, 2s, 2px, 2py, 2pz; H: 1s each)
    wf.num_basis = 7
    wf.num_shells = 5
    wf.shells = [
        Shell(
            type=0,
            center_idx=0,
            exponents=np.array([20.0]),
            coefficients=np.array([[1.0]]),
        ),  # O 1s
        Shell(
            type=0,
            center_idx=0,
            exponents=np.array([2.0]),
            coefficients=np.array([[1.0]]),
        ),  # O 2s
        Shell(
            type=1,
            center_idx=0,
            exponents=np.array([2.0]),
            coefficients=np.array([[1.0]]),
        ),  # O 2p (px, py, pz)
        Shell(
            type=0,
            center_idx=1,
            exponents=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        ),  # H1 1s
        Shell(
            type=0,
            center_idx=2,
            exponents=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        ),  # H2 1s
    ]
    wf.num_shells = 5

    # Create properly normalized MO coefficients
    # For water, we have 7 basis functions
    # We need to occupy 5 orbitals (10 electrons)
    # Each MO should be orthonormal: C @ S @ C.T = I

    n_mo = 7
    n_basis = 7
    n_electrons = 10

    # Start with random coefficients
    rng = np.random.RandomState(42)
    C = rng.randn(n_mo, n_basis)

    # Gram-Schmidt orthogonalization
    for i in range(n_mo):
        # Normalize column i
        C[:, i] = C[:, i] / np.linalg.norm(C[:, i])

        # Subtract projection from other columns
        for j in range(i):
            proj = np.dot(C[:, j], C[:, i]) * C[:, j]
            C[:, i] = C[:, i] - proj

    # Verify orthonormality: C @ C.T should be close to identity
    I = np.eye(n_basis)
    print(f"Orthonormality check: ||I - C @ C.T|| = {np.linalg.norm(I - C @ C.T):.6f}")

    # Set coefficients and occupations
    wf.coefficients = C
    # Create reasonable energy levels (bonding orbital lower, antibonding higher)
    # Water molecular orbitals: 1b1, 2a1, 1b2, 3a1, 1b1*, 2a1*, 3a1*, 4a1
    wf.energies = np.array([-1.2, -0.7, -0.5, -0.3, -0.1, 0.5, 0.8])

    # Occupy lowest 5 orbitals (10 electrons)
    wf.occupations = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0])

    # Calculate density matrices
    wf.calculate_density_matrices()
    wf.calculate_overlap_matrix()

    return wf


def analyze_mulliken_issue():
    """Analyze why Mulliken population is wrong"""
    print("=" * 60)
    print("Mulliken Population Issue Analysis")
    print("=" * 60)

    wf = create_water_with_normalized_mo()

    print(f"\nWavefunction info:")
    print(f"  Num atoms: {wf.num_atoms}")
    print(f"  Num basis: {wf.num_basis}")
    print(f"  Num electrons: {wf.num_electrons}")

    print(f"\nDensity matrix Ptot:")
    print(f"  trace(P): {np.trace(wf.Ptot):.6f}")
    print(f"  sum(diag(P)): {np.sum(np.diag(wf.Ptot)):.6f}")

    print(f"\nOverlap matrix S:")
    print(f"  trace(S): {np.trace(wf.overlap_matrix):.6f}")
    print(f"  sum(diag(S)): {np.sum(np.diag(wf.overlap_matrix)):.6f}")

    # Calculate PS = P @ S
    PS = wf.Ptot @ wf.overlap_matrix

    print(f"\nPS = P @ S:")
    print(f"  trace(PS): {np.trace(PS):.6f}")
    print(f"  sum(diag(PS)): {np.sum(np.diag(PS)):.6f}")
    print(f"  Expected sum: {wf.num_electrons}")

    # Test the current Mulliken calculation
    from pymultiwfn.analysis.population.mulliken import (
        calculate_mulliken_population_and_charges,
    )

    total_pop, total_charges, _, _, _ = calculate_mulliken_population_and_charges(
        wf, wf.overlap_matrix
    )

    print(f"\nCurrent Mulliken population calculation:")
    print(f"  Total populations: {total_pop}")
    print(f"  Sum: {np.sum(total_pop):.6f}")
    print(f"  Expected: {wf.num_electrons}")
    print(f"  Difference: {np.sum(total_pop) - wf.num_electrons:.6f}")

    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)

    # Check what the current code does
    from pymultiwfn.core.data import Wavefunction

    # Get atomic basis indices
    atom_to_bfs = wf.get_atomic_basis_indices()

    print("\nCurrent code calculates population as:")
    print("  for each atom:")
    print("    1. Extract submatrix PS[bfs_i, all_bfs]")
    print("    2. Sum ALL elements in this submatrix")
    print()
    print("BUT CORRECT formula is:")
    print("  for each atom:")
    print("    1. Extract block PS[bfs_i, bfs_i]")
    print("    2. Sum ONLY diagonal elements of this block")
    print()
    print("EXAMPLE (2 atoms, 1 basis each):")
    print("  P = [[1, 0], [0, 1]] (PS = P @ S, S = I)")
    print("  PS[bfs_0, all_bfs] = [[1, 0], [0, 1]]  (sum of ALL elements = 2)")
    print("  PS[bfs_0, bfs_0] = [[1]]  (diagonal element = 1)")
    print("  PS[bfs_1, bfs_1] = [[1]]  (diagonal element = 1)")
    print()
    print(
        "  WRONG calculation: sum(PS[bfs_0, all_bfs]) + sum(PS[bfs_1, all_bfs]) = 2 + 1 = 3"
    )
    print(
        "  CORRECT calculation: sum(diag(PS[bfs_0, bfs_0])) + sum(diag(PS[bfs_1, bfs_1])) = 1 + 1 = 2"
    )
    print()
    print("This is why we're getting 60.28 instead of 10.0!")

    print("\n" + "=" * 60)
    print("SOLUTION")
    print("=" * 60)
    print("Fix: In pymultiwfn/analysis/population/mulliken.py:")
    print("  Change line ~57-60 to:")
    print("  # Old (WRONG):")
    print(
        "  # total_atomic_populations[i] = np.sum(PS_tot_element_wise[np.ix_(bfs_i, range(num_basis))])"
    )
    print()
    print("  # New (CORRECT):")
    print("  # atom_ps_block = PS_tot_element_wise[bfs_i, :][:, bfs_i]")
    print("  # total_atomic_populations[i] = np.sum(np.diag(atom_ps_block))")

    return total_pop, total_charges, wf


if __name__ == "__main__":
    analyze_mulliken_issue()
