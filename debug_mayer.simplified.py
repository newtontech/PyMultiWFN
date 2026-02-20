#!/usr/bin/env python3

"""

Debug script for Mayer bond order calculation.

"""

import numpy as np

from pymultiwfn.io.loader import load_wavefunction

from pymultiwfn.analysis.bonding.mayer import calculate_mayer_bond_order

# Load H2 wavefunction

wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"

print(f"Loading wavefunction from: {wfn_path}")

wfn = load_wavefunction(wfn_path)


print(f"\n=== Wavefunction Info ===")

print(f"Number of atoms: {wfn.num_atoms}")

print(f"Number of basis functions: {wfn.num_basis}")

print(f"Is unrestricted: {wfn.is_unrestricted}")


print(f"\n=== Density Matrices ===")

if wfn.Ptot is not None:

    print(f"Ptot shape: {wfn.Ptot.shape}")

    print(f"Ptot sum: {np.sum(wfn.Ptot):.6f}")

    print(f"Ptot diagonal sum: {np.sum(np.diag(wfn.Ptot)):.6f}")

else:

    print("Ptot is None - calculating...")

    wfn.calculate_density_matrices()

    if wfn.Ptot is not None:

        print(f"Ptot calculated - shape: {wfn.Ptot.shape}")

        print(f"Ptot sum: {np.sum(wfn.Ptot):.6f}")


print(f"\n=== Overlap Matrix ===")

if wfn.overlap_matrix is not None:

    print(f"Overlap matrix shape: {wfn.overlap_matrix.shape}")

    print(f"Overlap matrix trace: {np.trace(wfn.overlap_matrix):.6f}")

else:

    print("Overlap matrix is None!")


print(f"\n=== Atomic Basis Indices ===")

atomic_indices = wfn.get_atomic_basis_indices()

for i, indices in enumerate(atomic_indices):

    print(f"Atom {i}: basis functions {indices}")


print(f"\n=== Calculating Mayer Bond Order ===")

result = calculate_mayer_bond_order(wfn)

bond_matrix = result["total"]


print(f"Bond order matrix:")

print(bond_matrix)

print(f"\nH-H bond order (off-diagonal): {bond_matrix[0, 1]:.6f}")

print(f"Expected: ~1.0")


# Manual calculation for debugging

print(f"\n=== Manual Calculation ===")

if wfn.Ptot is not None and wfn.overlap_matrix is not None:

    PS = wfn.Ptot @ wfn.overlap_matrix

    print(f"PS matrix shape: {PS.shape}")

    print(f"PS sum: {np.sum(PS):.6f}")

    # Extract H-H block

    h1_indices = atomic_indices[0]

    h2_indices = atomic_indices[1]

    ps_ij = PS[np.ix_(h1_indices, h2_indices)]

    ps_ji = PS[np.ix_(h2_indices, h1_indices)]

    print(f"\nPS_ij shape: {ps_ij.shape}")

    print(f"PS_ij:\n{ps_ij}")

    print(f"\nPS_ji:\n{ps_ji}")

    print(f"\nElement-wise product:\n{ps_ij * ps_ji}")

    print(f"\nSum (should be bond order): {np.sum(ps_ij * ps_ji):.6f}")
