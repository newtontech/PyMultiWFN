#!/usr/bin/env python3
"""
Debug overlap matrix calculation for H2 WFN file
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
print(f"Loading WFN file: {wfn_path}")

wfn = load_wavefunction(wfn_path)

print(f"\nWavefunction info:")
print(f"  Num atoms: {wfn.num_atoms}")
print(f"  Num basis: {wfn.num_basis}")
print(f"  Num MOs: {wfn.num_mos}")
print(f"  Num electrons: {wfn.num_electrons}")

print(f"\nOverlap matrix:")
print(f"  Shape: {wfn.overlap_matrix.shape}")
print(f"  Trace: {np.trace(wfn.overlap_matrix):.6f}")
print(f"  Diagonal sum: {np.sum(np.diag(wfn.overlap_matrix)):.6f}")
print(f"  Max diagonal: {np.max(np.diag(wfn.overlap_matrix)):.6f}")
print(f"  Min diagonal: {np.min(np.diag(wfn.overlap_matrix)):.6f}")

# Check if it's identity
is_identity = np.allclose(wfn.overlap_matrix, np.eye(wfn.num_basis))
print(f"  Is identity: {is_identity}")

# Check off-diagonal elements
off_diag = wfn.overlap_matrix - np.diag(np.diag(wfn.overlap_matrix))
max_off_diag = np.max(np.abs(off_diag))
print(f"  Max off-diagonal: {max_off_diag:.6f}")

# Show first 5x5 block
print(f"\nFirst 5x5 block of overlap matrix:")
print(wfn.overlap_matrix[:5, :5])

# Calculate density matrices
wfn.calculate_density_matrices()

print(f"\nDensity matrix info:")
print(f"  Ptot trace: {np.trace(wfn.Ptot):.6f}")
print(f"  Expected trace (num_electrons): {wfn.num_electrons:.6f}")

# Calculate PS = P @ S
PS = wfn.Ptot @ wfn.overlap_matrix
print(f"\nPS = P @ S:")
print(f"  Trace: {np.trace(PS):.6f}")
print(f"  Expected trace (num_electrons): {wfn.num_electrons:.6f}")

# Calculate bond orders
from pymultiwfn.analysis.bonding.mayer import calculate_mayer_bond_order

mayer_result = calculate_mayer_bond_order(wfn)
mayer_bo = mayer_result["total"]

print(f"\nMayer bond order matrix (2x2 for H2):")
print(mayer_bo)
print(f"  H-H bond order: {mayer_bo[0, 1]:.6f}")

from pymultiwfn.analysis.bonding.bondorder import calculate_wiberg_bond_order

wiberg_result = calculate_wiberg_bond_order(wfn)
wiberg_bo = wiberg_result["total"]

print(f"\nWiberg bond order matrix (2x2 for H2):")
print(wiberg_bo)
print(f"  H-H bond order: {wiberg_bo[0, 1]:.6f}")

# Calculate relative difference
if mayer_bo[0, 1] != 0:
    rel_diff = abs(mayer_bo[0, 1] - wiberg_bo[0, 1]) / mayer_bo[0, 1] * 100
    print(f"\nRelative difference: {rel_diff:.2f}%")
else:
    print(f"\nRelative difference: N/A (Mayer bond order is 0)")
