#!/usr/bin/env python3
"""
Debug script for density and overlap matrices.
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
print(f"Loading wavefunction from: {wfn_path}")
wfn = load_wavefunction(wfn_path)

# Calculate density matrices
wfn.calculate_density_matrices()

print(f"\n=== Density Matrix (Ptot) ===")
print(f"Shape: {wfn.Ptot.shape}")
print(f"Sum of all elements: {np.sum(wfn.Ptot):.6f}")
print(f"Trace (sum of diagonal): {np.trace(wfn.Ptot):.6f}")
print(f"Min value: {np.min(wfn.Ptot):.6e}")
print(f"Max value: {np.max(wfn.Ptot):.6f}")

print(f"\n=== Overlap Matrix (S) ===")
print(f"Shape: {wfn.overlap_matrix.shape}")
print(f"Trace: {np.trace(wfn.overlap_matrix):.6f}")
print(f"Min value: {np.min(wfn.overlap_matrix):.6e}")
print(f"Max value: {np.max(wfn.overlap_matrix):.6f}")

print(f"\n=== PS Matrix (Ptot @ S) ===")
PS = wfn.Ptot @ wfn.overlap_matrix
print(f"Shape: {PS.shape}")
print(f"Sum of all elements: {np.sum(PS):.6f}")
print(f"Trace: {np.trace(PS):.6f}")
print(f"Min value: {np.min(PS):.6e}")
print(f"Max value: {np.max(PS):.6f}")

# Check if PS is symmetric
print(f"\n=== Symmetry Check ===")
PS_T = PS.T
diff = np.abs(PS - PS_T)
print(f"Max absolute difference between PS and PS.T: {np.max(diff):.6e}")
print(f"Is PS symmetric? {np.allclose(PS, PS.T)}")

# Check element-wise product
print(f"\n=== Element-wise Product (PS * PS.T) ===")
PS_product = PS * PS.T
print(f"Sum of all elements: {np.sum(PS_product):.6f}")
print(f"Trace: {np.trace(PS_product):.6f}")
print(f"Min value: {np.min(PS_product):.6e}")
print(f"Max value: {np.max(PS_product):.6f}")

# Check if this matches the manual calculation
atomic_indices = wfn.get_atomic_basis_indices()
h1_indices = atomic_indices[0]
h2_indices = atomic_indices[1]

ps_ij = PS[np.ix_(h1_indices, h2_indices)]
ps_ji = PS[np.ix_(h2_indices, h1_indices)]

print(f"\n=== Block-wise Calculation ===")
print(f"PS_ij shape: {ps_ij.shape}")
print(f"PS_ji shape: {ps_ji.shape}")
print(f"Element-wise product sum: {np.sum(ps_ij * ps_ji):.6f}")
print(f"Expected H-H bond order: ~1.0")

# Check if PS_ij and PS_ji are transposes
print(f"\n=== PS_ij vs PS_ji ===")
print(f"Is PS_ij.T equal to PS_ji? {np.allclose(ps_ij.T, ps_ji)}")

# Try the correct formula: trace(PS_ij @ PS_ji)
print(f"\n=== Trace Formula ===")
print(f"trace(PS_ij @ PS_ji): {np.trace(ps_ij @ ps_ji):.6f}")

# Try element-wise sum: sum(PS_ij * PS_ji.T)
print(f"\n=== Element-wise with Transpose ===")
print(f"sum(PS_ij * PS_ji.T): {np.sum(ps_ij * ps_ji.T):.6f}")
