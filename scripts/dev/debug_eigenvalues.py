#!/usr/bin/env python3
"""
Check eigenvalues of overlap matrix
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
wfn = load_wavefunction(wfn_path)

# Check eigenvalues of overlap matrix
S = wfn.overlap_matrix

print(f"Overlap matrix shape: {S.shape}")
print(f"Overlap matrix trace: {np.trace(S):.6f}")
print(f"Overlap matrix determinant: {np.linalg.det(S):.6e}")

# Compute eigenvalues
eigvals = np.linalg.eigvalsh(S)

print(f"\nEigenvalues of overlap matrix:")
print(f"  Min: {np.min(eigvals):.6e}")
print(f"  Max: {np.max(eigvals):.6e}")
print(f"  Mean: {np.mean(eigvals):.6f}")
print(f"  Num negative: {np.sum(eigvals < 0)}")
print(f"  Num zero: {np.sum(np.abs(eigvals) < 1e-10)}")

# Show first 10 eigenvalues
print(f"\nFirst 10 eigenvalues:")
for i, eigval in enumerate(eigvals[:10]):
    print(f"  {i:2d}: {eigval:.6e}")

# Check condition number
condition_number = np.max(eigvals) / np.min(eigvals[np.abs(eigvals) > 1e-10])
print(f"\nCondition number: {condition_number:.6e}")

# Check symmetry
is_symmetric = np.allclose(S, S.T, rtol=1e-10)
print(f"\nIs symmetric: {is_symmetric}")

# Check if positive definite
is_positive_definite = np.all(eigvals > 0)
print(f"Is positive definite: {is_positive_definite}")
