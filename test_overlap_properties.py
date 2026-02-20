#!/usr/bin/env python3
"""
Test script to check overlap matrix properties.
"""

import sys

sys.path.insert(0, "/home/yhm/software/PyMultiWFN")

from pymultiwfn.io.loader import load_wavefunction
import numpy as np

# Load the WFN file
wfn_path = "consistency_verifier/examples/H2_CCSD.wfn"
wfn = load_wavefunction(wfn_path)

print(f"Overlap matrix shape: {wfn.overlap_matrix.shape}")
print(f"Trace of overlap matrix: {np.trace(wfn.overlap_matrix):.6f}")
print(f"Max diagonal element: {np.max(np.diag(wfn.overlap_matrix)):.6f}")
print(f"Min diagonal element: {np.min(np.diag(wfn.overlap_matrix)):.6f}")
print(f"Mean diagonal element: {np.mean(np.diag(wfn.overlap_matrix)):.6f}")

# Check symmetry
is_symmetric = np.allclose(wfn.overlap_matrix, wfn.overlap_matrix.T)
print(f"Is symmetric: {is_symmetric}")

# Check max off-diagonal
max_off_diag = np.max(np.abs(wfn.overlap_matrix - np.diag(np.diag(wfn.overlap_matrix))))
print(f"Max absolute off-diagonal: {max_off_diag:.6f}")

# Print first 10 diagonal elements
print("\nFirst 10 diagonal elements:")
for i in range(min(10, wfn.num_basis)):
    print(f"  {i:2d}: {wfn.overlap_matrix[i,i]:.6f}")
