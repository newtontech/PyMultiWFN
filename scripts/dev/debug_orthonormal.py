#!/usr/bin/env python3
"""
Check if MOs are orthonormal
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
wfn = load_wavefunction(wfn_path)

# Check if C^T @ S @ C = I
C = wfn.coefficients  # (nmo, nbasis)
S = wfn.overlap_matrix  # (nbasis, nbasis)

print(f"C shape: {C.shape}")
print(f"S shape: {S.shape}")

# Compute C @ S @ C.T (orthonormality check)
CSCT = C @ S @ C.T

print(f"\nC @ S @ C.T shape: {CSCT.shape}")

# Check diagonal
diagonal = np.diag(CSCT)
print(f"Diagonal elements: {diagonal[:10]}")
print(f"Diagonal range: [{np.min(diagonal):.6f}, {np.max(diagonal):.6f}]")
print(f"Should be ~1.0 for orthonormal MOs")

# Check off-diagonal
off_diag = CSCT - np.diag(np.diag(CSCT))
max_off_diag = np.max(np.abs(off_diag))
print(f"\nMax off-diagonal: {max_off_diag:.6f}")
print(f"Should be ~0.0 for orthonormal MOs")

# Check specific MOs
print(f"\nFirst few MOs:")
for i in range(3):
    norm_squared = np.sum(C[i, :] * C[i, :])
    norm_s = np.sqrt(C[i, :] @ S @ C[i, :])
    print(f"  MO {i}: |C|² = {norm_squared:.6f}, <C_i|S|C_i> = {norm_s:.6f}")
