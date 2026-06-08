#!/usr/bin/env python3
"""
Test script to verify that _extract_wfn_basis_functions() returns 34 basis functions.
"""

import sys

sys.path.insert(0, "/home/yhm/software/PyMultiWFN")

from pymultiwfn.io.loader import load_wavefunction

# Load the WFN file
wfn_path = "consistency_verifier/examples/H2_CCSD.wfn"
wfn = load_wavefunction(wfn_path)

print(f"Number of atoms: {wfn.num_atoms}")
print(f"Number of shells: {len(wfn.shells)}")
print(f"Number of basis functions (num_basis): {wfn.num_basis}")
print(
    f"Overlap matrix shape: {wfn.overlap_matrix.shape if wfn.overlap_matrix is not None else 'None'}"
)

# Count basis functions from shells
basis_count = 0
for shell in wfn.shells:
    if shell.type == 0:
        basis_count += 1
    elif shell.type == 1:
        basis_count += 3
    elif shell.type == 2:
        basis_count += 6

print(f"Counted basis functions from shells: {basis_count}")
print(f"Mismatch: {basis_count - wfn.num_basis} functions")

# Check if overlap matrix dimensions match
if wfn.overlap_matrix is not None:
    if wfn.overlap_matrix.shape[0] != wfn.num_basis:
        print(
            f"ERROR: Overlap matrix dimension {wfn.overlap_matrix.shape[0]} doesn't match num_basis {wfn.num_basis}"
        )
    else:
        print(f"✅ Overlap matrix dimensions match num_basis")
