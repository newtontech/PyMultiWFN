#!/usr/bin/env python3
"""
Debug script for shell information.
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
print(f"Loading wavefunction from: {wfn_path}")
wfn = load_wavefunction(wfn_path)

print(f"\n=== Atoms ===")
for i, atom in enumerate(wfn.atoms):
    print(f"Atom {i}: {atom.element} at ({atom.x:.6f}, {atom.y:.6f}, {atom.z:.6f})")

print(f"\n=== Shells ===")
print(f"Total number of shells: {len(wfn.shells)}")
for i, shell in enumerate(wfn.shells):
    print(f"\nShell {i}:")
    print(f"  Type: {shell.type}")
    print(f"  Center atom index: {shell.center_idx}")
    print(f"  Number of exponents: {len(shell.exponents)}")
    print(f"  Number of coefficients: {len(shell.coefficients)}")

print(f"\n=== Atomic Basis Indices ===")
atomic_indices = wfn.get_atomic_basis_indices()
for i, indices in atomic_indices.items():
    print(f"Atom {i}: basis functions {indices} (count: {len(indices)})")

print(f"\n=== Expected vs Actual ===")
print(f"Number of atoms: {wfn.num_atoms}")
print(f"Number of basis functions: {wfn.num_basis}")
print(
    f"Total basis functions from atoms: {sum(len(indices) for indices in atomic_indices.values())}"
)
