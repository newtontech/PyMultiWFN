#!/usr/bin/env python3
"""
Debug script for electron count and density matrix.
"""

import numpy as np
from pymultiwfn.io.loader import load_wavefunction

# Load H2 wavefunction
wfn_path = "/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn"
print(f"Loading wavefunction from: {wfn_path}")
wfn = load_wavefunction(wfn_path)

print(f"\n=== Wavefunction Info ===")
print(f"Number of electrons: {wfn.num_electrons}")
print(f"Charge: {wfn.charge}")
print(f"Multiplicity: {wfn.multiplicity}")
print(f"Is unrestricted: {wfn.is_unrestricted}")

# Calculate density matrices
wfn.calculate_density_matrices()

print(f"\n=== Density Matrix (Ptot) ===")
print(f"Shape: {wfn.Ptot.shape}")
print(f"Trace (should equal num_electrons): {np.trace(wfn.Ptot):.6f}")
print(f"Expected: {wfn.num_electrons:.6f}")

print(f"\n=== Orbitals ===")
if wfn.coefficients is not None:
    print(f"Number of MOs: {wfn.coefficients.shape[0]}")
    print(f"Number of basis functions: {wfn.coefficients.shape[1]}")

if wfn.occupations is not None:
    print(f"\nOccupations:")
    print(f"  First 10: {wfn.occupations[:10]}")
    print(f"  Sum of occupations: {np.sum(wfn.occupations):.6f}")

if wfn.energies is not None:
    print(f"\nEnergies (first 10):")
    print(f"  {wfn.energies[:10]}")

# Manual density matrix calculation
if wfn.coefficients is not None and wfn.occupations is not None:
    print(f"\n=== Manual Density Matrix Calculation ===")
    # Density matrix = C * n * C^T
    C = wfn.coefficients
    n = wfn.occupations
    P_manual = (C.T * n) @ C
    print(f"Manual Ptot trace: {np.trace(P_manual):.6f}")
    print(f"Matches calculated Ptot? {np.allclose(P_manual, wfn.Ptot)}")

    # Check if they are close
    diff = np.abs(P_manual - wfn.Ptot)
    print(f"Max difference: {np.max(diff):.6e}")
