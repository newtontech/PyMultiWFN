#!/usr/bin/env python3
"""
Debug density matrix calculation for H2 WFN file
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

print(f"\nMO coefficients info:")
print(f"  Shape: {wfn.coefficients.shape}")
print(f"  First few elements of first MO:")
print(wfn.coefficients[0, :10])

print(f"\nOccupations info:")
if wfn.occupations is not None:
    print(f"  Occupations: {wfn.occupations}")
    print(f"  Occupied MOs: {np.where(wfn.occupations > 0)[0]}")
    print(f"  Sum of occupations: {np.sum(wfn.occupations):.6f}")
else:
    print(f"  Occupations: None")

# Calculate density matrix
wfn.calculate_density_matrices()

print(f"\nDensity matrix info:")
print(f"  Palpha shape: {wfn.Palpha.shape}")
print(f"  Pbeta shape: {wfn.Pbeta.shape}")
print(f"  Ptot shape: {wfn.Ptot.shape}")

print(f"\nDensity matrix traces:")
print(f"  Palpha trace: {np.trace(wfn.Palpha):.6f}")
print(f"  Pbeta trace: {np.trace(wfn.Pbeta):.6f}")
print(f"  Ptot trace: {np.trace(wfn.Ptot):.6f}")
print(f"  Expected (num_electrons): {wfn.num_electrons:.6f}")

# Check manual calculation
if wfn.coefficients is not None and wfn.occupations is not None:
    print(f"\nManual density calculation:")
    print(f"  C shape: {wfn.coefficients.shape}")
    print(f"  occ shape: {wfn.occupations.shape}")

    # Manual calculation: P = C^T @ diag(occ) @ C
    # But our C is (nmo, nbasis), so:
    # P = (C * occ[:, np.newaxis]).T @ C
    P_manual = np.einsum(
        "oi,oj->ij", wfn.coefficients * wfn.occupations[:, np.newaxis], wfn.coefficients
    )

    print(f"  P_manual trace: {np.trace(P_manual):.6f}")
    print(f"  Ptot trace: {np.trace(wfn.Ptot):.6f}")
    print(f"  Are they equal? {np.allclose(P_manual, wfn.Ptot)}")

    # Check if density matrix is symmetric
    print(f"  Is Ptot symmetric? {np.allclose(wfn.Ptot, wfn.Ptot.T)}")

    # Show first 5x5 block
    print(f"\nFirst 5x5 block of Ptot:")
    print(wfn.Ptot[:5, :5])
