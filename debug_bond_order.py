#!/usr/bin/env python3
"""
Debug script to analyze bond order calculation issues
"""

import numpy as np
from pathlib import Path
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding.mayer import calculate_mayer_bond_order


def analyze_wavefunction(wfn_path):
    """Analyze wavefunction and bond order calculation."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {Path(wfn_path).name}")
    print(f"{'='*70}")

    wfn = load_wavefunction(str(wfn_path))

    print(f"\n# Basic Info")
    print(f"  Number of atoms: {len(wfn.atoms)}")
    print(f"  Number of basis functions: {wfn.num_basis}")
    print(f"  Number of MOs: {wfn.coefficients.shape[0]}")
    print(f"  Number of electrons: {wfn.num_electrons}")

    print(f"\n# Density Matrix")
    print(f"  Palpha shape: {wfn.Palpha.shape}")
    print(f"  Palpha trace: {np.trace(wfn.Palpha):.6f}")
    print(f"  Ptot trace: {np.trace(wfn.Ptot):.6f}")
    print(f"  Ptot max: {np.max(np.abs(wfn.Ptot)):.6f}")
    print(f"  Ptot min: {np.min(wfn.Ptot):.6f}")

    print(f"\n# Overlap Matrix")
    print(f"  Shape: {wfn.overlap_matrix.shape}")
    print(f"  Trace: {np.trace(wfn.overlap_matrix):.6f}")
    print(
        f"  Max off-diagonal: {np.max(np.abs(wfn.overlap_matrix - np.eye(wfn.num_basis))):.6f}"
    )
    print(
        f"  Is near identity: {np.allclose(wfn.overlap_matrix, np.eye(wfn.num_basis), atol=1e-3)}"
    )

    # Calculate PS = P @ S
    PS = wfn.Ptot @ wfn.overlap_matrix
    print(f"\n# PS = P @ S")
    print(f"  Shape: {PS.shape}")
    print(f"  Trace: {np.trace(PS):.6f}")
    print(f"  Max element: {np.max(np.abs(PS)):.6f}")
    print(f"  Min element: {np.min(PS):.6f}")

    # Calculate bond order
    result = calculate_mayer_bond_order(wfn)
    bond_matrix = result["total"]

    print(f"\n# Bond Order Matrix")
    print(f"  Shape: {bond_matrix.shape}")
    print(f"  Trace: {np.trace(bond_matrix):.6f}")
    print(f"  Max bond: {np.max(bond_matrix):.6f}")
    print(f"  Min bond: {np.min(bond_matrix):.6f}")
    print(f"\n  Full matrix:")
    print(f"    {bond_matrix}")

    # Find main bond
    max_val = 0
    max_i, max_j = 0, 0
    for i in range(bond_matrix.shape[0]):
        for j in range(i + 1, bond_matrix.shape[1]):
            if bond_matrix[i, j] > max_val:
                max_val = bond_matrix[i, j]
                max_i, max_j = i, j

    print(f"\n  Main bond: {max_i}-{max_j} = {max_val:.6f}")

    # Show atomic information
    print(f"\n# Atoms")
    for i, atom in enumerate(wfn.atoms):
        print(
            f"  Atom {i}: {atom.element} at ({atom.x:.4f}, {atom.y:.4f}, {atom.z:.4f})"
        )


if __name__ == "__main__":
    test_data_dir = Path("/home/yhm/software/PyMultiWFN/consistency_verifier/examples")

    # Analyze H2
    h2_path = test_data_dir / "H2_CCSD.wfn"
    if h2_path.exists():
        analyze_wavefunction(h2_path)

    # Analyze C2H2
    c2h2_path = test_data_dir / "C2H2.wfn"
    if c2h2_path.exists():
        analyze_wavefunction(c2h2_path)

    print(f"\n{'='*70}")
    print("Analysis complete")
    print(f"{'='*70}")
