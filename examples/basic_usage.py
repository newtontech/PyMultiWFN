#!/usr/bin/env python3
"""
Example: Basic PyMultiWFN Usage

This script demonstrates fundamental PyMultiWFN operations:
1. Loading wavefunction files
2. Accessing molecular properties
3. Calculating electron density
4. Computing bond orders

Usage:
    python basic_usage.py molecule.wfn
"""

import sys
import numpy as np
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order


def main(filepath):
    """Demonstrate basic PyMultiWFN usage."""
    
    print("=" * 60)
    print("PyMultiWFN Basic Usage Example")
    print("=" * 60)
    
    # 1. Load Wavefunction
    print("\n1. Loading Wavefunction")
    print("-" * 60)
    try:
        wfn = load_wavefunction(filepath)
        print(f"✓ Successfully loaded: {filepath}")
        print(f"  Title: {wfn.title}")
        print(f"  Method: {wfn.method}")
        print(f"  Basis: {wfn.basis_set_name}")
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print("  Please provide a valid .wfn or .fch file")
        return
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # 2. Molecular Properties
    print("\n2. Molecular Properties")
    print("-" * 60)
    print(f"  Number of atoms: {wfn.num_atoms}")
    print(f"  Number of electrons: {wfn.num_electrons}")
    print(f"  Charge: {wfn.charge}")
    print(f"  Multiplicity: {wfn.multiplicity}")
    print(f"  Basis functions: {wfn.num_basis}")
    
    print("\n  Atoms:")
    for i, atom in enumerate(wfn.atoms):
        print(f"    {i:2d}. {atom.element:2s} ({atom.x:8.4f}, {atom.y:8.4f}, {atom.z:8.4f})")
    
    # 3. Electron Density
    print("\n3. Electron Density Calculation")
    print("-" * 60)
    
    # Sample points: atomic positions and origin
    coords = np.array([[atom.x, atom.y, atom.z] for atom in wfn.atoms])
    coords = np.vstack([coords, [0.0, 0.0, 0.0]])  # Add origin
    
    density = calc_density(wfn, coords)
    
    print("  Density at atomic positions:")
    for i, atom in enumerate(wfn.atoms):
        print(f"    {atom.element:2s}: {density[i]:.6f}")
    print(f"  Density at origin: {density[-1]:.6f}")
    
    # 4. Bond Order Analysis
    print("\n4. Bond Order Analysis")
    print("-" * 60)
    
    if wfn.overlap_matrix is not None and wfn.Ptot is not None:
        bond_orders = calculate_mayer_bond_order(wfn)
        bo_matrix = bond_orders['total']
        
        print("  Mayer Bond Orders (strength > 0.5):")
        printed = set()
        for i in range(wfn.num_atoms):
            for j in range(i+1, wfn.num_atoms):
                bo = bo_matrix[i, j]
                if bo > 0.5:  # Only print significant bonds
                    atom_i = wfn.atoms[i].element
                    atom_j = wfn.atoms[j].element
                    print(f"    {atom_i}-{atom_j} ({i}-{j}): {bo:.4f}")
                    printed.add((i, j))
        
        if not printed:
            print("    No significant bonds found (threshold: 0.5)")
    else:
        print("  ⚠ Bond order calculation requires overlap and density matrices")
        print("    These may not be available in all wavefunction files")
    
    # 5. Summary
    print("\n5. Summary")
    print("-" * 60)
    print(f"  ✓ Loaded wavefunction successfully")
    print(f"  ✓ Analyzed {wfn.num_atoms} atoms, {wfn.num_electrons} electrons")
    print(f"  ✓ Calculated density at {len(coords)} points")
    if wfn.overlap_matrix is not None:
        print(f"  ✓ Computed bond order matrix")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python basic_usage.py <wavefunction_file>")
        print("Example: python basic_usage.py molecule.wfn")
        sys.exit(1)
    
    main(sys.argv[1])
