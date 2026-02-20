#!/usr/bin/env python3
"""
Example: Bond Order Analysis

This script demonstrates bond order analysis:
1. Calculating Mayer and Wiberg bond orders
2. Comparing different methods
3. Identifying bond types
4. Generating bond order reports

Usage:
    python bond_analysis.py molecule.wfn
"""

import sys
import numpy as np
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding.bondorder import (
    calculate_mayer_bond_order,
    calculate_wiberg_bond_order,
)


def classify_bond(bond_order):
    """Classify bond type based on bond order."""
    if bond_order < 0.3:
        return "No bond"
    elif bond_order < 0.7:
        return "Weak"
    elif bond_order < 1.3:
        return "Single"
    elif bond_order < 1.7:
        return "Partial double"
    elif bond_order < 2.3:
        return "Double"
    elif bond_order < 2.7:
        return "Partial triple"
    elif bond_order < 3.3:
        return "Triple"
    else:
        return "Strong multiple"


def analyze_bonds(wfn):
    """Perform comprehensive bond order analysis."""
    
    print("\n" + "=" * 60)
    print("Bond Order Analysis")
    print("=" * 60)
    
    # Check if required matrices are available
    if wfn.overlap_matrix is None:
        print("\n⚠ Warning: Overlap matrix not available")
        print("Bond order calculation requires overlap matrix.")
        print("Please use a wavefunction file that includes overlap matrix data.")
        return
    
    if wfn.Ptot is None:
        print("\n⚠ Warning: Density matrix not available")
        print("Bond order calculation requires density matrix.")
        return
    
    # 1. Calculate Mayer bond orders
    print("\n1. Mayer Bond Orders")
    print("-" * 60)
    
    mayer = calculate_mayer_bond_order(wfn)
    mayer_bo = mayer['total']
    
    # Print all significant bonds
    bonds = []
    for i in range(wfn.num_atoms):
        for j in range(i+1, wfn.num_atoms):
            bo = mayer_bo[i, j]
            if bo > 0.1:  # Threshold for "significant" bond
                atom_i = wfn.atoms[i].element
                atom_j = wfn.atoms[j].element
                bonds.append((i, j, atom_i, atom_j, bo))
    
    # Sort by bond order
    bonds.sort(key=lambda x: x[4], reverse=True)
    
    print(f"  Found {len(bonds)} significant bonds (BO > 0.1):")
    print("\n  Atom I  Atom J  |  Bond Order  |  Type")
    print("  " + "-" * 50)
    
    for i, j, atom_i, atom_j, bo in bonds:
        bond_type = classify_bond(bo)
        print(f"  {atom_i:2s} ({i:2d})  {atom_j:2s} ({j:2d})  |  {bo:8.4f}   |  {bond_type}")
    
    # 2. Compare Mayer vs Wiberg
    print("\n2. Mayer vs Wiberg Comparison")
    print("-" * 60)
    
    wiberg = calculate_wiberg_bond_order(wfn)
    wiberg_bo = wiberg['total']
    
    print("  Bond         |  Mayer   |  Wiberg  |  Difference")
    print("  " + "-" * 55)
    
    for i, j, atom_i, atom_j, mayer_val in bonds[:10]:  # Top 10 bonds
        wiberg_val = wiberg_bo[i, j]
        diff = abs(mayer_val - wiberg_val)
        print(f"  {atom_i:2s}-{atom_j:2s} ({i:2d}-{j:2d}) |  {mayer_val:7.4f} |  {wiberg_val:7.4f} |  {diff:7.4f}")
    
    # 3. Bond order statistics
    print("\n3. Bond Order Statistics")
    print("-" * 60)
    
    # Extract upper triangle (unique bonds)
    upper_triangle = []
    for i in range(wfn.num_atoms):
        for j in range(i+1, wfn.num_atoms):
            upper_triangle.append(mayer_bo[i, j])
    
    upper_triangle = np.array(upper_triangle)
    
    print(f"  Total unique atom pairs: {len(upper_triangle)}")
    print(f"  Bonds with BO > 0.1: {np.sum(upper_triangle > 0.1)}")
    print(f"  Bonds with BO > 0.5: {np.sum(upper_triangle > 0.5)}")
    print(f"  Bonds with BO > 1.0: {np.sum(upper_triangle > 1.0)}")
    print(f"  Bonds with BO > 2.0: {np.sum(upper_triangle > 2.0)}")
    print(f"\n  Max bond order: {upper_triangle.max():.4f}")
    print(f"  Mean bond order: {upper_triangle.mean():.4f}")
    print(f"  Median bond order: {np.median(upper_triangle):.4f}")
    
    # 4. Atom connectivity
    print("\n4. Atom Connectivity")
    print("-" * 60)
    
    for i, atom in enumerate(wfn.atoms):
        # Find bonds for this atom
        connected = []
        for j in range(wfn.num_atoms):
            if i != j and mayer_bo[i, j] > 0.3:
                connected.append((j, wfn.atoms[j].element, mayer_bo[i, j]))
        
        # Sort by bond order
        connected.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  {atom.element:2s} ({i:2d}): {len(connected)} connections")
        for j, elem, bo in connected:
            print(f"    → {elem:2s} ({j:2d}): {bo:.4f}")
    
    # 5. Bond order matrix (if small molecule)
    if wfn.num_atoms <= 10:
        print("\n5. Bond Order Matrix")
        print("-" * 60)
        
        # Print header
        print("      ", end="")
        for i in range(wfn.num_atoms):
            print(f"{wfn.atoms[i].element:6s}", end="")
        print()
        
        # Print matrix
        for i in range(wfn.num_atoms):
            print(f"{wfn.atoms[i].element:4s} ", end="")
            for j in range(wfn.num_atoms):
                if i == j:
                    print(f"{'---':>6s}", end="")
                else:
                    print(f"{mayer_bo[i, j]:6.3f}", end="")
            print()
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    
    if len(sys.argv) != 2:
        print("Usage: python bond_analysis.py <wavefunction_file>")
        print("Example: python bond_analysis.py molecule.wfn")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        wfn = load_wavefunction(filepath)
        print(f"\nLoaded: {filepath}")
        print(f"Molecule: {wfn.num_atoms} atoms, {wfn.num_electrons} electrons")
        analyze_bonds(wfn)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
