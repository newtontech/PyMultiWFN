#!/usr/bin/env python
"""
Debug script to check overlap matrix properties.

This script loads a WFN file, calculates the overlap matrix,
and verifies its properties: symmetry, positivity, integration, and range.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from pymultiwfn.io.parsers.wfn import WFNLoader
    from pymultiwfn.integrals import calculate_overlap_matrix
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project root and dependencies are installed.")
    sys.exit(1)


def check_overlap_matrix(wfn_file: str):
    """
    Load WFN file, calculate overlap matrix, and check properties.

    Parameters
    ----------
    wfn_file : str
        Path to WFN file
    """
    print("=" * 50)
    print("Overlap Matrix Debug Report")
    print("=" * 50)
    print(f"WFN file: {wfn_file}")
    print()

    # Step 1: Load WFN file
    print("Step 1: Loading WFN file...")
    try:
        loader = WFNLoader(wfn_file)
        wfn = loader.load()
        print(f"✅ Loaded WFN file")
        print(f"   Atoms: {len(wfn.atoms)}")
        print(f"   Electrons: {wfn.num_electrons}")
        print(f"   Basis functions (num_basis): {wfn.num_basis}")
        print(f"   MOs: {wfn.num_mos}")
        print()
    except Exception as e:
        print(f"❌ Failed to load WFN file: {e}")
        return

    # Step 2: Calculate overlap matrix
    print("Step 2: Calculating overlap matrix...")
    try:
        overlap_matrix = calculate_overlap_matrix(wfn, use_cache=True, verbose=False)
        print(f"✅ Calculated overlap matrix")
        print(f"   Shape: {overlap_matrix.shape}")
        print()
    except Exception as e:
        print(f"❌ Failed to calculate overlap matrix: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Check symmetry
    print("Step 3: Checking symmetry...")
    symmetry_diff = np.max(np.abs(overlap_matrix - overlap_matrix.T))
    symmetry_pass = symmetry_diff < 1e-10
    if symmetry_pass:
        print(f"✅ Symmetry PASS (max diff = {symmetry_diff:.2e})")
    else:
        print(f"❌ Symmetry FAIL (max diff = {symmetry_diff:.2e})")
    print()

    # Step 4: Check diagonal elements (positivity)
    print("Step 4: Checking diagonal elements...")
    diagonal = np.diag(overlap_matrix)
    min_diagonal = np.min(diagonal)
    max_diagonal = np.max(diagonal)
    all_positive = np.all(diagonal > 0)
    if all_positive:
        print(f"✅ Diagonal elements PASS (all > 0)")
        print(f"   Min: {min_diagonal:.6f}")
        print(f"   Max: {max_diagonal:.6f}")
    else:
        print(f"❌ Diagonal elements FAIL (not all > 0)")
        print(f"   Min: {min_diagonal:.6f}")
        print(f"   Max: {max_diagonal:.6f}")
    print()

    # Step 5: Check trace (integration)
    print("Step 5: Checking trace (integration)...")
    trace = np.trace(overlap_matrix)
    expected_electrons = wfn.num_electrons
    trace_diff = abs(trace - expected_electrons)
    trace_pass = trace_diff < 0.1  # Allow 0.1 difference
    if trace_pass:
        print(f"✅ Trace PASS ({trace:.4f}, expected: ~{expected_electrons} electrons)")
        print(f"   Difference: {trace_diff:.4f}")
    else:
        print(f"⚠️  Trace WARNING ({trace:.4f}, expected: ~{expected_electrons} electrons)")
        print(f"   Difference: {trace_diff:.4f}")
        print(f"   Note: Trace may not equal electrons due to basis set normalization")
    print()

    # Step 6: Check range
    print("Step 6: Checking value range...")
    min_value = np.min(overlap_matrix)
    max_value = np.max(overlap_matrix)
    range_ok = (min_value >= -1e-10) and (max_value <= 1 + 1e-10)
    if range_ok:
        print(f"✅ Range PASS (values in [0, 1])")
        print(f"   Min: {min_value:.6f}")
        print(f"   Max: {max_value:.6f}")
    else:
        print(f"❌ Range FAIL (values not in [0, 1])")
        print(f"   Min: {min_value:.6f}")
        print(f"   Max: {max_value:.6f}")
    print()

    # Step 7: Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Matrix size: {overlap_matrix.shape}")
    print(f"Symmetry: {'✅ PASS' if symmetry_pass else '❌ FAIL'}")
    print(f"Diagonal: {'✅ PASS' if all_positive else '❌ FAIL'}")
    print(f"Trace: {trace:.4f} (expected: ~{expected_electrons})")
    print(f"Range: {'✅ PASS' if range_ok else '❌ FAIL'}")
    print()

    # Step 8: Print first 5x5 submatrix
    print("First 5x5 submatrix:")
    submatrix = overlap_matrix[:5, :5]
    for row in submatrix:
        row_str = " ".join([f"{x:8.4f}" for x in row])
        print(f"[{row_str}]")
    print()

    # Step 9: Check if overlap matrix is identity
    is_identity = np.allclose(overlap_matrix, np.eye(overlap_matrix.shape[0]), atol=1e-6)
    if is_identity:
        print("⚠️  WARNING: Overlap matrix is identity matrix!")
        print("   This means overlap calculation is not working correctly.")
    else:
        print("✅ Overlap matrix is not identity (good!)")

    # Overall result
    print()
    print("=" * 50)
    all_pass = symmetry_pass and all_positive and range_ok
    if all_pass:
        print("✅ OVERALL: All checks passed!")
    else:
        print("❌ OVERALL: Some checks failed!")
    print("=" * 50)


def main():
    """Main function."""
    # Default WFN file
    default_wfn = "tests/data/H2.wfn"

    # Get WFN file from command line or use default
    if len(sys.argv) > 1:
        wfn_file = sys.argv[1]
    else:
        wfn_file = default_wfn

    # Check if file exists
    if not Path(wfn_file).exists():
        print(f"❌ WFN file not found: {wfn_file}")
        print(f"   Looking for: {Path(wfn_file).absolute()}")
        print()
        print("Usage: python pymultiwfn/debug_overlap_matrix.py [wfn_file]")
        print()
        print("Searching for WFN files...")
        project_root = Path(__file__).parent.parent
        for wfn_path in project_root.rglob("*.wfn"):
            print(f"   Found: {wfn_path.relative_to(project_root)}")
        sys.exit(1)

    # Run check
    check_overlap_matrix(wfn_file)


if __name__ == "__main__":
    main()
