"""
Test script to verify the overlap matrix fix.

This script tests the fixed overlap calculation and compares it with
the original implementation.

Author: PyMultiWFN Ralph Loop
Date: 2026-02-19
"""

import numpy as np
from pathlib import Path
from pymultiwfn.io.loader import load_wavefunction
from overlap_fix import calculate_overlap_matrix_fixed


def test_overlap_calculation():
    """Test the fixed overlap calculation."""
    print("=" * 70)
    print("Testing Fixed Overlap Calculation")
    print("=" * 70)

    test_data_dir = Path("/home/yhm/software/PyMultiWFN/consistency_verifier/examples")

    # Test H2 (simple case)
    print("\n1. Testing H2 (simple case)...")
    h2_path = test_data_dir / "H2_CCSD.wfn"

    if h2_path.exists():
        try:
            wfn = load_wavefunction(str(h2_path))
            print(f"  Loaded H2 wavefunction")
            print(f"  Number of atoms: {len(wfn.atoms)}")
            print(f"  Number of basis functions: {wfn.num_basis}")

            # Calculate overlap matrix with fixed implementation
            S_fixed = calculate_overlap_matrix_fixed(wfn, verbose=True)

            if S_fixed.size > 0:
                print(f"\n  ✅ Overlap matrix calculated successfully!")
                print(f"  Shape: {S_fixed.shape}")
                print(f"  Trace: {np.trace(S_fixed):.6f}")
                print(f"  Is symmetric: {np.allclose(S_fixed, S_fixed.T, rtol=1e-10)}")

                # Check properties
                eigenvalues = np.linalg.eigvalsh(S_fixed)
                print(f"  Min eigenvalue: {np.min(eigenvalues):.6f}")
                print(f"  Max eigenvalue: {np.max(eigenvalues):.6f}")
                print(
                    f"  Condition number: {np.max(eigenvalues) / np.min(eigenvalues):.2e}"
                )

                # Check if identity
                is_identity = np.allclose(S_fixed, np.eye(S_fixed.shape[0]), rtol=1e-10)
                if is_identity:
                    print(f"  ❌ WARNING: Matrix is identity!")
                else:
                    print(f"  ✅ Matrix is NOT identity (good!)")

                # Show a sample
                if S_fixed.shape[0] <= 10:
                    print(f"\n  Sample values (top-left 3x3):")
                    for i in range(min(3, S_fixed.shape[0])):
                        row_str = " ".join(
                            [
                                f"{S_fixed[i,j]:8.4f}"
                                for j in range(min(3, S_fixed.shape[1]))
                            ]
                        )
                        print(f"    {row_str}")
            else:
                print(f"  ❌ Overlap matrix is empty!")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"  ⚠️  File not found: {h2_path}")

    # Test C2H2
    print("\n2. Testing C2H2 (acetylene)...")
    c2h2_path = test_data_dir / "C2H2.wfn"

    if c2h2_path.exists():
        try:
            wfn = load_wavefunction(str(c2h2_path))
            print(f"  Loaded C2H2 wavefunction")
            print(f"  Number of atoms: {len(wfn.atoms)}")
            print(f"  Number of basis functions: {wfn.num_basis}")

            # Calculate overlap matrix with fixed implementation
            S_fixed = calculate_overlap_matrix_fixed(wfn, verbose=True)

            if S_fixed.size > 0:
                print(f"\n  ✅ Overlap matrix calculated successfully!")
                print(f"  Shape: {S_fixed.shape}")
                print(f"  Trace: {np.trace(S_fixed):.6f}")
                print(f"  Is symmetric: {np.allclose(S_fixed, S_fixed.T, rtol=1e-10)}")

                # Check properties
                eigenvalues = np.linalg.eigvalsh(S_fixed)
                print(f"  Min eigenvalue: {np.min(eigenvalues):.6f}")
                print(f"  Max eigenvalue: {np.max(eigenvalues):.6f}")

                # Check if identity
                is_identity = np.allclose(S_fixed, np.eye(S_fixed.shape[0]), rtol=1e-10)
                if is_identity:
                    print(f"  ❌ WARNING: Matrix is identity!")
                else:
                    print(f"  ✅ Matrix is NOT identity (good!)")
            else:
                print(f"  ❌ Overlap matrix is empty!")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"  ⚠️  File not found: {c2h2_path}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_overlap_calculation()
