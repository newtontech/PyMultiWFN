"""

Debug script for overlap matrix properties.





This script analyzes the overlap matrix calculated from wavefunction files

to help diagnose bond order calculation issues.





Author: PyMultiWFN Ralph Loop

Date: 2026-02-19

"""

import numpy as np

from pathlib import Path

from pymultiwfn.io.loader import load_wavefunction

from pymultiwfn.integrals.overlap import calculate_overlap_matrix


def analyze_overlap_matrix(wfn, wfn_name, verbose=True):
    """

    Analyze overlap matrix properties.





    Args:

        wfn: Wavefunction object

        wfn_name: Name of the wavefunction (for output)

        verbose: Whether to print detailed information





    Returns:

        Dictionary with analysis results

    """

    print("\n" + "=" * 70)

    print(f"Analyzing: {wfn_name}")

    print("=" * 70)

    # Basic wavefunction info

    if verbose:

        print("\nWavefunction Info:")

        print(f"  Number of atoms: {len(wfn.atoms)}")

        print(f"  Number of basis functions: {wfn.num_basis}")

        print(f"  Number of electrons: {wfn.num_electrons}")

        print(f"  Charge: {wfn.charge}")

        print(f"  Multiplicity: {wfn.multiplicity}")

        print(f"  Is unrestricted: {wfn.is_unrestricted}")

    # Calculate overlap matrix

    print("\nCalculating overlap matrix...")

    try:

        S = calculate_overlap_matrix(wfn, use_cache=True, verbose=False)

    except Exception as e:

        print(f"❌ Error calculating overlap matrix: {e}")

        return None

    if S.size == 0:

        print(f"❌ Empty overlap matrix returned")

        return None

    # Basic matrix properties

    print("\nMatrix Properties:")

    print(f"  Shape: {S.shape}")

    print(f"  Dtype: {S.dtype}")

    print(f"  Size: {S.size}")

    # 1. Symmetry check

    print("\n" + "1. Symmetry Check".center(70))

    is_symmetric = np.allclose(S, S.T, rtol=1e-10, atol=1e-12)

    print(f"  Is symmetric (S == S.T): {is_symmetric}")

    if not is_symmetric:

        diff = np.abs(S - S.T)

        print(f"  Max asymmetry: {np.max(diff):.2e}")

        print(f"  Mean asymmetry: {np.mean(diff):.2e}")

    # 2. Positivity check

    print("\n" + "2. Positivity Check".center(70))

    min_diagonal = np.min(np.diag(S))

    max_diagonal = np.max(np.diag(S))

    print(f"  Min diagonal element: {min_diagonal:.6f}")

    print(f"  Max diagonal element: {max_diagonal:.6f}")

    print(f"  All diagonal > 0: {np.all(np.diag(S) > 0)}")

    # 3. Trace check

    print("\n" + "3. Trace Check".center(70))

    trace = np.trace(S)

    print(f"  Trace(S): {trace:.6f}")

    print(f"  Num electrons: {wfn.num_electrons}")

    print(f"  Ratio (trace / electrons): {trace / wfn.num_electrons:.6f}")

    # For normalized basis functions, trace should be close to num_electrons

    # But this depends on basis set normalization

    if abs(trace - wfn.num_electrons) < 0.1:

        print(f"  ✅ Trace matches number of electrons")

    else:

        print(f"  ⚠️  Trace differs from number of electrons")

        print(f"     Difference: {abs(trace - wfn.num_electrons):.6f}")

    # 4. Eigenvalue check

    print("\n" + "4. Eigenvalue Analysis".center(70))

    eigenvalues = np.linalg.eigvalsh(S)

    print(f"  Number of eigenvalues: {len(eigenvalues)}")

    print(f"  Min eigenvalue: {np.min(eigenvalues):.6f}")

    print(f"  Max eigenvalue: {np.max(eigenvalues):.6f}")

    print(f"  Condition number: {np.max(eigenvalues) / np.min(eigenvalues):.2e}")

    print(f"  All eigenvalues > 0: {np.all(eigenvalues > 0)}")

    if not np.all(eigenvalues > 0):

        negative_count = np.sum(eigenvalues < 0)

        print(f"  ❌ {negative_count} negative eigenvalues found!")

        print(f"     Negative eigenvalues: {eigenvalues[eigenvalues < 0]}")

    # 5. Range check

    print("\n" + "5. Range Check".center(70))

    min_val = np.min(S)

    max_val = np.max(S)

    print(f"  Min value: {min_val:.6f}")

    print(f"  Max value: {max_val:.6f}")

    off_diag = S - np.diag(np.diag(S))

    print(f"  Off-diagonal min: {np.min(off_diag):.6f}")

    print(f"  Off-diagonal max: {np.max(off_diag):.6f}")

    # 6. Normalization check

    print("\n" + "6. Normalization Check".center(70))

    # For normalized GTOs, diagonal elements should be close to 1.0

    diag_mean = np.mean(np.diag(S))

    print(f"  Mean diagonal: {diag_mean:.6f}")

    if abs(diag_mean - 1.0) < 0.1:

        print(f"  ✅ Diagonal elements appear normalized (≈1.0)")

    else:

        print(f"  ⚠️  Diagonal elements not normalized (expected ≈1.0)")

    # 7. Check for identity matrix

    print("\n" + "7. Identity Check".center(70))

    is_identity = np.allclose(S, np.eye(S.shape[0]), rtol=1e-10, atol=1e-12)

    print(f"  Is identity matrix: {is_identity}")

    if is_identity:

        print(f"  ❌ WARNING: Overlap matrix is identity!")

        print(f"     This suggests overlap calculation is not working.")

        print(f"     Bond order calculations will be incorrect.")

    # 8. Print matrix (small matrices only)

    if S.shape[0] <= 10:

        print("\n" + "8. Matrix Display".center(70))

        print("  Full matrix:")

        for row in S:

            row_str = " ".join([f"{val:8.4f}" for val in row])

            print(f"    {row_str}")

    else:

        print("\n" + "8. Matrix Display".center(70))

        print(f"  Matrix too large ({S.shape[0]}x{S.shape[0]}), showing corners...")

        print("  Top-left 5x5:")

        for i in range(min(5, S.shape[0])):

            row_vals = [f"{S[i,j]:8.4f}" for j in range(min(5, S.shape[1]))]

            print(f"    {' '.join(row_vals)}")

        print("  Bottom-right 5x5:")

        row_start = max(0, S.shape[1] - 5)

        for i in range(max(0, S.shape[0] - 5), S.shape[0]):

            row_vals = [f"{S[i,j]:8.4f}" for j in range(row_start, S.shape[1])]

            print(f"    {' '.join(row_vals)}")

    # Return analysis results

    return {
        "wfn_name": wfn_name,
        "shape": S.shape,
        "is_symmetric": is_symmetric,
        "min_diagonal": min_diagonal,
        "max_diagonal": max_diagonal,
        "trace": trace,
        "eigenvalues": eigenvalues,
        "condition_number": np.max(eigenvalues) / np.min(eigenvalues),
        "is_identity": is_identity,
        "matrix": S,
    }


def main():
    """Main function."""

    print("=" * 70)

    print("PyMultiWFN Overlap Matrix Debugger")

    print("=" * 70)

    test_data_dir = Path("/home/yhm/software/PyMultiWFN/consistency_verifier/examples")

    # Test files

    test_files = [
        ("H2_CCSD.wfn", "H2 (CCSD/cc-pVTZ)"),
        ("C2H2.wfn", "C2H2 (acetylene)"),
        ("C2H4_HF.wfn", "C2H4 (ethene)"),
        ("H2O_m3ub3lyp.wfn", "H2O"),
    ]

    results = []

    for filename, description in test_files:

        wfn_path = test_data_dir / filename

        if not wfn_path.exists():

            print(f"\n⚠️  Skipping {filename}: File not found")

            continue

        try:

            wfn = load_wavefunction(str(wfn_path))

            result = analyze_overlap_matrix(wfn, description, verbose=True)

            if result:

                results.append(result)

        except Exception as e:

            print(f"\n❌ Error loading {filename}: {e}")

            import traceback

            traceback.print_exc()

    # Summary

    print("\n" + "=" * 70)

    print("SUMMARY")

    print("=" * 70)

    if results:

        print(f"\nAnalyzed {len(results)} wavefunction(s)")

        # Check for issues

        issues = []

        for result in results:

            if not result["is_symmetric"]:

                issues.append(f"{result['wfn_name']}: Not symmetric")

            if result["is_identity"]:

                issues.append(f"{result['wfn_name']}: Identity matrix (ERROR!)")

            if np.min(result["eigenvalues"]) <= 0:

                issues.append(f"{result['wfn_name']}: Has non-positive eigenvalues")

        if issues:

            print("\n⚠️  ISSUES FOUND:")

            for issue in issues:

                print(f"  - {issue}")

        else:

            print("\n✅ All matrices pass basic checks!")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":

    main()
