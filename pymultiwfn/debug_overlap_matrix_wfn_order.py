#!/usr/bin/env python3
"""
Fix for overlap matrix calculation - use WFN original basis function ordering
"""

import numpy as np
from pymultiwfn.io.parsers.wfn import WFNLoader
from pymultiwfn.integrals.overlap import (
    _calculate_primitive_overlap,
)


def calculate_overlap_matrix_wfn_order(wfn):
    """
    Calculate overlap matrix using WFN file's original basis function ordering.

    This is critical for ensuring that the overlap matrix indices match
    the MO coefficient indices in the WFN file.

    Args:
        wfn: Wavefunction object with WFN metadata

    Returns:
        Overlap matrix S (nbasis x nbasis)
    """
    # Check if WFN metadata exists
    if not hasattr(wfn, "_type_assignments") or not hasattr(wfn, "_centre_assignments"):
        raise ValueError(
            "Wavefunction missing WFN metadata (_type_assignments or _centre_assignments)"
        )

    type_assignments = wfn._type_assignments
    centre_assignments = wfn._centre_assignments

    num_basis = len(type_assignments)
    overlap_matrix = np.zeros((num_basis, num_basis))

    print(f"Calculating overlap matrix for {num_basis} basis functions...")
    print(f"Using WFN original ordering (type_assignments and centre_assignments)")

    # Build basis function mapping: basis_idx -> (shell, primitive_idx)
    # We need to find which shell and primitive each WFN basis function corresponds to

    # First, build a map from (atom_idx, shell_type) to shell index
    shell_map = {}
    for shell_idx, shell in enumerate(wfn.shells):
        key = (shell.center_idx, shell.type)
        if key not in shell_map:
            shell_map[key] = []
        shell_map[key].append(shell_idx)

    # Track which primitives we've used from each shell
    primitive_counters = {key: 0 for key in shell_map.keys()}

    # For each WFN basis function, find the corresponding primitive exponent
    wfn_exponents = []
    for i, (centre_idx, wfn_type) in enumerate(
        zip(centre_assignments, type_assignments)
    ):
        # Convert WFN type to shell type
        # WFN: 1=S, 2=P_x, 3=P_y, 4=P_z, 5-10=D, 11+=F
        # Shell: 0=S, 1=P, 2=D, 3=F
        if wfn_type == 1:
            shell_type = 0  # S
        elif wfn_type in [2, 3, 4]:
            shell_type = 1  # P
        elif 5 <= wfn_type <= 10:
            shell_type = 2  # D
        elif wfn_type >= 11:
            shell_type = 3  # F
        else:
            raise ValueError(f"Unknown WFN type: {wfn_type}")

        key = (centre_idx, shell_type)

        if key not in shell_map:
            raise ValueError(
                f"No shell found for atom {centre_idx}, shell_type {shell_type}"
            )

        # Get the first shell that matches (assuming each (atom, type) has only one shell)
        shell_idx = shell_map[key][0]
        shell = wfn.shells[shell_idx]

        # Get the next primitive exponent
        prim_idx = primitive_counters[key]
        if prim_idx >= len(shell.exponents):
            raise ValueError(
                f"Not enough primitives in shell {shell_idx} for basis function {i}"
            )

        exponent = shell.exponents[prim_idx]
        coefficient = shell.coefficients.flatten()[
            0
        ]  # Assuming all coefficients are 1.0

        wfn_exponents.append(
            {
                "type": wfn_type,
                "center": centre_idx,
                "coords": tuple(wfn.atoms[centre_idx].coord),
                "exponent": exponent,
                "coefficient": coefficient,
            }
        )

        primitive_counters[key] += 1

    print(f"Extracted {len(wfn_exponents)} WFN basis functions with exponents")

    # Calculate overlap matrix
    for i in range(num_basis):
        bf_i = wfn_exponents[i]
        coords_i = bf_i["coords"]
        alpha = bf_i["exponent"]
        type_i = bf_i["type"]

        for j in range(i, num_basis):
            bf_j = wfn_exponents[j]
            coords_j = bf_j["coords"]
            beta = bf_j["exponent"]
            type_j = bf_j["type"]

            # Calculate overlap
            S_ij = _calculate_primitive_overlap(
                type_i, type_j, coords_i, coords_j, alpha, beta
            )

            overlap_matrix[i, j] = S_ij
            if i != j:
                overlap_matrix[j, i] = S_ij  # Symmetric

    return overlap_matrix


def check_overlap_matrix_properties(S, wfn):
    """Check properties of the overlap matrix."""
    print("\n" + "=" * 50)
    print("Overlap Matrix Properties Check")
    print("=" * 50)

    # Symmetry
    symmetry_diff = np.max(np.abs(S - S.T))
    symmetry_pass = symmetry_diff < 1e-10
    print(f"\nSymmetry (S == S.T):")
    print(f"  Max difference: {symmetry_diff:.2e}")
    print(f"  Result: {'✅ PASS' if symmetry_pass else '❌ FAIL'}")

    # Diagonal positivity
    diag_min = np.min(np.diag(S))
    diag_max = np.max(np.diag(S))
    diag_pass = diag_min > 0
    print(f"\nDiagonal elements (S[i,i] > 0):")
    print(f"  Min value: {diag_min:.6f}")
    print(f"  Max value: {diag_max:.6f}")
    print(f"  Result: {'✅ PASS' if diag_pass else '❌ FAIL'}")

    # Trace (should be close to number of electrons)
    trace = np.trace(S)
    trace_diff = abs(trace - wfn.num_electrons)
    trace_pass = trace_diff < 0.1
    print(f"\nTrace (should be ≈ {wfn.num_electrons} electrons):")
    print(f"  Trace value: {trace:.6f}")
    print(f"  Difference: {trace_diff:.6f}")
    print(f"  Result: {'✅ PASS' if trace_pass else '❌ FAIL'}")

    # Value range
    S_min = np.min(S)
    S_max = np.max(S)
    range_pass = (S_min >= -1e-10) and (S_max <= 1.0)
    print(f"\nValue range (should be in [0, 1]):")
    print(f"  Min value: {S_min:.6f}")
    print(f"  Max value: {S_max:.6f}")
    print(f"  Result: {'✅ PASS' if range_pass else '❌ FAIL'}")

    # Print matrix statistics
    print("\n" + "=" * 50)
    print("Matrix Statistics")
    print("=" * 50)
    print(f"Mean: {np.mean(S):.6f}")
    print(f"Std: {np.std(S):.6f}")
    print(f"Non-zero elements: {np.count_nonzero(S)}/{S.size}")

    # Print first 5x5 submatrix
    print("\n" + "=" * 50)
    print("First 5x5 Submatrix")
    print("=" * 50)
    submatrix_size = min(5, S.shape[0])
    for i in range(submatrix_size):
        row_str = " ".join([f"{S[i,j]:8.4f}" for j in range(submatrix_size)])
        print(f"[{row_str}]")

    # Overall summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    all_pass = symmetry_pass and diag_pass and trace_pass and range_pass
    if all_pass:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed!")

    return all_pass


if __name__ == "__main__":
    # Test with H2
    wfn_file = "consistency_verifier/examples/H2_CCSD.wfn"

    print(f"Loading WFN file: {wfn_file}")
    loader = WFNLoader(wfn_file)
    wfn = loader.load()

    print(f"\nNumber of atoms: {len(wfn.atoms)}")
    print(f"Number of electrons: {wfn.num_electrons}")
    print(f"Number of shells: {len(wfn.shells)}")
    print(f"Number of basis functions (from WFN): {len(wfn._type_assignments)}")

    # Calculate overlap matrix using WFN ordering
    print("\nCalculating overlap matrix using WFN original ordering...")
    S = calculate_overlap_matrix_wfn_order(wfn)

    print(f"\nOverlap matrix shape: {S.shape}")
    print(f"Overlap matrix dtype: {S.dtype}")

    # Check properties
    all_pass = check_overlap_matrix_properties(S, wfn)

    print(f"\n✅ Test completed!")
