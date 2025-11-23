"""
Bond order analysis module for PyMultiWFN.

This module implements various bond order analysis methods including:
- Mayer bond order analysis
- Mulliken bond order analysis
- Orbital decomposition of bond orders
- Orbital occupancy-perturbed Mayer bond order
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ...core.data import Wavefunction


def calculate_mayer_bond_order(wfn: Wavefunction) -> Dict[str, np.ndarray]:
    """
    Calculate Mayer bond order matrices using optimized NumPy operations.

    For restricted closed-shell wavefunctions, only total bond order is returned.
    For unrestricted wavefunctions, returns alpha, beta, and total bond orders.

    Args:
        wfn: Wavefunction object with density matrices and overlap matrix

    Returns:
        Dictionary containing bond order matrices:
        - 'total': Total bond order matrix (n_atoms x n_atoms)
        - 'alpha': Alpha bond order matrix (for unrestricted)
        - 'beta': Beta bond order matrix (for unrestricted)
    """

    # Check required matrices
    if wfn.overlap_matrix is None:
        raise ValueError("Overlap matrix is required for bond order calculation")

    if wfn.Ptot is None:
        wfn.calculate_density_matrices()
        if wfn.Ptot is None:
            raise ValueError("Total density matrix could not be calculated")

    n_atoms = wfn.num_atoms
    atom_to_bfs = wfn.get_atomic_basis_indices()

    # Calculate P*S matrices
    PS_total = wfn.Ptot @ wfn.overlap_matrix

    # Initialize bond order matrices
    bnd_total = np.zeros((n_atoms, n_atoms))

    # Calculate total bond order using vectorized operations
    for i in range(n_atoms):
        bfs_i = atom_to_bfs.get(i, [])
        if not bfs_i:
            continue

        for j in range(i+1, n_atoms):
            bfs_j = atom_to_bfs.get(j, [])
            if not bfs_j:
                continue

            # Extract submatrices and compute bond order using vectorized operations
            ps_ij = PS_total[np.ix_(bfs_i, bfs_j)]
            ps_ji = PS_total[np.ix_(bfs_j, bfs_i)]

            # Vectorized calculation: sum of element-wise products
            accum = np.sum(ps_ij * ps_ji)
            bnd_total[i, j] = accum
            bnd_total[j, i] = accum  # Symmetric matrix

    # Set diagonal elements as sum of row elements (Mayer valence)
    for i in range(n_atoms):
        bnd_total[i, i] = np.sum(bnd_total[i, :])

    result = {'total': bnd_total}

    # For unrestricted wavefunctions, calculate alpha and beta bond orders
    if wfn.is_unrestricted:
        if wfn.Palpha is None or wfn.Pbeta is None:
            raise ValueError("Alpha and beta density matrices are required for unrestricted calculation")

        PS_alpha = wfn.Palpha @ wfn.overlap_matrix
        PS_beta = wfn.Pbeta @ wfn.overlap_matrix

        bnd_alpha = np.zeros((n_atoms, n_atoms))
        bnd_beta = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            bfs_i = atom_to_bfs.get(i, [])
            if not bfs_i:
                continue

            for j in range(i+1, n_atoms):
                bfs_j = atom_to_bfs.get(j, [])
                if not bfs_j:
                    continue

                # Vectorized calculations for alpha and beta
                ps_alpha_ij = PS_alpha[np.ix_(bfs_i, bfs_j)]
                ps_alpha_ji = PS_alpha[np.ix_(bfs_j, bfs_i)]
                accum_alpha = np.sum(ps_alpha_ij * ps_alpha_ji)

                ps_beta_ij = PS_beta[np.ix_(bfs_i, bfs_j)]
                ps_beta_ji = PS_beta[np.ix_(bfs_j, bfs_i)]
                accum_beta = np.sum(ps_beta_ij * ps_beta_ji)

                bnd_alpha[i, j] = accum_alpha
                bnd_alpha[j, i] = accum_alpha
                bnd_beta[i, j] = accum_beta
                bnd_beta[j, i] = accum_beta

        # Scale by 2 for unrestricted (following Multiwfn convention)
        bnd_alpha = 2 * bnd_alpha
        bnd_beta = 2 * bnd_beta

        # Set diagonal elements
        for i in range(n_atoms):
            bnd_alpha[i, i] = np.sum(bnd_alpha[i, :])
            bnd_beta[i, i] = np.sum(bnd_beta[i, :])

        result['alpha'] = bnd_alpha
        result['beta'] = bnd_beta

    return result


def calculate_mulliken_bond_order(wfn: Wavefunction) -> Dict[str, np.ndarray]:
    """
    Calculate Mulliken bond order matrices using optimized NumPy operations.

    Args:
        wfn: Wavefunction object with density matrices and overlap matrix

    Returns:
        Dictionary containing bond order matrices:
        - 'total': Total bond order matrix (n_atoms x n_atoms)
        - 'alpha': Alpha bond order matrix (for unrestricted)
        - 'beta': Beta bond order matrix (for unrestricted)
    """

    if wfn.overlap_matrix is None:
        raise ValueError("Overlap matrix is required for Mulliken bond order calculation")

    # Ensure density matrices are available
    if wfn.Ptot is None and wfn.Palpha is None and wfn.Pbeta is None:
        wfn.calculate_density_matrices()

    n_atoms = wfn.num_atoms
    atom_to_bfs = wfn.get_atomic_basis_indices()

    # Initialize bond order matrices
    bnd_total = np.zeros((n_atoms, n_atoms))

    if not wfn.is_unrestricted:
        # Closed-shell case
        if wfn.Ptot is not None:
            PS_total = wfn.Ptot * wfn.overlap_matrix  # Element-wise multiplication
        elif wfn.Palpha is not None:
            # Fallback: use alpha density matrix for closed-shell
            PS_total = wfn.Palpha * wfn.overlap_matrix
        else:
            raise ValueError("No density matrix available for Mulliken bond order calculation")

        # Calculate bond orders using vectorized operations
        for i in range(n_atoms):
            bfs_i = atom_to_bfs.get(i, [])
            if not bfs_i:
                continue

            for j in range(i+1, n_atoms):
                bfs_j = atom_to_bfs.get(j, [])
                if not bfs_j:
                    continue

                # Vectorized sum over basis function pairs
                bnd_total[i, j] = np.sum(PS_total[np.ix_(bfs_i, bfs_j)])
                bnd_total[j, i] = bnd_total[i, j]  # Symmetric matrix

        # Scale by 2 for closed-shell
        bnd_total = 2 * bnd_total

        # Set diagonal elements
        for i in range(n_atoms):
            bnd_total[i, i] = np.sum(bnd_total[i, :])

    else:
        # Unrestricted case
        if wfn.Palpha is None or wfn.Pbeta is None:
            raise ValueError("Alpha and beta density matrices are required for unrestricted calculation")

        PS_alpha = wfn.Palpha * wfn.overlap_matrix
        PS_beta = wfn.Pbeta * wfn.overlap_matrix

        bnd_alpha = np.zeros((n_atoms, n_atoms))
        bnd_beta = np.zeros((n_atoms, n_atoms))

        # Calculate alpha and beta bond orders using vectorized operations
        for i in range(n_atoms):
            bfs_i = atom_to_bfs.get(i, [])
            if not bfs_i:
                continue

            for j in range(i+1, n_atoms):
                bfs_j = atom_to_bfs.get(j, [])
                if not bfs_j:
                    continue

                # Vectorized calculations for alpha and beta
                bnd_alpha[i, j] = np.sum(PS_alpha[np.ix_(bfs_i, bfs_j)])
                bnd_alpha[j, i] = bnd_alpha[i, j]
                bnd_beta[i, j] = np.sum(PS_beta[np.ix_(bfs_i, bfs_j)])
                bnd_beta[j, i] = bnd_beta[i, j]

        # Scale by 2 for unrestricted
        bnd_alpha = 2 * bnd_alpha
        bnd_beta = 2 * bnd_beta

        # Set diagonal elements
        for i in range(n_atoms):
            bnd_alpha[i, i] = np.sum(bnd_alpha[i, :])
            bnd_beta[i, i] = np.sum(bnd_beta[i, :])

        bnd_total = bnd_alpha + bnd_beta

    result = {'total': bnd_total}
    if wfn.is_unrestricted:
        result['alpha'] = bnd_alpha
        result['beta'] = bnd_beta

    return result


def get_bond_orders_above_threshold(
    bond_matrix: np.ndarray,
    threshold: float = 0.01,
    atom_names: Optional[List[str]] = None
) -> List[Tuple[int, int, float]]:
    """
    Get bond orders above a specified threshold.

    Args:
        bond_matrix: Bond order matrix (n_atoms x n_atoms)
        threshold: Minimum bond order value to include
        atom_names: Optional list of atom names for labeling

    Returns:
        List of tuples (atom1_idx, atom2_idx, bond_order)

    Raises:
        ValueError: If bond_matrix is not a 2D square matrix or threshold is negative
    """
    # Input validation
    if bond_matrix.ndim != 2 or bond_matrix.shape[0] != bond_matrix.shape[1]:
        raise ValueError("bond_matrix must be a square 2D matrix")

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    n_atoms = bond_matrix.shape[0]

    if atom_names is not None:
        if len(atom_names) != n_atoms:
            raise ValueError("atom_names length must match bond_matrix dimension")
    else:
        atom_names = [f"Atom{i+1}" for i in range(n_atoms)]

    significant_bonds = []

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            bond_order = bond_matrix[i, j]
            if abs(bond_order) >= threshold:
                significant_bonds.append((i, j, bond_order))

    # Sort by bond order magnitude (descending)
    significant_bonds.sort(key=lambda x: abs(x[2]), reverse=True)

    return significant_bonds


def print_bond_orders(
    bond_results: Dict[str, np.ndarray],
    threshold: float = 0.01,
    atom_names: Optional[List[str]] = None
):
    """
    Print bond orders in a formatted way.

    Args:
        bond_results: Dictionary from calculate_mayer_bond_order or calculate_mulliken_bond_order
        threshold: Minimum bond order to print
        atom_names: Optional list of atom names

    Raises:
        ValueError: If bond_results is missing required keys or matrices have inconsistent dimensions
    """
    # Input validation
    if 'total' not in bond_results:
        raise ValueError("bond_results must contain 'total' key")

    total_matrix = bond_results['total']
    if total_matrix.ndim != 2 or total_matrix.shape[0] != total_matrix.shape[1]:
        raise ValueError("Total bond matrix must be a square 2D matrix")

    n_atoms = total_matrix.shape[0]

    if atom_names is not None:
        if len(atom_names) != n_atoms:
            raise ValueError("atom_names length must match bond matrix dimension")
    else:
        atom_names = [f"Atom{i+1}" for i in range(n_atoms)]

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    print(f"Bond orders with absolute value >= {threshold:.6f}")

    if 'alpha' in bond_results and 'beta' in bond_results:
        # Unrestricted case
        alpha_matrix = bond_results['alpha']
        beta_matrix = bond_results['beta']

        # Validate alpha and beta matrices
        if alpha_matrix.shape != total_matrix.shape or beta_matrix.shape != total_matrix.shape:
            raise ValueError("Alpha and beta bond matrices must have same dimensions as total matrix")

        count = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                alpha_bo = alpha_matrix[i, j]
                beta_bo = beta_matrix[i, j]
                total_bo = alpha_bo + beta_bo

                if abs(total_bo) >= threshold:
                    count += 1
                    print(f" #{count:3d}: {i+1:3d}({atom_names[i]}) - {j+1:3d}({atom_names[j]}) "
                          f"Alpha: {alpha_bo:10.6f} Beta: {beta_bo:10.6f} Total: {total_bo:10.6f}")

        print("\nNote: The 'Total' bond orders shown above are more meaningful than the below ones.")
        print("If you are not familiar with related theory, you can simply ignore below output")
        print(f"\nBond order from mixed alpha&beta density matrix >= {threshold:.6f}")

        count = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                total_bo = total_matrix[i, j]
                if abs(total_bo) >= threshold:
                    count += 1
                    print(f" #{count:3d}: {i+1:3d}({atom_names[i]}) - {j+1:3d}({atom_names[j]}) {total_bo:14.8f}")
    else:
        # Closed-shell case
        count = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                total_bo = total_matrix[i, j]
                if abs(total_bo) >= threshold:
                    count += 1
                    print(f" #{count:3d}: {i+1:3d}({atom_names[i]}) - {j+1:3d}({atom_names[j]}) {total_bo:14.8f}")


def calculate_fragment_bond_order(
    bond_matrix: np.ndarray,
    fragment1: List[int],
    fragment2: List[int]
) -> float:
    """
    Calculate total bond order between two fragments.

    Args:
        bond_matrix: Bond order matrix
        fragment1: List of atom indices in fragment 1
        fragment2: List of atom indices in fragment 2

    Returns:
        Total bond order between the two fragments

    Raises:
        ValueError: If bond_matrix is not a square matrix or atom indices are out of bounds
    """
    # Input validation
    if bond_matrix.ndim != 2 or bond_matrix.shape[0] != bond_matrix.shape[1]:
        raise ValueError("bond_matrix must be a square 2D matrix")

    n_atoms = bond_matrix.shape[0]

    # Validate fragment indices
    for atom_idx in fragment1 + fragment2:
        if atom_idx < 0 or atom_idx >= n_atoms:
            raise ValueError(f"Atom index {atom_idx} is out of bounds (0-{n_atoms-1})")

    total_bond = 0.0

    for i in fragment1:
        for j in fragment2:
            total_bond += bond_matrix[i, j]

    return total_bond


def decompose_mulliken_bond_order(
    wfn: Wavefunction,
    atom1: int,
    atom2: int
) -> Dict[str, Union[float, List[Tuple[int, float, float, float]]]]:
    """
    Decompose Mulliken bond order between two atoms to orbital contributions.

    Args:
        wfn: Wavefunction object
        atom1: Index of first atom (0-based)
        atom2: Index of second atom (0-based)

    Returns:
        Dictionary with decomposition results

    Raises:
        ValueError: If required matrices are missing or atom indices are invalid
    """
    # Input validation
    if wfn.overlap_matrix is None:
        raise ValueError("Overlap matrix is required for decomposition")

    if wfn.coefficients is None:
        raise ValueError("MO coefficients are required for decomposition")

    if atom1 < 0 or atom1 >= wfn.num_atoms:
        raise ValueError(f"atom1 index {atom1} is out of bounds (0-{wfn.num_atoms-1})")

    if atom2 < 0 or atom2 >= wfn.num_atoms:
        raise ValueError(f"atom2 index {atom2} is out of bounds (0-{wfn.num_atoms-1})")

    if atom1 == atom2:
        raise ValueError("atom1 and atom2 must be different atoms")

    atom_to_bfs = wfn.get_atomic_basis_indices()
    bfs1 = atom_to_bfs.get(atom1, [])
    bfs2 = atom_to_bfs.get(atom2, [])

    if not bfs1 or not bfs2:
        raise ValueError(f"No basis functions found for atoms {atom1} and {atom2}")

    n_mo = wfn.coefficients.shape[0]
    contributions = []
    total_bond = 0.0

    # Ensure occupations are available
    if wfn.occupations is None:
        wfn._infer_occupations()

    for mo_idx in range(n_mo):
        if wfn.occupations is not None and wfn.occupations[mo_idx] == 0:
            continue

        contrib = 0.0
        for mu in bfs1:
            for nu in bfs2:
                contrib += (wfn.occupations[mo_idx] if wfn.occupations is not None else 2.0) * \
                          wfn.coefficients[mo_idx, mu] * wfn.coefficients[mo_idx, nu] * \
                          wfn.overlap_matrix[mu, nu]

        # Scale by 2 for closed-shell
        if not wfn.is_unrestricted:
            contrib *= 2

        total_bond += contrib

        energy = wfn.energies[mo_idx] if wfn.energies is not None else 0.0
        occupation = wfn.occupations[mo_idx] if wfn.occupations is not None else 0.0

        contributions.append((mo_idx, occupation, energy, contrib))

    return {
        'total_bond_order': total_bond,
        'orbital_contributions': contributions
    }


def calculate_wiberg_bond_order(wfn: Wavefunction) -> Dict[str, np.ndarray]:
    """
    Calculate Wiberg bond order matrices.

    Wiberg bond order is defined as: W_AB = sum_{mu in A} sum_{nu in B} (PS)_mu,nu * (PS)_nu,mu
    This is essentially the same as Mayer bond order for closed-shell systems.

    Args:
        wfn: Wavefunction object with density matrices and overlap matrix

    Returns:
        Dictionary containing bond order matrices:
        - 'total': Total Wiberg bond order matrix (n_atoms x n_atoms)
    """
    # For closed-shell systems, Wiberg bond order is the same as Mayer bond order
    return calculate_mayer_bond_order(wfn)


def calculate_fuzzy_bond_order(
    wfn: Wavefunction,
    method: str = "mayer",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Calculate bond orders using fuzzy atom partitioning.

    Args:
        wfn: Wavefunction object
        method: Bond order method ("mayer", "mulliken", "wiberg")
        **kwargs: Additional arguments for specific methods

    Returns:
        Dictionary containing bond order matrices
    """
    from ..population.fuzzy_atoms import calculate_fuzzy_population

    # Calculate fuzzy atomic populations
    fuzzy_pops = calculate_fuzzy_population(wfn)

    if method.lower() == "mayer":
        bond_results = calculate_mayer_bond_order(wfn)
    elif method.lower() == "mulliken":
        bond_results = calculate_mulliken_bond_order(wfn)
    elif method.lower() == "wiberg":
        bond_results = calculate_wiberg_bond_order(wfn)
    else:
        raise ValueError(f"Unknown bond order method: {method}")

    # For now, return standard bond orders
    # Future enhancement: implement fuzzy bond orders based on fuzzy populations
    return bond_results


def get_bond_order_statistics(bond_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for bond order matrix.

    Args:
        bond_matrix: Bond order matrix (n_atoms x n_atoms)

    Returns:
        Dictionary with bond order statistics
    """
    # Input validation
    if bond_matrix.ndim != 2 or bond_matrix.shape[0] != bond_matrix.shape[1]:
        raise ValueError("bond_matrix must be a square 2D matrix")

    n_atoms = bond_matrix.shape[0]

    # Extract off-diagonal elements (bond orders)
    off_diag_indices = np.triu_indices(n_atoms, k=1)
    bond_orders = bond_matrix[off_diag_indices]

    # Calculate statistics
    stats = {
        'mean': np.mean(bond_orders),
        'std': np.std(bond_orders),
        'max': np.max(bond_orders),
        'min': np.min(bond_orders),
        'median': np.median(bond_orders),
        'num_bonds': len(bond_orders),
        'num_significant_bonds': np.sum(np.abs(bond_orders) >= 0.1),
        'total_bond_order': np.sum(bond_orders)
    }

    return stats


def compare_bond_orders(
    bond_matrix1: np.ndarray,
    bond_matrix2: np.ndarray,
    method: str = "absolute"
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compare two bond order matrices.

    Args:
        bond_matrix1: First bond order matrix
        bond_matrix2: Second bond order matrix
        method: Comparison method ("absolute", "relative", "correlation")

    Returns:
        Dictionary with comparison results
    """
    # Input validation
    if bond_matrix1.shape != bond_matrix2.shape:
        raise ValueError("Bond matrices must have the same shape")

    if bond_matrix1.ndim != 2 or bond_matrix1.shape[0] != bond_matrix1.shape[1]:
        raise ValueError("Bond matrices must be square 2D matrices")

    n_atoms = bond_matrix1.shape[0]

    # Extract off-diagonal elements
    off_diag_indices = np.triu_indices(n_atoms, k=1)
    bonds1 = bond_matrix1[off_diag_indices]
    bonds2 = bond_matrix2[off_diag_indices]

    comparison = {}

    if method == "absolute":
        diff = bonds1 - bonds2
        comparison['mean_absolute_error'] = np.mean(np.abs(diff))
        comparison['max_absolute_error'] = np.max(np.abs(diff))
        comparison['rmsd'] = np.sqrt(np.mean(diff**2))

    elif method == "relative":
        # Avoid division by zero
        mask = np.abs(bonds2) > 1e-10
        rel_diff = np.zeros_like(bonds1)
        rel_diff[mask] = (bonds1[mask] - bonds2[mask]) / bonds2[mask]
        comparison['mean_relative_error'] = np.mean(np.abs(rel_diff[mask]))
        comparison['max_relative_error'] = np.max(np.abs(rel_diff[mask]))

    elif method == "correlation":
        correlation = np.corrcoef(bonds1, bonds2)[0, 1]
        comparison['correlation_coefficient'] = correlation

    else:
        raise ValueError(f"Unknown comparison method: {method}")

    return comparison