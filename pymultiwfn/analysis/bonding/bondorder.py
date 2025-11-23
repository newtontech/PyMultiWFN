"""
Bond order analysis module.
Implements various bond order calculations including Mayer, Mulliken, and Wiberg bond orders.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pymultiwfn.core.data import Wavefunction


def calculate_mayer_bond_order(wfn: Wavefunction) -> Dict[str, np.ndarray]:
    """
    Calculate Mayer bond order matrices.

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
    if wfn.Ptot is None or wfn.overlap_matrix is None:
        raise ValueError("Density matrix and overlap matrix are required for bond order calculation")

    n_atoms = wfn.num_atoms
    n_basis = wfn.num_basis

    # Get atomic basis function indices
    atom_to_bfs = wfn.get_atomic_basis_indices()

    # Initialize bond order matrices
    bnd_total = np.zeros((n_atoms, n_atoms))
    bnd_alpha = np.zeros((n_atoms, n_atoms))
    bnd_beta = np.zeros((n_atoms, n_atoms))

    # Calculate P*S matrices
    PS_total = wfn.Ptot @ wfn.overlap_matrix

    # Calculate total bond order (for closed-shell or generalized Wiberg for open-shell)
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            bfs_i = atom_to_bfs[i]
            bfs_j = atom_to_bfs[j]

            if not bfs_i or not bfs_j:
                continue

            accum = 0.0
            for mu in bfs_i:
                for nu in bfs_j:
                    accum += PS_total[mu, nu] * PS_total[nu, mu]

            bnd_total[i, j] = accum

    # Copy to symmetric part
    bnd_total += bnd_total.T

    # Set diagonal elements as sum of row elements
    for i in range(n_atoms):
        bnd_total[i, i] = np.sum(bnd_total[i, :])

    # For unrestricted wavefunctions, calculate alpha and beta bond orders
    if wfn.is_unrestricted and wfn.Palpha is not None and wfn.Pbeta is not None:
        PS_alpha = wfn.Palpha @ wfn.overlap_matrix
        PS_beta = wfn.Pbeta @ wfn.overlap_matrix

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                bfs_i = atom_to_bfs[i]
                bfs_j = atom_to_bfs[j]

                if not bfs_i or not bfs_j:
                    continue

                accum_alpha = 0.0
                accum_beta = 0.0
                for mu in bfs_i:
                    for nu in bfs_j:
                        accum_alpha += PS_alpha[mu, nu] * PS_alpha[nu, mu]
                        accum_beta += PS_beta[mu, nu] * PS_beta[nu, mu]

                bnd_alpha[i, j] = accum_alpha
                bnd_beta[i, j] = accum_beta

        # Copy to symmetric part and scale by 2 for unrestricted
        bnd_alpha = 2 * (bnd_alpha + bnd_alpha.T)
        bnd_beta = 2 * (bnd_beta + bnd_beta.T)

        # Set diagonal elements
        for i in range(n_atoms):
            bnd_alpha[i, i] = np.sum(bnd_alpha[i, :])
            bnd_beta[i, i] = np.sum(bnd_beta[i, :])

    result = {'total': bnd_total}
    if wfn.is_unrestricted:
        result['alpha'] = bnd_alpha
        result['beta'] = bnd_beta

    return result


def calculate_mulliken_bond_order(wfn: Wavefunction) -> Dict[str, np.ndarray]:
    """
    Calculate Mulliken bond order matrices.

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

    n_atoms = wfn.num_atoms
    atom_to_bfs = wfn.get_atomic_basis_indices()

    # Initialize bond order matrices
    bnd_total = np.zeros((n_atoms, n_atoms))
    bnd_alpha = np.zeros((n_atoms, n_atoms))
    bnd_beta = np.zeros((n_atoms, n_atoms))

    if not wfn.is_unrestricted or wfn.Ptot is not None:
        # Closed-shell or total density matrix available
        if wfn.Ptot is not None:
            PS_total = wfn.Ptot * wfn.overlap_matrix  # Element-wise multiplication
        else:
            # Fallback: use alpha density matrix for closed-shell
            PS_total = wfn.Palpha * wfn.overlap_matrix

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                bfs_i = atom_to_bfs[i]
                bfs_j = atom_to_bfs[j]

                if not bfs_i or not bfs_j:
                    continue

                bnd_total[i, j] = np.sum(PS_total[np.ix_(bfs_i, bfs_j)])

        # Scale by 2 for closed-shell
        bnd_total = 2 * (bnd_total + bnd_total.T)

        # Set diagonal elements
        for i in range(n_atoms):
            bnd_total[i, i] = np.sum(bnd_total[i, :])

    if wfn.is_unrestricted and wfn.Palpha is not None and wfn.Pbeta is not None:
        # Unrestricted case
        PS_alpha = wfn.Palpha * wfn.overlap_matrix
        PS_beta = wfn.Pbeta * wfn.overlap_matrix

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                bfs_i = atom_to_bfs[i]
                bfs_j = atom_to_bfs[j]

                if not bfs_i or not bfs_j:
                    continue

                bnd_alpha[i, j] = np.sum(PS_alpha[np.ix_(bfs_i, bfs_j)])
                bnd_beta[i, j] = np.sum(PS_beta[np.ix_(bfs_i, bfs_j)])

        # Scale by 2 for unrestricted
        bnd_alpha = 2 * (bnd_alpha + bnd_alpha.T)
        bnd_beta = 2 * (bnd_beta + bnd_beta.T)

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
    """
    n_atoms = bond_matrix.shape[0]

    if atom_names is None:
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
    """
    n_atoms = bond_results['total'].shape[0]

    if atom_names is None:
        atom_names = [f"Atom{i+1}" for i in range(n_atoms)]

    print(f"Bond orders with absolute value >= {threshold:.6f}")

    if 'alpha' in bond_results and 'beta' in bond_results:
        # Unrestricted case
        alpha_matrix = bond_results['alpha']
        beta_matrix = bond_results['beta']
        total_matrix = bond_results['total']

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
        total_matrix = bond_results['total']
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
    """
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
    """
    if wfn.overlap_matrix is None or wfn.coefficients is None:
        raise ValueError("Overlap matrix and MO coefficients are required for decomposition")

    atom_to_bfs = wfn.get_atomic_basis_indices()
    bfs1 = atom_to_bfs[atom1]
    bfs2 = atom_to_bfs[atom2]

    if not bfs1 or not bfs2:
        raise ValueError(f"No basis functions found for atoms {atom1} and {atom2}")

    n_mo = wfn.coefficients.shape[0]
    contributions = []
    total_bond = 0.0

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