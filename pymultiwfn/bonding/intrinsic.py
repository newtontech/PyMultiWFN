"""Intrinsic Bond Order (IBO) implementation (Issue 3).

This module implements intrinsic bond order analysis with bond polarity
correction. The intrinsic bond order provides a measure of the intrinsic
strength of chemical bonds, accounting for ionic character.

The IBO is calculated using the formula:
    IBO_ij = (PS_ij)^2 * (1 - polarity_correction)

where PS is the density-overlap matrix product and the polarity correction
accounts for electronegativity differences between atoms.

References:
- Mayer, I. (1984). Chem. Phys. Lett. 97, 270-274.
- Wiberg, K. B. (1968). Tetrahedron 24, 1083-1096.
- Knizhnik, A. et al. (2019). J. Comput. Chem. 40, 1458-1467.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Pauling electronegativity values for common elements
ELECTRONEGATIVITY = {
    "H": 2.20,
    "He": 4.16,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Ne": 4.79,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Ar": 3.30,
    "K": 0.82,
    "Ca": 1.00,
    "Sc": 1.36,
    "Ti": 1.54,
    "V": 1.63,
    "Cr": 1.66,
    "Mn": 1.55,
    "Fe": 1.83,
    "Co": 1.88,
    "Ni": 1.91,
    "Cu": 1.90,
    "Zn": 1.65,
    "Ga": 1.81,
    "Ge": 2.01,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Kr": 3.00,
    "Rb": 0.82,
    "Sr": 0.95,
    "Y": 1.22,
    "Zr": 1.33,
    "Nb": 1.6,
    "Mo": 2.16,
    "Tc": 1.9,
    "Ru": 2.2,
    "Rh": 2.28,
    "Pd": 2.20,
    "Ag": 1.93,
    "Cd": 1.69,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.1,
    "I": 2.66,
    "Xe": 2.60,
    "Cs": 0.79,
    "Ba": 0.89,
}


@dataclass
class IntrinsicBondResult:
    """Results from intrinsic bond order calculation.

    Attributes:
        bond_order_matrix: NxN matrix of intrinsic bond orders
        polarity_matrix: NxN matrix of bond polarity corrections
        wiberg_matrix: NxN matrix of Wiberg bond orders (for comparison)
        n_atoms: Number of atoms in the molecule
    """

    bond_order_matrix: np.ndarray
    polarity_matrix: np.ndarray
    wiberg_matrix: np.ndarray
    n_atoms: int

    def get_bond_order(self, atom_i: int, atom_j: int) -> float:
        """Get intrinsic bond order between two atoms.

        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)

        Returns:
            Intrinsic bond order value
        """
        return float(self.bond_order_matrix[atom_i, atom_j])

    def get_polarity(self, atom_i: int, atom_j: int) -> float:
        """Get bond polarity correction between two atoms.

        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)

        Returns:
            Bond polarity correction (0-1)
        """
        return float(self.polarity_matrix[atom_i, atom_j])


def get_electronegativity(element: str) -> float:
    """Get Pauling electronegativity for an element.

    Args:
        element: Atomic symbol (e.g., 'C', 'H', 'O')

    Returns:
        Pauling electronegativity value

    Raises:
        ValueError: If element is not in the electronegativity table
    """
    element = element.capitalize()
    if element not in ELECTRONEGATIVITY:
        raise ValueError(f"No electronegativity defined for element: {element}")
    return ELECTRONEGATIVITY[element]


def calculate_bond_polarity(
    element_i: str,
    element_j: str,
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    atom_i_indices: List[int],
    atom_j_indices: List[int],
) -> float:
    """Calculate bond polarity correction between two atoms.

    The bond polarity correction accounts for the ionic character of a bond
    based on electronegativity differences and electron density distribution.

    Args:
        element_i: Element symbol for atom i
        element_j: Element symbol for atom j
        density_matrix: Total density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_i_indices: List of basis function indices for atom i
        atom_j_indices: List of basis function indices for atom j

    Returns:
        Bond polarity correction factor (0-1)
        0 = purely covalent, 1 = purely ionic
    """
    # Get electronegativity values
    chi_i = get_electronegativity(element_i)
    chi_j = get_electronegativity(element_j)

    # Calculate electronegativity difference
    delta_chi = abs(chi_i - chi_j)

    # Calculate Mulliken population on each atom
    # P_i = sum_{mu in i} (PS)_mu,mu
    if len(atom_i_indices) > 0 and len(atom_j_indices) > 0:
        # Extract submatrices
        PS = density_matrix @ overlap_matrix

        # Calculate atomic populations
        pop_i = sum(PS[mu, mu] for mu in atom_i_indices)
        pop_j = sum(PS[nu, nu] for nu in atom_j_indices)

        # Calculate charge transfer
        # Positive value indicates electron transfer from i to j
        charge_transfer = abs(pop_i - pop_j) / max(pop_i + pop_j, 1e-10)
    else:
        charge_transfer = 0.0

    # Combine electronegativity and charge transfer effects
    # Pauling's formula: ionic character ≈ 1 - exp(-0.25 * Δχ²)
    ionic_character = 1.0 - np.exp(-0.25 * delta_chi**2)

    # Weight by charge transfer
    polarity = 0.5 * (ionic_character + min(charge_transfer, 1.0))

    return min(max(polarity, 0.0), 1.0)


def calculate_wiberg_bond_order(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    atom_i_indices: List[int],
    atom_j_indices: List[int],
) -> float:
    """Calculate Wiberg bond order between two atoms.

    The Wiberg bond order is defined as:
        W_ij = sum_{mu in i} sum_{nu in j} (PS)_mu,nu * (PS)_nu,mu

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_i_indices: List of basis function indices for atom i
        atom_j_indices: List of basis function indices for atom j

    Returns:
        Wiberg bond order value
    """
    if len(atom_i_indices) == 0 or len(atom_j_indices) == 0:
        return 0.0

    # Calculate PS matrix
    PS = density_matrix @ overlap_matrix

    # Extract submatrices for atoms i and j
    PS_ij = PS[np.ix_(atom_i_indices, atom_j_indices)]
    PS_ji = PS[np.ix_(atom_j_indices, atom_i_indices)]

    # Wiberg bond order: trace(PS_ij @ PS_ji)
    bond_order = np.trace(PS_ij @ PS_ji)

    return max(float(bond_order), 0.0)


def calculate_intrinsic_bond_order(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    element_i: str,
    element_j: str,
    atom_i_indices: List[int],
    atom_j_indices: List[int],
) -> Tuple[float, float, float]:
    """Calculate intrinsic bond order between two atoms.

    The intrinsic bond order is calculated as:
        IBO_ij = W_ij * (1 - polarity_correction)

    where W_ij is the Wiberg bond order and the polarity correction
    accounts for ionic character.

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        element_i: Element symbol for atom i
        element_j: Element symbol for atom j
        atom_i_indices: List of basis function indices for atom i
        atom_j_indices: List of basis function indices for atom j

    Returns:
        Tuple of (intrinsic_bond_order, polarity, wiberg_bond_order)
    """
    # Calculate Wiberg bond order
    wiberg_bo = calculate_wiberg_bond_order(
        density_matrix, overlap_matrix, atom_i_indices, atom_j_indices
    )

    # Calculate bond polarity
    polarity = calculate_bond_polarity(
        element_i,
        element_j,
        density_matrix,
        overlap_matrix,
        atom_i_indices,
        atom_j_indices,
    )

    # Intrinsic bond order = Wiberg BO * (1 - polarity)
    # This reduces the bond order for polar bonds
    intrinsic_bo = wiberg_bo * (1.0 - polarity)

    return intrinsic_bo, polarity, wiberg_bo


def calculate_intrinsic_bond_order_matrix(
    wfn, density_matrix: Optional[np.ndarray] = None
) -> IntrinsicBondResult:
    """Calculate intrinsic bond order matrix for all atom pairs.

    Args:
        wfn: Wavefunction object containing molecular data
        density_matrix: Optional density matrix (uses wfn.Ptot if not provided)

    Returns:
        IntrinsicBondResult containing bond order matrices

    Raises:
        ValueError: If required matrices are not available
    """
    # Check required data
    if wfn.overlap_matrix is None:
        raise ValueError("Overlap matrix is required for bond order calculation")

    # Use provided density matrix or get from wavefunction
    if density_matrix is None:
        if wfn.Ptot is None:
            wfn.calculate_density_matrices()
            if wfn.Ptot is None:
                raise ValueError("Density matrix could not be calculated")
        density_matrix = wfn.Ptot

    n_atoms = wfn.num_atoms

    # Get atomic basis function indices
    atom_to_bfs = wfn.get_atomic_basis_indices()

    # Initialize matrices
    ibo_matrix = np.zeros((n_atoms, n_atoms))
    polarity_matrix = np.zeros((n_atoms, n_atoms))
    wiberg_matrix = np.zeros((n_atoms, n_atoms))

    # Calculate bond orders for all atom pairs
    for i in range(n_atoms):
        bfs_i = atom_to_bfs.get(i, [])

        for j in range(i + 1, n_atoms):
            bfs_j = atom_to_bfs.get(j, [])

            if not bfs_i or not bfs_j:
                continue

            # Get element symbols
            element_i = wfn.atoms[i].element
            element_j = wfn.atoms[j].element

            # Calculate intrinsic bond order
            ibo, polarity, wiberg = calculate_intrinsic_bond_order(
                density_matrix, wfn.overlap_matrix, element_i, element_j, bfs_i, bfs_j
            )

            # Store in matrices (symmetric)
            ibo_matrix[i, j] = ibo
            ibo_matrix[j, i] = ibo
            polarity_matrix[i, j] = polarity
            polarity_matrix[j, i] = polarity
            wiberg_matrix[i, j] = wiberg
            wiberg_matrix[j, i] = wiberg

    # Set diagonal to zero (no self-bonding)
    np.fill_diagonal(ibo_matrix, 0.0)
    np.fill_diagonal(wiberg_matrix, 0.0)

    return IntrinsicBondResult(
        bond_order_matrix=ibo_matrix,
        polarity_matrix=polarity_matrix,
        wiberg_matrix=wiberg_matrix,
        n_atoms=n_atoms,
    )


def intrinsic_bond_order(
    wfn, atom_i: int, atom_j: int, density_matrix: Optional[np.ndarray] = None
) -> float:
    """Calculate intrinsic bond order between two specific atoms.

    This is a convenience function for calculating the IBO between
    a single pair of atoms.

    Args:
        wfn: Wavefunction object
        atom_i: Index of atom i (0-based)
        atom_j: Index of atom j (0-based)
        density_matrix: Optional density matrix

    Returns:
        Intrinsic bond order value

    Raises:
        ValueError: If atom indices are invalid
    """
    n_atoms = wfn.num_atoms
    if not (0 <= atom_i < n_atoms and 0 <= atom_j < n_atoms):
        raise ValueError(f"Atom indices must be in range [0, {n_atoms})")
    if atom_i == atom_j:
        raise ValueError("Atom indices must be different")

    # Get atomic basis function indices
    atom_to_bfs = wfn.get_atomic_basis_indices()
    bfs_i = atom_to_bfs.get(atom_i, [])
    bfs_j = atom_to_bfs.get(atom_j, [])

    if not bfs_i or not bfs_j:
        return 0.0

    # Use provided density matrix or get from wavefunction
    if density_matrix is None:
        if wfn.Ptot is None:
            wfn.calculate_density_matrices()
            if wfn.Ptot is None:
                raise ValueError("Density matrix could not be calculated")
        density_matrix = wfn.Ptot

    # Get element symbols
    element_i = wfn.atoms[atom_i].element
    element_j = wfn.atoms[atom_j].element

    # Calculate intrinsic bond order
    ibo, _, _ = calculate_intrinsic_bond_order(
        density_matrix, wfn.overlap_matrix, element_i, element_j, bfs_i, bfs_j
    )

    return ibo
