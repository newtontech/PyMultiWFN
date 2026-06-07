"""Fuzzy bond order implementation (Issue 20).

This module implements fuzzy bond order analysis based on the fuzzy atom
partitioning method. The fuzzy bond order (FBO) is calculated using fuzzy
overlap populations and provides a robust measure of bond strength.

References:
- Mayer, I. (1984). Chem. Phys. Lett. 97, 270-274.
- Wiberg, K. B. (1968). Tetrahedron 24, 1083-1096.
"""

import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass


# van der Waals radii (in Angstroms)
VDW_RADII = {
    'H': 1.20,  'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47,
    'P': 1.80,  'S': 1.80,  'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'B': 1.92,  'Si': 2.10, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90,
}


@dataclass
class FuzzyAtom:
    """Represents a fuzzy atom with shared electron density.

    Attributes:
        atom_index: 0-based index of the atom
        element: Atomic symbol (e.g., 'C', 'H', 'O')
        coordinates: 3D coordinates (x, y, z) in atomic units or Angstroms
        vdwa_radius: van der Waals radius in Angstroms
        fuzzy_factor: Fuzzy partition factor (0.0-1.0, default 0.5)
    """
    atom_index: int
    element: str
    coordinates: np.ndarray
    vdwa_radius: float
    fuzzy_factor: float = 0.5

    def __post_init__(self):
        """Validate fuzzy atom parameters."""
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates, dtype=np.float64)
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must be 3D array, got {self.coordinates.shape}")
        if not (0.0 < self.fuzzy_factor < 1.0):
            raise ValueError(f"Fuzzy factor must be in (0, 1), got {self.fuzzy_factor}")

    @property
    def symbol(self) -> str:
        """Get atomic symbol."""
        return self.element

    def fuzzy_overlap_radius(self, other: 'FuzzyAtom') -> float:
        """Calculate fuzzy overlap radius with another atom.

        Args:
            other: Another FuzzyAtom

        Returns:
            Overlap radius in Angstroms
        """
        avg_factor = (self.fuzzy_factor + other.fuzzy_factor) / 2
        overlap = avg_factor * (self.vdwa_radius + other.vdwa_radius) / 2
        return overlap


def fuzzy_bond_order(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    shells: list,
    atom_i: int,
    atom_j: int,
    fuzzy_factor: float = 0.5
) -> float:
    """Calculate fuzzy bond order between two atoms.

    The fuzzy bond order is calculated by summing over all AOs on each atom:
    FBO_ij = fuzzy_factor * sum_{mu in i, nu in j} 2 * P_{mu,nu} * S_{mu,nu}

    where P is the density matrix, S is the overlap matrix, and the sum
    is over all atomic orbitals (AOs) centered on atoms i and j.

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        shells: List of Shell objects containing center_idx
        atom_i: Index of atom i (0-based)
        atom_j: Index of atom j (0-based)
        fuzzy_factor: Fuzzy partition factor (default 0.5)

    Returns:
        Fuzzy bond order value

    Raises:
        ValueError: If atom indices are invalid
    """
    # Validate atom indices
    if atom_i < 0 or atom_j < 0:
        raise ValueError(f"Atom indices must be non-negative, got {atom_i}, {atom_j}")
    if atom_i == atom_j:
        raise ValueError("Atom indices must be different")

    # Find AO indices for each atom
    ao_indices_i = [idx for idx, shell in enumerate(shells) if shell.center_idx == atom_i]
    ao_indices_j = [idx for idx, shell in enumerate(shells) if shell.center_idx == atom_j]

    if not ao_indices_i:
        raise ValueError(f"No AOs found for atom {atom_i}")
    if not ao_indices_j:
        raise ValueError(f"No AOs found for atom {atom_j}")

    # Calculate fuzzy bond order by summing over AO pairs
    # Factor of 2 for electron pairs
    bond_order = 0.0
    for mu in ao_indices_i:
        for nu in ao_indices_j:
            # Sum 2 * P_{mu,nu} * S_{mu,nu} for all AO pairs
            bond_order += 2.0 * density_matrix[mu, nu] * overlap_matrix[mu, nu]

    # Apply fuzzy factor
    bond_order *= fuzzy_factor

    # Ensure bond order is positive
    bond_order = max(0.0, abs(bond_order))

    return float(bond_order)


def calculate_fuzzy_bond_order_matrix(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    shells: list,
    fuzzy_factor: float = 0.5
) -> np.ndarray:
    """Calculate fuzzy bond order matrix.

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        shells: List of Shell objects containing center_idx
        fuzzy_factor: Fuzzy partition factor (default 0.5)

    Returns:
        Symmetric bond order matrix (n_atoms x n_atoms)
    """
    # Determine number of atoms from shells
    atom_indices = sorted(set(shell.center_idx for shell in shells))
    n_atoms = max(atom_indices) + 1

    # Initialize bond order matrix
    bond_order_matrix = np.zeros((n_atoms, n_atoms))

    # Calculate bond order for each atom pair
    for i in atom_indices:
        for j in atom_indices:
            if i < j:  # Only calculate upper triangle
                try:
                    bo = fuzzy_bond_order(density_matrix, overlap_matrix,
                                         shells, i, j, fuzzy_factor)
                    bond_order_matrix[i, j] = bo
                    bond_order_matrix[j, i] = bo  # Symmetric
                except ValueError:
                    # Skip if no AOs for either atom
                    pass

    return bond_order_matrix


def get_default_vdwa_radius(element: str) -> float:
    """Get default van der Waals radius for an element.

    Args:
        element: Atomic symbol

    Returns:
        van der Waals radius in Angstroms
    """
    element = element.capitalize()
    if element not in VDW_RADII:
        raise ValueError(f"No van der Waals radius defined for element: {element}")
    return VDW_RADII[element]
