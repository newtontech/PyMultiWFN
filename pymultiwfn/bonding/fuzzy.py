"""Fuzzy bond order implementation (Issue 20).

This module implements fuzzy bond order analysis based on the fuzzy atom
partitioning method. The fuzzy bond order (FBO) is calculated using fuzzy
overlap populations and provides a robust measure of bond strength.

References:
- Mayer, I. (1984). Chem. Phys. Lett. 97, 270-274.
- Wiberg, K. B. (1968). Tetrahedron 24, 1083-1096.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# van der Waals radii (in Angstroms)
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "B": 1.92,
    "Si": 2.10,
    "Ge": 2.11,
    "As": 1.85,
    "Se": 1.90,
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
            raise ValueError(
                f"Coordinates must be 3D array, got {self.coordinates.shape}"
            )
        if not (0.0 < self.fuzzy_factor <= 1.0):
            raise ValueError(f"Fuzzy factor must be in (0, 1], got {self.fuzzy_factor}")

    @property
    def symbol(self) -> str:
        """Get atomic symbol."""
        return self.element

    def fuzzy_overlap_radius(self, other: "FuzzyAtom") -> float:
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
    atom_i: int,
    atom_j: int,
    fuzzy_factor: float = 1.0,
    atomic_basis_indices: Optional[Dict[int, List[int]]] = None,
) -> float:
    """Calculate fuzzy bond order between two atoms.

    The current AO-partition implementation uses the same block contraction as
    Mayer/Wiberg bond order and applies ``fuzzy_factor`` as an optional damping
    factor:
    FBO_AB = fuzzy_factor * trace((P*S)_AB @ (P*S)_BA)

    where P is the density matrix and S is the overlap matrix.

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_i: Index of atom i (1-based)
        atom_j: Index of atom j (1-based)
        fuzzy_factor: Fuzzy partition factor (default 1.0)
        atomic_basis_indices: Optional mapping from 0-based atom index to AO indices.
            If omitted, atom_i/atom_j are interpreted as matrix row/column indices.

    Returns:
        Fuzzy bond order value

    Raises:
        IndexError: If atom indices are out of range
    """
    _validate_matrices(density_matrix, overlap_matrix)
    _validate_fuzzy_factor(fuzzy_factor)

    bond_order_matrix = calculate_fuzzy_bond_order_matrix(
        density_matrix,
        overlap_matrix,
        fuzzy_factor=fuzzy_factor,
        atomic_basis_indices=atomic_basis_indices,
    )

    i_idx = atom_i - 1
    j_idx = atom_j - 1
    n_atoms = bond_order_matrix.shape[0]
    if i_idx < 0 or i_idx >= n_atoms:
        raise IndexError(f"Atom index {atom_i} out of range [1, {n_atoms}]")
    if j_idx < 0 or j_idx >= n_atoms:
        raise IndexError(f"Atom index {atom_j} out of range [1, {n_atoms}]")
    if i_idx == j_idx:
        raise ValueError("Atom indices must be different")

    return float(bond_order_matrix[i_idx, j_idx])


def calculate_fuzzy_bond_order_matrix(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    fuzzy_factor: float = 1.0,
    atomic_basis_indices: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Calculate fuzzy bond order matrix.

    Args:
        density_matrix: Density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        fuzzy_factor: Fuzzy partition factor (default 1.0)
        atomic_basis_indices: Optional mapping from 0-based atom index to AO indices.
            If omitted, the returned matrix is at the input matrix dimension.

    Returns:
        Symmetric bond order matrix
    """
    _validate_matrices(density_matrix, overlap_matrix)
    _validate_fuzzy_factor(fuzzy_factor)

    ps_matrix = density_matrix @ overlap_matrix

    if atomic_basis_indices is None:
        bond_order_matrix = fuzzy_factor * ps_matrix
    else:
        n_atoms = len(atomic_basis_indices)
        bond_order_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        for atom_a in range(n_atoms):
            bfs_a = atomic_basis_indices.get(atom_a, [])
            for atom_b in range(atom_a + 1, n_atoms):
                bfs_b = atomic_basis_indices.get(atom_b, [])
                if not bfs_a or not bfs_b:
                    continue
                block = ps_matrix[np.ix_(bfs_a, bfs_b)]
                reverse_block = ps_matrix[np.ix_(bfs_b, bfs_a)]
                value = fuzzy_factor * float(np.trace(block @ reverse_block))
                bond_order_matrix[atom_a, atom_b] = value
                bond_order_matrix[atom_b, atom_a] = value

    # Make symmetric and non-negative for deterministic public behavior.
    bond_order_matrix = np.abs((bond_order_matrix + bond_order_matrix.T) / 2)

    # Zero out diagonal (no self-bonding)
    np.fill_diagonal(bond_order_matrix, 0.0)

    return bond_order_matrix


def _validate_matrices(density_matrix: np.ndarray, overlap_matrix: np.ndarray) -> None:
    """Validate density and overlap matrices used by fuzzy bond order helpers."""
    if not isinstance(density_matrix, np.ndarray) or not isinstance(
        overlap_matrix, np.ndarray
    ):
        raise TypeError("density_matrix and overlap_matrix must be numpy arrays")
    if density_matrix.ndim != 2 or overlap_matrix.ndim != 2:
        raise ValueError("density_matrix and overlap_matrix must be 2D matrices")
    if density_matrix.shape[0] != density_matrix.shape[1]:
        raise ValueError("density_matrix must be square")
    if overlap_matrix.shape[0] != overlap_matrix.shape[1]:
        raise ValueError("overlap_matrix must be square")
    if density_matrix.shape != overlap_matrix.shape:
        raise ValueError(
            "density_matrix and overlap_matrix must have the same shape, "
            f"got {density_matrix.shape} and {overlap_matrix.shape}"
        )


def _validate_fuzzy_factor(fuzzy_factor: float) -> None:
    """Validate fuzzy partition factor."""
    if not isinstance(fuzzy_factor, (float, int)):
        raise TypeError("fuzzy_factor must be numeric")
    if not (0.0 < float(fuzzy_factor) <= 1.0):
        raise ValueError(f"fuzzy_factor must be in (0, 1], got {fuzzy_factor}")


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
