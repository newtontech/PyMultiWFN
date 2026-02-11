"""
Simple approximate overlap matrix for testing.

This module provides a simplified overlap matrix calculation
for testing purposes when the full implementation is not available.
"""

import numpy as np
from typing import List, Tuple
from ..core.data import Wavefunction


def calculate_simple_overlap_matrix(wfn: Wavefunction) -> np.ndarray:
    """
    Calculate a simple approximate overlap matrix.

    This is a simplified version for testing:
    - Diagonal elements: 1.0 (normalized basis functions)
    - Off-diagonal elements: Based on distance between basis function centers
    - Uses exponential decay: S_ij = exp(-alpha * r_ij^2)

    Args:
        wfn: Wavefunction object

    Returns:
        Approximate overlap matrix (nbasis x nbasis)
    """
    if wfn.num_basis == 0:
        return np.array([])

    nbasis = wfn.num_basis
    overlap_matrix = np.zeros((nbasis, nbasis))

    # Get basis function centers (atoms)
    atom_coords = np.array([atom.coord for atom in wfn.atoms])

    # Map each basis function to its atom center
    # We need to know which basis function belongs to which atom
    # For simplicity, assume basis functions are evenly distributed across atoms
    atom_indices = _get_basis_to_atom_mapping(wfn)

    # Calculate overlap matrix
    for i in range(nbasis):
        atom_i = atom_indices[i]
        coord_i = atom_coords[atom_i]

        for j in range(i, nbasis):
            atom_j = atom_indices[j]
            coord_j = atom_coords[atom_j]

            if i == j:
                # Diagonal: normalized basis function
                overlap_matrix[i, j] = 1.0
            elif atom_i == atom_j:
                # Same atom: high overlap
                overlap_matrix[i, j] = 0.3  # Approximate
            else:
                # Different atoms: based on distance
                distance = np.linalg.norm(coord_i - coord_j)
                # Exponential decay
                alpha = 1.0  # Decay parameter
                overlap_matrix[i, j] = np.exp(-alpha * distance**2)

            # Symmetric
            overlap_matrix[j, i] = overlap_matrix[i, j]

    return overlap_matrix


def _get_basis_to_atom_mapping(wfn: Wavefunction) -> np.ndarray:
    """
    Map each basis function to its atom index.

    Args:
        wfn: Wavefunction object

    Returns:
        Array of shape (nbasis,) with atom indices for each basis function
    """
    atom_indices = []

    # Go through shells and extract atom assignments
    bf_counter = 0
    for shell in wfn.shells:
        atom_idx = shell.center_idx

        # Number of functions in this shell
        if shell.type == -1:  # SP shell
            num_functions = 4  # 1 S + 3 P
        elif shell.type == 0:  # S shell
            num_functions = 1
        elif shell.type == 1:  # P shell
            num_functions = 3
        elif shell.type == 2:  # D shell (spherical)
            num_functions = 5
        elif shell.type == 3:  # F shell (spherical)
            num_functions = 7
        else:
            num_functions = 1  # Default

        # Add atom index for each basis function
        for _ in range(num_functions):
            atom_indices.append(atom_idx)
            bf_counter += 1

    return np.array(atom_indices)
