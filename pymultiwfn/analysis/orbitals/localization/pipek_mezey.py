"""
Pipek-Mezey orbital localization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pymultiwfn.core.data import Wavefunction


class PipekMezeyLocalizer:
    """
    Implements Pipek-Mezey orbital localization.

    Pipek-Mezey localization maximizes the sum of squared atomic populations
    to produce chemically intuitive localized orbitals.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Pipek-Mezey localizer.

        Args:
            wavefunction: Wavefunction object containing MO coefficients and overlap matrix
        """
        self.wfn = wavefunction

        if self.wfn.overlap_matrix is None:
            raise ValueError("Overlap matrix is required for Pipek-Mezey localization")
        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for Pipek-Mezey localization")

    def localize_orbitals(self, orbital_indices: List[int], max_iterations: int = 100,
                         convergence_threshold: float = 1e-8, is_beta: bool = False) -> Dict:
        """
        Localize a set of molecular orbitals using Pipek-Mezey method.

        Args:
            orbital_indices: List of orbital indices to localize
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for localization
            is_beta: Whether to localize beta orbitals

        Returns:
            Dictionary containing:
            - 'localized_coefficients': Localized orbital coefficients
            - 'transformation_matrix': Transformation matrix
            - 'iterations': Number of iterations performed
            - 'converged': Whether localization converged
        """
        if is_beta:
            coefficients = self.wfn.coefficients_beta
        else:
            coefficients = self.wfn.coefficients

        if coefficients is None:
            raise ValueError("MO coefficients not available")

        # Select orbitals to localize
        n_orbitals = len(orbital_indices)
        if n_orbitals < 2:
            raise ValueError("At least 2 orbitals required for localization")

        C_local = coefficients[orbital_indices, :].copy()
        S = self.wfn.overlap_matrix

        # Initialize transformation matrix
        U = np.eye(n_orbitals)

        converged = False
        iteration = 0

        for iteration in range(max_iterations):
            max_change = 0.0

            # Iterate over all orbital pairs
            for i in range(n_orbitals):
                for j in range(i + 1, n_orbitals):
                    # Calculate rotation angle for this pair
                    angle = self._calculate_rotation_angle(C_local[i, :], C_local[j, :], S)

                    if abs(angle) > 1e-12:
                        # Apply rotation
                        c = np.cos(angle)
                        s = np.sin(angle)

                        # Rotate orbitals
                        temp_i = c * C_local[i, :] - s * C_local[j, :]
                        temp_j = s * C_local[i, :] + c * C_local[j, :]
                        C_local[i, :] = temp_i
                        C_local[j, :] = temp_j

                        # Update transformation matrix
                        temp_u_i = c * U[i, :] - s * U[j, :]
                        temp_u_j = s * U[i, :] + c * U[j, :]
                        U[i, :] = temp_u_i
                        U[j, :] = temp_u_j

                        max_change = max(max_change, abs(angle))

            # Check convergence
            if max_change < convergence_threshold:
                converged = True
                break

        # Create full coefficient matrix with localized orbitals
        full_coefficients = coefficients.copy()
        for idx, local_idx in enumerate(orbital_indices):
            full_coefficients[local_idx, :] = C_local[idx, :]

        return {
            'localized_coefficients': full_coefficients,
            'transformation_matrix': U,
            'iterations': iteration + 1,
            'converged': converged
        }

    def _calculate_rotation_angle(self, C_i: np.ndarray, C_j: np.ndarray, S: np.ndarray) -> float:
        """
        Calculate optimal rotation angle for Pipek-Mezey localization.

        Args:
            C_i: Coefficients for orbital i
            C_j: Coefficients for orbital j
            S: Overlap matrix

        Returns:
            Optimal rotation angle
        """
        # Calculate atomic populations
        P_i = self._calculate_atomic_populations(C_i, S)
        P_j = self._calculate_atomic_populations(C_j, S)
        P_ij = self._calculate_atomic_populations_mixed(C_i, C_j, S)

        # Calculate gradient and Hessian
        grad = 0.0
        hess = 0.0

        for atom_idx in range(len(self.wfn.atoms)):
            grad += 2 * P_ij[atom_idx] * (P_i[atom_idx] - P_j[atom_idx])
            hess += 2 * (P_ij[atom_idx]**2 - (P_i[atom_idx] - P_j[atom_idx])**2)

        # Calculate rotation angle
        if abs(hess) > 1e-12:
            angle = -grad / hess
            # Limit rotation angle
            angle = np.clip(angle, -np.pi/4, np.pi/4)
        else:
            angle = 0.0

        return angle

    def _calculate_atomic_populations(self, C: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Calculate atomic populations for an orbital."""
        n_atoms = len(self.wfn.atoms)
        populations = np.zeros(n_atoms)

        # Calculate basis function contributions
        basis_contributions = C * np.dot(S, C)

        # Sum to atoms
        basis_idx = 0
        for atom_idx, atom in enumerate(self.wfn.atoms):
            for shell in atom.basis_functions:
                n_functions = shell.get_n_functions()
                populations[atom_idx] += np.sum(
                    basis_contributions[basis_idx:basis_idx + n_functions]
                )
                basis_idx += n_functions

        return populations

    def _calculate_atomic_populations_mixed(self, C_i: np.ndarray, C_j: np.ndarray,
                                          S: np.ndarray) -> np.ndarray:
        """Calculate mixed atomic populations for two orbitals."""
        n_atoms = len(self.wfn.atoms)
        populations = np.zeros(n_atoms)

        # Calculate mixed basis function contributions
        mixed_contributions = 0.5 * (C_i * np.dot(S, C_j) + C_j * np.dot(S, C_i))

        # Sum to atoms
        basis_idx = 0
        for atom_idx, atom in enumerate(self.wfn.atoms):
            for shell in atom.basis_functions:
                n_functions = shell.get_n_functions()
                populations[atom_idx] += np.sum(
                    mixed_contributions[basis_idx:basis_idx + n_functions]
                )
                basis_idx += n_functions

        return populations