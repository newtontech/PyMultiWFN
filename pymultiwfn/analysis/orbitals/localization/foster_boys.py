"""
Foster-Boys orbital localization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pymultiwfn.core.data import Wavefunction


class FosterBoysLocalizer:
    """
    Implements Foster-Boys orbital localization.

    Foster-Boys localization minimizes the spatial extent of orbitals
    to produce localized orbitals with minimal spatial overlap.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Foster-Boys localizer.

        Args:
            wavefunction: Wavefunction object containing MO coefficients and dipole integrals
        """
        self.wfn = wavefunction

        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for Foster-Boys localization")

        # Note: In practice, dipole moment integrals would be needed
        # For now, we'll use a simplified approach

    def localize_orbitals(self, orbital_indices: List[int], max_iterations: int = 100,
                         convergence_threshold: float = 1e-8, is_beta: bool = False) -> Dict:
        """
        Localize a set of molecular orbitals using Foster-Boys method.

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
                    angle = self._calculate_rotation_angle(C_local[i, :], C_local[j, :])

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

    def _calculate_rotation_angle(self, C_i: np.ndarray, C_j: np.ndarray) -> float:
        """
        Calculate optimal rotation angle for Foster-Boys localization.

        Args:
            C_i: Coefficients for orbital i
            C_j: Coefficients for orbital j

        Returns:
            Optimal rotation angle
        """
        # Simplified Foster-Boys implementation
        # In practice, this would use dipole moment integrals
        # For now, we'll use a heuristic based on orbital overlap

        # Calculate approximate orbital centers
        center_i = self._calculate_orbital_center(C_i)
        center_j = self._calculate_orbital_center(C_j)

        # Calculate distance between centers
        distance = np.linalg.norm(center_i - center_j)

        # Simple heuristic: rotate to maximize separation
        # This is a simplified version - real Foster-Boys uses dipole integrals
        if distance > 1e-6:
            # Calculate angle based on current overlap
            overlap = np.dot(C_i, C_j)
            angle = 0.25 * np.arcsin(np.clip(overlap, -1, 1))
        else:
            angle = 0.0

        return angle

    def _calculate_orbital_center(self, C: np.ndarray) -> np.ndarray:
        """Calculate approximate center of an orbital."""
        center = np.zeros(3)
        total_weight = 0.0

        basis_idx = 0
        for atom in self.wfn.atoms:
            for shell in atom.basis_functions:
                n_functions = shell.get_n_functions()

                # Use squared coefficients as weights
                weights = C[basis_idx:basis_idx + n_functions]**2
                total_weight += np.sum(weights)

                # Add weighted atomic position
                center += atom.coords * np.sum(weights)

                basis_idx += n_functions

        if total_weight > 0:
            center /= total_weight

        return center