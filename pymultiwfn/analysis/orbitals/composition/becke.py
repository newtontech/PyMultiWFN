"""
Becke population analysis for orbital composition.
"""

import numpy as np
from typing import Dict, List, Optional
from pymultiwfn.core.data import Wavefunction


class BeckeAnalyzer:
    """
    Analyzes orbital compositions using Becke space partitioning.

    Becke method uses atomic weight functions to partition space
    and determine atomic contributions to molecular orbitals.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Becke analyzer.

        Args:
            wavefunction: Wavefunction object containing molecular information
        """
        self.wfn = wavefunction

        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for Becke analysis")

    def calculate_orbital_composition(self, mo_idx: int, coords: np.ndarray,
                                    is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate Becke composition for a specific molecular orbital.

        Args:
            mo_idx: 0-based index of the molecular orbital
            coords: (N_points, 3) array of grid coordinates
            is_beta: Whether to analyze beta orbitals

        Returns:
            Dictionary containing:
            - 'atom_contributions': Contributions from each atom
            - 'weights': Becke weights at each grid point
        """
        if is_beta:
            coefficients = self.wfn.coefficients_beta
        else:
            coefficients = self.wfn.coefficients

        if coefficients is None:
            raise ValueError("MO coefficients not available")

        if not (0 <= mo_idx < coefficients.shape[0]):
            raise IndexError(f"Molecular orbital index {mo_idx} out of range")

        # Calculate orbital density on grid
        orbital_density = self._calculate_orbital_density(mo_idx, coords, is_beta)

        # Calculate Becke weights
        weights = self._calculate_becke_weights(coords)

        # Calculate atomic contributions
        atom_contributions = self._integrate_contributions(orbital_density, weights, coords)

        return {
            'atom_contributions': atom_contributions,
            'weights': weights
        }

    def _calculate_orbital_density(self, mo_idx: int, coords: np.ndarray,
                                 is_beta: bool = False) -> np.ndarray:
        """Calculate orbital density on grid points."""
        from pymultiwfn.analysis.orbitals import OrbitalAnalyzer

        analyzer = OrbitalAnalyzer(self.wfn)
        mo_values = analyzer.calculate_mo_on_grid(mo_idx, coords, is_beta)
        return mo_values**2

    def _calculate_becke_weights(self, coords: np.ndarray) -> np.ndarray:
        """Calculate Becke weights at grid points."""
        n_points = coords.shape[0]
        n_atoms = len(self.wfn.atoms)

        # Calculate distances from each atom
        distances = np.zeros((n_atoms, n_points))
        for i, atom in enumerate(self.wfn.atoms):
            coords_centered = coords - atom.coords
            distances[i, :] = np.linalg.norm(coords_centered, axis=1)

        # Calculate Becke weights
        weights = np.ones((n_atoms, n_points))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate Becke step function
                mu_ij = (distances[i, :] - distances[j, :]) / distances[i, :]
                s_ij = self._becke_step_function(mu_ij)

                # Apply step function to weights
                weights[i, :] *= s_ij
                weights[j, :] *= (1 - s_ij)

        # Normalize weights
        total_weights = np.sum(weights, axis=0)
        mask = total_weights > 1e-12
        weights[:, mask] /= total_weights[mask]

        return weights

    def _becke_step_function(self, mu: np.ndarray) -> np.ndarray:
        """Becke step function for space partitioning."""
        # Becke's original step function
        s = 1.5 * mu - 0.5 * mu**3
        return 0.5 * (1 - s)

    def _integrate_contributions(self, orbital_density: np.ndarray,
                               weights: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Integrate orbital density with Becke weights."""
        n_atoms = weights.shape[0]
        atom_contributions = np.zeros(n_atoms)

        # Simple integration (assuming uniform grid spacing)
        # In practice, this would use proper numerical integration weights
        for i in range(n_atoms):
            atom_contributions[i] = np.sum(orbital_density * weights[i, :])

        # Normalize contributions
        total = np.sum(atom_contributions)
        if total > 0:
            atom_contributions /= total

        return atom_contributions