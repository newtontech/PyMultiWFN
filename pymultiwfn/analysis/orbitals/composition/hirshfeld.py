"""
Hirshfeld population analysis for orbital composition.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.density import calc_density


class HirshfeldAnalyzer:
    """
    Analyzes orbital compositions using Hirshfeld space partitioning.

    Hirshfeld method uses promolecular densities to partition space
    and determine atomic contributions to molecular orbitals.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Hirshfeld analyzer.

        Args:
            wavefunction: Wavefunction object containing molecular information
        """
        self.wfn = wavefunction

        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for Hirshfeld analysis")

    def calculate_orbital_composition(self, mo_idx: int, coords: np.ndarray,
                                    is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate Hirshfeld composition for a specific molecular orbital.

        Args:
            mo_idx: 0-based index of the molecular orbital
            coords: (N_points, 3) array of grid coordinates
            is_beta: Whether to analyze beta orbitals

        Returns:
            Dictionary containing:
            - 'atom_contributions': Contributions from each atom
            - 'weights': Hirshfeld weights at each grid point
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

        # Calculate Hirshfeld weights
        weights = self._calculate_hirshfeld_weights(coords)

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

    def _calculate_hirshfeld_weights(self, coords: np.ndarray) -> np.ndarray:
        """Calculate Hirshfeld weights at grid points."""
        n_points = coords.shape[0]
        n_atoms = len(self.wfn.atoms)

        # Calculate promolecular density for each atom
        promolecular_densities = np.zeros((n_atoms, n_points))

        for i, atom in enumerate(self.wfn.atoms):
            promolecular_densities[i, :] = self._calculate_atomic_density(
                atom, coords
            )

        # Calculate total promolecular density
        total_promolecular_density = np.sum(promolecular_densities, axis=0)

        # Calculate Hirshfeld weights
        weights = np.zeros((n_atoms, n_points))
        for i in range(n_atoms):
            # Avoid division by zero
            mask = total_promolecular_density > 1e-12
            weights[i, mask] = promolecular_densities[i, mask] / total_promolecular_density[mask]

        return weights

    def _calculate_atomic_density(self, atom, coords: np.ndarray) -> np.ndarray:
        """Calculate atomic density using Slater-type orbitals approximation."""
        # Simplified atomic density calculation
        # In practice, this would use tabulated atomic densities
        # or more sophisticated atomic wavefunctions

        atomic_number = atom.atomic_number
        coords_centered = coords - atom.coords
        distances = np.linalg.norm(coords_centered, axis=1)

        # Simple exponential approximation
        # For better accuracy, use tabulated atomic densities
        if atomic_number <= 2:  # H, He
            zeta = atomic_number
        else:
            zeta = atomic_number * 0.7  # Approximate screening

        density = np.exp(-2 * zeta * distances)

        # Normalize (approximate)
        density *= atomic_number / (4 * np.pi * np.sum(density))

        return density

    def _integrate_contributions(self, orbital_density: np.ndarray,
                               weights: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Integrate orbital density with Hirshfeld weights."""
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