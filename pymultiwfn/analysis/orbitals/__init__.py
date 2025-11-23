"""
Module for analyzing molecular orbitals.
"""

import numpy as np
from typing import Optional, Tuple

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.basis import evaluate_basis

# Import composition analysis
from .composition import (
    MullikenAnalyzer,
    SCPAAnalyzer,
    HirshfeldAnalyzer,
    BeckeAnalyzer,
    FragmentAnalyzer
)

# Import localization methods
from .localization import (
    PipekMezeyLocalizer,
    FosterBoysLocalizer
)

class OrbitalAnalyzer:
    def __init__(self, wavefunction: Wavefunction):
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction object must contain MO coefficients.")
        self.wfn = wavefunction

    def get_orbital_properties(self, mo_idx: int, is_beta: bool = False) -> dict:
        """
        Retrieves properties (energy, occupation) for a given molecular orbital.

        Args:
            mo_idx: The 0-based index of the molecular orbital.
            is_beta: If True, retrieve properties for beta orbital.

        Returns:
            A dictionary containing 'energy' and 'occupation'.
        """
        if is_beta:
            energies = self.wfn.energies_beta
            occupations = self.wfn.occupations_beta
            if energies is None or occupations is None:
                raise ValueError("Beta orbital properties not available or calculated.")
        else:
            energies = self.wfn.energies
            occupations = self.wfn.occupations
            if energies is None or occupations is None:
                raise ValueError("Alpha orbital properties not available or calculated.")

        if not (0 <= mo_idx < len(energies)):
            raise IndexError(f"Molecular orbital index {mo_idx} out of range.")

        return {
            "energy": energies[mo_idx],
            "occupation": occupations[mo_idx]
        }

    def calculate_mo_on_grid(self, mo_idx: int, coords: np.ndarray, is_beta: bool = False) -> np.ndarray:
        """
        Calculates the value of a specific molecular orbital on a grid of points.

        Args:
            mo_idx: The 0-based index of the molecular orbital.
            coords: (N_points, 3) array of Cartesian coordinates where to evaluate the MO.
            is_beta: If True, calculate for beta orbital.

        Returns:
            (N_points,) array of MO values at each grid point.
        """
        if is_beta:
            mo_coefficients = self.wfn.coefficients_beta
            if mo_coefficients is None:
                raise ValueError("Beta MO coefficients not available.")
        else:
            mo_coefficients = self.wfn.coefficients
            if mo_coefficients is None:
                raise ValueError("Alpha MO coefficients not available.")

        if not (0 <= mo_idx < mo_coefficients.shape[0]):
            raise IndexError(f"Molecular orbital index {mo_idx} out of range for available MOs.")

        # Evaluate all basis functions at the grid points
        phi = evaluate_basis(self.wfn, coords)  # (N_points, N_basis)

        # Get the coefficients for the specific molecular orbital
        # mo_coefficients is (N_MOs, N_basis)
        c_mo = mo_coefficients[mo_idx, :]  # (N_basis,)

        # Calculate MO value: psi_i(r) = sum_mu c_mu_i * phi_mu(r)
        # Result will be (N_points,)
        mo_values = np.dot(phi, c_mo)

        return mo_values

    def calculate_mo_density_on_grid(self, mo_idx: int, coords: np.ndarray, is_beta: bool = False) -> np.ndarray:
        """
        Calculates the density contribution of a specific molecular orbital on a grid of points.
        (i.e., psi_i(r)^2).

        Args:
            mo_idx: The 0-based index of the molecular orbital.
            coords: (N_points, 3) array of Cartesian coordinates where to evaluate the MO density.
            is_beta: If True, calculate for beta orbital.

        Returns:
            (N_points,) array of MO density values at each grid point.
        """
        mo_values = self.calculate_mo_on_grid(mo_idx, coords, is_beta)
        return mo_values**2
