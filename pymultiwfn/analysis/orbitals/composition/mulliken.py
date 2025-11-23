"""
Mulliken population analysis for orbital composition.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pymultiwfn.core.data import Wavefunction


class MullikenAnalyzer:
    """
    Analyzes orbital compositions using Mulliken population analysis.

    This class implements the traditional Mulliken method for determining
    atomic orbital contributions to molecular orbitals.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Mulliken analyzer.

        Args:
            wavefunction: Wavefunction object containing MO coefficients and overlap matrix
        """
        self.wfn = wavefunction

        if self.wfn.overlap_matrix is None:
            raise ValueError("Overlap matrix is required for Mulliken analysis")
        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for Mulliken analysis")

    def calculate_orbital_composition(self, mo_idx: int, is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate Mulliken composition for a specific molecular orbital.

        Args:
            mo_idx: 0-based index of the molecular orbital
            is_beta: Whether to analyze beta orbitals (for unrestricted calculations)

        Returns:
            Dictionary containing:
            - 'basis_contributions': Contributions from each basis function
            - 'atom_contributions': Contributions from each atom
            - 'shell_contributions': Contributions from each shell type
        """
        if is_beta:
            coefficients = self.wfn.coefficients_beta
        else:
            coefficients = self.wfn.coefficients

        if coefficients is None:
            raise ValueError("MO coefficients not available")

        if not (0 <= mo_idx < coefficients.shape[0]):
            raise IndexError(f"Molecular orbital index {mo_idx} out of range")

        S = self.wfn.overlap_matrix
        C = coefficients[mo_idx, :]  # Coefficients for this MO

        # Calculate basis function contributions
        # Q_mu = C_mu * sum_nu(C_nu * S_mu_nu)
        basis_contributions = C * np.dot(S, C)

        # Calculate atomic contributions
        atom_contributions = self._sum_to_atoms(basis_contributions)

        # Calculate shell contributions
        shell_contributions = self._sum_to_shells(basis_contributions)

        return {
            'basis_contributions': basis_contributions,
            'atom_contributions': atom_contributions,
            'shell_contributions': shell_contributions
        }

    def calculate_all_orbital_compositions(self, is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate Mulliken compositions for all molecular orbitals.

        Args:
            is_beta: Whether to analyze beta orbitals

        Returns:
            Dictionary containing 2D arrays of contributions:
            - 'basis_contributions': (n_mos, n_basis)
            - 'atom_contributions': (n_mos, n_atoms)
            - 'shell_contributions': (n_mos, n_shell_types)
        """
        if is_beta:
            coefficients = self.wfn.coefficients_beta
        else:
            coefficients = self.wfn.coefficients

        if coefficients is None:
            raise ValueError("MO coefficients not available")

        S = self.wfn.overlap_matrix
        n_mos, n_basis = coefficients.shape

        # Calculate all basis function contributions
        # For each MO: Q_mu = C_mu * sum_nu(C_nu * S_mu_nu)
        basis_contributions = np.zeros((n_mos, n_basis))
        for i in range(n_mos):
            C = coefficients[i, :]
            basis_contributions[i, :] = C * np.dot(S, C)

        # Calculate atomic contributions
        atom_contributions = np.zeros((n_mos, len(self.wfn.atoms)))
        for i in range(n_mos):
            atom_contributions[i, :] = self._sum_to_atoms(basis_contributions[i, :])

        # Calculate shell contributions
        shell_types = self._get_shell_types()
        shell_contributions = np.zeros((n_mos, len(shell_types)))
        for i in range(n_mos):
            shell_contributions[i, :] = self._sum_to_shells(basis_contributions[i, :])

        return {
            'basis_contributions': basis_contributions,
            'atom_contributions': atom_contributions,
            'shell_contributions': shell_contributions
        }

    def _sum_to_atoms(self, basis_contributions: np.ndarray) -> np.ndarray:
        """Sum basis function contributions to atomic contributions."""
        atom_contributions = np.zeros(len(self.wfn.atoms))

        # Group basis functions by atom
        basis_idx = 0
        for atom_idx, atom in enumerate(self.wfn.atoms):
            for shell in atom.basis_functions:
                n_functions = shell.get_n_functions()
                atom_contributions[atom_idx] += np.sum(
                    basis_contributions[basis_idx:basis_idx + n_functions]
                )
                basis_idx += n_functions

        return atom_contributions

    def _sum_to_shells(self, basis_contributions: np.ndarray) -> np.ndarray:
        """Sum basis function contributions to shell type contributions."""
        shell_types = self._get_shell_types()
        shell_contributions = np.zeros(len(shell_types))

        # Map shell types to indices
        shell_type_map = {shell_type: i for i, shell_type in enumerate(shell_types)}

        basis_idx = 0
        for atom in self.wfn.atoms:
            for shell in atom.basis_functions:
                shell_type = shell.shell_type
                n_functions = shell.get_n_functions()
                shell_idx = shell_type_map[shell_type]
                shell_contributions[shell_idx] += np.sum(
                    basis_contributions[basis_idx:basis_idx + n_functions]
                )
                basis_idx += n_functions

        return shell_contributions

    def _get_shell_types(self) -> List[str]:
        """Get unique shell types in the basis set."""
        shell_types = set()
        for atom in self.wfn.atoms:
            for shell in atom.basis_functions:
                shell_types.add(shell.shell_type)
        return sorted(shell_types)