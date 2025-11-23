"""
SCPA (Square of Coefficients Partition Analysis) for orbital composition.
"""

import numpy as np
from typing import Dict, List, Optional
from pymultiwfn.core.data import Wavefunction


class SCPAAnalyzer:
    """
    Analyzes orbital compositions using SCPA (Square of Coefficients Partition Analysis).

    SCPA is a modified population analysis method that uses squared coefficients
    normalized by the sum of squared coefficients.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the SCPA analyzer.

        Args:
            wavefunction: Wavefunction object containing MO coefficients
        """
        self.wfn = wavefunction

        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for SCPA analysis")

    def calculate_orbital_composition(self, mo_idx: int, is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate SCPA composition for a specific molecular orbital.

        Args:
            mo_idx: 0-based index of the molecular orbital
            is_beta: Whether to analyze beta orbitals

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

        C = coefficients[mo_idx, :]
        C_squared = C**2
        total_squared = np.sum(C_squared)

        # Normalize to get contributions
        if total_squared > 0:
            basis_contributions = C_squared / total_squared
        else:
            basis_contributions = np.zeros_like(C_squared)

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
        Calculate SCPA compositions for all molecular orbitals.

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

        n_mos, n_basis = coefficients.shape

        # Calculate all basis function contributions
        basis_contributions = np.zeros((n_mos, n_basis))
        for i in range(n_mos):
            C = coefficients[i, :]
            C_squared = C**2
            total_squared = np.sum(C_squared)
            if total_squared > 0:
                basis_contributions[i, :] = C_squared / total_squared

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