"""
Fragment analysis for orbital composition.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pymultiwfn.core.data import Wavefunction


class FragmentAnalyzer:
    """
    Analyzes orbital compositions for molecular fragments.

    This class allows defining molecular fragments and analyzing
    their contributions to molecular orbitals using various methods.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the fragment analyzer.

        Args:
            wavefunction: Wavefunction object containing molecular information
        """
        self.wfn = wavefunction
        self.fragments = []

        if self.wfn.coefficients is None:
            raise ValueError("MO coefficients are required for fragment analysis")

    def define_fragment(self, name: str, atom_indices: List[int]) -> None:
        """
        Define a molecular fragment.

        Args:
            name: Name of the fragment
            atom_indices: List of atom indices (0-based) in the fragment
        """
        # Validate atom indices
        for idx in atom_indices:
            if not (0 <= idx < len(self.wfn.atoms)):
                raise IndexError(f"Atom index {idx} out of range")

        self.fragments.append({
            'name': name,
            'atom_indices': atom_indices,
            'basis_indices': self._get_fragment_basis_indices(atom_indices)
        })

    def calculate_fragment_compositions(self, method: str = 'mulliken',
                                      is_beta: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate fragment compositions for all molecular orbitals.

        Args:
            method: Analysis method ('mulliken', 'scpa', 'hirshfeld', 'becke')
            is_beta: Whether to analyze beta orbitals

        Returns:
            Dictionary containing fragment contributions for each orbital
        """
        if not self.fragments:
            raise ValueError("No fragments defined. Use define_fragment() first.")

        if is_beta:
            coefficients = self.wfn.coefficients_beta
        else:
            coefficients = self.wfn.coefficients

        if coefficients is None:
            raise ValueError("MO coefficients not available")

        n_mos = coefficients.shape[0]
        n_fragments = len(self.fragments)

        # Initialize results
        fragment_contributions = np.zeros((n_mos, n_fragments))

        # Calculate contributions based on method
        if method == 'mulliken':
            fragment_contributions = self._calculate_mulliken_fragment_compositions(coefficients)
        elif method == 'scpa':
            fragment_contributions = self._calculate_scpa_fragment_compositions(coefficients)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Create result dictionary
        results = {}
        for i, fragment in enumerate(self.fragments):
            results[fragment['name']] = fragment_contributions[:, i]

        return results

    def _calculate_mulliken_fragment_compositions(self, coefficients: np.ndarray) -> np.ndarray:
        """Calculate fragment compositions using Mulliken method."""
        if self.wfn.overlap_matrix is None:
            raise ValueError("Overlap matrix required for Mulliken analysis")

        n_mos, n_basis = coefficients.shape
        n_fragments = len(self.fragments)

        fragment_contributions = np.zeros((n_mos, n_fragments))

        S = self.wfn.overlap_matrix

        for i in range(n_mos):
            C = coefficients[i, :]
            basis_contributions = C * np.dot(S, C)

            for frag_idx, fragment in enumerate(self.fragments):
                # Sum contributions from basis functions in this fragment
                frag_contribution = 0.0
                for basis_idx in fragment['basis_indices']:
                    frag_contribution += basis_contributions[basis_idx]
                fragment_contributions[i, frag_idx] = frag_contribution

        return fragment_contributions

    def _calculate_scpa_fragment_compositions(self, coefficients: np.ndarray) -> np.ndarray:
        """Calculate fragment compositions using SCPA method."""
        n_mos, n_basis = coefficients.shape
        n_fragments = len(self.fragments)

        fragment_contributions = np.zeros((n_mos, n_fragments))

        for i in range(n_mos):
            C = coefficients[i, :]
            C_squared = C**2
            total_squared = np.sum(C_squared)

            if total_squared > 0:
                basis_contributions = C_squared / total_squared

                for frag_idx, fragment in enumerate(self.fragments):
                    # Sum contributions from basis functions in this fragment
                    frag_contribution = 0.0
                    for basis_idx in fragment['basis_indices']:
                        frag_contribution += basis_contributions[basis_idx]
                    fragment_contributions[i, frag_idx] = frag_contribution

        return fragment_contributions

    def _get_fragment_basis_indices(self, atom_indices: List[int]) -> List[int]:
        """Get basis function indices for atoms in a fragment."""
        basis_indices = []
        current_basis_idx = 0

        for atom_idx, atom in enumerate(self.wfn.atoms):
            if atom_idx in atom_indices:
                # Add all basis functions for this atom
                for shell in atom.basis_functions:
                    n_functions = shell.get_n_functions()
                    for i in range(n_functions):
                        basis_indices.append(current_basis_idx + i)
                    current_basis_idx += n_functions
            else:
                # Skip basis functions for atoms not in fragment
                for shell in atom.basis_functions:
                    current_basis_idx += shell.get_n_functions()

        return basis_indices

    def get_fragment_info(self) -> List[Dict]:
        """Get information about defined fragments."""
        return [
            {
                'name': frag['name'],
                'atom_indices': frag['atom_indices'],
                'n_atoms': len(frag['atom_indices']),
                'n_basis_functions': len(frag['basis_indices'])
            }
            for frag in self.fragments
        ]

    def clear_fragments(self) -> None:
        """Clear all defined fragments."""
        self.fragments = []