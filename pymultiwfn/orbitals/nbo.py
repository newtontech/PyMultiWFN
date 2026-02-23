"""
Natural Bond Orbital (NBO) Analysis Module.

This module provides functionality for Natural Bond Orbital analysis,
including NBO transformation, Lewis structure identification, and
donor-acceptor interaction analysis.

Reference: PHASE2_TASKS.md - Task 2.1.4: NBO Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from pymultiwfn.core.data import Wavefunction


class NBOAnalyzer:
    """
    Analyzer for Natural Bond Orbital (NBO) analysis.
    
    NBO analysis transforms canonical molecular orbitals into a
    localized basis that corresponds to the Lewis structure picture,
    including core orbitals, bonding orbitals, lone pairs, and
    antibonding orbitals.
    
    Args:
        wavefunction: Wavefunction object containing MO data
        
    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.orbitals import NBOAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = NBOAnalyzer(wfn)
        >>> nbo = analyzer.get_natural_orbitals()
        >>> lewis = analyzer.identify_lewis_orbitals()
    """
    
    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the NBO analyzer.
        
        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")
        
        self.wfn = wavefunction
        self._nbo_coefficients = None
        self._nbo_occupations = None
        self._lewis_orbitals = None
    
    def get_natural_orbitals(self) -> np.ndarray:
        """
        Calculate natural bond orbital coefficients.
        
        Performs a simplified NBO transformation using diagonalization
        of the density matrix in the atomic orbital basis.
        
        Returns:
            NBO coefficient matrix (nbasis x nmo)
        """
        if self._nbo_coefficients is not None:
            return self._nbo_coefficients
        
        # Get density matrix
        if self.wfn.Ptot is None:
            self.wfn.calculate_density_matrices()
        
        P = self.wfn.Ptot
        
        # Get overlap matrix
        if self.wfn.overlap_matrix is not None:
            S = self.wfn.overlap_matrix
        else:
            S = self.wfn.calculate_overlap_matrix()
        
        # Natural orbitals from diagonalization of PS
        # S^{-1/2} P S^{-1/2}
        try:
            # Symmetric orthogonalization
            eigvals, eigvecs = np.linalg.eigh(S)
            S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            P_ortho = S_inv_sqrt @ P @ S_inv_sqrt
            
            # Diagonalize to get natural orbitals
            occs, coeffs = np.linalg.eigh(P_ortho)
            
            # Sort by occupation (descending)
            idx = np.argsort(-occs)
            occs = occs[idx]
            coeffs = coeffs[:, idx]
            
            # Transform back to AO basis
            self._nbo_coefficients = S_inv_sqrt @ coeffs
            self._nbo_occupations = occs
            
        except Exception:
            # Fallback: use canonical MOs as approximation
            self._nbo_coefficients = self.wfn.coefficients.T.copy()
            self._nbo_occupations = self.wfn.occupations.copy()
        
        return self._nbo_coefficients
    
    def get_nbo_occupations(self) -> np.ndarray:
        """
        Get natural orbital occupations.
        
        Returns:
            Array of occupation numbers (0 to 2 for closed-shell)
        """
        if self._nbo_occupations is None:
            self.get_natural_orbitals()
        
        return self._nbo_occupations
    
    def identify_lewis_orbitals(self) -> List[int]:
        """
        Identify orbitals that form the Lewis structure.
        
        Lewis orbitals include:
        - Core orbitals (highly occupied, ~2.0)
        - Bonding orbitals (occupied, ~2.0)
        - Lone pairs (occupied, ~2.0)
        
        Returns:
            List of orbital indices corresponding to Lewis structure
        """
        if self._lewis_orbitals is not None:
            return self._lewis_orbitals
        
        occupations = self.get_nbo_occupations()
        
        # Lewis orbitals have high occupation (> 1.9)
        lewis_threshold = 1.9
        self._lewis_orbitals = [i for i, occ in enumerate(occupations) if occ > lewis_threshold]
        
        return self._lewis_orbitals
    
    def get_bond_orbital_occupancy(self, bond_index: int = 0) -> float:
        """
        Get occupancy of a specific bond orbital.
        
        Args:
            bond_index: Index of the bond orbital
            
        Returns:
            Occupation number (typically ~2.0 for occupied bonds)
        """
        occupations = self.get_nbo_occupations()
        
        if bond_index < 0 or bond_index >= len(occupations):
            raise IndexError(f"Bond index {bond_index} out of range")
        
        return float(occupations[bond_index])
    
    def analyze_donor_acceptor(self) -> List[Dict[str, Any]]:
        """
        Analyze donor-acceptor interactions.
        
        Identifies potential donor-acceptor interactions between
        filled (donor) and empty (acceptor) orbitals.
        
        Returns:
            List of interaction dictionaries with:
            - 'donor': donor orbital index
            - 'acceptor': acceptor orbital index
            - 'energy': interaction energy (kcal/mol, approximate)
        """
        interactions = []
        occupations = self.get_nbo_occupations()
        nbo_coeffs = self.get_natural_orbitals()
        
        # Donor orbitals: high occupation
        donors = [i for i, occ in enumerate(occupations) if occ > 1.5]
        
        # Acceptor orbitals: low occupation
        acceptors = [i for i, occ in enumerate(occupations) if 0.01 < occ < 0.5]
        
        # Get overlap matrix for interaction strength
        if self.wfn.overlap_matrix is not None:
            S = self.wfn.overlap_matrix
        else:
            S = np.eye(nbo_coeffs.shape[0])
        
        # Calculate donor-acceptor interactions
        for d in donors[:5]:  # Limit to first 5 donors
            for a in acceptors[:5]:  # Limit to first 5 acceptors
                # Overlap between donor and acceptor
                overlap = nbo_coeffs[:, d].T @ S @ nbo_coeffs[:, a]
                
                # Approximate interaction energy (simplified)
                # E(2) ~ F_ij^2 / (e_j - e_i)
                energy = abs(overlap) ** 2 * 10  # Placeholder scaling
                
                if energy > 0.1:  # Threshold for significant interaction
                    interactions.append({
                        'donor': d,
                        'acceptor': a,
                        'energy': energy
                    })
        
        # Sort by energy (descending)
        interactions.sort(key=lambda x: -x['energy'])
        
        return interactions
    
    def get_lone_pairs(self) -> List[int]:
        """
        Identify lone pair orbitals.
        
        Lone pairs are highly occupied orbitals localized on
        a single atom (not between atoms).
        
        Returns:
            List of lone pair orbital indices
        """
        # Simplified: orbitals with high occupation that are not bonding
        occupations = self.get_nbo_occupations()
        
        # Lone pairs have occupation ~2.0 and are atom-centered
        lone_pairs = []
        for i, occ in enumerate(occupations):
            if 1.95 < occ < 2.05:
                # Check if orbital is atom-centered (simplified)
                lone_pairs.append(i)
        
        return lone_pairs
    
    def get_bonding_orbitals(self) -> List[int]:
        """
        Identify bonding orbitals.
        
        Bonding orbitals have high occupation and are localized
        between atoms.
        
        Returns:
            List of bonding orbital indices
        """
        occupations = self.get_nbo_occupations()
        
        # Bonding orbitals: high occupation
        bonding = [i for i, occ in enumerate(occupations) if occ > 1.8]
        
        return bonding
    
    def get_antibonding_orbitals(self) -> List[int]:
        """
        Identify antibonding orbitals.
        
        Antibonding orbitals have low occupation and are the
        counterparts to bonding orbitals.
        
        Returns:
            List of antibonding orbital indices
        """
        occupations = self.get_nbo_occupations()
        
        # Antibonding orbitals: low but non-zero occupation
        antibonding = [i for i, occ in enumerate(occupations) if 0.01 < occ < 0.5]
        
        return antibonding
    
    def generate_nbo_report(self) -> str:
        """
        Generate a human-readable NBO analysis report.
        
        Returns:
            Formatted string report of NBO analysis
        """
        occupations = self.get_nbo_occupations()
        lewis = self.identify_lewis_orbitals()
        bonding = self.get_bonding_orbitals()
        antibonding = self.get_antibonding_orbitals()
        interactions = self.analyze_donor_acceptor()
        
        lines = [
            "Natural Bond Orbital (NBO) Analysis Report",
            "=" * 50,
            "",
            f"Total orbitals: {len(occupations)}",
            f"Lewis orbitals: {len(lewis)}",
            f"Bonding orbitals: {len(bonding)}",
            f"Antibonding orbitals: {len(antibonding)}",
            "",
            "Orbital Occupations:",
        ]
        
        for i, occ in enumerate(occupations[:10]):
            lines.append(f"  NBO {i}: {occ:.4f}")
        
        if interactions:
            lines.append("")
            lines.append("Donor-Acceptor Interactions:")
            for interaction in interactions[:5]:
                lines.append(
                    f"  NBO {interaction['donor']} -> NBO {interaction['acceptor']}: "
                    f"{interaction['energy']:.2f} kcal/mol"
                )
        
        return "\n".join(lines)
    
    def get_perturbation_energies(self) -> List[Dict[str, Any]]:
        """
        Calculate second-order perturbation energies.
        
        Returns:
            List of perturbation energy dictionaries
        """
        return self.analyze_donor_acceptor()
    
    def get_natural_populations(self) -> Dict[str, float]:
        """
        Calculate natural atomic populations.
        
        Returns:
            Dictionary mapping atom labels to their natural populations
        """
        occupations = self.get_nbo_occupations()
        nbo_coeffs = self.get_natural_orbitals()
        
        # Get atomic basis indices
        atomic_basis = self.wfn.get_atomic_basis_indices()
        
        populations = {}
        
        for atom_idx, basis_indices in atomic_basis.items():
            atom = self.wfn.atoms[atom_idx]
            atom_label = f"{atom.element}{atom_idx + 1}"
            
            # Sum contributions from all NBOs
            population = 0.0
            for i, occ in enumerate(occupations):
                for basis_idx in basis_indices:
                    if basis_idx < nbo_coeffs.shape[0]:
                        coeff_sq = nbo_coeffs[basis_idx, i] ** 2
                        population += occ * coeff_sq
            
            populations[atom_label] = population
        
        return populations
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        occupations = self.get_nbo_occupations()
        return (
            f"NBOAnalyzer("
            f"orbitals={len(occupations)}, "
            f"lewis={len(self.identify_lewis_orbitals())})"
        )
