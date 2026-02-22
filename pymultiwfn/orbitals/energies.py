"""
Orbital energy analysis module.

This module provides functionality for analyzing molecular orbital energies,
including HOMO-LUMO gap calculation, Fermi level determination, and energy
diagram generation.

Reference: PHASE2_TASKS.md - Task 2.1.1: MO Energy Analysis
"""

import numpy as np
from typing import Dict, Optional, Any

from pymultiwfn.core.data import Wavefunction


class OrbitalsAnalyzer:
    """
    Analyzer for molecular orbital energies and properties.
    
    This class provides methods to extract and analyze orbital energies
    from wavefunction data, including HOMO-LUMO gap calculation and
    Fermi level determination.
    
    Args:
        wavefunction: Wavefunction object containing MO data
        
    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.orbitals import OrbitalsAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = OrbitalsAnalyzer(wfn)
        >>> print(f"HOMO energy: {analyzer.homo_energy:.4f} a.u.")
        >>> print(f"HOMO-LUMO gap: {analyzer.gap:.4f} a.u.")
    """
    
    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the orbital analyzer.
        
        Args:
            wavefunction: Wavefunction object with MO energies and occupations
            
        Raises:
            ValueError: If wavefunction doesn't contain MO energies
        """
        if wavefunction.energies is None:
            raise ValueError("Wavefunction must contain MO energies")
        
        self.wfn = wavefunction
        self._homo_index = None
        self._fermi_level = None
    
    @property
    def alpha_energies(self) -> np.ndarray:
        """Get alpha orbital energies (Hartree)."""
        return self.wfn.energies
    
    @property
    def beta_energies(self) -> Optional[np.ndarray]:
        """
        Get beta orbital energies (Hartree).
        
        Returns None for restricted calculations where beta energies
        are identical to alpha energies.
        """
        return self.wfn.energies_beta
    
    @property
    def homo_index(self) -> int:
        """
        Get the index of the HOMO (Highest Occupied Molecular Orbital).
        
        The HOMO is identified as the highest energy orbital with
        significant occupation (> 0.5 for restricted calculations).
        
        Returns:
            Zero-based index of the HOMO
        """
        if self._homo_index is not None:
            return self._homo_index
        
        # Find HOMO based on occupations
        # Use threshold of 0.5 to avoid counting nearly-unoccupied orbitals
        occupations = self.wfn.occupations
        threshold = 0.5
        
        # HOMO is the highest significantly occupied orbital
        occupied_indices = np.where(occupations > threshold)[0]
        
        if len(occupied_indices) == 0:
            # Fall back to any orbital with non-zero occupation
            occupied_indices = np.where(occupations > 1e-6)[0]
        
        if len(occupied_indices) == 0:
            # No occupied orbitals (unlikely but handle it)
            self._homo_index = 0
        else:
            # HOMO is the highest occupied orbital
            self._homo_index = int(occupied_indices[-1])
        
        return self._homo_index
    
    @property
    def homo_energy(self) -> float:
        """
        Get the energy of the HOMO (Hartree).
        
        Returns:
            HOMO energy in Hartree
        """
        return float(self.alpha_energies[self.homo_index])
    
    @property
    def lumo_index(self) -> int:
        """
        Get the index of the LUMO (Lowest Unoccupied Molecular Orbital).
        
        The LUMO is the lowest energy orbital with zero occupation,
        immediately following the HOMO.
        
        Returns:
            Zero-based index of the LUMO
        """
        return self.homo_index + 1
    
    @property
    def lumo_energy(self) -> float:
        """
        Get the energy of the LUMO (Hartree).
        
        Returns:
            LUMO energy in Hartree
        """
        return float(self.alpha_energies[self.lumo_index])
    
    @property
    def gap(self) -> float:
        """
        Calculate the HOMO-LUMO energy gap.
        
        The gap is calculated as LUMO_energy - HOMO_energy.
        
        Returns:
            HOMO-LUMO gap in Hartree
        """
        return self.lumo_energy - self.homo_energy
    
    @property
    def fermi_level(self) -> float:
        """
        Calculate the Fermi level.
        
        The Fermi level is approximated as the midpoint between
        HOMO and LUMO energies: (HOMO + LUMO) / 2
        
        Returns:
            Fermi level in Hartree
        """
        if self._fermi_level is not None:
            return self._fermi_level
        
        # Fermi level is midpoint between HOMO and LUMO
        self._fermi_level = (self.homo_energy + self.lumo_energy) / 2.0
        
        return self._fermi_level
    
    def get_energy_diagram(self, n_orbitals: int = 10) -> Dict[str, Any]:
        """
        Generate data for orbital energy diagram.
        
        Returns orbital energies and occupations centered around
        the Fermi level, suitable for visualization.
        
        Args:
            n_orbitals: Number of orbitals to include (default: 10)
            
        Returns:
            Dictionary containing:
                - 'energies': array of orbital energies (Hartree)
                - 'occupations': array of orbital occupations
                - 'indices': array of orbital indices
                
        Example:
            >>> data = analyzer.get_energy_diagram(n_orbitals=5)
            >>> print(data['energies'])
            >>> print(data['occupations'])
        """
        # Select orbitals around Fermi level
        center = self.homo_index
        half = n_orbitals // 2
        
        start_idx = max(0, center - half)
        end_idx = min(len(self.alpha_energies), start_idx + n_orbitals)
        
        # Adjust if we're at the boundaries
        if end_idx - start_idx < n_orbitals:
            start_idx = max(0, end_idx - n_orbitals)
        
        # Extract energies and occupations for selected orbitals
        energies = self.alpha_energies[start_idx:end_idx]
        occupations = self.wfn.occupations[start_idx:end_idx]
        indices = np.arange(start_idx, end_idx)
        
        return {
            'energies': energies,
            'occupations': occupations,
            'indices': indices,
            'homo_index': self.homo_index,
            'lumo_index': self.lumo_index,
            'fermi_level': self.fermi_level
        }
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"OrbitalsAnalyzer("
            f"HOMO={self.homo_index} at {self.homo_energy:.4f} Ha, "
            f"LUMO={self.lumo_index} at {self.lumo_energy:.4f} Ha, "
            f"gap={self.gap:.4f} Ha)"
        )
