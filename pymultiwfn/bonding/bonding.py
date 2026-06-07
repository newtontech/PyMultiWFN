"""Main bonding analysis class (Issue 20-22).

This module provides the main Bonding class for advanced bond analysis
including fuzzy bond order, intrinsic bond order, and delocalization indices.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple

from ..io import load
from .fuzzy import FuzzyAtom, fuzzy_bond_order, calculate_fuzzy_bond_order_matrix
from .intrinsic import intrinsic_bond_order, calculate_intrinsic_bond_order_matrix, IntrinsicBondResult


class Bonding:
    """Main class for bonding analysis.
    
    Provides methods for calculating various bond order metrics including
    fuzzy bond order, intrinsic bond order, and delocalization indices.
    
    Attributes:
        wfn: Wavefunction object containing molecular data
        atoms: List of atoms in the molecule
        natoms: Number of atoms
        fuzzy_atoms: List of FuzzyAtom objects for fuzzy analysis
        
    Example:
        >>> from pymultiwfn import Bonding
        >>> bond = Bonding('molecule.fch')
        >>> fbo = bond.get_fuzzy_bond_order(atom_i=0, atom_j=1)
        >>> print(f"Fuzzy BO: {fbo:.3f}")
    """
    
    def __init__(self, wfn: Union[str, Path, object]):
        """Initialize Bonding analysis.
        
        Args:
            wfn: Path to wavefunction file or Wavefunction object
            
        Raises:
            FileNotFoundError: If wavefunction file doesn't exist
            ValueError: If wavefunction is invalid
        """
        if isinstance(wfn, (str, Path)):
            self.wfn = load(wfn)
        else:
            self.wfn = wfn
            
        self.atoms = self.wfn.atoms
        self.natoms = len(self.atoms)
        self._fuzzy_atoms = None
        self._fuzzy_matrix = None
        
    @property
    def fuzzy_atoms(self) -> List[FuzzyAtom]:
        """Get or create fuzzy atoms for the molecule."""
        if self._fuzzy_atoms is None:
            self._fuzzy_atoms = self._create_fuzzy_atoms()
        return self._fuzzy_atoms
    
    def _create_fuzzy_atoms(self) -> List[FuzzyAtom]:
        """Create fuzzy atom objects from wavefunction data."""
        fuzzy_atoms = []
        for i, atom in enumerate(self.atoms):
            # Convert coordinates from Bohr to Angstrom if needed
            # Atom has x, y, z attributes
            coords = np.array([atom.x, atom.y, atom.z], dtype=np.float64) * 0.529177  # Bohr to Angstrom
                
            fa = FuzzyAtom(
                atom_index=i,
                element=atom.element,
                coordinates=coords,
                vdwa_radius=self._get_vdw_radius(atom.element),
                fuzzy_factor=0.5
            )
            fuzzy_atoms.append(fa)
        return fuzzy_atoms
    
    def _get_vdw_radius(self, element: str) -> float:
        """Get van der Waals radius for an element."""
        from .fuzzy import VDW_RADII
        return VDW_RADII.get(element, 1.70)  # Default to carbon radius
    
    def get_fuzzy_bond_order(self, atom_i: int, atom_j: int) -> float:
        """Calculate fuzzy bond order between two atoms.
        
        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)
            
        Returns:
            Fuzzy bond order value
            
        Raises:
            ValueError: If atom indices are invalid or wavefunction data is missing
        """
        if not (0 <= atom_i < self.natoms and 0 <= atom_j < self.natoms):
            raise ValueError(f"Atom indices must be in range [0, {self.natoms})")
        if atom_i == atom_j:
            raise ValueError("Atom indices must be different")
            
        # Get density and overlap matrices from wavefunction
        if self.wfn.Ptot is None:
            raise ValueError("Density matrix (Ptot) not available in wavefunction")
        if self.wfn.overlap_matrix is None:
            raise ValueError("Overlap matrix not available in wavefunction")
        if self.wfn.shells is None or len(self.wfn.shells) == 0:
            raise ValueError("Shell information not available in wavefunction")
            
        return fuzzy_bond_order(
            density_matrix=self.wfn.Ptot,
            overlap_matrix=self.wfn.overlap_matrix,
            shells=self.wfn.shells,
            atom_i=atom_i,
            atom_j=atom_j,
            fuzzy_factor=0.5
        )
    
    def get_fuzzy_bond_order_matrix(self) -> np.ndarray:
        """Calculate fuzzy bond order matrix for all atom pairs.
        
        Returns:
            NxN matrix of fuzzy bond orders
            
        Raises:
            ValueError: If wavefunction data is missing
        """
        if self._fuzzy_matrix is None:
            # Get density and overlap matrices from wavefunction
            if self.wfn.Ptot is None:
                raise ValueError("Density matrix (Ptot) not available in wavefunction")
            if self.wfn.overlap_matrix is None:
                raise ValueError("Overlap matrix not available in wavefunction")
            if self.wfn.shells is None or len(self.wfn.shells) == 0:
                raise ValueError("Shell information not available in wavefunction")
                
            self._fuzzy_matrix = calculate_fuzzy_bond_order_matrix(
                density_matrix=self.wfn.Ptot,
                overlap_matrix=self.wfn.overlap_matrix,
                shells=self.wfn.shells,
                fuzzy_factor=0.5
            )
        return self._fuzzy_matrix
    
    @property
    def vdwa_radii(self) -> List[float]:
        """Get van der Waals radii for all atoms."""
        return [self._get_vdw_radius(atom.element) for atom in self.atoms]

    @property
    def fuzzy_factor(self) -> float:
        """Get fuzzy partition factor."""
        return 0.5

    def get_intrinsic_bond_order(self, atom_i: int, atom_j: int) -> float:
        """Calculate intrinsic bond order between two atoms.
        
        The intrinsic bond order (IBO) accounts for bond polarity
        and provides a measure of the intrinsic covalent character
        of a bond.
        
        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)
            
        Returns:
            Intrinsic bond order value
            
        Raises:
            ValueError: If atom indices are invalid
        """
        if not (0 <= atom_i < self.natoms and 0 <= atom_j < self.natoms):
            raise ValueError(f"Atom indices must be in range [0, {self.natoms})")
        if atom_i == atom_j:
            raise ValueError("Atom indices must be different")

        return intrinsic_bond_order(self.wfn, atom_i, atom_j)

    def get_intrinsic_bond_order_matrix(self) -> IntrinsicBondResult:
        """Calculate intrinsic bond order matrix for all atom pairs.
        
        Returns:
            IntrinsicBondResult containing:
            - bond_order_matrix: NxN matrix of intrinsic bond orders
            - polarity_matrix: NxN matrix of bond polarity corrections
            - wiberg_matrix: NxN matrix of Wiberg bond orders
        """
        return calculate_intrinsic_bond_order_matrix(self.wfn)