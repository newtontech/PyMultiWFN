"""Main bonding analysis class (Issue 20-22).

This module provides the main Bonding class for advanced bond analysis
including fuzzy bond order, intrinsic bond order, and delocalization indices.
"""

import numpy as np
from pathlib import Path
from typing import Union, List

from ..io import load
from .fuzzy import FuzzyAtom, fuzzy_bond_order, calculate_fuzzy_bond_order_matrix


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
        >>> fbo = bond.get_fuzzy_bond_order(atom_i=1, atom_j=2)
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
        self.natoms = self.wfn.num_atoms
        self.fuzzy_factor = 1.0
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
            coords = atom.coord * 0.529177
                
            fa = FuzzyAtom(
                atom_index=i,
                element=atom.element,
                coordinates=coords,
                vdwa_radius=self._get_vdw_radius(atom.element),
                fuzzy_factor=self.fuzzy_factor,
            )
            fuzzy_atoms.append(fa)
        return fuzzy_atoms
    
    def _get_vdw_radius(self, element: str) -> float:
        """Get van der Waals radius for an element."""
        from .fuzzy import VDW_RADII
        return VDW_RADII.get(element, 1.70)  # Default to carbon radius
    
    def get_fuzzy_bond_order(self, atom_i: int, atom_j: int) -> float:
        """Calculate fuzzy bond order between two atoms.

        The public ``Bonding`` API follows Multiwfn-style 1-based atom indices.
        
        Args:
            atom_i: Index of first atom (1-based)
            atom_j: Index of second atom (1-based)
            
        Returns:
            Fuzzy bond order value
            
        Raises:
            ValueError: If atom indices are invalid
        """
        if not (1 <= atom_i <= self.natoms and 1 <= atom_j <= self.natoms):
            raise ValueError(f"Atom indices must be in range [1, {self.natoms}]")
        if atom_i == atom_j:
            raise ValueError("Atom indices must be different")

        density_matrix, overlap_matrix = self._get_density_and_overlap_matrices()
        return fuzzy_bond_order(
            density_matrix,
            overlap_matrix,
            atom_i,
            atom_j,
            fuzzy_factor=self.fuzzy_factor,
            atomic_basis_indices=self.wfn.get_atomic_basis_indices(),
        )
    
    def get_fuzzy_bond_order_matrix(self) -> np.ndarray:
        """Calculate fuzzy bond order matrix for all atom pairs.
        
        Returns:
            NxN matrix of fuzzy bond orders
        """
        if self._fuzzy_matrix is None:
            density_matrix, overlap_matrix = self._get_density_and_overlap_matrices()
            self._fuzzy_matrix = calculate_fuzzy_bond_order_matrix(
                density_matrix,
                overlap_matrix,
                fuzzy_factor=self.fuzzy_factor,
                atomic_basis_indices=self.wfn.get_atomic_basis_indices(),
            )
        return self._fuzzy_matrix

    def _get_density_and_overlap_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return available density and overlap matrices or raise a clear error."""
        if self.wfn.Ptot is None:
            self.wfn.calculate_density_matrices()
        if self.wfn.overlap_matrix is None:
            self.wfn.calculate_overlap_matrix()
        if self.wfn.Ptot is None or self.wfn.overlap_matrix is None:
            raise ValueError(
                "Wavefunction must provide density and overlap matrices"
            )
        return self.wfn.Ptot, self.wfn.overlap_matrix
