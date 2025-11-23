"""
Core data structures for PyMultiWFN.
Defines Atom, BasisSet, and Wavefunction classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class Atom:
    """Represents an atom in the system."""
    element: str
    index: int  # Atomic number (Z)
    x: float    # Bohr
    y: float    # Bohr
    z: float    # Bohr
    charge: float # Nuclear charge (can be different from Z if ECP is used)
    
    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

@dataclass
class Shell:
    """Represents a shell of basis functions (e.g., S, P, SP, D)."""
    type: int # 0=S, 1=P, -1=SP, 2=D, etc.
    center_idx: int # Index of the atom this shell belongs to
    exponents: np.ndarray # Exponents of primitives
    coefficients: np.ndarray # Contraction coefficients
    # For SP shells, coefficients will be (2, N) array, row 0 for S, row 1 for P

@dataclass
class Wavefunction:
    """
    The central data object representing the electronic wavefunction.
    """
    # System info
    atoms: List[Atom] = field(default_factory=list)
    num_electrons: float = 0.0
    charge: int = 0
    multiplicity: int = 1
    
    # Basis set info
    shells: List[Shell] = field(default_factory=list)
    num_basis: int = 0 # Total number of basis functions
    num_primitives: int = 0
    
    # Electronic structure
    # Coefficients matrix: (nmo, nbasis)
    # Note: Multiwfn stores as (nmo, nbasis), but typical Python libs might use (nbasis, nmo).
    # We stick to Multiwfn convention: rows are orbitals.
    coefficients: Optional[np.ndarray] = None 
    energies: Optional[np.ndarray] = None # Orbital energies
    occupations: Optional[np.ndarray] = None # Orbital occupations
    
    # Unrestricted / Open Shell
    is_unrestricted: bool = False
    coefficients_beta: Optional[np.ndarray] = None
    energies_beta: Optional[np.ndarray] = None
    occupations_beta: Optional[np.ndarray] = None
    
    # Metadata
    title: str = ""
    method: str = ""
    basis_set_name: str = ""

    def add_atom(self, element: str, z: int, x: float, y: float, z_coord: float, charge: float = None):
        if charge is None:
            charge = float(z)
        self.atoms.append(Atom(element, z, x, y, z_coord, charge))

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)
