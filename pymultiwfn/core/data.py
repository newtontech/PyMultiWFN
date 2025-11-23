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
    num_atomic_orbitals: int = 0 # Alias for num_basis, sometimes called AOs
    num_primitives: int = 0
    num_shells: int = 0
    
    # Electronic structure
    # Coefficients matrix: (nmo, nbasis)
    # Note: Multiwfn stores as (nmo, nbasis), but typical Python libs might use (nbasis, nmo).
    # We stick to Multiwfn convention: rows are orbitals.
    coefficients: Optional[np.ndarray] = None 
    energies: Optional[np.ndarray] = None # Orbital energies
    occupations: Optional[np.ndarray] = None # Orbital occupations
    
    # Overlap matrix
    overlap_matrix: Optional[np.ndarray] = None
    
    # Unrestricted / Open Shell
    is_unrestricted: bool = False
    coefficients_beta: Optional[np.ndarray] = None
    energies_beta: Optional[np.ndarray] = None
    occupations_beta: Optional[np.ndarray] = None
    
    # New attributes for density matrices
    Palpha: Optional[np.ndarray] = None
    Pbeta: Optional[np.ndarray] = None
    Ptot: Optional[np.ndarray] = None
    overlap_matrix: Optional[np.ndarray] = None # Overlap matrix (S_uv)
    
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

    def _infer_occupations(self):
        """Infers orbital occupations based on num_electrons, multiplicity and orbital energies."""
        if self.occupations is not None and self.occupations_beta is not None:
            return # Already set

        # Determine alpha and beta electron counts based on multiplicity
        # num_electrons is total electrons
        alpha_electrons = (self.num_electrons + self.multiplicity - 1) / 2
        beta_electrons = (self.num_electrons - self.multiplicity + 1) / 2

        # For alpha orbitals
        if self.coefficients is not None and self.energies is not None:
            nmo_alpha = self.coefficients.shape[0]
            self.occupations = np.zeros(nmo_alpha)
            
            # Sort by energy to determine occupation
            # For restricted: total electrons / 2 are doubly occupied
            # For unrestricted: alpha_electrons are singly occupied in alpha MOs
            
            if self.is_unrestricted:
                # Sort by energy and fill
                sorted_indices = np.argsort(self.energies)
                occupied_alpha_indices = sorted_indices[:int(alpha_electrons)]
                self.occupations[occupied_alpha_indices] = 1.0 # Occupation of 1 for alpha
            else: # Restricted
                sorted_indices = np.argsort(self.energies)
                occupied_alpha_indices = sorted_indices[:int(self.num_electrons / 2)] # For restricted, alpha and beta share MOs
                self.occupations[occupied_alpha_indices] = 2.0 # Occupation of 2 for restricted

        # For beta orbitals (only if unrestricted)
        if self.is_unrestricted and self.coefficients_beta is not None and self.energies_beta is not None:
            nmo_beta = self.coefficients_beta.shape[0]
            self.occupations_beta = np.zeros(nmo_beta)
            sorted_indices = np.argsort(self.energies_beta)
            occupied_beta_indices = sorted_indices[:int(beta_electrons)]
            self.occupations_beta[occupied_beta_indices] = 1.0 # Occupation of 1 for beta
            
    def calculate_density_matrices(self):
        """
        Calculates the alpha, beta, and total density matrices
        from MO coefficients and occupations.
        """
        self._infer_occupations()

        nbasis = self.num_basis

        # Alpha density matrix
        if self.coefficients is not None and self.occupations is not None:
            # P_uv = sum_i occ_i * C_ui * C_vi
            # Here, C is (nmo, nbasis), so C_ui is coefficients[i, u]
            # Sum over i (MOs)
            self.Palpha = np.einsum(
                'io,jo->ij',
                (self.coefficients * self.occupations[:, np.newaxis]), # (nmo, nbasis) * (nmo, 1)
                self.coefficients
            )
        else:
            self.Palpha = np.zeros((nbasis, nbasis))

        # Beta density matrix
        if self.is_unrestricted and self.coefficients_beta is not None and self.occupations_beta is not None:
            self.Pbeta = np.einsum(
                'io,jo->ij',
                (self.coefficients_beta * self.occupations_beta[:, np.newaxis]),
                self.coefficients_beta
            )
        else:
            self.Pbeta = np.zeros((nbasis, nbasis))

        self.Ptot = self.Palpha + self.Pbeta
        
    def get_atomic_basis_indices(self) -> Dict[int, List[int]]:
        """
        Returns a dictionary mapping atom index (0-based) to a list of its basis function indices.
        
        Returns:
            Dict[int, List[int]]: {atom_idx: [bf_idx1, bf_idx2, ...]}
        """
        atom_to_bfs = {i: [] for i in range(self.num_atoms)}
        
        bf_idx_counter = 0
        for shell in self.shells:
            # Determine number of basis functions in this shell
            # S=1, P=3, D=5, F=7 (for pure angular momentum)
            # SP shell (type -1) counts as 1 S and 3 P => 4 functions
            l_value = shell.type # 0 for S, 1 for P, 2 for D, ...
            
            num_bfs_in_shell = 0
            if l_value == -1: # SP shell
                num_bfs_in_shell = 4 # 1 s-type + 3 p-type
            elif l_value >= 0: # S, P, D, F, ...
                num_bfs_in_shell = 2 * l_value + 1
            else:
                raise ValueError(f"Unknown shell type: {l_value}")

            # Assign basis function indices to the atom
            for _ in range(num_bfs_in_shell):
                if shell.center_idx < self.num_atoms: # Ensure atom index is valid
                    atom_to_bfs[shell.center_idx].append(bf_idx_counter)
                bf_idx_counter += 1
        
        # Verify that total basis functions match
        total_bfs_assigned = sum(len(bfs) for bfs in atom_to_bfs.values())
        if total_bfs_assigned != self.num_basis:
            # This can happen if FCHK has basis functions not assigned to an atom, or issue in parsing.
            # For now, just a warning.
            print(f"Warning: Mismatch in total basis functions assigned ({total_bfs_assigned}) "
                  f"vs expected ({self.num_basis}). This may indicate a parsing issue or "
                  f"basis functions not associated with a specific atom (e.g., ghost atoms).")

        return atom_to_bfs
