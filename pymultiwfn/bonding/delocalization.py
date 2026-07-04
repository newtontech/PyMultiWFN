"""Delocalization Index Implementation (Issue 4).

This module implements electron pair delocalization index analysis for
investigating electron sharing and aromaticity. The delocalization index (DI)
measures the number of electron pairs shared between atoms, providing
insights into bond character and aromatic systems.

References:
- Bader, R.F.W. & Stephens, M.E. (1975). J. Am. Chem. Soc. 97, 7391-7399.
- Angyan, J.G. et al. (2006). J. Chem. Sci. 118, 159-169.
- Matito, E. et al. (2005). J. Phys. Chem. A 109, 5600-5609.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DelocalizationResult:
    """Container for delocalization index results.

    Attributes:
        atom_i: Index of first atom (0-based)
        atom_j: Index of second atom (0-based)
        di_value: Delocalization index value in electrons
        bond_type: Classification of bond type (single, double, aromatic, etc.)
    """

    atom_i: int
    atom_j: int
    di_value: float
    bond_type: str = "unknown"

    def __repr__(self) -> str:
        return f"DI({self.atom_i},{self.atom_j}) = {self.di_value:.4f} e"


def delocalization_index(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    atom_i_indices: List[int],
    atom_j_indices: List[int],
) -> float:
    """Calculate the 2-center delocalization index between two atoms.

    The delocalization index δ(A,B) measures the number of electron pairs
    shared between atoms A and B. It is calculated using:

    δ(A,B) = 2 * Σ_μ∈A Σ_ν∈B |P_μν * S_μν|

    where P is the density matrix and S is the overlap matrix.

    Args:
        density_matrix: Total density matrix (AO basis), shape (n_ao, n_ao)
        overlap_matrix: Overlap matrix (AO basis), shape (n_ao, n_ao)
        atom_i_indices: List of basis function indices belonging to atom i
        atom_j_indices: List of basis function indices belonging to atom j

    Returns:
        Delocalization index value in electrons

    Raises:
        ValueError: If matrices have incompatible shapes or indices are empty
    """
    # Validate inputs
    if density_matrix.shape != overlap_matrix.shape:
        raise ValueError(
            f"Density matrix shape {density_matrix.shape} must match "
            f"overlap matrix shape {overlap_matrix.shape}"
        )

    if not atom_i_indices or not atom_j_indices:
        return 0.0

    n_ao = density_matrix.shape[0]

    # Validate indices
    for idx in atom_i_indices + atom_j_indices:
        if idx < 0 or idx >= n_ao:
            raise ValueError(f"Basis function index {idx} out of range [0, {n_ao})")

    # Calculate DI using the formula: δ(A,B) = 2 * Σ_μ∈A Σ_ν∈B |P_μν * S_μν|
    di_value = 0.0
    for mu in atom_i_indices:
        for nu in atom_j_indices:
            # Use absolute value to ensure positive DI
            di_value += abs(density_matrix[mu, nu] * overlap_matrix[mu, nu])

    # Multiply by 2 for electron pairs
    di_value *= 2.0

    return float(di_value)


def three_center_delocalization_index(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    atom_a_indices: List[int],
    atom_b_indices: List[int],
    atom_c_indices: List[int],
) -> float:
    """Calculate the 3-center delocalization index.

    The 3-center DI measures electron sharing among three atoms,
    important for analyzing multi-center bonds and aromaticity.

    δ(A,B,C) = 2 * Σ_μ∈A Σ_ν∈B Σ_λ∈C |P_μνλ * S_μνλ|

    For practical implementation, we use the approximation:
    δ(A,B,C) ≈ √[δ(A,B) * δ(A,C) * δ(B,C)] / 2

    Args:
        density_matrix: Total density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_a_indices: List of basis function indices for atom A
        atom_b_indices: List of basis function indices for atom B
        atom_c_indices: List of basis function indices for atom C

    Returns:
        3-center delocalization index value in electrons
    """
    # Calculate 2-center DIs for all pairs
    di_ab = delocalization_index(
        density_matrix, overlap_matrix, atom_a_indices, atom_b_indices
    )
    di_ac = delocalization_index(
        density_matrix, overlap_matrix, atom_a_indices, atom_c_indices
    )
    di_bc = delocalization_index(
        density_matrix, overlap_matrix, atom_b_indices, atom_c_indices
    )

    # Use geometric mean approximation for 3-center DI
    # δ(A,B,C) ≈ √[δ(A,B) * δ(A,C) * δ(B,C)] / 2
    if di_ab > 0 and di_ac > 0 and di_bc > 0:
        di_3c = np.sqrt(di_ab * di_ac * di_bc) / 2.0
    else:
        di_3c = 0.0

    return float(di_3c)


def calculate_di_matrix(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    atom_basis_indices: Dict[int, List[int]],
    natoms: int,
) -> np.ndarray:
    """Calculate the full delocalization index matrix.

    Args:
        density_matrix: Total density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_basis_indices: Dictionary mapping atom index to basis function indices
        natoms: Number of atoms in the system

    Returns:
        Symmetric natoms x natoms matrix of delocalization indices
    """
    di_matrix = np.zeros((natoms, natoms))

    for i in range(natoms):
        for j in range(i + 1, natoms):
            indices_i = atom_basis_indices.get(i, [])
            indices_j = atom_basis_indices.get(j, [])

            if indices_i and indices_j:
                di_val = delocalization_index(
                    density_matrix, overlap_matrix, indices_i, indices_j
                )
                di_matrix[i, j] = di_val
                di_matrix[j, i] = di_val

    return di_matrix


def classify_bond_from_di(di_value: float) -> str:
    """Classify bond type based on delocalization index value.

    Args:
        di_value: Delocalization index in electrons

    Returns:
        Bond type string: 'single', 'double', 'triple', 'aromatic', or 'weak'
    """
    if di_value < 0.3:
        return "weak"
    elif di_value < 0.8:
        return "single"
    elif di_value < 1.3:
        return "aromatic"
    elif di_value < 1.8:
        return "double"
    elif di_value < 2.5:
        return "triple"
    else:
        return "very_strong"


def calculate_aromaticity_index(
    di_matrix: np.ndarray,
    ring_atoms: List[int],
) -> float:
    """Calculate aromaticity index from DI matrix for a ring system.

    The aromaticity index based on DI (also related to PDI - Para
    Delocalization Index) measures electron delocalization in rings.

    For a ring with n atoms, we compute:
    AI = (1/n) * Σ_<i,j> δ(i,j) / δ_ref

    where δ_ref is the reference DI for a perfect aromatic bond (~1.4 e).

    Args:
        di_matrix: Full delocalization index matrix
        ring_atoms: List of atom indices forming the ring (0-based)

    Returns:
        Aromaticity index (1.0 = perfect aromatic, <1.0 = less aromatic)
    """
    if len(ring_atoms) < 3:
        return 0.0

    # Reference DI for perfect aromatic bond (e.g., benzene)
    di_ref = 1.4  # electrons

    # Calculate average DI for adjacent atoms in ring
    n_atoms = len(ring_atoms)
    total_di = 0.0

    for i in range(n_atoms):
        # Adjacent atoms in ring (with periodic boundary)
        j = (i + 1) % n_atoms
        atom_i = ring_atoms[i]
        atom_j = ring_atoms[j]
        total_di += di_matrix[atom_i, atom_j]

    # Average DI per bond
    avg_di = total_di / n_atoms

    # Normalize to aromaticity index
    aromaticity_index = avg_di / di_ref

    return float(aromaticity_index)


def calculate_pdi(
    di_matrix: np.ndarray,
    ring_atoms: List[int],
) -> float:
    """Calculate Para-Delocalization Index (PDI) for a six-membered ring.

    PDI measures electron delocalization between para positions in
    six-membered rings, useful for aromaticity analysis.

    PDI = (δ(1,4) + δ(2,5) + δ(3,6)) / 3

    Reference: Matito, E. et al. (2005). J. Phys. Chem. A 109, 5600.

    Args:
        di_matrix: Full delocalization index matrix
        ring_atoms: List of 6 atom indices forming the ring (0-based)

    Returns:
        PDI value in electrons

    Raises:
        ValueError: If ring doesn't have exactly 6 atoms
    """
    if len(ring_atoms) != 6:
        raise ValueError(f"PDI requires exactly 6 ring atoms, got {len(ring_atoms)}")

    # Para positions: (1,4), (2,5), (3,6) - 0-indexed: (0,3), (1,4), (2,5)
    pdi = (
        di_matrix[ring_atoms[0], ring_atoms[3]]
        + di_matrix[ring_atoms[1], ring_atoms[4]]
        + di_matrix[ring_atoms[2], ring_atoms[5]]
    ) / 3.0

    return float(pdi)


def calculate_flu(
    di_matrix: np.ndarray,
    ring_atoms: List[int],
    natoms_total: int = None,
) -> float:
    """Calculate Aromatic Fluctuation Index (FLU) for a ring.

    FLU measures the fluctuation of electron delocalization relative
    to a reference aromatic system. Lower FLU indicates more aromatic.

    FLU = (1/n) * Σ_i [(δ_ref - δ_i) / δ_ref]^2

    where δ_i is the sum of DIs for atom i with its neighbors in the ring.

    Reference: Matito, E. et al. (2006). Chem. Phys. Lett. 420, 287.

    Args:
        di_matrix: Full delocalization index matrix
        ring_atoms: List of atom indices forming the ring (0-based)
        natoms_total: Total number of atoms (unused, kept for API compatibility)

    Returns:
        FLU value (0 = perfectly aromatic, higher = less aromatic)
    """
    if len(ring_atoms) < 3:
        return 1.0

    n_ring = len(ring_atoms)

    # Reference DI sum for aromatic atom (benzene reference ~2.8 e total)
    # Each C in benzene has DI ~1.4 with each of 2 neighbors
    di_ref_sum = 2.8

    flu = 0.0
    for i in range(n_ring):
        # Get DI with two neighbors in ring
        j_prev = ring_atoms[(i - 1) % n_ring]
        j_next = ring_atoms[(i + 1) % n_ring]
        atom_i = ring_atoms[i]

        di_sum = di_matrix[atom_i, j_prev] + di_matrix[atom_i, j_next]

        # FLU contribution
        if di_ref_sum > 0:
            flu += ((di_ref_sum - di_sum) / di_ref_sum) ** 2

    flu /= n_ring

    return float(flu)


class DelocalizationIndex:
    """Main class for delocalization index analysis.

    Provides methods for calculating 2-center and 3-center delocalization
    indices, aromaticity indices, and bond classification.

    Attributes:
        wfn: Wavefunction object containing molecular data
        natoms: Number of atoms
        density_matrix: Total density matrix (AO basis)
        overlap_matrix: Overlap matrix (AO basis)
        atom_basis_indices: Mapping from atom index to basis function indices

    Example:
        >>> from pymultiwfn.bonding import DelocalizationIndex
        >>> di = DelocalizationIndex(wfn)
        >>> di_val = di.get_delocalization_index(atom_i=0, atom_j=1)
        >>> print(f"DI: {di_val:.3f} e")
    """

    def __init__(self, wfn: Union[str, Path, object]):
        """Initialize DelocalizationIndex analysis.

        Args:
            wfn: Path to wavefunction file or Wavefunction object

        Raises:
            FileNotFoundError: If wavefunction file doesn't exist
            ValueError: If wavefunction is invalid or missing required data
        """
        # Import here to avoid circular imports
        from ..io import load

        if isinstance(wfn, (str, Path)):
            self.wfn = load(wfn)
        else:
            self.wfn = wfn

        self.natoms = self.wfn.num_atoms
        self.atoms = self.wfn.atoms

        # Get density and overlap matrices
        self._validate_wavefunction()
        self.density_matrix = self._get_density_matrix()
        self.overlap_matrix = self._get_overlap_matrix()
        self.atom_basis_indices = self._get_atom_basis_indices()

        # Cached DI matrix
        self._di_matrix = None

    def _validate_wavefunction(self) -> None:
        """Validate that wavefunction has required data."""
        if self.wfn is None:
            raise ValueError("Wavefunction object is None")

        if self.natoms == 0:
            raise ValueError("Wavefunction has no atoms")

    def _get_density_matrix(self) -> np.ndarray:
        """Get or calculate the total density matrix."""
        if self.wfn.Ptot is not None:
            return self.wfn.Ptot

        # Calculate if not present
        if hasattr(self.wfn, "calculate_density_matrices"):
            self.wfn.calculate_density_matrices()
            if self.wfn.Ptot is not None:
                return self.wfn.Ptot

        # Fallback: construct from coefficients and occupations
        if self.wfn.coefficients is not None and self.wfn.occupations is not None:
            # P = C * occ * C^T (for MOs as rows)
            C = self.wfn.coefficients
            occ = self.wfn.occupations
            P = np.einsum("oi,oj->ij", C * occ[:, np.newaxis], C)
            return P

        raise ValueError("Cannot construct density matrix from wavefunction")

    def _get_overlap_matrix(self) -> np.ndarray:
        """Get or calculate the overlap matrix."""
        if self.wfn.overlap_matrix is not None:
            return self.wfn.overlap_matrix

        # Calculate if method exists
        if hasattr(self.wfn, "calculate_overlap_matrix"):
            self.wfn.calculate_overlap_matrix()
            if self.wfn.overlap_matrix is not None:
                return self.wfn.overlap_matrix

        # Fallback: use identity matrix (crude approximation)
        n_basis = self.wfn.num_basis
        return np.eye(n_basis)

    def _get_atom_basis_indices(self) -> Dict[int, List[int]]:
        """Get mapping from atom index to basis function indices."""
        if hasattr(self.wfn, "get_atomic_basis_indices"):
            return self.wfn.get_atomic_basis_indices()

        # Fallback: assign basis functions equally (approximation)
        n_basis = self.wfn.num_basis
        indices = {}
        basis_per_atom = max(1, n_basis // self.natoms)

        for i in range(self.natoms):
            start = i * basis_per_atom
            end = min((i + 1) * basis_per_atom, n_basis)
            indices[i] = list(range(start, end))

        # Handle remaining basis functions
        remaining = n_basis - self.natoms * basis_per_atom
        if remaining > 0:
            for i in range(remaining):
                atom_idx = i % self.natoms
                bf_idx = self.natoms * basis_per_atom + i
                indices[atom_idx].append(bf_idx)

        return indices

    def get_delocalization_index(self, atom_i: int, atom_j: int) -> float:
        """Calculate 2-center delocalization index between two atoms.

        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)

        Returns:
            Delocalization index value in electrons

        Raises:
            ValueError: If atom indices are invalid
        """
        if not (0 <= atom_i < self.natoms and 0 <= atom_j < self.natoms):
            raise ValueError(
                f"Atom indices must be in range [0, {self.natoms}), "
                f"got {atom_i} and {atom_j}"
            )

        if atom_i == atom_j:
            return 0.0

        indices_i = self.atom_basis_indices.get(atom_i, [])
        indices_j = self.atom_basis_indices.get(atom_j, [])

        return delocalization_index(
            self.density_matrix, self.overlap_matrix, indices_i, indices_j
        )

    def get_three_center_di(self, atom_a: int, atom_b: int, atom_c: int) -> float:
        """Calculate 3-center delocalization index.

        Args:
            atom_a: Index of first atom (0-based)
            atom_b: Index of second atom (0-based)
            atom_c: Index of third atom (0-based)

        Returns:
            3-center delocalization index value in electrons

        Raises:
            ValueError: If atom indices are invalid or not distinct
        """
        if len({atom_a, atom_b, atom_c}) != 3:
            raise ValueError("Three center DI requires three distinct atoms")

        for atom in [atom_a, atom_b, atom_c]:
            if not (0 <= atom < self.natoms):
                raise ValueError(f"Atom index {atom} out of range [0, {self.natoms})")

        indices_a = self.atom_basis_indices.get(atom_a, [])
        indices_b = self.atom_basis_indices.get(atom_b, [])
        indices_c = self.atom_basis_indices.get(atom_c, [])

        return three_center_delocalization_index(
            self.density_matrix,
            self.overlap_matrix,
            indices_a,
            indices_b,
            indices_c,
        )

    def get_di_matrix(self) -> np.ndarray:
        """Calculate full delocalization index matrix.

        Returns:
            Symmetric natoms x natoms matrix of delocalization indices
        """
        if self._di_matrix is None:
            self._di_matrix = calculate_di_matrix(
                self.density_matrix,
                self.overlap_matrix,
                self.atom_basis_indices,
                self.natoms,
            )
        return self._di_matrix

    def get_bond_type(self, atom_i: int, atom_j: int) -> str:
        """Classify bond type based on delocalization index.

        Args:
            atom_i: Index of first atom (0-based)
            atom_j: Index of second atom (0-based)

        Returns:
            Bond type string
        """
        di_val = self.get_delocalization_index(atom_i, atom_j)
        return classify_bond_from_di(di_val)

    def get_aromaticity_index(self, ring_atoms: List[int]) -> float:
        """Calculate aromaticity index for a ring.

        Args:
            ring_atoms: List of atom indices forming the ring (0-based)

        Returns:
            Aromaticity index (1.0 = perfect aromatic)
        """
        di_matrix = self.get_di_matrix()
        return calculate_aromaticity_index(di_matrix, ring_atoms)

    def get_pdi(self, ring_atoms: List[int]) -> float:
        """Calculate Para-Delocalization Index for a six-membered ring.

        Args:
            ring_atoms: List of 6 atom indices (0-based)

        Returns:
            PDI value in electrons

        Raises:
            ValueError: If ring doesn't have exactly 6 atoms
        """
        di_matrix = self.get_di_matrix()
        return calculate_pdi(di_matrix, ring_atoms)

    def get_flu(self, ring_atoms: List[int]) -> float:
        """Calculate Aromatic Fluctuation Index for a ring.

        Args:
            ring_atoms: List of atom indices forming the ring (0-based)

        Returns:
            FLU value (lower = more aromatic)
        """
        di_matrix = self.get_di_matrix()
        return calculate_flu(di_matrix, ring_atoms)

    def is_aromatic_ring(self, ring_atoms: List[int], threshold: float = 0.85) -> bool:
        """Determine if a ring is aromatic based on aromaticity index.

        Args:
            ring_atoms: List of atom indices forming the ring (0-based)
            threshold: Aromaticity threshold (default 0.85)

        Returns:
            True if ring is aromatic, False otherwise
        """
        ai = self.get_aromaticity_index(ring_atoms)
        return ai >= threshold

    def get_all_dihedral_dis(
        self, ring_atoms: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """Get all delocalization indices for atom pairs in a ring.

        Args:
            ring_atoms: List of atom indices forming the ring (0-based)

        Returns:
            Dictionary mapping (atom_i, atom_j) pairs to DI values
        """
        di_matrix = self.get_di_matrix()
        result = {}

        for i, atom_i in enumerate(ring_atoms):
            for j, atom_j in enumerate(ring_atoms):
                if i < j:
                    result[(atom_i, atom_j)] = di_matrix[atom_i, atom_j]

        return result
