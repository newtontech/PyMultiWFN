"""
Density of States (DOS) analysis module for PyMultiWFN.

This module provides functionality to calculate various types of DOS:
- Total DOS (TDOS)
- Projected DOS (PDOS)
- Overlap Population DOS (OPDOS)
- Local Density of States (LDOS)
- Crystal Orbital Hamilton Population (COHP)

The implementation follows the approach used in Multiwfn, supporting different
broadening functions (Gaussian, Lorentzian, Pseudo-Voigt) and fragment definitions.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
from enum import Enum
from dataclasses import dataclass, field

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import BOHR_TO_ANGSTROM, AU_TO_EV


class BroadeningFunction(Enum):
    """Enumeration of available broadening functions."""
    GAUSSIAN = 1
    LORENTZIAN = 2
    PSEUDO_VOIGT = 3


class CompositionMethod(Enum):
    """Methods for calculating orbital compositions."""
    MULLIKEN = 1
    SCPA = 2
    HIRSHFELD = 3
    BECKE = 4


class FragmentType(Enum):
    """Types of fragment definitions."""
    BASIS_FUNCTION = 1  # Fragment defined by basis functions
    ATOM = 2  # Fragment defined by atoms
    MOLECULAR_ORBITAL = 3  # Fragment defined by MOs


@dataclass
class DOSConfig:
    """Configuration parameters for DOS calculations."""
    # Energy range (in atomic units)
    energy_min: float = -0.8
    energy_max: float = 0.2
    energy_shift: float = 0.0
    energy_step: float = 0.01

    # Broadening parameters
    fwhm: float = 0.05  # Full width at half maximum
    broadening_func: BroadeningFunction = BroadeningFunction.GAUSSIAN
    pseudo_voigt_weight: float = 0.5  # Weight of Gaussian in Pseudo-Voigt

    # Units
    use_ev: bool = False
    unit_str: str = " a.u."

    # Display parameters
    scale_curve: float = 0.1
    show_degneracy: bool = False

    # Composition method
    composition_method: CompositionMethod = CompositionMethod.MULLIKEN


@dataclass
class Fragment:
    """Defines a fragment for DOS calculations."""
    indices: List[int]  # Basis function, atom, or MO indices
    fragment_type: FragmentType
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Fragment_{id(self)}"


@dataclass
class DOSResult:
    """Container for DOS calculation results."""
    # Energy grid
    energies: np.ndarray

    # DOS data
    tdos: np.ndarray  # Total DOS
    pdos: Optional[np.ndarray] = None  # Projected DOS (n_fragments, n_points)
    opdos: Optional[np.ndarray] = None  # Overlap population DOS
    cohp: Optional[np.ndarray] = None  # Crystal Orbital Hamilton Population
    ldos: Optional[np.ndarray] = None  # Local DOS

    # Additional data
    fragment_names: List[str] = field(default_factory=list)
    homo_energy: Optional[float] = None
    lumo_energy: Optional[float] = None

    # Configuration used
    config: Optional[DOSConfig] = None


class DOSAnalyzer:
    """
    Main class for Density of States analysis.

    This class provides methods to calculate various types of DOS from
    wavefunction data, supporting different fragment definitions and
    visualization options.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize DOS analyzer.

        Args:
            wavefunction: Wavefunction object containing MO energies, coefficients, etc.
        """
        self.wf = wavefunction
        self.fragments: List[Fragment] = []
        self.config = DOSConfig()

        # Validate wavefunction has required data
        if self.wf.mo_energies is None:
            raise ValueError("Wavefunction must have MO energies for DOS analysis")
        if self.wf.mo_coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients for DOS analysis")

    def set_config(self, config: DOSConfig) -> None:
        """Set DOS calculation configuration."""
        self.config = config

    def add_fragment(self, fragment: Fragment) -> None:
        """Add a fragment for PDOS calculations."""
        self.fragments.append(fragment)

    def clear_fragments(self) -> None:
        """Clear all defined fragments."""
        self.fragments.clear()

    def _generate_energy_grid(self) -> np.ndarray:
        """Generate energy grid for DOS calculations."""
        if self.config.use_ev:
            e_min = self.config.energy_min * AU_TO_EV + self.config.energy_shift
            e_max = self.config.energy_max * AU_TO_EV + self.config.energy_shift
            step = self.config.energy_step * AU_TO_EV
        else:
            e_min = self.config.energy_min + self.config.energy_shift
            e_max = self.config.energy_max + self.config.energy_shift
            step = self.config.energy_step

        return np.arange(e_min, e_max + step, step)

    def _broadening_function(self, energy_diff: float, fwhm: float) -> float:
        """
        Apply broadening function to energy difference.

        Args:
            energy_diff: Energy difference (E - E_MO)
            fwhm: Full width at half maximum

        Returns:
            Broadened contribution
        """
        if self.config.broadening_func == BroadeningFunction.GAUSSIAN:
            # Gaussian: A * exp(-4*ln(2)*(x/FWHM)^2)
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-0.5 * (energy_diff / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        elif self.config.broadening_func == BroadeningFunction.LORENTZIAN:
            # Lorentzian: (FWHM/2π) / ((x-E0)^2 + (FWHM/2)^2)
            gamma = fwhm / 2
            return (gamma / np.pi) / (energy_diff ** 2 + gamma ** 2)

        elif self.config.broadening_func == BroadeningFunction.PSEUDO_VOIGT:
            # Pseudo-Voigt: weighted sum of Gaussian and Lorentzian
            gauss_contrib = self._broadening_function(energy_diff, fwhm)

            # Temporarily set to Gaussian for calculation
            old_func = self.config.broadening_func
            self.config.broadening_func = BroadeningFunction.LORENTZIAN
            lor_contrib = self._broadening_function(energy_diff, fwhm)
            self.config.broadening_func = old_func

            return (self.config.pseudo_voigt_weight * gauss_contrib +
                    (1 - self.config.pseudo_voigt_weight) * lor_contrib)

        return 0.0

    def _calculate_mo_compositions(self) -> np.ndarray:
        """
        Calculate MO compositions for defined fragments.

        Returns:
            Array of shape (n MOs, n fragments) with compositions
        """
        if not self.fragments:
            return np.array([])

        n_mo = len(self.wf.mo_energies)
        n_fragments = len(self.fragments)
        compositions = np.zeros((n_mo, n_fragments))

        for i, fragment in enumerate(self.fragments):
            if fragment.fragment_type == FragmentType.BASIS_FUNCTION:
                compositions[:, i] = self._calculate_basis_function_compositions(fragment.indices)
            elif fragment.fragment_type == FragmentType.ATOM:
                compositions[:, i] = self._calculate_atomic_compositions(fragment.indices)
            elif fragment.fragment_type == FragmentType.MOLECULAR_ORBITAL:
                compositions[:, i] = self._calculate_mo_fragment_compositions(fragment.indices)

        return compositions

    def _calculate_basis_function_compositions(self, basis_indices: List[int]) -> np.ndarray:
        """Calculate compositions for basis function fragment."""
        n_mo = len(self.wf.mo_energies)
        compositions = np.zeros(n_mo)

        if self.config.composition_method == CompositionMethod.MULLIKEN:
            # Need overlap matrix for Mulliken analysis
            if self.wf.overlap_matrix is None:
                self.wf.calculate_overlap_matrix()

            S = self.wf.overlap_matrix
            C = self.wf.mo_coefficients

            for i_mo in range(n_mo):
                # Sum over basis functions in fragment: C_mu,i * C_nu,i * S_mu_nu
                for mu in basis_indices:
                    for nu in basis_indices:
                        if mu < S.shape[0] and nu < S.shape[0]:
                            compositions[i_mo] += C[mu, i_mo] * C[nu, i_mo] * S[mu, nu]

        return compositions

    def _calculate_atomic_compositions(self, atom_indices: List[int]) -> np.ndarray:
        """Calculate compositions for atomic fragment."""
        # Convert atom indices to basis function indices
        basis_indices = []
        atom_to_bfs = self.wf.get_atomic_basis_indices()

        for atom_idx in atom_indices:
            if atom_idx in atom_to_bfs:
                basis_indices.extend(atom_to_bfs[atom_idx])

        return self._calculate_basis_function_compositions(basis_indices)

    def _calculate_mo_fragment_compositions(self, mo_indices: List[int]) -> np.ndarray:
        """Calculate compositions for MO fragment."""
        n_mo = len(self.wf.mo_energies)
        compositions = np.zeros(n_mo)

        for i_mo in range(n_mo):
            if i_mo in mo_indices:
                compositions[i_mo] = 1.0

        return compositions

    def _calculate_tdos(self, energy_grid: np.ndarray) -> np.ndarray:
        """Calculate Total Density of States."""
        tdos = np.zeros_like(energy_grid)

        mo_energies = self.wf.mo_energies.copy()
        mo_occupations = self.wf.mo_occupations.copy()

        # Apply energy unit conversion if needed
        if self.config.use_ev:
            mo_energies *= AU_TO_EV

        # Apply energy shift
        mo_energies += self.config.energy_shift

        for i_mo, (e_mo, occ) in enumerate(zip(mo_energies, mo_occupations)):
            if occ <= 0:
                continue

            # Add broadened contribution to each energy point
            for i_e, e_grid in enumerate(energy_grid):
                contribution = self._broadening_function(e_grid - e_mo, self.config.fwhm)
                tdos[i_e] += contribution * occ

        # Apply scaling
        tdos *= self.config.scale_curve

        return tdos

    def _calculate_pdos(self, energy_grid: np.ndarray, compositions: np.ndarray) -> np.ndarray:
        """Calculate Projected Density of States."""
        if compositions.size == 0:
            return np.array([])

        n_fragments = compositions.shape[1]
        n_points = len(energy_grid)
        pdos = np.zeros((n_fragments, n_points))

        mo_energies = self.wf.mo_energies.copy()
        mo_occupations = self.wf.mo_occupations.copy()

        # Apply energy unit conversion if needed
        if self.config.use_ev:
            mo_energies *= AU_TO_EV

        # Apply energy shift
        mo_energies += self.config.energy_shift

        for i_frag in range(n_fragments):
            for i_mo, (e_mo, occ, comp) in enumerate(zip(mo_energies, mo_occupations, compositions[:, i_frag])):
                if occ <= 0 or comp <= 0:
                    continue

                # Add broadened contribution
                for i_e, e_grid in enumerate(energy_grid):
                    contribution = self._broadening_function(e_grid - e_mo, self.config.fwhm)
                    pdos[i_frag, i_e] += contribution * occ * comp

        # Apply scaling
        pdos *= self.config.scale_curve

        return pdos

    def calculate_dos(self) -> DOSResult:
        """
        Calculate Density of States.

        Returns:
            DOSResult object containing calculated DOS data
        """
        # Generate energy grid
        energy_grid = self._generate_energy_grid()

        # Calculate TDOS
        tdos = self._calculate_tdos(energy_grid)

        # Calculate fragment compositions if fragments are defined
        compositions = self._calculate_mo_compositions()
        pdos = None
        if compositions.size > 0:
            pdos = self._calculate_pdos(energy_grid, compositions)

        # Find HOMO/LUMO energies
        homo_energy = None
        lumo_energy = None

        mo_energies = self.wf.mo_energies.copy()
        mo_occupations = self.wf.mo_occupations.copy()

        # Apply energy unit conversion if needed
        if self.config.use_ev:
            mo_energies *= AU_TO_EV

        # Apply energy shift
        mo_energies_shifted = mo_energies + self.config.energy_shift

        occupied_indices = np.where(mo_occupations > 0)[0]
        virtual_indices = np.where(mo_occupations == 0)[0]

        if len(occupied_indices) > 0:
            homo_energy = mo_energies_shifted[occupied_indices[-1]]
        if len(virtual_indices) > 0:
            lumo_energy = mo_energies_shifted[virtual_indices[0]]

        # Get fragment names
        fragment_names = [frag.name for frag in self.fragments]

        return DOSResult(
            energies=energy_grid,
            tdos=tdos,
            pdos=pdos,
            fragment_names=fragment_names,
            homo_energy=homo_energy,
            lumo_energy=lumo_energy,
            config=self.config
        )

    def calculate_ldos(self, point: np.ndarray, energy_grid: np.ndarray) -> np.ndarray:
        """
        Calculate Local Density of States at a given point.

        Args:
            point: 3D coordinates (in Bohr)
            energy_grid: Energy grid points

        Returns:
            LDOS values at the point for each energy
        """
        # This would require evaluating basis functions at the point
        # For now, return a placeholder implementation
        ldos = np.zeros_like(energy_grid)

        # TODO: Implement actual LDOS calculation
        # This involves:
        # 1. Evaluating basis functions at the point
        # 2. Calculating orbital contributions at the point
        # 3. Applying broadening functions

        return ldos

    def get_frontier_orbital_info(self) -> Dict[str, Union[float, int]]:
        """
        Get information about frontier orbitals.

        Returns:
            Dictionary with HOMO/LUMO energies and indices
        """
        mo_energies = self.wf.mo_energies.copy()
        mo_occupations = self.wf.mo_occupations.copy()

        # Apply energy unit conversion if needed
        if self.config.use_ev:
            mo_energies *= AU_TO_EV

        # Apply energy shift
        mo_energies_shifted = mo_energies + self.config.energy_shift

        occupied_indices = np.where(mo_occupations > 0)[0]
        virtual_indices = np.where(mo_occupations == 0)[0]

        info = {}

        if len(occupied_indices) > 0:
            homo_idx = occupied_indices[-1]
            info['homo_index'] = homo_idx
            info['homo_energy'] = mo_energies_shifted[homo_idx]
            info['homo_occupation'] = mo_occupations[homo_idx]

        if len(virtual_indices) > 0:
            lumo_idx = virtual_indices[0]
            info['lumo_index'] = lumo_idx
            info['lumo_energy'] = mo_energies_shifted[lumo_idx]
            info['lumo_occupation'] = mo_occupations[lumo_idx]

        if len(occupied_indices) > 0 and len(virtual_indices) > 0:
            info['gap'] = info['lumo_energy'] - info['homo_energy']

        return info


# Utility functions for creating common fragment types

def create_atomic_fragment(wavefunction: Wavefunction, atom_indices: List[int],
                         name: str = "") -> Fragment:
    """Create a fragment from atomic indices."""
    if not name:
        name = f"Atoms_{','.join(map(str, atom_indices))}"
    return Fragment(
        indices=atom_indices,
        fragment_type=FragmentType.ATOM,
        name=name
    )


def create_basis_function_fragment(basis_indices: List[int],
                                 name: str = "") -> Fragment:
    """Create a fragment from basis function indices."""
    if not name:
        name = f"Basis_{','.join(map(str, basis_indices))}"
    return Fragment(
        indices=basis_indices,
        fragment_type=FragmentType.BASIS_FUNCTION,
        name=name
    )


def create_molecular_orbital_fragment(mo_indices: List[int],
                                    name: str = "") -> Fragment:
    """Create a fragment from molecular orbital indices."""
    if not name:
        name = f"MOs_{','.join(map(str, mo_indices))}"
    return Fragment(
        indices=mo_indices,
        fragment_type=FragmentType.MOLECULAR_ORBITAL,
        name=name
    )


def create_elemental_fragments(wavefunction: Wavefunction) -> List[Fragment]:
    """
    Create fragments for each unique element in the system.

    Args:
        wavefunction: Wavefunction object

    Returns:
        List of fragments, one for each element
    """
    element_atoms = {}
    for i, atom in enumerate(wavefunction.atoms):
        element = atom.element
        if element not in element_atoms:
            element_atoms[element] = []
        element_atoms[element].append(i)

    fragments = []
    for element, atom_indices in element_atoms.items():
        fragment = create_atomic_fragment(wavefunction, atom_indices, element)
        fragments.append(fragment)

    return fragments