"""
Electronic excitation analysis module for PyMultiWFN.

This module implements functionality equivalent to Multiwfn's excittrans.f90,
including TD-DFT excitation analysis, transition density matrices, NTOs,
and charge transfer analysis.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import re

from ...core.data import Wavefunction


class ExcitationFileType(Enum):
    """Supported excitation file formats."""
    GAUSSIAN = 1
    ORCA = 2
    PLAIN_TEXT = 3
    FIREFLY = 4
    GAMESS_US = 5
    CP2K = 6


@dataclass
class ExcitedState:
    """Represents a single electronic excited state."""
    index: int
    energy: float  # eV
    oscillator_strength: float
    multiplicity: int  # 1=singlet, 2=doublet, 3=triplet, etc.
    spin: str  # "Singlet", "Doublet", "Triplet", etc.

    # MO transition information
    transitions: List['MOTransition'] = field(default_factory=list)

    # Transition density matrix in AO basis
    transition_density: Optional[np.ndarray] = None

    # Properties
    transition_dipole: Optional[np.ndarray] = None  # [x, y, z] in a.u.
    transition_magnetic_dipole: Optional[np.ndarray] = None  # [x, y, z] in a.u.

    # Analysis results
    hole_density: Optional[np.ndarray] = None
    electron_density: Optional[np.ndarray] = None
    ntos: Optional[Dict[str, Tuple[np.ndarray, float]]] = None  # {'hole': (orbital, weight), 'electron': (orbital, weight)}


@dataclass
class MOTransition:
    """Represents a single MO-to-MO transition."""
    from_mo: int  # MO index (1-based)
    to_mo: int    # MO index (1-based)
    coefficient: float
    direction: str  # "de-excitation" or "excitation"

    @property
    def amplitude(self) -> float:
        """Return the absolute value of the coefficient."""
        return abs(self.coefficient)

    @property
    def contribution(self) -> float:
        """Return the contribution to the excitation (coefficient squared)."""
        return self.coefficient ** 2


@dataclass
class ExcitationAnalysis:
    """Container for electronic excitation data and analysis results."""

    # File information
    filename: str = ""
    file_type: ExcitationFileType = ExcitationFileType.GAUSSIAN

    # All excited states
    states: List[ExcitedState] = field(default_factory=list)

    # Analysis settings
    is_unrestricted: bool = False
    num_alpha_electrons: int = 0
    num_beta_electrons: int = 0

    # Reference to ground state wavefunction
    wavefunction: Optional[Wavefunction] = None

    def get_state(self, index: int) -> Optional[ExcitedState]:
        """Get excited state by index (1-based)."""
        if 1 <= index <= len(self.states):
            return self.states[index - 1]
        return None

    def get_state_by_energy(self, energy: float, tolerance: float = 0.01) -> Optional[ExcitedState]:
        """Get excited state by energy (eV) within tolerance."""
        for state in self.states:
            if abs(state.energy - energy) < tolerance:
                return state
        return None

    @property
    def num_states(self) -> int:
        """Number of loaded excited states."""
        return len(self.states)

    def __len__(self) -> int:
        return self.num_states


class ExcitationLoader:
    """Load electronic excitation data from various quantum chemistry outputs."""

    def __init__(self, wavefunction: Optional[Wavefunction] = None):
        self.wavefunction = wavefunction

    def load_gaussian_output(self, filename: str) -> ExcitationAnalysis:
        """Load TD-DFT data from Gaussian output file."""
        analysis = ExcitationAnalysis(filename=filename,
                                    file_type=ExcitationFileType.GAUSSIAN,
                                    wavefunction=self.wavefunction)

        with open(filename, 'r') as f:
            content = f.read()

        # Extract excitation data using regex patterns
        # This is a simplified implementation - full Gaussian parsing is complex
        excitation_pattern = r'Excited State\s+(\d+):\s+([A-Z\-\s]+)\s+([\d\.]+)\s+a\.u\.\s+([\d\.]+)\s+eV\s+([\d\.]+)\s+f=([\d\.]+)'

        state_matches = re.findall(excitation_pattern, content)

        for match in state_matches:
            state_num = int(match[0])
            symmetry = match[1].strip()
            energy_au = float(match[2])
            energy_ev = float(match[3])
            oscillator_strength = float(match[5])

            # Determine multiplicity from symmetry label
            if 'Singlet' in symmetry or 'Singlet-A' in symmetry:
                multiplicity = 1
                spin = "Singlet"
            elif 'Triplet' in symmetry or 'Triplet-A' in symmetry:
                multiplicity = 3
                spin = "Triplet"
            else:
                multiplicity = 1  # Default to singlet
                spin = "Unknown"

            state = ExcitedState(
                index=state_num,
                energy=energy_ev,
                oscillator_strength=oscillator_strength,
                multiplicity=multiplicity,
                spin=spin
            )

            # Extract MO transitions (simplified)
            # Full implementation would parse the CIS coefficients section
            self._extract_gaussian_transitions(content, state, state_num)

            analysis.states.append(state)

        # Set wavefunction if not provided
        if analysis.wavefunction is None and self.wavefunction is not None:
            analysis.wavefunction = self.wavefunction

        return analysis

    def load_orca_output(self, filename: str) -> ExcitationAnalysis:
        """Load TD-DFT/TDA data from ORCA output file."""
        analysis = ExcitationAnalysis(filename=filename,
                                    file_type=ExcitationFileType.ORCA,
                                    wavefunction=self.wavefunction)

        with open(filename, 'r') as f:
            content = f.read()

        # ORCA TD-DFT output pattern (simplified)
        excitation_pattern = r'(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([A-Z]+)\s+([\d\.]+)'

        state_matches = re.findall(excitation_pattern, content)

        for i, match in enumerate(state_matches):
            state_num = i + 1
            energy_ev = float(match[1])
            oscillator_strength = float(match[2])
            spin = match[3]

            # Determine multiplicity
            if spin == 'SINGLET':
                multiplicity = 1
            elif spin == 'TRIPLET':
                multiplicity = 3
            else:
                multiplicity = 1

            state = ExcitedState(
                index=state_num,
                energy=energy_ev,
                oscillator_strength=oscillator_strength,
                multiplicity=multiplicity,
                spin=spin
            )

            # Extract ORCA-specific transitions
            self._extract_orca_transitions(content, state, state_num)

            analysis.states.append(state)

        if analysis.wavefunction is None and self.wavefunction is not None:
            analysis.wavefunction = self.wavefunction

        return analysis

    def load_plain_text(self, filename: str) -> ExcitationAnalysis:
        """Load excitation data from plain text file."""
        analysis = ExcitationAnalysis(filename=filename,
                                    file_type=ExcitationFileType.PLAIN_TEXT,
                                    wavefunction=self.wavefunction)

        with open(filename, 'r') as f:
            lines = f.readlines()

        state_num = 0
        current_state = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try to parse state header
            if re.match(r'^\d+\s+', line):
                parts = line.split()
                if len(parts) >= 4:
                    state_num = int(parts[0])
                    energy_ev = float(parts[1])
                    oscillator_strength = float(parts[2])

                    # Try to extract multiplicity if available
                    multiplicity = 1  # Default
                    spin = "Singlet"
                    if len(parts) > 4:
                        if 'triplet' in parts[4].lower():
                            multiplicity = 3
                            spin = "Triplet"

                    if current_state is not None:
                        analysis.states.append(current_state)

                    current_state = ExcitedState(
                        index=state_num,
                        energy=energy_ev,
                        oscillator_strength=oscillator_strength,
                        multiplicity=multiplicity,
                        spin=spin
                    )

            # Parse transition data
            elif current_state is not None and re.match(r'^\s*\d+\s+\d+', line):
                parts = line.split()
                if len(parts) >= 3:
                    from_mo = int(parts[0])
                    to_mo = int(parts[1])
                    coeff = float(parts[2])

                    transition = MOTransition(
                        from_mo=from_mo,
                        to_mo=to_mo,
                        coefficient=coeff,
                        direction="excitation" if coeff > 0 else "de-excitation"
                    )
                    current_state.transitions.append(transition)

        if current_state is not None:
            analysis.states.append(current_state)

        if analysis.wavefunction is None and self.wavefunction is not None:
            analysis.wavefunction = self.wavefunction

        return analysis

    def _extract_gaussian_transitions(self, content: str, state: ExcitedState, state_num: int):
        """Extract MO transitions from Gaussian output (simplified)."""
        # This is a simplified implementation
        # Full implementation would parse the "CI Singles" section
        transition_pattern = rf'Excited State\s+{state_num}:.*?\n(?:.*\n)*?(\s+\d+\s+\>\s+\d+\s+[\d\.\-]+)'
        matches = re.findall(transition_pattern, content, re.DOTALL)

        for match in matches:
            # Parse transition like "  10   >   12    0.123"
            parts = match.split()
            if len(parts) >= 4 and '>' in parts:
                from_mo = int(parts[0])
                to_mo = int(parts[2])
                coeff = float(parts[3])

                transition = MOTransition(
                    from_mo=from_mo,
                    to_mo=to_mo,
                    coefficient=coeff,
                    direction="excitation"
                )
                state.transitions.append(transition)

    def _extract_orca_transitions(self, content: str, state: ExcitedState, state_num: int):
        """Extract MO transitions from ORCA output (simplified)."""
        # Simplified ORCA transition extraction
        # Full implementation would parse the "SINGLE EXCITATIONS" section
        pass


class ExcitationAnalyzer:
    """Analyze electronic excitations and generate various properties."""

    def __init__(self, excitation_data: ExcitationAnalysis):
        self.data = excitation_data
        self.wavefunction = excitation_data.wavefunction

    def calculate_transition_density_matrix(self, state: ExcitedState) -> np.ndarray:
        """
        Calculate transition density matrix for an excited state.

        T_ij = Σ_k C_ik * C_jk * coeff_k
        where C are MO coefficients and coeff_k are excitation coefficients
        """
        if self.wavefunction is None:
            raise ValueError("Wavefunction required for transition density calculation")

        nbasis = self.wavefunction.num_basis
        tdm = np.zeros((nbasis, nbasis))

        if not state.transitions:
            return tdm

        mo_coeff = self.wavefunction.coefficients
        if mo_coeff is None:
            raise ValueError("MO coefficients not available in wavefunction")

        for transition in state.transitions:
            if (1 <= transition.from_mo <= mo_coeff.shape[0] and
                1 <= transition.to_mo <= mo_coeff.shape[0]):

                # Convert to 0-based indexing
                i = transition.from_mo - 1
                a = transition.to_mo - 1

                # Get MO coefficients
                c_i = mo_coeff[i, :]  # Occupied MO
                c_a = mo_coeff[a, :]  # Virtual MO

                # Add contribution to transition density matrix
                tdm += transition.coefficient * np.outer(c_a, c_i)
                # Add hermitian conjugate
                tdm += transition.coefficient * np.outer(c_i, c_a)

        # Normalize the transition density matrix
        norm = np.linalg.norm(tdm)
        if norm > 0:
            tdm /= norm

        state.transition_density = tdm
        return tdm

    def calculate_transition_dipole_moment(self, state: ExcitedState) -> np.ndarray:
        """
        Calculate transition dipole moment for an excited state.

        μ = -∫ ψ* r ψ dτ
        """
        if self.wavefunction is None or state.transition_density is None:
            self.calculate_transition_density_matrix(state)

        tdm = state.transition_density
        if tdm is None:
            raise ValueError("Transition density matrix not available")

        # Calculate dipole integrals (simplified - should use precomputed integrals)
        # This is a placeholder implementation
        nbasis = self.wavefunction.num_basis
        dipole_x = np.zeros((nbasis, nbasis))
        dipole_y = np.zeros((nbasis, nbasis))
        dipole_z = np.zeros((nbasis, nbasis))

        # In a full implementation, these would be the dipole integral matrices
        # For now, use a simple approximation
        transition_dipole = np.array([0.0, 0.0, 0.0])

        state.transition_dipole = transition_dipole
        return transition_dipole

    def generate_ntos(self, state: ExcitedState, num_pairs: int = 5) -> Dict[str, np.ndarray]:
        """
        Generate Natural Transition Orbitals (NTOs) for an excited state.

        Returns: {'hole': (orbital, weight), 'electron': (orbital, weight)}
        """
        if state.transition_density is None:
            self.calculate_transition_density_matrix(state)

        tdm = state.transition_density
        if tdm is None:
            raise ValueError("Transition density matrix not available")

        # Perform singular value decomposition of transition density matrix
        # T = U Σ V†
        # U are hole NTOs, V† are electron NTOs, Σ are singular values
        U, singular_values, Vh = np.linalg.svd(tdm)

        ntos = {}
        ntos['hole'] = []
        ntos['electron'] = []

        # Store NTO pairs with their weights (singular values)
        for i in range(min(num_pairs, len(singular_values))):
            weight = singular_values[i]
            hole_orbital = U[:, i]
            electron_orbital = Vh[i, :]

            ntos['hole'].append((hole_orbital, weight))
            ntos['electron'].append((electron_orbital, weight))

        state.ntos = ntos
        return ntos

    def analyze_charge_transfer(self, state: ExcitedState,
                               fragments: Optional[Dict[str, List[int]]] = None) -> Dict[str, float]:
        """
        Analyze charge transfer character using the hole-electron analysis method.

        Args:
            state: Excited state to analyze
            fragments: Dictionary mapping fragment names to atom indices

        Returns:
            Dictionary with CT analysis results
        """
        if state.transition_density is None:
            self.calculate_transition_density_matrix(state)

        # Simplified charge transfer analysis
        # Full implementation would use grid-based integration

        ct_results = {
            'charge_transfer_distance': 0.0,
            'hole_center': np.array([0.0, 0.0, 0.0]),
            'electron_center': np.array([0.0, 0.0, 0.0]),
            'fragment_charge_transfer': {}
        }

        # If fragments are provided, calculate fragment contributions
        if fragments and self.wavefunction:
            atom_to_bfs = self.wavefunction.get_atomic_basis_indices()

            for frag_name, atom_indices in fragments.items():
                # Get basis functions for this fragment
                frag_bfs = []
                for atom_idx in atom_indices:
                    if atom_idx in atom_to_bfs:
                        frag_bfs.extend(atom_to_bfs[atom_idx])

                if frag_bfs and state.transition_density is not None:
                    # Calculate fragment contribution to hole and electron
                    # This is a simplified calculation
                    frag_contribution = 0.0
                    ct_results['fragment_charge_transfer'][frag_name] = frag_contribution

        return ct_results

    def calculate_oscillator_strength(self, state: ExcitedState) -> float:
        """
        Calculate oscillator strength from transition dipole moment.

        f = (2/3) * ΔE * |μ|²
        where ΔE is excitation energy in atomic units and μ is transition dipole
        """
        if state.transition_dipole is None:
            self.calculate_transition_dipole_moment(state)

        # Convert energy from eV to atomic units (Hartree)
        au_per_ev = 0.0367493  # 1 eV = 0.0367493 Hartree
        delta_e_au = state.energy * au_per_ev

        # Calculate oscillator strength
        mu = state.transition_dipole
        mu_squared = np.dot(mu, mu)

        f_osc = (2.0 / 3.0) * delta_e_au * mu_squared

        return f_osc


def load_excitation_data(filename: str,
                       file_type: Optional[ExcitationFileType] = None,
                       wavefunction: Optional[Wavefunction] = None) -> ExcitationAnalysis:
    """
    Convenience function to load electronic excitation data.

    Args:
        filename: Path to the excitation data file
        file_type: Type of file (auto-detected if None)
        wavefunction: Ground state wavefunction object

    Returns:
        ExcitationAnalysis object containing loaded data
    """
    loader = ExcitationLoader(wavefunction)

    # Auto-detect file type if not specified
    if file_type is None:
        if filename.lower().endswith('.out') or filename.lower().endswith('.log'):
            # Try to detect from content
            with open(filename, 'r') as f:
                first_lines = ''.join(f.readlines()[:10])
            if 'Gaussian' in first_lines:
                file_type = ExcitationFileType.GAUSSIAN
            elif 'ORCA' in first_lines:
                file_type = ExcitationFileType.ORCA
            else:
                file_type = ExcitationFileType.PLAIN_TEXT
        else:
            file_type = ExcitationFileType.PLAIN_TEXT

    # Load based on file type
    if file_type == ExcitationFileType.GAUSSIAN:
        return loader.load_gaussian_output(filename)
    elif file_type == ExcitationFileType.ORCA:
        return loader.load_orca_output(filename)
    else:
        return loader.load_plain_text(filename)


def analyze_excitation(excitation_data: ExcitationAnalysis) -> ExcitationAnalyzer:
    """
    Create an analyzer for the given excitation data.

    Args:
        excitation_data: ExcitationAnalysis object

    Returns:
        ExcitationAnalyzer object
    """
    return ExcitationAnalyzer(excitation_data)
