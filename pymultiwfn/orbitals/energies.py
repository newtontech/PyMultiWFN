"""
Orbital energy analysis module.

This module provides functionality for analyzing molecular orbital energies,
including HOMO-LUMO gap calculation, Fermi level determination, and energy
diagram generation.

Reference: PHASE2_TASKS.md - Task 2.1.1: MO Energy Analysis
"""

from typing import Any, Dict, Optional

import numpy as np

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
            "energies": energies,
            "occupations": occupations,
            "indices": indices,
            "homo_index": self.homo_index,
            "lumo_index": self.lumo_index,
            "fermi_level": self.fermi_level,
        }

    def get_composition(self, mo_index: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate AO contribution to a molecular orbital.

        Uses squared MO coefficients as a measure of AO contribution
        (Mulliken population approach).

        Args:
            mo_index: Zero-based index of the molecular orbital

        Returns:
            Dictionary mapping atom labels to their orbital contributions:
            {'C1': {'2s': 0.35, '2p_z': 0.45}, 'H1': {'1s': 0.20}}

        Raises:
            IndexError: If mo_index is out of range
            ValueError: If negative mo_index is provided
        """
        # Validate input
        if mo_index < 0:
            raise ValueError(f"MO index must be non-negative, got {mo_index}")

        n_mo = len(self.alpha_energies)
        if mo_index >= n_mo:
            raise IndexError(f"MO index {mo_index} out of range (0-{n_mo-1})")

        # Get MO coefficients
        if self.wfn.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        coeffs = self.wfn.coefficients[mo_index, :]  # MO coefficients for this orbital

        # Calculate contributions (squared coefficients)
        contributions = coeffs**2

        # Normalize to sum to 1.0
        total = np.sum(contributions)
        if total > 1e-10:
            contributions = contributions / total

        # Get atomic basis indices mapping
        atomic_basis = self.wfn.get_atomic_basis_indices()

        # Build composition by atom
        composition = {}

        for atom_idx, basis_indices in atomic_basis.items():
            atom = self.wfn.atoms[atom_idx]
            atom_label = f"{atom.element}{atom_idx + 1}"

            # Sum contributions for this atom
            atom_contrib = sum(
                contributions[i] for i in basis_indices if i < len(contributions)
            )

            if atom_contrib > 1e-6:  # Only include significant contributions
                composition[atom_label] = {}

                # Assign orbital types based on shell types for this atom
                atom_shells = [s for s in self.wfn.shells if s.center_idx == atom_idx]

                # Map shell types to orbital labels (simplified)
                for shell in atom_shells:
                    shell_type = shell.type
                    if shell_type == 0:  # S shell
                        orb_type = "s"
                    elif shell_type == 1:  # P shell
                        orb_type = "p"
                    elif shell_type == 2:  # D shell
                        orb_type = "d"
                    elif shell_type == 3:  # F shell
                        orb_type = "f"
                    else:
                        orb_type = f"type{shell_type}"

                    # Distribute atom contribution among orbital types
                    if orb_type not in composition[atom_label]:
                        composition[atom_label][orb_type] = 0.0

                # Proportionally distribute atom contribution
                n_types = len(composition[atom_label])
                if n_types > 0:
                    per_type = atom_contrib / n_types
                    for orb_type in composition[atom_label]:
                        composition[atom_label][orb_type] = per_type

        return composition

    def get_dominant_orbital_type(self, mo_index: int) -> str:
        """
        Identify the dominant orbital type for a molecular orbital.

        Args:
            mo_index: Zero-based index of the molecular orbital

        Returns:
            One of: 's', 'p', 'd', 'f', or 'mixed'
        """
        composition = self.get_composition(mo_index)

        # Sum contributions by orbital type
        type_sums = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}

        for atom_contrib in composition.values():
            for orb_type, contrib in atom_contrib.items():
                if orb_type.startswith("s"):
                    type_sums["s"] += contrib
                elif orb_type.startswith("p"):
                    type_sums["p"] += contrib
                elif orb_type.startswith("d"):
                    type_sums["d"] += contrib
                elif orb_type.startswith("f"):
                    type_sums["f"] += contrib

        # Find dominant type
        max_type = max(type_sums, key=type_sums.get)
        max_value = type_sums[max_type]

        # If dominant type has > 50% contribution, return it
        if max_value > 0.5:
            return max_type
        else:
            return "mixed"

    def get_orbital_localization(self, mo_index: int) -> Dict[str, float]:
        """
        Calculate orbital localization on each atom.

        Args:
            mo_index: Zero-based index of the molecular orbital

        Returns:
            Dictionary mapping atom labels to their total contribution:
            {'C1': 0.65, 'H1': 0.20, 'H2': 0.15}
        """
        composition = self.get_composition(mo_index)

        # Sum contributions for each atom
        localization = {}
        for atom_label, orb_contrib in composition.items():
            localization[atom_label] = sum(orb_contrib.values())

        return localization

    def generate_composition_report(self, mo_index: int) -> str:
        """
        Generate a human-readable composition report for a molecular orbital.

        Args:
            mo_index: Zero-based index of the molecular orbital

        Returns:
            Formatted string report of orbital composition
        """
        composition = self.get_composition(mo_index)
        localization = self.get_orbital_localization(mo_index)
        dominant_type = self.get_dominant_orbital_type(mo_index)
        energy = self.alpha_energies[mo_index]

        lines = [
            f"MO #{mo_index} Composition Report",
            "=" * 40,
            f"Energy: {energy:.6f} Ha ({energy * 27.2114:.3f} eV)",
            f"Dominant Type: {dominant_type.upper()}",
            "",
            "Atomic Contributions:",
        ]

        # Sort atoms by contribution (descending)
        sorted_atoms = sorted(localization.items(), key=lambda x: -x[1])

        for atom_label, atom_total in sorted_atoms:
            lines.append(f"  {atom_label}: {atom_total*100:.2f}%")
            orb_contrib = composition[atom_label]
            sorted_orbs = sorted(orb_contrib.items(), key=lambda x: -x[1])
            for orb_type, contrib in sorted_orbs:
                if contrib > 0.01:  # Only show > 1% contributions
                    lines.append(f"    {orb_type}: {contrib*100:.2f}%")

        return "\n".join(lines)

    def get_orbital_symmetry(self, mo_index: int) -> Optional[str]:
        """
        Get orbital symmetry label (if available).

        Note: This is a placeholder for future symmetry analysis.

        Args:
            mo_index: Zero-based index of the molecular orbital

        Returns:
            Symmetry label or None if not available
        """
        # Placeholder - symmetry analysis requires molecular point group
        return None

    def get_orbital_overlap(self, mo_i: int, mo_j: int) -> float:
        """
        Calculate overlap between two molecular orbitals.

        The orbital overlap is calculated as:
        S_ij = C_i^T * S * C_j
        where S is the AO overlap matrix and C are MO coefficients.

        For normalized orbitals, S_ii = 1.0 (self-overlap).

        Args:
            mo_i: Zero-based index of first molecular orbital
            mo_j: Zero-based index of second molecular orbital

        Returns:
            Orbital overlap value (typically in range [-1, 1])

        Raises:
            ValueError: If MO indices are negative
            IndexError: If MO indices are out of range
        """
        # Validate input
        if mo_i < 0 or mo_j < 0:
            raise ValueError(f"MO indices must be non-negative, got ({mo_i}, {mo_j})")

        n_mo = len(self.alpha_energies)
        if mo_i >= n_mo or mo_j >= n_mo:
            raise IndexError(f"MO index out of range (0-{n_mo-1})")

        # Get MO coefficients
        if self.wfn.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        # Get or calculate AO overlap matrix
        if self.wfn.overlap_matrix is not None:
            S_ao = self.wfn.overlap_matrix
        else:
            # Calculate overlap matrix
            S_ao = self.wfn.calculate_overlap_matrix()

        C_i = self.wfn.coefficients[mo_i, :]
        C_j = self.wfn.coefficients[mo_j, :]

        # Calculate MO overlap: S_ij = C_i^T * S_ao * C_j
        overlap = float(C_i @ S_ao @ C_j)

        return overlap

    def get_overlap_matrix(self, mo_indices: list) -> np.ndarray:
        """
        Generate overlap matrix for a subset of molecular orbitals.

        Args:
            mo_indices: List of MO indices to include in the matrix

        Returns:
            Symmetric overlap matrix where S[i,j] = overlap(MO_i, MO_j)
            Diagonal elements are 1.0 for normalized orbitals.

        Example:
            >>> S = analyzer.get_overlap_matrix([0, 1, 2])
            >>> print(S.shape)  # (3, 3)
        """
        n = len(mo_indices)
        overlap_matrix = np.eye(n)  # Start with identity (self-overlap = 1)

        # Fill off-diagonal elements
        for i, mo_i in enumerate(mo_indices):
            for j, mo_j in enumerate(mo_indices):
                if i != j:
                    overlap_matrix[i, j] = self.get_orbital_overlap(mo_i, mo_j)

        return overlap_matrix

    def get_bonding_character(self, mo_i: int, mo_j: int) -> str:
        """
        Determine bonding character of interaction between two orbitals.

        Args:
            mo_i: Zero-based index of first molecular orbital
            mo_j: Zero-based index of second molecular orbital

        Returns:
            One of: 'bonding', 'antibonding', 'non-bonding', or 'mixed'
        """
        overlap = self.get_orbital_overlap(mo_i, mo_j)

        if abs(overlap) < 0.1:
            return "non-bonding"
        elif overlap > 0.3:
            return "bonding"
        elif overlap < -0.3:
            return "antibonding"
        else:
            return "mixed"

    def get_interaction_strength(self, mo_i: int, mo_j: int) -> float:
        """
        Calculate orbital interaction strength (absolute overlap).

        Interaction strength is defined as |S_ij|, representing the
        magnitude of orbital coupling regardless of phase.

        Args:
            mo_i: Zero-based index of first molecular orbital
            mo_j: Zero-based index of second molecular orbital

        Returns:
            Interaction strength in range [0, 1]
        """
        overlap = self.get_orbital_overlap(mo_i, mo_j)
        return abs(overlap)

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"OrbitalsAnalyzer("
            f"HOMO={self.homo_index} at {self.homo_energy:.4f} Ha, "
            f"LUMO={self.lumo_index} at {self.lumo_energy:.4f} Ha, "
            f"gap={self.gap:.4f} Ha)"
        )
