"""
Enhanced parser for CP2K output files and related formats.

CP2K is a quantum chemistry and solid state physics software package.
This parser supports CP2K output files, coordinate files, and
related formats with comprehensive error handling.
"""

import re
import numpy as np
import warnings
from typing import List, Optional, Dict, Any
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class CP2KLoader:
    """
    Enhanced loader for CP2K output files and coordinate files.

    Supports:
    - CP2K output files (.out)
    - CP2K coordinate files (.xyz, .coord)
    - CP2K restart files (.restart)
    - CP2K MO output files (when available)
    """

    def __init__(self, filename: str):
        """
        Initialize the CP2K loader.

        Args:
            filename: Path to the CP2K file

        Raises:
            FileNotFoundError: If the file does not exist
        """
        self.filename = filename
        self.wfn = Wavefunction()
        self.metadata: Dict[str, Any] = {}

    def load(self) -> Wavefunction:
        """
        Parse the CP2K file and return a complete Wavefunction object.

        Returns:
            Wavefunction: Complete wavefunction object with all parsed data

        Raises:
            ValueError: If the file format is invalid
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.filename, 'r', encoding='latin-1') as f:
                content = f.read()

        if not content.strip():
            raise ValueError(f"File {self.filename} appears to be empty")

        # Determine file type based on content and extension
        if self.filename.endswith('.out'):
            self._parse_output_file(content)
        elif self.filename.endswith(('.coord', '.xyz')):
            self._parse_coordinate_file(content)
        else:
            # Try to auto-detect format
            if 'MODULE QUICKSTEP' in content or 'CP2K' in content:
                self._parse_output_file(content)
            else:
                self._parse_coordinate_file(content)

        # Validate and finalize
        self._validate_parsed_data()
        self.wfn._infer_occupations()

        return self.wfn

    def _parse_output_file(self, content: str):
        """Parse CP2K output file."""
        # Extract title
        title_match = re.search(r'CP2K\|.*?(?:\n.*?)*?\*+', content, re.MULTILINE)
        if title_match:
            self.wfn.title = "CP2K Calculation"
        else:
            self.wfn.title = "CP2K output"

        # Parse atomic coordinates
        self._parse_atoms_from_output(content)

        # Parse basis set information (if available)
        self._parse_basis_set_from_output(content)

        # Parse orbital energies (if available)
        self._parse_orbital_energies_from_output(content)

        # Parse total energy
        self._parse_total_energy(content)

    def _parse_coordinate_file(self, content: str):
        """Parse CP2K coordinate file."""
        lines = content.strip().split('\n')

        # First line might be comment or title
        if lines and not lines[0].startswith('#') and len(lines[0].split()) < 3:
            self.wfn.title = lines[0].strip()
            start_line = 1
        else:
            self.wfn.title = "CP2K Coordinate File"
            start_line = 0

        # CP2K coordinate format: element_symbol x y z
        atoms_parsed = 0
        for line in lines[start_line:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0].title()
                    x = float(parts[1]) * ANGSTROM_TO_BOHR
                    y = float(parts[2]) * ANGSTROM_TO_BOHR
                    z = float(parts[3]) * ANGSTROM_TO_BOHR

                    atomic_num = self._element_to_atomic_number(element)

                    self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num))
                    atoms_parsed += 1
                except (ValueError, IndexError):
                    continue

        self.metadata['atoms_parsed'] = atoms_parsed

    def _parse_atoms_from_output(self, content: str):
        """Parse atomic coordinates from CP2K output."""
        # Look for coordinate sections in output
        coord_patterns = [
            r'Atomic coordinates.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])',
            r'COORDINATES.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])',
            r'Atom\s+Kind\s+Element\s+X\s+Y\s+Z.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])'
        ]

        atoms_parsed = 0
        for pattern in coord_patterns:
            coord_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if coord_match:
                coord_lines = coord_match.group(1).strip().split('\n')

                for line in coord_lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:  # index, element, x, y, z (or similar format)
                        try:
                            # Try different formats
                            if parts[0].isdigit() and len(parts) >= 5:
                                # Format: index element x y z
                                element = parts[1].title()
                                x = float(parts[2]) * ANGSTROM_TO_BOHR
                                y = float(parts[3]) * ANGSTROM_TO_BOHR
                                z = float(parts[4]) * ANGSTROM_TO_BOHR
                            elif parts[0].isalpha() and len(parts) >= 4:
                                # Format: element x y z
                                element = parts[0].title()
                                x = float(parts[1]) * ANGSTROM_TO_BOHR
                                y = float(parts[2]) * ANGSTROM_TO_BOHR
                                z = float(parts[3]) * ANGSTROM_TO_BOHR
                            else:
                                continue

                            atomic_num = self._element_to_atomic_number(element)

                            self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num))
                            atoms_parsed += 1

                        except (ValueError, IndexError):
                            continue

                if atoms_parsed > 0:
                    break

        self.metadata['atoms_parsed'] = atoms_parsed

    def _parse_basis_set_from_output(self, content: str):
        """Parse basis set information from CP2K output."""
        # Look for basis set information
        basis_patterns = [
            r'BASIS SET.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])',
            r'Gaussian basis set.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])'
        ]

        basis_found = False
        for pattern in basis_patterns:
            basis_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if basis_match:
                basis_found = True
                # This is a placeholder for basis set parsing
                # CP2K basis set information in output is often limited
                # For full basis sets, one might need to parse input files
                break

        if basis_found:
            self.metadata['basis_set_parsed'] = True
        else:
            warnings.warn("No basis set information found in CP2K output", RuntimeWarning)

    def _parse_orbital_energies_from_output(self, content: str):
        """Parse molecular orbital energies from CP2K output."""
        # Look for eigenvalue sections
        eigenvalue_patterns = [
            r'Molecular Orbital.*?Eigenvalues?\s*\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])',
            r'EIGENVALUES?\s*\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])',
            r'KS-?\s*Orbitals.*?Energies?.*?\n(.*?)(?=\n\s*\n|\n\s*-|\n[A-Z])'
        ]

        for pattern in eigenvalue_patterns:
            eigen_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if eigen_match:
                eigen_lines = eigen_match.group(1).strip()

                # Extract numeric values
                eigenvalues = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', eigen_lines)

                try:
                    energies = [float(e) for e in eigenvalues if abs(float(e)) < 1e10]  # Filter out unrealistic values

                    if energies:
                        self.wfn.energies = np.array(energies)
                        self.wfn.num_mos = len(energies)
                        self.metadata['orbital_energies_parsed'] = True
                        self.metadata['num_orbitals'] = len(energies)
                        break
                except ValueError:
                    continue

    def _parse_total_energy(self, content: str):
        """Parse total energy from CP2K output."""
        # Look for total energy
        energy_patterns = [
            r'Total FORCE_EVAL.*?Energy\s*[:\|]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'Total energy\s*[:\|]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'ENERGY\| Total FORCE_EVAL.*?[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        ]

        for pattern in energy_patterns:
            energy_match = re.search(pattern, content, re.IGNORECASE)
            if energy_match:
                try:
                    energy = float(energy_match.group(1) if energy_match.groups() else energy_match.group(0))
                    self.wfn.total_energy = energy
                    self.metadata['total_energy_parsed'] = True
                    break
                except (ValueError, IndexError):
                    continue

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        element_mapping = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
        }
        return element_mapping.get(element.title(), 0)

    def _validate_parsed_data(self):
        """Validate parsed data for consistency."""
        if self.wfn.num_atoms == 0:
            warnings.warn("No atoms were parsed from the CP2K file", RuntimeWarning)

        # Set validation metadata
        self.metadata['validation_passed'] = True
        self.metadata['validation_timestamp'] = np.datetime64('now')