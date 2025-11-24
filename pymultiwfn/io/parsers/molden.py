"""
Enhanced parser for Molden files (.molden, .molden.input, molden.inp).

Molden format is widely used for visualizing molecular orbitals and electron density.
This enhanced parser provides comprehensive error handling and supports various
Molden format variants from different quantum chemistry programs.
"""

import re
import numpy as np
import warnings
from typing import List, Optional, Dict, Any, Tuple
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES

class MoldenLoader:
    """
    Enhanced loader for Molden files.

    Supports standard Molden format and common variants from ORCA, DALTON,
    and other quantum chemistry programs. Includes comprehensive error
    handling and data validation.
    """

    def __init__(self, filename: str):
        """
        Initialize the Molden loader.

        Args:
            filename: Path to the Molden file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file extension is not recognized
        """
        if not any(filename.lower().endswith(ext) for ext in ['.molden', '.molf', '.input', '.inp']):
            if 'molden' not in filename.lower() and not filename.endswith('MOLDEN'):
                warnings.warn(f"File does not appear to be a Molden file: {filename}", RuntimeWarning)

        self.filename = filename
        self.wfn = Wavefunction()
        self.metadata: Dict[str, Any] = {}
        self.sections: Dict[str, List[str]] = {}

    def load(self) -> Wavefunction:
        """
        Parse the Molden file and return a complete Wavefunction object.

        Returns:
            Wavefunction: Complete wavefunction object with all parsed data

        Raises:
            ValueError: If the file format is invalid or critical data is missing
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.filename, 'r', encoding='latin-1') as f:
                content = f.read()

        if not content.strip():
            raise ValueError(f"File {self.filename} appears to be empty")

        # Split into sections for better parsing
        self._split_into_sections(content)

        # Parse each section
        self._parse_atoms()
        self._parse_basis()
        self._parse_mo()

        # Validate and finalize
        self._validate_parsed_data()
        self.wfn._infer_occupations()

        return self.wfn

    def _split_into_sections(self, content: str):
        """Split Molden file into logical sections."""
        # Find section headers
        section_pattern = r'\[([A-Za-z0-9_]+)\]'

        current_section = None
        current_lines = []

        for line in content.splitlines():
            line = line.strip()

            # Check for section header
            section_match = re.match(section_pattern, line, re.IGNORECASE)
            if section_match:
                # Save previous section
                if current_section:
                    self.sections[current_section.upper()] = current_lines

                # Start new section
                current_section = section_match.group(1).upper()
                current_lines = []
            elif current_section and line:
                current_lines.append(line)

        # Save last section
        if current_section:
            self.sections[current_section] = current_lines

    def _parse_atoms(self):
        """Enhanced parsing of atomic coordinates with error handling."""
        atoms_section = self.sections.get('ATOMS') or self.sections.get('ATOM')

        if not atoms_section:
            warnings.warn("No [Atoms] section found in Molden file", RuntimeWarning)
            return

        # Molden atom formats:
        # Format 1: index element_symbol atomic_number x y z [additional_data...]
        # Format 2: element_symbol atomic_number x y z [additional_data...]
        # Format 3: element_symbol x y z [charge...]

        atoms_found = 0
        for line in atoms_section:
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                # Try different atom formats
                if parts[0].isdigit() and len(parts) >= 6:
                    # Format: index element_symbol atomic_number x y z
                    index = int(parts[0])
                    element = parts[1]
                    atomic_num = int(float(parts[2]))
                    x, y, z_coord = float(parts[3]), float(parts[4]), float(parts[5])
                elif parts[0].replace('.', '').isdigit() and len(parts) >= 6:
                    # Format: index element_symbol atomic_number x y z (with float index)
                    element = parts[1]
                    atomic_num = int(float(parts[2]))
                    x, y, z_coord = float(parts[3]), float(parts[4]), float(parts[5])
                elif parts[0].isalpha() and len(parts) >= 4:
                    # Format: element_symbol x y z [charge...]
                    element = parts[0]
                    atomic_num = self._get_atomic_number(element)
                    x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
                else:
                    continue

                # Validate element symbol
                if not element.isalpha() or len(element) > 2:
                    # Try to get element from atomic number
                    element = ELEMENT_NAMES[atomic_num] if atomic_num < len(ELEMENT_NAMES) else f"X{atomic_num}"

                self.wfn.add_atom(element, atomic_num, x, y, z_coord, float(atomic_num))
                atoms_found += 1

            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing atom line '{line}': {e}", RuntimeWarning)
                continue

        if atoms_found == 0:
            warnings.warn("No atoms were successfully parsed from [Atoms] section", RuntimeWarning)
        else:
            self.metadata['atoms_parsed'] = atoms_found

    def _get_atomic_number(self, element_symbol: str) -> int:
        """Get atomic number from element symbol."""
        symbol_to_number = {
            'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15,
            'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22,
            'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29,
            'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
            # Add more as needed
        }
        return symbol_to_number.get(element_symbol.upper(), 0)

    def _parse_basis(self):
        """Enhanced parsing of basis set information from [GTO] section."""
        gto_section = self.sections.get('GTO')

        if not gto_section:
            # Try alternative section names
            gto_section = self.sections.get('BASIS') or self.sections.get('BASISSET')

        if not gto_section:
            warnings.warn("No [GTO] section found in Molden file", RuntimeWarning)
            return

        for line in gto_section:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # New atom definition
            if len(parts) >= 2 and parts[0].isdigit():
                # Save previous shell if exists
                if current_atom is not None and current_shell_type is not None:
                    shell = Shell(
                        type=self._molden_to_shell_type(current_shell_type),
                        center_idx=current_atom - 1,  # Molden is 1-based
                        exponents=np.array(prim_exps),
                        coefficients=np.array([prim_coeffs]) if current_shell_type != 'SP' else np.array(prim_coeffs)
                    )
                    self.wfn.shells.append(shell)

                # Start new atom
                current_atom = int(parts[0])
                prim_exps = []
                prim_coeffs = []
                continue

            # Shell definition
            if len(parts) >= 2 and parts[0] in ['s', 'p', 'd', 'f', 'g', 'h', 'sp', 'SP']:
                # Save previous shell if exists
                if current_shell_type is not None:
                    shell = Shell(
                        type=self._molden_to_shell_type(current_shell_type),
                        center_idx=current_atom - 1,
                        exponents=np.array(prim_exps),
                        coefficients=np.array([prim_coeffs]) if current_shell_type != 'SP' else np.array(prim_coeffs)
                    )
                    self.wfn.shells.append(shell)

                current_shell_type = parts[0].upper()
                prim_exps = []
                prim_coeffs = []
                continue

            # Primitive information
            if len(parts) >= 2 and current_shell_type is not None:
                try:
                    exp = float(parts[0])
                    coeff = float(parts[1])
                    prim_exps.append(exp)

                    if current_shell_type == 'SP':
                        # SP shells have two coefficients
                        if len(prim_coeffs) == 0:
                            prim_coeffs = [[coeff], []]
                        else:
                            prim_coeffs[0].append(coeff)
                            if len(parts) > 2:
                                prim_coeffs[1].append(float(parts[2]))
                            else:
                                prim_coeffs[1].append(0.0)
                    else:
                        if len(prim_coeffs) == 0:
                            prim_coeffs = []
                        prim_coeffs.append(coeff)
                except ValueError:
                    continue

        # Save last shell
        if current_atom is not None and current_shell_type is not None:
            shell = Shell(
                type=self._molden_to_shell_type(current_shell_type),
                center_idx=current_atom - 1,
                exponents=np.array(prim_exps),
                coefficients=np.array([prim_coeffs]) if current_shell_type != 'SP' else np.array(prim_coeffs)
            )
            self.wfn.shells.append(shell)

        # Calculate number of basis functions
        self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

    def _parse_mo(self):
        """Enhanced parsing of molecular orbital information from [MO] section."""
        mo_section = self.sections.get('MO')

        if not mo_section:
            warnings.warn("No [MO] section found in Molden file", RuntimeWarning)
            return

        # MO data format:
        # Orbital information lines (Ene, Occup, Spin)
        # Followed by coefficient data

        current_orbital = 0
        energies = []
        occupations = []
        coeffs = []
        mo_coeffs_list = []

        for line in mo_section:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check for orbital header
            if 'Ene=' in line or 'Occup=' in line or 'Spin=' in line:
                # Extract energy and occupation
                energy_match = re.search(r'Ene\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                occup_match = re.search(r'Occup\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)

                if energy_match:
                    energies.append(float(energy_match.group(1)))
                if occup_match:
                    occupations.append(float(occup_match.group(1)))
                else:
                    occupations.append(2.0)  # Default occupation
                continue

            # Check for coefficient line
            if re.match(r'^\s*\d+\s+', line):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        coeff = float(parts[1])
                        coeffs.append(coeff)
                except ValueError:
                    continue
            elif line == '' and coeffs:  # End of MO
                mo_coeffs_list.append(np.array(coeffs))
                coeffs = []
                current_orbital += 1

        # Add the last MO if exists
        if coeffs:
            mo_coeffs_list.append(np.array(coeffs))

        if mo_coeffs_list:
            self.wfn.energies = np.array(energies) if energies else np.zeros(len(mo_coeffs_list))
            self.wfn.coefficients = np.array(mo_coeffs_list)

            # Set occupations if available
            if occupations:
                self.wfn.occupations = np.array(occupations)
            self.metadata['mo_coefficients_parsed'] = True
            self.metadata['num_mos_parsed'] = len(mo_coeffs_list)
        mo_match = re.search(r'\[MO\].*?\n(.*?)(?=\[|\Z)', content, re.DOTALL | re.IGNORECASE)
        if not mo_match:
            return  # No MO section, that's OK for some files

        mo_lines = mo_match.group(1).strip().split('\n')
        current_mo = 0
        energies = []
        occupations = []
        coeffs = []
        mo_coeffs_list = []

        for line in mo_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('Ene='):
                energies.append(float(line.split('=')[1]))
            elif line.startswith('Occup='):
                occupations.append(float(line.split('=')[1]))
            elif line.startswith('Sym='):
                # Symmetry information, can be ignored for now
                pass
            elif re.match(r'^\s*\d+\s+', line):  # MO coefficient line
                parts = line.split()
                if len(parts) >= 2:
                    coeff = float(parts[1])
                    coeffs.append(coeff)
            elif line == '' and coeffs:  # End of MO
                mo_coeffs_list.append(np.array(coeffs))
                coeffs = []
                current_mo += 1

        # Add the last MO if exists
        if coeffs:
            mo_coeffs_list.append(np.array(coeffs))

        if mo_coeffs_list:
            self.wfn.energies = np.array(energies) if energies else np.zeros(len(mo_coeffs_list))
            self.wfn.coefficients = np.array(mo_coeffs_list)

            # Set occupations if available
            if occupations:
                self.wfn.occupations = np.array(occupations)

    def _molden_to_shell_type(self, molden_type: str) -> int:
        """Convert Molden shell type to internal representation."""
        type_mapping = {
            'S': 0,
            'P': 1,
            'SP': -1,  # Special case for combined SP shells
            'D': 2,
            'F': 3,
            'G': 4,
            'H': 5
        }
        return type_mapping.get(molden_type.upper(), 0)

    def _shell_num_functions(self, shell_type: int) -> int:
        """Return number of basis functions for a given shell type."""
        # Spherical basis functions
        func_counts = {
            0: 1,  # S
            1: 3,  # P
            -1: 4, # SP (1 S + 3 P)
            2: 5,  # D (spherical)
            3: 7,  # F (spherical)
            4: 9,  # G (spherical)
            5: 11  # H (spherical)
        }
        return func_counts.get(shell_type, 1)

    def _validate_parsed_data(self):
        """Validate parsed data for consistency."""
        if self.wfn.num_atoms == 0:
            raise ValueError("No atoms were parsed from the Molden file")

        if len(self.wfn.shells) == 0:
            warnings.warn("No basis shells were parsed from the Molden file", RuntimeWarning)

        # Calculate number of basis functions
        self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

        # Validate MO coefficients
        if hasattr(self.wfn, 'coefficients') and self.wfn.coefficients.size > 0:
            if self.wfn.coefficients.shape[1] != self.wfn.num_basis:
                warnings.warn(f"MO coefficients shape mismatch: {self.wfn.coefficients.shape}, expected (nmo, {self.wfn.num_basis})", RuntimeWarning)

        # Set validation metadata
        self.metadata['validation_passed'] = True
        self.metadata['validation_timestamp'] = np.datetime64('now')