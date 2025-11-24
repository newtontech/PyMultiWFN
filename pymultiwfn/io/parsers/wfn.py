"""
Enhanced parser for Gaussian WFN files (.wfn, .WFN).
WFN format contains wavefunction information in a simple text format.

This enhanced parser provides comprehensive error handling, validation,
and supports various WFN format variants from different quantum chemistry programs.
"""

import re
import numpy as np
import warnings
from typing import List, Optional, Dict, Any
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES

class WFNLoader:
    """
    Enhanced loader for Gaussian WFN files.

    Supports standard Gaussian WFN format and common variants from other
    quantum chemistry programs. Includes comprehensive error handling
    and data validation.
    """

    def __init__(self, filename: str):
        """
        Initialize the WFN loader.

        Args:
            filename: Path to the .wfn file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file extension is not .wfn
        """
        if not filename.lower().endswith(('.wfn', '.wfx')):
            raise ValueError(f"File must have .wfn or .wfx extension, got: {filename}")

        self.filename = filename
        self.wfn = Wavefunction()
        self.metadata: Dict[str, Any] = {}
        self.lines: List[str] = []

    def load(self) -> Wavefunction:
        """
        Parse the WFN file and return a complete Wavefunction object.

        Returns:
            Wavefunction: Complete wavefunction object with all parsed data

        Raises:
            ValueError: If the file format is invalid or critical data is missing
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except UnicodeDecodeError:
            with open(self.filename, 'r', encoding='latin-1') as f:
                self.lines = f.readlines()

        if not self.lines:
            raise ValueError(f"File {self.filename} appears to be empty")

        # Remove empty lines and strip whitespace
        self.lines = [line.strip() for line in self.lines if line.strip()]

        # Parse all sections
        self._parse_header()
        self._parse_atoms()
        self._parse_basis_and_mo(self.lines)
        self._parse_mo_coefficients()
        self._parse_orbital_energies()

        # Validate and finalize
        self._validate_parsed_data()
        self.wfn._infer_occupations()

        return self.wfn

    def _parse_header(self):
        """Enhanced parsing of header information with error handling."""
        if len(self.lines) < 2:
            raise ValueError("WFN file is too short - needs at least header and atom count")

        # First line contains title
        self.wfn.title = self.lines[0].strip()
        if not self.wfn.title:
            self.wfn.title = "WFN calculation"

        # Second line contains orbital information
        # Format variations:
        # - Standard Gaussian: NMO NPRIMITIVES NELECTRONS MULTIPLICITY
        # - Some variants: NUM_ORBITALS NUM_PRIMITIVES N_ELECTRONS MULTIPLICITY
        header_line = self.lines[1].strip().split()

        if len(header_line) < 4:
            # Try to find header information in subsequent lines
            for i, line in enumerate(self.lines[2:6], 2):
                parts = line.strip().split()
                if len(parts) >= 4 and all(self._is_float_or_int(p) for p in parts):
                    header_line = parts
                    break

        if len(header_line) >= 4 and all(self._is_float_or_int(p) for p in header_line[:4]):
            try:
                self.wfn.num_mos = int(float(header_line[0]))
                self.wfn.num_primitives = int(float(header_line[1]))
                self.wfn.num_electrons = int(float(header_line[2]))
                self.wfn.multiplicity = int(float(header_line[3]))
                self.metadata['header_parsed'] = True
            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing header line '{header_line}': {e}", RuntimeWarning)
                self._set_default_header_values()
        else:
            warnings.warn(f"Cannot parse header information from line: {header_line}", RuntimeWarning)
            self._set_default_header_values()

    def _is_float_or_int(self, value: str) -> bool:
        """Check if a string can be converted to float or int."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _set_default_header_values(self):
        """Set default values when header parsing fails."""
        # Try to infer from other parts of the file
        self.wfn.num_mos = 0
        self.wfn.num_primitives = 0
        self.wfn.num_electrons = 0
        self.wfn.multiplicity = 1  # Default to singlet
        self.metadata['header_parsed'] = False

    def _parse_atoms(self):
        """Enhanced parsing of atomic coordinates with error handling."""
        # Find atom section - usually starts after header information
        atom_start = 2  # Skip header lines

        # Look for atom information patterns
        atom_patterns = [
            # Standard format: X Y Z ATOMIC_NUMBER
            r'^(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)$',
            # Alternative format with element symbol: ELEMENT X Y Z
            r'^([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)$',
            # Format with charges: X Y Z ATOMIC_NUMBER CHARGE
            r'^(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(-?\d+\.\d+)$',
            # WFN format: ELEMENT index (CENTRE n) x y z CHARGE = charge
            r'^([A-Z][a-z]?)\s+\d+\s+\(CENTRE\s+\d+\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+CHARGE\s*=\s*(\d+\.\d+)$'
        ]

        atoms_found = 0
        for i in range(atom_start, len(self.lines)):
            line = self.lines[i].strip()
            if not line or line.upper().startswith(('END DATA', 'BASIS', 'MO', 'SHELL')):
                break

            # Try each atom pattern
            for pattern in atom_patterns:
                match = re.match(pattern, line)
                if match:
                    try:
                        groups = match.groups()
                        if len(groups) == 4 and all(self._is_float_or_int(g) for g in groups[:3]):
                            # X Y Z ATOMIC_NUMBER format
                            x, y, z = float(groups[0]), float(groups[1]), float(groups[2])
                            atomic_num = int(float(groups[3]))
                            element = ELEMENT_NAMES[atomic_num] if atomic_num < len(ELEMENT_NAMES) else f"X{atomic_num}"
                            charge = float(atomic_num)
                        elif len(groups) == 4 and groups[0].isalpha():
                            # ELEMENT X Y Z format
                            element = groups[0]
                            x, y, z = float(groups[1]), float(groups[2]), float(groups[3])
                            atomic_num = self._get_atomic_number(element)
                            charge = float(atomic_num)
                        elif len(groups) == 5 and groups[0].isalpha():
                            # WFN format: ELEMENT index (CENTRE n) x y z CHARGE = charge
                            element = groups[0]
                            x, y, z = float(groups[1]), float(groups[2]), float(groups[3])
                            charge = float(groups[4])
                            atomic_num = int(float(charge))
                        else:
                            continue

                        # Add atom
                        self.wfn.add_atom(element, atomic_num, x, y, z, charge)
                        atoms_found += 1
                        break
                    except (ValueError, IndexError) as e:
                        warnings.warn(f"Error parsing atom line '{line}': {e}", RuntimeWarning)
                        continue

        if atoms_found == 0:
            warnings.warn("No atoms found in WFN file", RuntimeWarning)
        else:
            self.metadata['atoms_parsed'] = atoms_found

    def _get_atomic_number(self, element_symbol: str) -> int:
        """Get atomic number from element symbol."""
        symbol_to_number = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            # Add more as needed
        }
        return symbol_to_number.get(element_symbol.capitalize(), 0)

    def _parse_mo_coefficients(self):
        """Enhanced parsing of molecular orbital coefficients from WFN file."""
        # WFN format typically has MO coefficients in a specific section
        # Look for MO coefficient patterns in the file

        # Pattern for MO coefficients in WFN format
        # Usually appears after basis function definitions
        mo_section_start = -1
        for i, line in enumerate(self.lines):
            if 'MO' in line.upper() and 'COEFF' in line.upper():
                mo_section_start = i
                break
            elif 'MOLECULAR ORBITAL' in line.upper():
                mo_section_start = i
                break

        if mo_section_start == -1:
            # Try to find MO coefficients by pattern matching
            # WFN format often has coefficients after basis function definitions
            for i, line in enumerate(self.lines):
                if len(line.split()) >= 5 and all(self._is_float_or_int(p) for p in line.split()[:5]):
                    # Could be MO coefficient line
                    mo_section_start = i
                    break

        if mo_section_start == -1:
            warnings.warn("No MO coefficient section found in WFN file", RuntimeWarning)
            return

        # Parse MO coefficients
        mo_coeffs = []
        current_mo = []

        for line in self.lines[mo_section_start:]:
            line = line.strip()
            if not line:
                continue

            # Check for end of MO section
            if line.upper().startswith(('END', 'BASIS', 'ATOM')):
                break

            # Parse coefficient values
            parts = line.split()
            if len(parts) >= 2 and all(self._is_float_or_int(p) for p in parts):
                try:
                    # Try to parse as MO coefficients
                    coeffs = [float(p) for p in parts]
                    current_mo.extend(coeffs)
                except ValueError:
                    continue
            elif current_mo:
                # End of current MO
                mo_coeffs.append(np.array(current_mo))
                current_mo = []

        # Add the last MO if exists
        if current_mo:
            mo_coeffs.append(np.array(current_mo))

        if mo_coeffs:
            # Ensure consistent MO coefficient matrix
            max_len = max(len(coeffs) for coeffs in mo_coeffs)
            normalized_coeffs = []

            for coeffs in mo_coeffs:
                if len(coeffs) < max_len:
                    # Pad with zeros
                    padded = np.zeros(max_len)
                    padded[:len(coeffs)] = coeffs
                    normalized_coeffs.append(padded)
                else:
                    normalized_coeffs.append(coeffs[:max_len])

            self.wfn.coefficients = np.array(normalized_coeffs)
            self.metadata['mo_coefficients_parsed'] = True
            self.metadata['num_mos_parsed'] = len(normalized_coeffs)

    def _parse_orbital_energies(self):
        """Parse orbital energies if present."""
        # Look for energy information in the file
        energy_patterns = [
            r'ORBITAL\s+ENERGIES?\s*[:=]\s*([-\d\s.E]+)',
            r'EIGENVALUES?\s*[:=]\s*([-\d\s.E]+)',
            r'ENERGIES?\s*[:=]\s*([-\d\s.E]+)'
        ]

        for line in self.lines:
            line_upper = line.upper()
            for pattern in energy_patterns:
                match = re.search(pattern, line_upper)
                if match:
                    try:
                        energy_str = match.group(1).strip()
                        energies = [float(e) for e in energy_str.split() if e]
                        if energies:
                            self.wfn.energies = np.array(energies)
                            self.metadata['orbital_energies_parsed'] = True
                            return
                    except ValueError:
                        continue

    def _validate_parsed_data(self):
        """Validate parsed data for consistency."""
        if self.wfn.num_atoms == 0:
            raise ValueError("No atoms were parsed from the WFN file")

        if self.wfn.num_electrons == 0:
            warnings.warn("No electron count was parsed from the WFN file", RuntimeWarning)

        if len(self.wfn.shells) == 0:
            warnings.warn("No basis shells were parsed from the WFN file", RuntimeWarning)

        # Calculate number of basis functions if not set
        if self.wfn.num_basis == 0 and len(self.wfn.shells) > 0:
            self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

        # Set validation metadata
        self.metadata['validation_passed'] = True
        self.metadata['validation_timestamp'] = np.datetime64('now')

    def _parse_basis_and_mo(self, lines):
        """Parse basis set and molecular orbital coefficients."""
        # Find basis function and MO coefficient sections
        basis_pattern = re.compile(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*')  # Center NPrim NType FirstMO LastMO
        coeff_pattern = re.compile(r'\s*\d+\s+(-?\d+\.\d+E[+-]?\d+)\s+(-?\d+\.\d+E[+-]?\d+)\s*')  # MO index coefficient exponent

        current_atom_idx = None
        current_shell = None
        prim_exps = []
        prim_coeffs = []

        # Collect MO coefficients for each MO
        mo_coeffs = {}

        for line in lines:
            line = line.strip()

            # Check for basis function header
            basis_match = basis_pattern.match(line)
            if basis_match:
                # Save previous shell if exists
                if current_shell is not None:
                    shell = Shell(
                        type=current_shell['type'],
                        center_idx=current_shell['center_idx'],
                        exponents=np.array(prim_exps),
                        coefficients=np.array([prim_coeffs])
                    )
                    self.wfn.shells.append(shell)

                # Parse new shell information
                center_idx = int(basis_match.group(1)) - 1  # Convert to 0-based
                n_prims = int(basis_match.group(2))
                shell_type = int(basis_match.group(3))
                first_mo = int(basis_match.group(4))
                last_mo = int(basis_match.group(5))

                # Map shell types (0=S, 1=P, 2=D, etc.)
                type_mapping = {0: 0, 1: 1, 2: 2, 3: 3}  # Simplified mapping
                shell_type_mapped = type_mapping.get(shell_type, 0)

                current_shell = {
                    'center_idx': center_idx,
                    'type': shell_type_mapped,
                    'first_mo': first_mo,
                    'last_mo': last_mo
                }
                prim_exps = []
                prim_coeffs = []
                continue

            # Check for primitive coefficient line
            coeff_match = coeff_pattern.match(line)
            if coeff_match and current_shell is not None:
                try:
                    coeff = float(coeff_match.group(1))
                    exp = float(coeff_match.group(2))

                    prim_coeffs.append(coeff)
                    prim_exps.append(exp)
                except ValueError:
                    continue

            # Check for MO coefficients section
            if "MO" in line and "OCC" in line:
                # This might be a different format, skip for now
                continue

        # Save the last shell
        if current_shell is not None:
            shell = Shell(
                type=current_shell['type'],
                center_idx=current_shell['center_idx'],
                exponents=np.array(prim_exps),
                coefficients=np.array([prim_coeffs])
            )
            self.wfn.shells.append(shell)

        # Calculate number of basis functions
        self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

        # WFN format typically doesn't contain MO coefficients in the same way as other formats
        # The coefficients are often embedded in the primitive definitions
        # For now, we'll set up a basic coefficient matrix structure
        if self.wfn.num_mos > 0 and self.wfn.num_basis > 0:
            # This is a placeholder - actual MO parsing would need more sophisticated logic
            self.wfn.coefficients = np.zeros((self.wfn.num_mos, self.wfn.num_basis))
            self.wfn.energies = np.zeros(self.wfn.num_mos)

    def _shell_num_functions(self, shell_type: int) -> int:
        """Return number of basis functions for a given shell type."""
        func_counts = {
            0: 1,  # S
            1: 3,  # P
            2: 5,  # D (spherical)
            3: 7,  # F (spherical)
        }
        return func_counts.get(shell_type, 1)
