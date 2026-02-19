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
        # Note: _parse_mo_coefficients_wfn() is called internally by _parse_basis_and_mo()

        # Validate and finalize
        self._validate_parsed_data()
        # Don't infer occupations since we already parsed them from WFN file
        # self.wfn._infer_occupations()

        return self.wfn

    def _parse_header(self):
        """Enhanced parsing of header information with error handling."""
        if len(self.lines) < 2:
            raise ValueError("WFN file is too short - needs at least header and atom count")

        # First line contains title
        self.wfn.title = self.lines[0].strip()
        if not self.wfn.title:
            self.wfn.title = "WFN calculation"

        # Second line contains orbital information in Gaussian WFN format:
        # GAUSSIAN N MOL ORBITALS P PRIMITIVES N NUCLEI
        header_line = self.lines[1].strip()

        # Parse Gaussian format: "GAUSSIAN 28 MOL ORBITALS 34 PRIMITIVES 2 NUCLEI"
        if "GAUSSIAN" in header_line and "ORBITALS" in header_line:
            try:
                parts = header_line.split()
                # Find positions of keywords
                orbitals_idx = parts.index("ORBITALS")
                primitives_idx = parts.index("PRIMITIVES")
                nuclei_idx = parts.index("NUCLEI")

                # The number before "ORBITALS" might have "MOL" before it
                # So we need to look 2 positions back if "MOL" is there
                if orbitals_idx >= 2 and parts[orbitals_idx - 1] == "MOL":
                    self.wfn.num_mos = int(parts[orbitals_idx - 2])
                else:
                    self.wfn.num_mos = int(parts[orbitals_idx - 1])

                self.wfn.num_primitives = int(parts[primitives_idx - 1])
                num_nuclei = int(parts[nuclei_idx - 1])

                # Store nuclei count for validation
                self.metadata['num_nuclei'] = num_nuclei
                self.metadata['header_parsed'] = True
            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing Gaussian header line '{header_line}': {e}", RuntimeWarning)
                self._set_default_header_values()
        else:
            # Try to parse as simple format: NMO NPRIMITIVES NELECTRONS MULTIPLICITY
            header_parts = header_line.split()
            if len(header_parts) >= 4 and all(self._is_float_or_int(p) for p in header_parts[:4]):
                try:
                    self.wfn.num_mos = int(float(header_parts[0]))
                    self.wfn.num_primitives = int(float(header_parts[1]))
                    self.wfn.num_electrons = int(float(header_parts[2]))
                    self.wfn.multiplicity = int(float(header_parts[3]))
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
        # WFN format: ELEMENT index (CENTRE n) x y z CHARGE = charge
        # Example: H    1    (CENTRE  1)   0.00000000  0.00000000  0.70160240  CHARGE =  1.0
        atom_pattern = re.compile(
            r'^([A-Z][a-z]?)\s+\d+\s+\(CENTRE\s+\d+\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+CHARGE\s*=\s*(\d+\.\d+)$'
        )

        atoms_found = 0
        total_nuclear_charge = 0.0
        for i in range(atom_start, len(self.lines)):
            line = self.lines[i].strip()
            if not line or line.upper().startswith(('CENTRE ASSIGNMENTS', 'TYPE ASSIGNMENTS', 'EXPONENTS', 'MO', 'END DATA')):
                break

            # Try WFN atom pattern
            match = atom_pattern.match(line)
            if match:
                try:
                    element = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    charge = float(match.group(5))
                    atomic_num = int(charge)

                    # Add atom
                    self.wfn.add_atom(element, atomic_num, x, y, z, charge)
                    atoms_found += 1
                    total_nuclear_charge += charge
                except (ValueError, IndexError) as e:
                    warnings.warn(f"Error parsing atom line '{line}': {e}", RuntimeWarning)
                    continue

        if atoms_found == 0:
            warnings.warn("No atoms found in WFN file", RuntimeWarning)
        else:
            self.metadata['atoms_parsed'] = atoms_found

        # Calculate electron count from nuclear charges and molecular charge
        # For WFN files, the header doesn't contain electron count directly
        # We calculate: num_electrons = total_nuclear_charge - molecular_charge
        # If molecular charge is not specified (default 0), then num_electrons = total_nuclear_charge
        if total_nuclear_charge > 0:
            self.wfn.num_electrons = total_nuclear_charge - self.wfn.charge
            self.metadata['electrons_calculated_from_atoms'] = True

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

    def _parse_mo_coefficients_wfn(self):
        """
        Parse MO coefficients from WFN format.

        WFN format has sections like:
        MO    1     MO 0.0        OCC NO =    1.9643096  ORB. ENERGY =    0.000000
          0.48513550D-01  0.87496186D-01  0.12901307D+00 ...
        """
        # Find all MO sections
        mo_sections = []
        current_mo = None

        for line in self.lines:
            line_upper = line.upper().strip()

            # Stop at END DATA
            if line_upper.startswith('END DATA'):
                break

            # Check for MO header line FIRST (before parsing coefficients)
            # Format: MO    1     MO 0.0        OCC NO =    X  ORB. ENERGY =    Y
            if line_upper.startswith('MO') and 'OCC NO' in line_upper:
                # Save previous MO if exists
                if current_mo is not None and current_mo['coefficients']:
                    mo_sections.append(current_mo)

                # Parse new MO header
                # Extract occupation and energy
                occ_match = re.search(r'OCC\s+NO\s*=\s*([-+]?\d*\.?\d+(?:[ED][-+]?\d+)?)', line_upper)
                energy_match = re.search(r'ORB\.\s*ENERGY\s*=\s*([-+]?\d*\.?\d+(?:[ED][-+]?\d+)?)', line_upper)

                occupation = float(occ_match.group(1).replace('D', 'E')) if occ_match else 0.0
                energy = float(energy_match.group(1).replace('D', 'E')) if energy_match else 0.0

                current_mo = {
                    'occupation': occupation,
                    'energy': energy,
                    'coefficients': []
                }

            # Parse coefficient lines
            # Only parse if we're in an MO section AND the line doesn't start with 'MO'
            elif current_mo is not None and line and not line_upper.startswith('MO'):
                # Try to parse as coefficients
                parts = line.split()
                for part in parts:
                    try:
                        # Handle FORTRAN D notation
                        coeff_str = part.upper().replace('D', 'E')
                        coeff = float(coeff_str)
                        current_mo['coefficients'].append(coeff)
                    except ValueError:
                        pass

        # Add the last MO
        if current_mo is not None and current_mo['coefficients']:
            mo_sections.append(current_mo)

        if not mo_sections:
            warnings.warn("No MO coefficients found in WFN file", RuntimeWarning)
            return

        # Extract occupations, energies, and coefficients
        occupations = []
        energies = []
        coefficients_list = []

        for mo in mo_sections:
            occupations.append(mo['occupation'])
            energies.append(mo['energy'])
            coefficients_list.append(mo['coefficients'])

        # Use num_basis as the expected number of coefficients
        expected_coeffs = self.wfn.num_basis

        # Ensure all MOs have the expected number of coefficients
        if coefficients_list and expected_coeffs > 0:
            normalized_coeffs = []
            for coeffs in coefficients_list:
                if len(coeffs) < expected_coeffs:
                    # Pad with zeros
                    padded = list(coeffs) + [0.0] * (expected_coeffs - len(coeffs))
                    normalized_coeffs.append(padded)
                else:
                    # Truncate to expected length
                    normalized_coeffs.append(coeffs[:expected_coeffs])

            self.wfn.coefficients = np.array(normalized_coeffs)
            self.wfn.occupations = np.array(occupations)
            self.wfn.energies = np.array(energies)

            # Update num_mos from actual parsed data
            self.wfn.num_mos = len(normalized_coeffs)

            self.metadata['mo_coefficients_parsed'] = True
            self.metadata['num_mos_parsed'] = len(normalized_coeffs)
            self.metadata['occupations_parsed'] = True
            self.metadata['energies_parsed'] = True

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
        """
        Parse basis set and molecular orbital coefficients from WFN format.

        WFN format structure:
        1. CENTRE ASSIGNMENTS - which atom each basis function belongs to
        2. TYPE ASSIGNMENTS - shell type for each basis function (1=S, 2=P, 3=D, 4=F, etc.)
        3. EXPONENTS - exponent for each primitive
        4. MO sections with coefficients
        """
        # Find all sections
        centre_assignments = []
        type_assignments = []
        exponents = []

        # Parse sections
        for line in lines:
            line_upper = line.upper().strip()

            # Parse CENTRE ASSIGNMENTS
            if line_upper.startswith('CENTRE ASSIGNMENTS'):
                parts = line.split()
                # Skip "CENTRE ASSIGNMENTS" prefix, get the numbers
                for part in parts[2:]:
                    try:
                        centre_assignments.append(int(part) - 1)  # Convert to 0-based
                    except ValueError:
                        pass

            # Parse TYPE ASSIGNMENTS
            elif line_upper.startswith('TYPE ASSIGNMENTS'):
                parts = line.split()
                # Skip "TYPE ASSIGNMENTS" prefix, get the numbers
                for part in parts[2:]:
                    try:
                        type_assignments.append(int(part))
                    except ValueError:
                        pass

            # Parse EXPONENTS
            elif line_upper.startswith('EXPONENTS'):
                parts = line.split()
                # Skip "EXPONENTS" prefix, get the numbers
                for part in parts[1:]:
                    try:
                        # Handle FORTRAN notation like 0.3387000D+02
                        exp_str = part.upper().replace('D', 'E')
                        exponents.append(float(exp_str))
                    except ValueError:
                        pass

        # Store parsed data
        self.metadata['centre_assignments'] = centre_assignments
        self.metadata['type_assignments'] = type_assignments
        self.metadata['exponents'] = exponents

        # Validate
        if not centre_assignments or not type_assignments:
            warnings.warn("Could not parse CENTRE or TYPE assignments", RuntimeWarning)
            return

        if len(centre_assignments) != len(type_assignments):
            warnings.warn(f"Mismatch between CENTRE assignments ({len(centre_assignments)}) and TYPE assignments ({len(type_assignments)})", RuntimeWarning)
            # Use the minimum length
            num_basis = min(len(centre_assignments), len(type_assignments))
        else:
            num_basis = len(centre_assignments)

        self.wfn.num_basis = num_basis

        # Group basis functions into shells by atom and type
        # In WFN format, each entry in TYPE ASSIGNMENTS is a single primitive/basis function
        # We need to group consecutive primitives of the same type on the same atom into shells

        shells_dict = {}  # Key: (atom_idx, shell_type), Value: list of (exponent, bf_idx)

        for i in range(num_basis):
            atom_idx = centre_assignments[i]
            shell_type = type_assignments[i]

            # Convert WFN shell type to our shell type
            # WFN: 1=S, 2=P, 3=SP, 4=D, 5=5D, 6=6D, 7=F, 8=5F, 9=6F, 10=F...
            # Our: S=0, P=1, SP=-1, D=2, F=3
            if shell_type == 1:
                our_type = 0  # S
            elif shell_type == 2:
                our_type = 1  # P
            elif shell_type == 3:
                our_type = -1  # SP shell (special case)
            elif shell_type in [4, 5, 6]:
                our_type = 2  # D (use 2 for all D types)
            elif shell_type in [7, 8, 9, 10]:
                our_type = 3  # F (use 3 for all F types)
            else:
                our_type = shell_type - 1  # Adjust for higher shells

            key = (atom_idx, our_type)

            if key not in shells_dict:
                shells_dict[key] = []

            # Get exponent for this basis function
            if i < len(exponents):
                shells_dict[key].append(exponents[i])
            else:
                shells_dict[key].append(0.0)  # Default if no exponent

        # Create Shell objects
        for (atom_idx, shell_type), shell_exponents in sorted(shells_dict.items()):
            if shell_exponents:
                # Create shell with dummy coefficients (WFN doesn't have contraction coefficients)
                # All primitives have coefficient 1.0
                shell = Shell(
                    type=shell_type,
                    center_idx=atom_idx,
                    exponents=np.array(shell_exponents),
                    coefficients=np.ones((1, len(shell_exponents)))
                )
                self.wfn.shells.append(shell)

        # Store centre_assignments in the wavefunction for accurate basis function indexing
        # This is critical for get_atomic_basis_indices() to work correctly
        self.wfn._centre_assignments = centre_assignments

        # DON'T recalculate num_basis from shells - use the value from WFN file
        # The num_basis from WFN file (based on centre_assignments) is the authoritative value
        # Recalculating from shells can cause mismatches because shell grouping may not match
        # the actual number of basis functions in the WFN file
        # self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

        # Set overlap matrix to identity as fallback (with correct size)
        if self.wfn.num_basis > 0:
            self.wfn.overlap_matrix = np.eye(self.wfn.num_basis)

        # Parse MO coefficients
        self._parse_mo_coefficients_wfn()

    def _shell_num_functions(self, shell_type: int) -> int:
        """Return number of basis functions for a given shell type."""
        # Use Cartesian coordinates (consistent with evaluate_basis)
        func_counts = {
            -1: 4,  # SP (1 S + 3 P)
            0: 1,   # S
            1: 3,   # P
            2: 6,   # D (Cartesian: xx, yy, zz, xy, xz, yz)
            3: 10,  # F (Cartesian: 10 functions)
        }
        return func_counts.get(shell_type, 1)
