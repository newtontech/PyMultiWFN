"""
Enhanced parser for Gaussian WFN files (.wfn, .WFN).
WFN format contains wavefunction information in a simple text format.

This enhanced parser provides comprehensive error handling, validation,
and supports various WFN format variants from different quantum chemistry programs.
"""

import logging
import re
import warnings
from typing import Any, Dict, List

import numpy as np

from pymultiwfn.core.data import Shell, Wavefunction
from pymultiwfn.core.definitions import get_atomic_number

logger = logging.getLogger(__name__)


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
        if not filename.lower().endswith((".wfn", ".wfx")):
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
            with open(self.filename, "r", encoding="utf-8") as f:
                self.lines = f.readlines()
        except UnicodeDecodeError:
            with open(self.filename, "r", encoding="latin-1") as f:
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
            raise ValueError(
                "WFN file is too short - needs at least header and atom count"
            )

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
                self.metadata["num_nuclei"] = num_nuclei
                self.metadata["header_parsed"] = True
            except (ValueError, IndexError) as e:
                warnings.warn(
                    f"Error parsing Gaussian header line '{header_line}': {e}",
                    RuntimeWarning,
                )
                self._set_default_header_values()
        else:
            # Try to parse as simple format: NMO NPRIMITIVES NELECTRONS MULTIPLICITY
            header_parts = header_line.split()
            if len(header_parts) >= 4 and all(
                self._is_float_or_int(p) for p in header_parts[:4]
            ):
                try:
                    self.wfn.num_mos = int(float(header_parts[0]))
                    self.wfn.num_primitives = int(float(header_parts[1]))
                    self.wfn.num_electrons = int(float(header_parts[2]))
                    self.wfn.multiplicity = int(float(header_parts[3]))
                    self.metadata["header_parsed"] = True
                except (ValueError, IndexError) as e:
                    warnings.warn(
                        f"Error parsing header line '{header_line}': {e}",
                        RuntimeWarning,
                    )
                    self._set_default_header_values()
            else:
                warnings.warn(
                    f"Cannot parse header information from line: {header_line}",
                    RuntimeWarning,
                )
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
        self.metadata["header_parsed"] = False

    def _parse_atoms(self):
        """Enhanced parsing of atomic coordinates with error handling."""
        # Find atom section - usually starts after header information
        atom_start = 2  # Skip header lines

        # Look for atom information patterns
        # WFN format: ELEMENT index (CENTRE n) x y z CHARGE = charge
        # Example: H    1    (CENTRE  1)   0.00000000  0.00000000  0.70160240  CHARGE =  1.0
        atom_pattern = re.compile(
            r"^([A-Z][a-z]?)\s+\d+\s+\(CENTRE\s+\d+\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+CHARGE\s*=\s*(\d+\.\d+)$"
        )

        atoms_found = 0
        total_nuclear_charge = 0.0
        for i in range(atom_start, len(self.lines)):
            line = self.lines[i].strip()
            if not line or line.upper().startswith(
                (
                    "CENTRE ASSIGNMENTS",
                    "TYPE ASSIGNMENTS",
                    "EXPONENTS",
                    "MO",
                    "END DATA",
                )
            ):
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
                    warnings.warn(
                        f"Error parsing atom line '{line}': {e}", RuntimeWarning
                    )
                    continue

        if atoms_found == 0:
            warnings.warn("No atoms found in WFN file", RuntimeWarning)
        else:
            self.metadata["atoms_parsed"] = atoms_found

        # Calculate electron count from nuclear charges and molecular charge
        # For WFN files, the header doesn't contain electron count directly
        # We calculate: num_electrons = total_nuclear_charge - molecular_charge
        # If molecular charge is not specified (default 0), then num_electrons = total_nuclear_charge
        if total_nuclear_charge > 0:
            self.wfn.num_electrons = total_nuclear_charge - self.wfn.charge
            self.metadata["electrons_calculated_from_atoms"] = True

    def _get_atomic_number(self, element_symbol: str) -> int:
        """Get atomic number from element symbol."""
        return get_atomic_number(element_symbol)

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
            if line_upper.startswith("END DATA"):
                break

            # Check for MO header line FIRST (before parsing coefficients)
            # Format: MO    1     MO 0.0        OCC NO =    X  ORB. ENERGY =    Y
            if line_upper.startswith("MO") and "OCC NO" in line_upper:
                # Save previous MO if exists
                if current_mo is not None and current_mo["coefficients"]:
                    mo_sections.append(current_mo)

                # Parse new MO header
                # Extract occupation and energy
                occ_match = re.search(
                    r"OCC\s+NO\s*=\s*([-+]?\d*\.?\d+(?:[ED][-+]?\d+)?)", line_upper
                )
                energy_match = re.search(
                    r"ORB\.\s*ENERGY\s*=\s*([-+]?\d*\.?\d+(?:[ED][-+]?\d+)?)",
                    line_upper,
                )

                occupation = (
                    float(occ_match.group(1).replace("D", "E")) if occ_match else 0.0
                )
                energy = (
                    float(energy_match.group(1).replace("D", "E"))
                    if energy_match
                    else 0.0
                )

                current_mo = {
                    "occupation": occupation,
                    "energy": energy,
                    "coefficients": [],
                }

            # Parse coefficient lines
            # Only parse if we're in an MO section AND the line doesn't start with 'MO'
            elif current_mo is not None and line and not line_upper.startswith("MO"):
                # Try to parse as coefficients
                parts = line.split()
                for part in parts:
                    try:
                        # Handle FORTRAN D notation
                        coeff_str = part.upper().replace("D", "E")
                        coeff = float(coeff_str)
                        current_mo["coefficients"].append(coeff)
                    except ValueError:
                        pass

        # Add the last MO
        if current_mo is not None and current_mo["coefficients"]:
            mo_sections.append(current_mo)

        if not mo_sections:
            warnings.warn("No MO coefficients found in WFN file", RuntimeWarning)
            return

        # Extract occupations, energies, and coefficients
        occupations = []
        energies = []
        coefficients_list = []

        for mo in mo_sections:
            occupations.append(mo["occupation"])
            energies.append(mo["energy"])
            coefficients_list.append(mo["coefficients"])

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

            self.metadata["mo_coefficients_parsed"] = True
            self.metadata["num_mos_parsed"] = len(normalized_coeffs)
            self.metadata["occupations_parsed"] = True
            self.metadata["energies_parsed"] = True

    def _validate_parsed_data(self):
        """Validate parsed data for consistency."""
        if self.wfn.num_atoms == 0:
            raise ValueError("No atoms were parsed from the WFN file")

        if self.wfn.num_electrons == 0:
            warnings.warn(
                "No electron count was parsed from the WFN file", RuntimeWarning
            )

        if len(self.wfn.shells) == 0:
            warnings.warn(
                "No basis shells were parsed from the WFN file", RuntimeWarning
            )

        # Calculate number of basis functions if not set
        if self.wfn.num_basis == 0 and len(self.wfn.shells) > 0:
            self.wfn.num_basis = sum(
                self._shell_num_functions(shell.type) for shell in self.wfn.shells
            )

        # Set validation metadata
        self.metadata["validation_passed"] = True
        self.metadata["validation_timestamp"] = np.datetime64("now")

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
            if line_upper.startswith("CENTRE ASSIGNMENTS"):
                parts = line.split()
                # Skip "CENTRE ASSIGNMENTS" prefix, get the numbers
                for part in parts[2:]:
                    try:
                        centre_assignments.append(int(part) - 1)  # Convert to 0-based
                    except ValueError:
                        pass

            # Parse TYPE ASSIGNMENTS
            elif line_upper.startswith("TYPE ASSIGNMENTS"):
                parts = line.split()
                # Skip "TYPE ASSIGNMENTS" prefix, get the numbers
                for part in parts[2:]:
                    try:
                        type_assignments.append(int(part))
                    except ValueError:
                        pass

            # Parse EXPONENTS
            elif line_upper.startswith("EXPONENTS"):
                parts = line.split()
                # Skip "EXPONENTS" prefix, get the numbers
                for part in parts[1:]:
                    try:
                        # Handle FORTRAN notation like 0.3387000D+02
                        exp_str = part.upper().replace("D", "E")
                        exponents.append(float(exp_str))
                    except ValueError:
                        pass

        # Store parsed data
        self.metadata["centre_assignments"] = centre_assignments
        self.metadata["type_assignments"] = type_assignments
        self.metadata["exponents"] = exponents

        # Validate
        if not centre_assignments or not type_assignments:
            warnings.warn("Could not parse CENTRE or TYPE assignments", RuntimeWarning)
            return

        if len(centre_assignments) != len(type_assignments):
            warnings.warn(
                f"Mismatch between CENTRE assignments ({len(centre_assignments)}) and TYPE assignments ({len(type_assignments)})",
                RuntimeWarning,
            )
            # Use the minimum length
            num_basis = min(len(centre_assignments), len(type_assignments))
        else:
            num_basis = len(centre_assignments)

        self.wfn.num_basis = num_basis

        # Group basis functions into shells by atom and type
        # In WFN format, each entry in TYPE ASSIGNMENTS is a single primitive/basis function
        # We need to group consecutive primitives of the same type on the same atom into shells

        shells_dict = (
            {}
        )  # Key: (atom_idx, shell_type), Value: list of (exponent, bf_idx)

        logger.debug("Basis function information (first 20 of %s):", num_basis)
        logger.debug("Index | Centre | WFN Type | GTO Type | Exponent")
        for i in range(min(20, num_basis)):
            atom_idx = centre_assignments[i]
            shell_type = type_assignments[i]
            exp_val = exponents[i] if i < len(exponents) else "N/A"
            logger.debug(
                "%5d | %6d | %8d | %7s | %s", i, atom_idx, shell_type, "?", exp_val
            )

        # Count WFN types
        wfn_type_counts = {}
        for shell_type in type_assignments:
            wfn_type_counts[shell_type] = wfn_type_counts.get(shell_type, 0) + 1
        logger.debug("WFN type counts: %s", dict(sorted(wfn_type_counts.items())))

        for i in range(num_basis):
            atom_idx = centre_assignments[i]
            shell_type = type_assignments[i]

            # Convert WFN shell type to our shell type
            # WFN: 1=S, 2=P_x, 3=P_y, 4=P_z, 5=D_xx, 6=D_yy, 7=D_zz, 8=D_xy, 9=D_xz, 10=D_yz, 11+=F...
            # Our: S=0, P=1, SP=-1, D=2, F=3
            # CRITICAL FIX: WFN type_assignments specify INDIVIDUAL basis functions, not shell types!
            # WFN types 2, 3, 4 are P_x, P_y, P_z (all P type)
            # WFN types 5-10 are D functions (all D type)
            # We should NOT map WFN types directly to shell types!

            if shell_type == 1:
                our_type = 0  # S
            elif shell_type in [2, 3, 4]:
                our_type = 1  # P (P_x, P_y, P_z all map to P shell)
            elif shell_type in [5, 6, 7, 8, 9, 10]:
                our_type = 2  # D (all D functions map to D shell)
            elif shell_type >= 11:
                our_type = 3  # F (all F functions map to F shell)
            else:
                our_type = shell_type - 1  # Fallback (should not happen)

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
                    coefficients=np.ones((1, len(shell_exponents))),
                )
                self.wfn.shells.append(shell)

        # Store centre_assignments in the wavefunction for accurate basis function indexing
        # This is critical for get_atomic_basis_indices() to work correctly
        self.wfn._centre_assignments = centre_assignments

        # Store type_assignments in the wavefunction for accurate overlap matrix calculation
        # This is critical for WFN files where shells may not accurately represent basis functions
        self.wfn._type_assignments = type_assignments

        # Store exponents for accurate basis function parameters
        self.wfn._exponents = exponents

        # DON'T recalculate num_basis from shells - use the value from WFN file
        # The num_basis from WFN file (based on centre_assignments) is the authoritative value
        # Recalculating from shells can cause mismatches because shell grouping may not match
        # the actual number of basis functions in the WFN file
        # self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

        # Calculate overlap matrix from basis set information
        # IMPORTANT: In WFN format, the MOs are typically expressed in an orthonormal basis,
        # so the overlap matrix should be the identity matrix.
        # Attempting to calculate overlap from Gaussian primitives is problematic because:
        # 1. WFN format doesn't specify primitive contraction coefficients
        # 2. The basis functions may already be orthogonalized
        # 3. MO coefficients are defined with respect to the orthonormal basis
        if self.wfn.num_basis > 0:
            # Use identity matrix for WFN format (orthonormal basis)
            self.wfn.overlap_matrix = np.eye(self.wfn.num_basis)
            logger.debug(
                "Using identity overlap matrix for WFN format: shape=%s",
                self.wfn.overlap_matrix.shape,
            )

            # Note: We're NOT normalizing the basis functions because the identity
            # overlap matrix already implies orthonormal basis functions
            # The MO coefficients and density matrices are already in the correct basis

        # Parse MO coefficients
        self._parse_mo_coefficients_wfn()

        # Normalize MO coefficients for correct density matrix calculation
        # WFN format stores unnormalized coefficients. Normalization ensures
        # the density matrix trace equals the number of electrons.
        self._normalize_mo_coefficients()

        # Use identity overlap matrix for WFN format (orthonormal basis)
        # This is a simplification that works for most bond order calculations

    def _shell_num_functions(self, shell_type: int) -> int:
        """Return number of basis functions for a given shell type."""
        # Use Cartesian coordinates (consistent with evaluate_basis)
        func_counts = {
            -1: 4,  # SP (1 S + 3 P)
            0: 1,  # S
            1: 3,  # P
            2: 6,  # D (Cartesian: xx, yy, zz, xy, xz, yz)
            3: 10,  # F (Cartesian: 10 functions)
        }
        return func_counts.get(shell_type, 1)

    def _calculate_wfn_overlap_matrix(self) -> np.ndarray:
        """
        Calculate overlap matrix using WFN-format-specific method.

        In WFN format, each entry in TYPE ASSIGNMENTS is a single basis function.
        This differs from standard Gaussian basis set formats where shells contain
        multiple basis functions (e.g., P shell has 3 functions: Px, Py, Pz).

        This method extracts basis functions directly from WFN data and uses the
        standard overlap calculation infrastructure.

        Args:
            None (uses self.wfn attributes)

        Returns:
            Overlap matrix of shape (num_basis, num_basis)

        Raises:
            ValueError: If required WFN data is missing
        """
        # Extract basis functions from WFN format data
        basis_functions = self._extract_wfn_basis_functions()

        if not basis_functions:
            raise ValueError("No basis functions extracted from WFN file")

        num_basis = len(basis_functions)

        if num_basis != self.wfn.num_basis:
            raise ValueError(
                f"Extracted {num_basis} basis functions, "
                f"but num_basis is {self.wfn.num_basis}"
            )

        # Initialize overlap matrix
        overlap_matrix = np.zeros((num_basis, num_basis))

        # Use the full overlap calculator from pymultiwfn.integrals.overlap
        # This ensures we get accurate P, D, F overlap values
        try:
            from pymultiwfn.integrals.overlap import _calculate_gto_overlap

            # Clear the cache to avoid recursion issues
            try:
                _calculate_gto_overlap.cache_clear()
            except AttributeError:
                pass  # Function might not have cache_clear method

            # Calculate all overlap integrals
            for i in range(num_basis):
                bf_i = basis_functions[i]

                for j in range(i, num_basis):
                    bf_j = basis_functions[j]

                    # Calculate overlap integral between bf_i and bf_j
                    # Each basis function has only 1 primitive, which is already
                    # included in the exponents and coefficients arrays
                    try:
                        S_ij = _calculate_gto_overlap(bf_i, bf_j, use_cache=False)
                    except RecursionError:
                        # If we hit recursion issues, use identity matrix fallback
                        raise Exception("Recursion error in overlap calculation")

                    overlap_matrix[i, j] = S_ij
                    if i != j:
                        overlap_matrix[j, i] = S_ij  # Symmetric matrix

            return overlap_matrix

        except Exception as e:
            warnings.warn(
                f"Full overlap calculation failed: {e}. "
                f"Using simple distance-based overlap calculation."
            )
            # Fallback: use simple distance-based overlap
            # This is not chemically accurate but better than identity matrix
            return self._calculate_distance_based_overlap(basis_functions)

    def _calculate_distance_based_overlap(
        self, basis_functions: List[dict]
    ) -> np.ndarray:
        """
        Calculate approximate overlap matrix as identity with minimal distance correction.

        Strategy:
        - Use identity matrix as base (basis functions are approximately orthonormal)
        - Add very small distance-based corrections for interatomic overlap
        - This preserves Mayer-Wiberg equality while acknowledging physical reality

        This is a fallback method when full GTO overlap calculation fails.

        Args:
            basis_functions: List of basis function dictionaries

        Returns:
            Overlap matrix of shape (num_basis, num_basis)
        """
        num_basis = len(basis_functions)

        # Start with identity matrix (orthonormal basis)
        overlap_matrix = np.eye(num_basis)

        # Very small correction for interatomic overlap
        alpha_distance = 0.01  # Bohr^-2 (extremely small - almost identity)

        for i in range(num_basis):
            bf_i = basis_functions[i]
            type_i = bf_i["type"]
            coords_i = np.array(bf_i["coords"])

            for j in range(i + 1, num_basis):
                bf_j = basis_functions[j]
                type_j = bf_j["type"]
                coords_j = np.array(bf_j["coords"])

                # Only add small corrections for same-type, different-center functions
                if type_i == type_j:
                    # Calculate distance
                    r2 = np.sum((coords_i - coords_j) ** 2)

                    # Small Gaussian decay correction
                    if r2 > 1e-6:  # Different centers
                        S_ij = np.exp(-alpha_distance * r2) - 1.0
                        overlap_matrix[i, j] = S_ij
                        overlap_matrix[j, i] = S_ij

        return overlap_matrix

    def _normalize_mo_coefficients(self):
        """
        Normalize MO coefficients for orthonormal basis.

        WFN format stores unnormalized MO coefficients. When using an
        orthonormal basis (identity overlap matrix), the coefficients must
        be normalized: ||C_i||^2 = 1 for each MO i.

        Normalization ensures:
        - Density matrix trace equals number of electrons
        - Correct bond order calculations
        - Proper population analysis

        For each MO i: C_i <- C_i / ||C_i||
        where ||C_i|| = sqrt(sum_u C_{ui}^2)
        """
        if self.wfn.coefficients is None:
            return

        logger.debug("Normalizing %s MO coefficients...", len(self.wfn.coefficients))

        # Normalize each MO coefficient vector
        for i in range(len(self.wfn.coefficients)):
            coeff_vector = self.wfn.coefficients[i, :]
            norm = np.sqrt(np.sum(coeff_vector**2))

            if norm > 1e-10:
                # Normalize the coefficient vector
                self.wfn.coefficients[i, :] /= norm
            else:
                # Warning: MO has near-zero norm
                warnings.warn(
                    f"MO {i} has near-zero norm ({norm:.2e}), skipping normalization",
                    RuntimeWarning,
                )

        logger.debug("MO coefficients normalized")

    def _extract_wfn_basis_functions(self) -> List[dict]:
        """
        Extract basis functions from WFN-format data.

        This method extracts basis functions directly from WFN file's
        CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS, creating
        one basis function per entry.

        For WFN format:
        - Each entry in TYPE ASSIGNMENTS is a single basis function
        - Type values represent specific basis function components:
          - Type 1 = S
          - Type 2 = Px
          - Type 3 = Py
          - Type 4 = Pz
          - Type 5-10 = D components (xx, yy, zz, xy, xz, yz)
          - Type 11-20 = F components
        - Each basis function has its own exponent and centre

        Returns:
            List of basis function dictionaries compatible with overlap calculation
        """
        if not hasattr(self.wfn, "_centre_assignments"):
            raise ValueError("WFN file missing centre assignments")
        if not hasattr(self.wfn, "_type_assignments"):
            raise ValueError("WFN file missing type assignments")
        if not hasattr(self.wfn, "_exponents"):
            raise ValueError("WFN file missing exponents")

        centre_assignments = self.wfn._centre_assignments
        type_assignments = self.wfn._type_assignments
        exponents = self.wfn._exponents

        basis_functions = []

        # Map WFN types directly to overlap calculation 'type' field
        # In overlap.py, type field uses:
        # - 0 = S
        # - 1 = Px, 2 = Py, 3 = Pz
        # - 4 = D_xx, 5 = D_yy, 6 = D_zz, 7 = D_xy, 8 = D_xz, 9 = D_yz
        # - 10-19 = F components
        wfn_type_to_overlap_type = {
            1: 0,  # S
            2: 1,  # Px
            3: 2,  # Py
            4: 3,  # Pz
            5: 4,  # D_xx
            6: 5,  # D_yy
            7: 6,  # D_zz
            8: 7,  # D_xy
            9: 8,  # D_xz
            10: 9,  # D_yz
            11: 10,  # F_xxx
            12: 11,  # F_yyy
            13: 12,  # F_zzz
            14: 13,  # F_xxy
            15: 14,  # F_xxz
            16: 15,  # F_xyy
            17: 16,  # F_yyz
            18: 17,  # F_xzz
            19: 18,  # F_yzz
            20: 19,  # F_xyz
        }

        # Map WFN types to shell types (for grouping)
        # - S = 0, P = 1, D = 2, F = 3
        wfn_type_to_shell_type = {
            1: 0,  # S
            2: 1,  # P
            3: 1,  # P
            4: 1,  # P
            5: 2,  # D
            6: 2,  # D
            7: 2,  # D
            8: 2,  # D
            9: 2,  # D
            10: 2,  # D
        }
        # Add F types (11-20) to shell type mapping
        for wfn_type in range(11, 21):
            wfn_type_to_shell_type[wfn_type] = 3  # F

        # Counter for generating unique basis function indices
        bf_idx = 0

        for i in range(len(centre_assignments)):
            centre_idx = centre_assignments[i]
            wfn_type = type_assignments[i]
            exp = exponents[i]

            atom = self.wfn.atoms[centre_idx]
            coords = tuple(atom.coord)

            # Map WFN type to overlap calculation type
            overlap_type = wfn_type_to_overlap_type.get(wfn_type, wfn_type - 1)
            shell_type = wfn_type_to_shell_type.get(wfn_type, 0)

            # Create basis function dictionary
            basis_functions.append(
                {
                    "type": overlap_type,
                    "center": centre_idx,
                    "coords": coords,
                    "exponents": np.array([exp]),
                    "coefficients": np.array([1.0]),
                    "shell_type": shell_type,
                    "shell_idx": i,
                    "bf_idx": bf_idx,
                }
            )
            bf_idx += 1

        return basis_functions
