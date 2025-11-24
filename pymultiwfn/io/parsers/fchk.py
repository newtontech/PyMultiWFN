"""
Parser for Gaussian Formatted Checkpoint (.fchk) files.
Enhanced version with comprehensive error handling and additional data parsing.
"""

import re
import numpy as np
import warnings
from typing import Optional, Dict, Any, List
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES

class FchkLoader:
    """
    Enhanced loader for Gaussian Formatted Checkpoint (.fchk) files.

    This parser extracts comprehensive wavefunction information from FCHK files,
    including molecular structure, basis sets, MO coefficients, and additional
    properties like density matrices and electrostatic data.
    """

    def __init__(self, filename: str):
        """
        Initialize the FCHK loader.

        Args:
            filename: Path to the .fchk file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file extension is not .fchk or .fch
        """
        if not filename.lower().endswith(('.fchk', '.fch')):
            raise ValueError(f"File must have .fchk or .fch extension, got: {filename}")

        self.filename = filename
        self.content = ""
        self.wfn = Wavefunction()
        self.metadata: Dict[str, Any] = {}

    def load(self) -> Wavefunction:
        """
        Parse the .fchk file and return a complete Wavefunction object.

        Returns:
            Wavefunction: Complete wavefunction object with all parsed data

        Raises:
            Exception: If there are critical parsing errors
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(self.filename, 'r', encoding='latin-1') as f:
                self.content = f.read()

        if not self.content.strip():
            raise ValueError(f"File {self.filename} appears to be empty")

        # Parse all sections in order
        self._parse_header()
        self._parse_atoms()
        self._parse_basis()
        self._parse_mo()
        self._parse_overlap_matrix()
        self._parse_density_matrices()
        self._parse_electrostatic_data()
        self._parse_additional_properties()

        # Validate and finalize
        self._validate_parsed_data()
        self.wfn._infer_occupations()

        return self.wfn

    def _read_section(self, label: str, dtype=float) -> np.ndarray:
        """
        Reads a data section from the fchk content.
        Format:
        Label                   Type   N=       Value
        Data...
        
        Handles both scalar and array data.
        """
        escaped_label = re.escape(label)
        # Pattern to capture scalar value on the same line OR N=count on the same line
        pattern_scalar_or_n = rf"{escaped_label}\s+[IR]\s+(?:N=\s*(\d+)\s*)?(.+)?(?=\n|$)"
        
        match = re.search(pattern_scalar_or_n, self.content)
        
        if not match:
            return np.array([])
            
        count_str = match.group(1)
        scalar_val_str = match.group(2)
        
        if count_str: # It's an array with N=count
            count = int(count_str)
            start_idx = self.content.find('\n', match.end()) + 1
            
            data_str = self.content[start_idx:]
            
            # Find the start of the next label to delimit the current data block
            next_label_match = re.search(r"\n[A-Z][a-zA-Z\s]+\s+[IR]", data_str)
            if next_label_match:
                data_chunk = data_str[:next_label_match.start()]
            else:
                data_chunk = data_str
                
            arr = np.fromstring(data_chunk, sep=' ', dtype=dtype, count=count)
            return arr
        elif scalar_val_str: # It's a scalar value on the same line
            try:
                # Need to find the actual scalar value at the end of the matched line
                # Example: "Number of electrons I 76" -> scalar_val_str will be "76"
                return np.array([dtype(scalar_val_str.strip())])
            except ValueError:
                return np.array([])
        return np.array([])

    def _parse_overlap_matrix(self):
        """Enhanced parsing of the overlap matrix from the fchk content."""
        if self.wfn.num_basis == 0:
            warnings.warn("num_basis is 0, cannot parse Overlap matrix.", RuntimeWarning)
            return

        overlap_vals = self._read_section("Overlap matrix", float)

        if len(overlap_vals) == 0:
            warnings.warn("No overlap matrix found in FCHK file.", RuntimeWarning)
            return

        expected_len = self.wfn.num_basis * (self.wfn.num_basis + 1) // 2
        if len(overlap_vals) != expected_len:
            warnings.warn(f"Overlap matrix size mismatch. Expected {expected_len}, got {len(overlap_vals)}.", RuntimeWarning)
            return

        # Reconstruct symmetric matrix from lower triangle
        overlap_mat = np.zeros((self.wfn.num_basis, self.wfn.num_basis))
        k = 0
        for i in range(self.wfn.num_basis):
            for j in range(i + 1):
                overlap_mat[i, j] = overlap_vals[k]
                overlap_mat[j, i] = overlap_vals[k]  # Symmetric
                k += 1

        self.wfn.overlap_matrix = overlap_mat
        self.metadata['overlap_matrix_available'] = True

    def _parse_density_matrices(self):
        """Parse density matrices (total, alpha, beta) if available."""
        # Total density matrix
        total_dens = self._read_section("Total SCF Density", float)
        if len(total_dens) > 0:
            expected_len = self.wfn.num_basis * (self.wfn.num_basis + 1) // 2
            if len(total_dens) == expected_len:
                self.wfn.density_matrix = self._reconstruct_symmetric_matrix(total_dens)
                self.metadata['total_density_available'] = True

        # Alpha density matrix
        alpha_dens = self._read_section("Alpha SCF Density", float)
        if len(alpha_dens) > 0:
            expected_len = self.wfn.num_basis * (self.wfn.num_basis + 1) // 2
            if len(alpha_dens) == expected_len:
                self.wfn.density_matrix_alpha = self._reconstruct_symmetric_matrix(alpha_dens)
                self.metadata['alpha_density_available'] = True

        # Beta density matrix
        beta_dens = self._read_section("Beta SCF Density", float)
        if len(beta_dens) > 0:
            expected_len = self.wfn.num_basis * (self.wfn.num_basis + 1) // 2
            if len(beta_dens) == expected_len:
                self.wfn.density_matrix_beta = self._reconstruct_symmetric_matrix(beta_dens)
                self.metadata['beta_density_available'] = True

    def _parse_electrostatic_data(self):
        """Parse electrostatic potential and related data."""
        # ESP charges
        esp_charges = self._read_section("ESP Charges", float)
        if len(esp_charges) > 0:
            if len(esp_charges) == self.wfn.num_atoms:
                self.wfn.esp_charges = esp_charges
                self.metadata['esp_charges_available'] = True

        # Mulliken charges
        mulliken_charges = self._read_section("Mulliken Atomic Charges", float)
        if len(mulliken_charges) > 0:
            if len(mulliken_charges) == self.wfn.num_atoms:
                self.wfn.mulliken_charges = mulliken_charges
                self.metadata['mulliken_charges_available'] = True

        # Natural charges
        natural_charges = self._read_section("Natural Population Analysis", float)
        if len(natural_charges) > 0:
            # NPA data format: [charge, core, valence, rydberg, total] for each atom
            if len(natural_charges) >= self.wfn.num_atoms:
                self.wfn.npa_charges = natural_charges[::5][:self.wfn.num_atoms]
                self.metadata['npa_charges_available'] = True

    def _parse_additional_properties(self):
        """Parse additional properties and metadata."""
        # Dipole moment
        dipole = self._read_section("Dipole Moment", float)
        if len(dipole) >= 3:
            self.wfn.dipole_moment = dipole[:3]
            self.metadata['dipole_available'] = True

        # Quadrupole moment
        quadrupole = self._read_section("Quadrupole Moment", float)
        if len(quadrupole) >= 6:
            self.wfn.quadrupole_moment = quadrupole[:6]
            self.metadata['quadrupole_available'] = True

        # SCF energy
        scf_energy = self._read_section("Total Energy", float)
        if len(scf_energy) > 0:
            self.wfn.scf_energy = scf_energy[0]
            self.metadata['scf_energy_available'] = True

        # Computational methods details
        self.metadata['scf_converged'] = self._read_section("SCF Converged", int)[0] if len(self._read_section("SCF Converged", int)) > 0 else None
        self.metadata['efield'] = self._read_section("Electric Field", float)

    def _reconstruct_symmetric_matrix(self, packed_array: np.ndarray) -> np.ndarray:
        """
        Reconstruct symmetric matrix from packed lower triangular format.

        Args:
            packed_array: Packed lower triangular matrix elements

        Returns:
            Full symmetric matrix
        """
        n = int((-1 + np.sqrt(1 + 8 * len(packed_array))) // 2)
        if n != self.wfn.num_basis:
            warnings.warn(f"Matrix reconstruction size mismatch: expected {self.wfn.num_basis}, got {n}", RuntimeWarning)

        matrix = np.zeros((self.wfn.num_basis, self.wfn.num_basis))
        k = 0
        for i in range(self.wfn.num_basis):
            for j in range(i + 1):
                matrix[i, j] = packed_array[k]
                matrix[j, i] = packed_array[k]
                k += 1
        return matrix

    def _validate_parsed_data(self):
        """Validate parsed data for consistency."""
        if self.wfn.num_atoms == 0:
            raise ValueError("No atoms were parsed from the FCHK file")

        if self.wfn.num_basis == 0:
            raise ValueError("No basis functions were parsed from the FCHK file")

        if self.wfn.num_electrons == 0:
            raise ValueError("No electron count was parsed from the FCHK file")

        # Validate basis set consistency
        if len(self.wfn.shells) == 0:
            raise ValueError("No basis shells were parsed from the FCHK file")

        # Validate MO coefficients
        if hasattr(self.wfn, 'coefficients') and self.wfn.coefficients.size > 0:
            if self.wfn.coefficients.shape[1] != self.wfn.num_basis:
                warnings.warn(f"MO coefficients shape mismatch: {self.wfn.coefficients.shape}, expected (nmo, {self.wfn.num_basis})", RuntimeWarning)

        # Store validation metadata
        self.metadata['validation_passed'] = True
        self.metadata['validation_timestamp'] = np.datetime64('now')

    def _parse_header(self):
        lines = self.content.splitlines()
        self.wfn.title = lines[0].strip()
        method_basis = lines[1].strip()
        self.wfn.method = method_basis.split()[0]
        self.wfn.basis_set_name = method_basis.split('/')[-1] if '/' in method_basis else "Unknown"
        
        # Parse scalar values
        nelec = self._read_section("Number of electrons", float)
        if len(nelec) > 0:
            self.wfn.num_electrons = nelec[0]
            
        # Multiplicity
        mult = self._read_section("Multiplicity", int)
        if len(mult) > 0:
            self.wfn.multiplicity = mult[0]

    def _parse_atoms(self):
        atomic_numbers = self._read_section("Atomic numbers", int)
        nuclear_charges = self._read_section("Nuclear charges", float)
        coords = self._read_section("Current cartesian coordinates", float)
        
        num_atoms = len(atomic_numbers)
        coords = coords.reshape((num_atoms, 3))
        
        for i in range(num_atoms):
            z = atomic_numbers[i]
            ele = ELEMENT_NAMES[z] if z < len(ELEMENT_NAMES) else "X"
            charge = nuclear_charges[i]
            x, y, z_coord = coords[i]
            self.wfn.add_atom(ele, z, x, y, z_coord, charge)

    def _parse_basis(self):
        shell_types = self._read_section("Shell types", int)
        shell_to_atom = self._read_section("Shell to atom map", int)
        shell_prims = self._read_section("Number of primitives per shell", int)
        prim_exps = self._read_section("Primitive exponents", float)
        contraction_coeffs = self._read_section("Contraction coefficients", float)
        
        # P(S=P) coefficients for SP shells
        sp_coeffs = self._read_section("P(S=P) Contraction coefficients", float)
        
        num_shells = len(shell_types)
        prim_idx = 0
        
        for i in range(num_shells):
            stype = shell_types[i]
            atom_idx = shell_to_atom[i] - 1 # FCHK is 1-based
            nprim = shell_prims[i]
            
            exps = prim_exps[prim_idx : prim_idx+nprim]
            coeffs = contraction_coeffs[prim_idx : prim_idx+nprim]
            
            # Handle SP shells (type -1)
            if stype == -1 and len(sp_coeffs) > 0:
                p_coeffs = sp_coeffs[prim_idx : prim_idx+nprim]
                # Combine S and P coeffs
                coeffs = np.vstack([coeffs, p_coeffs])
            
            shell = Shell(
                type=stype,
                center_idx=atom_idx,
                exponents=exps,
                coefficients=coeffs
            )
            self.wfn.shells.append(shell)
            
            prim_idx += nprim
            
        self.wfn.num_basis = self._read_section("Number of basis functions", int)[0]

    def _parse_mo(self):
        # Alpha
        energies = self._read_section("Alpha Orbital Energies", float)
        coeffs = self._read_section("Alpha MO coefficients", float)
        
        if len(energies) > 0:
            self.wfn.energies = energies
            # FCHK stores coeffs as flattened array. 
            # Multiwfn convention: (nmo, nbasis)
            # FCHK convention: (nbasis, nmo) ? No, usually it's blocked by orbital.
            # Let's assume (nmo, nbasis) for now or check size.
            nmo = len(energies)
            nbasis = self.wfn.num_basis
            if len(coeffs) == nmo * nbasis:
                self.wfn.coefficients = coeffs.reshape((nmo, nbasis))
            else:
                print(f"Warning: Alpha MO Coeffs size mismatch. Expected {nmo}*{nbasis}={nmo*nbasis}, got {len(coeffs)}")

        # Beta (if exists)
        energies_beta = self._read_section("Beta Orbital Energies", float)
        if len(energies_beta) > 0:
            self.wfn.is_unrestricted = True
            self.wfn.energies_beta = energies_beta
            coeffs_beta = self._read_section("Beta MO coefficients", float)
            nmo_beta = len(energies_beta)
            if len(coeffs_beta) == nmo_beta * self.wfn.num_basis:
                self.wfn.coefficients_beta = coeffs_beta.reshape((nmo_beta, self.wfn.num_basis))

