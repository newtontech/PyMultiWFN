"""
Parser for Gaussian Formatted Checkpoint (.fchk) files.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES # Corrected import

class FchkLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.content = ""
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parses the .fchk file and returns a Wavefunction object."""
        with open(self.filename, 'r') as f:
            self.content = f.read()
        
        self._parse_header()
        self._parse_atoms()
        self._parse_basis()
        self._parse_mo()
        self._parse_overlap_matrix()
        
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
        """Parses the overlap matrix from the fchk content."""
        if self.wfn.num_basis == 0:
            print("Warning: num_basis is 0, cannot parse Overlap matrix.")
            return

        overlap_vals = self._read_section("Overlap matrix", float)
        
        if len(overlap_vals) > 0:
            # Overlap matrix is typically stored as a flattened symmetric matrix
            # Gaussian stores lower triangle, column-wise.
            # Number of elements = nbasis * (nbasis + 1) / 2
            expected_len = self.wfn.num_basis * (self.wfn.num_basis + 1) // 2
            if len(overlap_vals) == expected_len:
                overlap_mat = np.zeros((self.wfn.num_basis, self.wfn.num_basis))
                k = 0
                for i in range(self.wfn.num_basis):
                    for j in range(i + 1):
                        overlap_mat[i, j] = overlap_vals[k]
                        overlap_mat[j, i] = overlap_vals[k] # Symmetric
                        k += 1
                self.wfn.overlap_matrix = overlap_mat
            else:
                print(f"Warning: Overlap matrix size mismatch. Expected {expected_len}, got {len(overlap_vals)}.")

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

