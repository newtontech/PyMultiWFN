"""
Parser for Gaussian Formatted Checkpoint (.fchk) files.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.constants import ELEMENT_NAMES

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
        
        return self.wfn

    def _read_section(self, label: str, dtype=float) -> np.ndarray:
        """
        Reads a data section from the fchk content.
        Format:
        Label                   Type   N=       Value
        Data...
        """
        # Regex to find the section header
        # Example: Number of electrons                    I               76
        # Example: Atomic numbers                         I   N=          24
        pattern = re.escape(label) + r"\s+[IR]\s+(?:N=\s+(\d+))?"
        match = re.search(pattern, self.content)
        
        if not match:
            return np.array([])
            
        # If N is not present in the header, the value is typically right there (scalar)
        if match.group(1) is None:
            # Try to read the scalar value at the end of the line
            line_end = self.content.find('\n', match.end())
            val_str = self.content[match.end():line_end].strip()
            return np.array([dtype(val_str)])
        
        count = int(match.group(1))
        start_idx = self.content.find('\n', match.end()) + 1
        
        # We need to read 'count' values. Fchk formats are fixed width but space separated usually works.
        # However, for robustness, we can just grab the next N tokens.
        # A simpler approach for large data is to find the start and estimate end, or just split.
        
        # Optimization: Slice the string to avoid processing the whole file
        # Fchk lines are usually 72-80 chars.
        # We can just split the substring starting from start_idx
        
        data_str = self.content[start_idx:]
        
        # Find the start of the next section to limit the split (heuristic: next line starting with capital letter and no leading spaces is rare in data block, but FCHK labels start at col 0)
        # Better: FCHK data lines start with space. Labels start with non-space.
        # Actually, FCHK data lines might not start with space.
        # Let's just tokenize until we have enough values.
        
        values = []
        tokens = data_str.split()
        # This might be slow for huge files, but okay for prototype.
        # For production, we should use numpy.fromstring or similar if possible, but FCHK formatting is weird (e.g. 1.234E-10).
        
        # NumPy's fromstring with sep=' ' handles scientific notation well.
        # We need to be careful not to read into the next section.
        # Let's find the next section header.
        
        # Heuristic: Find next line that looks like a label "Label   Type"
        next_label_match = re.search(r"\n[A-Z][a-zA-Z\s]+\s+[IR]", data_str)
        if next_label_match:
            data_chunk = data_str[:next_label_match.start()]
        else:
            data_chunk = data_str
            
        arr = np.fromstring(data_chunk, sep=' ', dtype=dtype, count=count)
        return arr

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

