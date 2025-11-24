"""
Parser for Gaussian WFX files (.wfx, .WFX).
WFX format is an extended version of WFN with more complete wavefunction information.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES

class WFXLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse WFX file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        self._parse_header(content)
        self._parse_atoms(content)
        self._parse_basis(content)
        self._parse_mo(content)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_header(self, content: str):
        """Parse header information."""
        # Extract title
        title_match = re.search(r'^Title\s*\n(.+?)(?=\n\w)', content, re.MULTILINE)
        if title_match:
            self.wfn.title = title_match.group(1).strip()

        # Extract number of atoms
        natom_match = re.search(r'Number of atoms\s*=\s*(\d+)', content)
        if natom_match:
            self.wfn.num_atoms = int(natom_match.group(1))

        # Extract number of electrons
        nelec_match = re.search(r'Number of electrons\s*=\s*(\d+)', content)
        if nelec_match:
            self.wfn.num_electrons = int(nelec_match.group(1))

        # Extract multiplicity
        mult_match = re.search(r'Number of alpha electrons\s*=\s*(\d+)', content)
        if mult_match:
            nalpha = int(mult_match.group(1))
            self.wfn.multiplicity = self.wfn.num_electrons - 2 * nalpha + 1

        # Extract number of MOs
        nmo_match = re.search(r'Number of MOs\s*=\s*(\d+)', content)
        if nmo_match:
            self.wfn.num_mos = int(nmo_match.group(1))

    def _parse_atoms(self, content: str):
        """Parse atomic coordinates."""
        # Look for atomic coordinates section
        atoms_section = re.search(
            r'Atomic coordinates\s*\n.*?\n(.*?)(?=\n\w)',
            content, re.DOTALL
        )

        if atoms_section:
            atom_lines = atoms_section.group(1).strip().split('\n')
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        atomic_num = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        charge = float(parts[4]) if len(parts) > 4 else float(atomic_num)

                        # Get element symbol
                        element = ELEMENT_NAMES[atomic_num] if atomic_num < len(ELEMENT_NAMES) else f"X{atomic_num}"

                        self.wfn.add_atom(element, atomic_num, x, y, z, charge)
                    except (ValueError, IndexError):
                        continue

    def _parse_basis(self, content: str):
        """Parse basis set information."""
        # Find basis function section
        basis_section = re.search(
            r'Basis functions\s*\n(.*?)(?=\n\w)',
            content, re.DOTALL
        )

        if not basis_section:
            return

        # Parse individual basis functions
        basis_pattern = re.compile(
            r'Center\s+(\d+)\s+Type\s+(\w+)\s+Nprims\s+(\d+)(.*?)(?=Center|\n\w)',
            re.DOTALL
        )

        basis_matches = basis_pattern.findall(basis_section.group(1))

        for center_idx, shell_type, n_prims, prim_data in basis_matches:
            try:
                center_idx = int(center_idx) - 1  # Convert to 0-based
                n_prims = int(n_prims)

                # Parse primitive data
                prim_lines = prim_data.strip().split('\n')
                exponents = []
                coefficients = []

                for line in prim_lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            exp = float(parts[0])
                            coeff = float(parts[1])
                            exponents.append(exp)
                            coefficients.append(coeff)
                        except ValueError:
                            continue

                # Convert shell type
                shell_type_num = self._shell_type_to_int(shell_type.upper())

                shell = Shell(
                    type=shell_type_num,
                    center_idx=center_idx,
                    exponents=np.array(exponents),
                    coefficients=np.array([coefficients])
                )
                self.wfn.shells.append(shell)

            except (ValueError, IndexError):
                continue

        # Calculate number of basis functions
        self.wfn.num_basis = sum(self._shell_num_functions(shell.type) for shell in self.wfn.shells)

    def _parse_mo(self, content: str):
        """Parse molecular orbital coefficients."""
        # Find MO coefficient section
        mo_section = re.search(
            r'Molecular Orbital coefficients\s*\n(.*?)(?=\n\w|$)',
            content, re.DOTALL
        )

        if not mo_section:
            return

        # Parse MO energies and coefficients
        mo_pattern = re.compile(
            r'MO\s+(\d+)\s+Energy\s*=\s*([-+]?\d*\.\d+E[+-]?\d+)(.*?)(?=MO\s+\d+|\n\w|$)',
            re.DOTALL
        )

        mo_matches = mo_pattern.findall(mo_section.group(1))

        energies = []
        coeff_matrices = []

        for mo_num, energy, coeff_data in mo_matches:
            try:
                energies.append(float(energy))

                # Parse coefficients
                coeff_lines = coeff_data.strip().split('\n')
                coeffs = []

                for line in coeff_lines:
                    parts = line.split()
                    for part in parts:
                        try:
                            coeff = float(part)
                            coeffs.append(coeff)
                        except ValueError:
                            continue

                coeff_matrices.append(np.array(coeffs))

            except ValueError:
                continue

        if coeff_matrices:
            # Ensure all coefficient vectors have the same length
            max_len = max(len(coeffs) for coeffs in coeff_matrices)
            normalized_coeffs = []

            for coeffs in coeff_matrices:
                if len(coeffs) < max_len:
                    # Pad with zeros if needed
                    padded = np.zeros(max_len)
                    padded[:len(coeffs)] = coeffs
                    normalized_coeffs.append(padded)
                else:
                    normalized_coeffs.append(coeffs[:max_len])

            self.wfn.coefficients = np.array(normalized_coeffs)
            self.wfn.energies = np.array(energies)

    def _shell_type_to_int(self, shell_type: str) -> int:
        """Convert shell type string to integer."""
        type_mapping = {
            'S': 0,
            'SP': -1,
            'P': 1,
            'D': 2,
            'F': 3,
            'G': 4,
            'H': 5
        }
        return type_mapping.get(shell_type.upper(), 0)

    def _shell_num_functions(self, shell_type: int) -> int:
        """Return number of basis functions for a given shell type."""
        func_counts = {
            0: 1,   # S
            1: 3,   # P
            -1: 4,  # SP (1 S + 3 P)
            2: 5,   # D (spherical)
            3: 7,   # F (spherical)
            4: 9,   # G (spherical)
            5: 11   # H (spherical)
        }
        return func_counts.get(shell_type, 1)