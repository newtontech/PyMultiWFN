"""
Parser for Molden files (.molden, .molden.input, molden.inp).
Molden format is widely used for visualizing molecular orbitals and electron density.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction, Shell
from pymultiwfn.core.definitions import ELEMENT_NAMES

class MoldenLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse Molden file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        self._parse_atoms(content)
        self._parse_basis(content)
        self._parse_mo(content)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_atoms(self, content: str):
        """Parse atomic coordinates from [Atoms] section."""
        atoms_match = re.search(r'\[Atoms\].*?\n(.*?)(?=\[|\Z)', content, re.DOTALL | re.IGNORECASE)
        if not atoms_match:
            raise ValueError("No [Atoms] section found in Molden file")

        atoms_lines = atoms_match.group(1).strip().split('\n')
        for line in atoms_lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:
                    # Format: atom_index element_symbol atomic_number x y z [charge...]
                    try:
                        # Some Molden files use different formats
                        if len(parts[0].replace('.', '').isdigit()) or parts[0].isdigit():
                            # Format: index element Z x y z
                            element = parts[1]
                            z = int(parts[2])
                            x, y, z_coord = float(parts[3]), float(parts[4]), float(parts[5])
                        else:
                            # Format: element x y z
                            element = parts[0]
                            z = ELEMENT_NAMES.index(element) + 1 if element in ELEMENT_NAMES else 0
                            x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])

                        self.wfn.add_atom(element, z, x, y, z_coord, float(z))
                    except (ValueError, IndexError):
                        continue

    def _parse_basis(self, content: str):
        """Parse basis set information from [GTO] section."""
        gto_match = re.search(r'\[GTO\].*?\n(.*?)(?=\[|\Z)', content, re.DOTALL | re.IGNORECASE)
        if not gto_match:
            raise ValueError("No [GTO] section found in Molden file")

        gto_lines = gto_match.group(1).strip().split('\n')
        current_atom = None
        current_shell_type = None
        prim_exps = []
        prim_coeffs = []

        for line in gto_lines:
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

    def _parse_mo(self, content: str):
        """Parse molecular orbital information from [MO] section."""
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