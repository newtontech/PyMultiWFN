"""
Parser for GROMACS coordinate files (.gro).
GRO format is used by GROMACS molecular dynamics software.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR, NM_TO_BOHR

class GROLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse GRO file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_gro(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_gro(self, lines):
        """Parse GRO format."""
        if len(lines) < 2:
            raise ValueError("GRO file is too short")

        # Line 1: Title/comment
        self.wfn.title = lines[0].strip() if lines else "GRO File"

        # Line 2: Number of atoms
        if len(lines) < 2:
            raise ValueError("GRO file missing atom count")

        try:
            num_atoms = int(lines[1].strip())
        except ValueError:
            raise ValueError("Cannot parse atom count from GRO file")

        # Atom coordinates (lines 3 to num_atoms+2)
        atom_block_start = 2
        atom_block_end = atom_block_start + num_atoms

        if len(lines) < atom_block_end:
            raise ValueError(f"GRO file expects {num_atoms} atoms but only {len(lines)-2} lines found")

        # Parse atom lines
        for i in range(atom_block_start, atom_block_end):
            line = lines[i].rstrip('\n')  # Don't strip trailing spaces as they may be important
            if len(line) < 25:
                continue

            try:
                # GRO format:
                # %5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f
                # residue_number residue_name atom_name atom_number x y z vx vy vz
                residue_num = int(line[0:5].strip())
                residue_name = line[5:10].strip()
                atom_name = line[10:15].strip()
                atom_num = int(line[15:20].strip())
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())

                # Optional velocities (not needed for wavefunction)
                vx = float(line[44:52].strip()) if len(line) > 44 else 0.0
                vy = float(line[52:60].strip()) if len(line) > 52 else 0.0
                vz = float(line[60:68].strip()) if len(line) > 60 else 0.0

                # GRO coordinates are in nm, convert to Bohr
                x_bohr = x * NM_TO_BOHR
                y_bohr = y * NM_TO_BOHR
                z_bohr = z * NM_TO_BOHR

                # Extract element from atom name
                element = self._extract_element_from_name(atom_name)
                atomic_num = self._element_to_atomic_number(element)

                # Add additional information as metadata
                atom_info = {
                    'residue_number': residue_num,
                    'residue_name': residue_name,
                    'atom_number': atom_num,
                    'velocity': [vx, vy, vz]
                }

                self.wfn.add_atom(element, atomic_num, x_bohr, y_bohr, z_bohr, float(atomic_num), atom_info)

            except (ValueError, IndexError):
                continue

        # Box vectors (last line, if present)
        if len(lines) > atom_block_end:
            box_line = lines[atom_block_end].strip()
            if box_line:
                self._parse_box_vectors(box_line)

    def _parse_box_vectors(self, box_line: str):
        """Parse box vectors from GRO file."""
        try:
            parts = box_line.split()
            if len(parts) >= 3:
                # GRO box format can be:
                # v1x v2y v3z (orthogonal box)
                # v1x v2y v3z v1y v1z v2x v2z v3x v3y (triclinic box)

                box_vectors = []
                for part in parts:
                    box_vectors.append(float(part) * NM_TO_BOHR)

                if len(box_vectors) >= 3:
                    self.wfn.box_vectors = {
                        'a': box_vectors[0],
                        'b': box_vectors[1],
                        'c': box_vectors[2]
                    }

                if len(box_vectors) == 9:
                    # Full triclinic box information
                    self.wfn.box_vectors.update({
                        'v1y': box_vectors[3],
                        'v1z': box_vectors[4],
                        'v2x': box_vectors[5],
                        'v2z': box_vectors[6],
                        'v3x': box_vectors[7],
                        'v3y': box_vectors[8]
                    })

        except ValueError:
            pass

    def _extract_element_from_name(self, atom_name: str) -> str:
        """Extract element symbol from atom name."""
        if not atom_name:
            return ''

        atom_name = atom_name.strip()

        # Common GROMACS naming conventions
        # For backbone atoms: N, CA, C, O, etc.
        if atom_name in ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'CF']:
            if atom_name == 'CA' or atom_name.startswith('C'):
                return 'C'
            elif atom_name.startswith('N'):
                return 'N'
            elif atom_name.startswith('O'):
                return 'O'

        # For side chain atoms: first character is usually the element
        if len(atom_name) >= 1:
            first_char = atom_name[0].upper()
            if first_char.isalpha():
                # Check for two-letter elements
                if len(atom_name) >= 2 and atom_name[1].islower():
                    two_letter = atom_name[:2].title()
                    if two_letter in ['Na', 'Cl', 'Br', 'Fe', 'Zn', 'Cu', 'Mg', 'Ca']:
                        return two_letter

                return first_char

        # Special cases for common atom names in force fields
        if atom_name.startswith('H'):
            return 'H'
        elif atom_name.startswith('C'):
            return 'C'
        elif atom_name.startswith('N'):
            return 'N'
        elif atom_name.startswith('O'):
            return 'O'
        elif atom_name.startswith('S'):
            return 'S'
        elif atom_name.startswith('P'):
            return 'P'

        # Default to carbon if unsure
        return 'C'

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        element_mapping = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
            'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
            'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
            'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
            'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
            'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        return element_mapping.get(element.title(), 0)