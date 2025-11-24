"""
Parser for MOPAC input files (.mop, .MOP).
MOPAC is a semi-empirical quantum chemistry program.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class MOPACLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse MOPAC file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_mopac(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_mopac(self, lines):
        """Parse MOPAC input format."""
        if not lines:
            raise ValueError("MOPAC file is empty")

        # First line: title (keywords and comments)
        first_line = lines[0].strip()
        if first_line:
            self.wfn.title = first_line

        # Determine coordinate format (0=XYZ, 1=XYZM, 2=TXYZ, etc.)
        coord_format = 0  # Default XYZ format

        # Look for format keyword in first line
        if 'XYZ' in first_line.upper():
            if 'XYZM' in first_line.upper():
                coord_format = 1
            elif 'TXYZ' in first_line.upper():
                coord_format = 2

        # Skip title lines (can be multiple)
        line_idx = 1
        while line_idx < len(lines) and not lines[line_idx].strip().split():
            line_idx += 1

        # Parse coordinates section
        while line_idx < len(lines):
            line = lines[line_idx].strip()

            # Stop at empty line or end of file
            if not line:
                break

            # Parse atom line based on format
            parts = line.split()
            if len(parts) >= 4:  # Minimum for coordinates
                try:
                    if coord_format == 0:  # XYZ format: element x y z
                        element = parts[0].title()
                        x = float(parts[1]) * ANGSTROM_TO_BOHR
                        y = float(parts[2]) * ANGSTROM_TO_BOHR
                        z = float(parts[3]) * ANGSTROM_TO_BOHR
                        charge = 0.0

                    elif coord_format == 1:  # XYZM format: element x y z charge
                        element = parts[0].title()
                        x = float(parts[1]) * ANGSTROM_TO_BOHR
                        y = float(parts[2]) * ANGSTROM_TO_BOHR
                        z = float(parts[3]) * ANGSTROM_TO_BOHR
                        charge = float(parts[4]) if len(parts) > 4 else 0.0

                    elif coord_format == 2:  # TXYZ format: atomic_num element x y z
                        atomic_num = int(parts[0])
                        element = parts[1].title()
                        x = float(parts[2]) * ANGSTROM_TO_BOHR
                        y = float(parts[3]) * ANGSTROM_TO_BOHR
                        z = float(parts[4]) * ANGSTROM_TO_BOHR
                        charge = 0.0

                        # Validate atomic number
                        if self._element_to_atomic_number(element) != atomic_num:
                            # Use atomic number if mismatch
                            element = self._atomic_number_to_element(atomic_num)

                    else:  # Default XYZ format
                        element = parts[0].title()
                        x = float(parts[1]) * ANGSTROM_TO_BOHR
                        y = float(parts[2]) * ANGSTROM_TO_BOHR
                        z = float(parts[3]) * ANGSTROM_TO_BOHR
                        charge = 0.0

                    # Get atomic number for wavefunction
                    atomic_num = self._element_to_atomic_number(element)

                    atom_info = {
                        'mopac_charge': charge,
                        'mopac_format': coord_format
                    }

                    self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num), atom_info)

                except (ValueError, IndexError):
                    # Skip malformed lines
                    pass

            line_idx += 1

        # Check for charge and multiplicity information
        # This would require more sophisticated parsing of MOPAC keywords
        # For now, use defaults

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

    def _atomic_number_to_element(self, atomic_num: int) -> str:
        """Convert atomic number to element symbol."""
        number_to_element = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
            19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
            27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
            35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
            43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
            59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
            67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
            83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
            91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
            99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
            106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
            113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
        }
        return number_to_element.get(atomic_num, 'C').title()