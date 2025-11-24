"""
Parser for PQR files (.pqr, .PQR).
PQR format is similar to PDB but includes partial charges and radii.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class PQRLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse PQR file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_pqr(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_pqr(self, lines):
        """Parse PQR format."""
        for line in lines:
            if not line or len(line) < 6:
                continue

            record_type = line[:6].strip()

            # Parse ATOM and HETATM records
            if record_type in ['ATOM', 'HETATM']:
                self._parse_atom_line(line)
            # Parse TITLE/HEADER records
            elif record_type in ['TITLE', 'HEADER']:
                self._parse_title_line(line)
            # Parse CRYST1 record for unit cell information
            elif record_type == 'CRYST1':
                self._parse_crystal_line(line)

    def _parse_atom_line(self, line: str):
        """Parse ATOM/HETATM line with charge and radius information."""
        # PQR format is similar to PDB but with charge and radius instead of occupancy and B-factor
        if len(line) < 54:  # Minimum length for coordinates
            return

        try:
            # Extract atom information
            atom_serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_seq = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            # PQR specific: charge and radius (instead of occupancy and temp_factor)
            # The format can vary, so try to parse from the remaining line
            remaining = line[54:].strip()
            charge = 0.0
            radius = 0.0

            # Try to parse charge and radius from the remaining part
            parts = remaining.split()
            if len(parts) >= 2:
                try:
                    charge = float(parts[0])
                    radius = float(parts[1])
                except ValueError:
                    # Try alternative parsing (some formats use columns)
                    try:
                        charge = float(line[54:62].strip()) if len(line) > 62 else 0.0
                        radius = float(line[62:70].strip()) if len(line) > 70 else 0.0
                    except ValueError:
                        pass
            elif len(parts) == 1:
                try:
                    charge = float(parts[0])
                except ValueError:
                    pass

            # Convert to Bohr
            x_bohr = x * ANGSTROM_TO_BOHR
            y_bohr = y * ANGSTROM_TO_BOHR
            z_bohr = z * ANGSTROM_TO_BOHR

            # Extract element from atom name
            element = self._extract_element_from_name(atom_name)
            element = element.title()
            atomic_num = self._element_to_atomic_number(element)

            # Add additional information as metadata, including charge and radius
            atom_info = {
                'serial': atom_serial,
                'name': atom_name,
                'residue_name': residue_name,
                'chain_id': chain_id,
                'residue_seq': residue_seq,
                'charge': charge,
                'radius': radius,
                'partial_charge': charge  # Alias for easier access
            }

            self.wfn.add_atom(element, atomic_num, x_bohr, y_bohr, z_bohr, float(atomic_num), atom_info)

        except (ValueError, IndexError):
            # Skip malformed lines
            return

    def _parse_title_line(self, line: str):
        """Parse TITLE/HEADER line."""
        title_text = line[10:].strip() if len(line) > 10 else ''
        if title_text:
            if self.wfn.title:
                self.wfn.title += ' ' + title_text
            else:
                self.wfn.title = title_text

    def _parse_crystal_line(self, line: str):
        """Parse CRYST1 line for unit cell information."""
        if len(line) < 54:
            return

        try:
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            alpha = float(line[33:40].strip())
            beta = float(line[40:47].strip())
            gamma = float(line[47:54].strip())
            space_group = line[55:66].strip() if len(line) > 55 else ''

            # Store as crystal information
            self.wfn.crystal_info = {
                'a': a * ANGSTROM_TO_BOHR,
                'b': b * ANGSTROM_TO_BOHR,
                'c': c * ANGSTROM_TO_BOHR,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'space_group': space_group
            }

        except (ValueError, IndexError):
            pass

    def _extract_element_from_name(self, atom_name: str) -> str:
        """Extract element symbol from atom name."""
        if not atom_name:
            return ''

        # Common patterns in PDB atom names
        atom_name = atom_name.strip()

        # If first character is a letter and second is a number, it's the element
        if len(atom_name) >= 2 and atom_name[0].isalpha() and atom_name[1].isdigit():
            return atom_name[0]

        # If it starts with H, it's probably hydrogen
        if atom_name.startswith('H'):
            return 'H'

        # For carbon, nitrogen, oxygen, etc.
        if atom_name.startswith('C'):
            return 'C'
        elif atom_name.startswith('N'):
            return 'N'
        elif atom_name.startswith('O'):
            return 'O'
        elif atom_name.startswith('S'):
            return 'S'
        elif atom_name.startswith('P'):
            return 'P'

        # Two-letter elements
        if len(atom_name) >= 2:
            two_letter = atom_name[:2].title()
            if two_letter in ['Na', 'Cl', 'Br', 'Fe', 'Zn', 'Cu', 'Mg', 'Ca']:
                return two_letter

        # Default to first character
        return atom_name[0] if atom_name else ''

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