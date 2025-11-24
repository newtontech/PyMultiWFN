"""
Parser for Turbomole coordinate files.
Turbomole is a quantum chemistry program package.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class TurbomoleLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse Turbomole file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_turbomole(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_turbomole(self, lines):
        """Parse Turbomole coordinate format."""
        # Find $coord section
        coord_start = None
        coord_end = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.upper() == '$COORD':
                coord_start = i
            elif coord_start is not None and line_stripped.upper() == '$END':
                coord_end = i
                break

        if coord_start is None:
            raise ValueError("No $coord section found in Turbomole file")

        # Parse coordinate lines
        coord_lines = lines[coord_start + 1:coord_end]

        for line in coord_lines:
            line = line.strip()
            if not line or line.startswith('$'):
                continue

            # Turbomole format: x y z element_symbol
            # Coordinates are in Bohr by default
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    element = parts[3].title()

                    # Validate element symbol
                    atomic_num = self._element_to_atomic_number(element)

                    # Turbomole coordinates are already in Bohr
                    x_bohr = x
                    y_bohr = y
                    z_bohr = z

                    atom_info = {
                        'turbomole_format': 'coord'
                    }

                    self.wfn.add_atom(element, atomic_num, x_bohr, y_bohr, z_bohr, float(atomic_num), atom_info)

                except (ValueError, IndexError):
                    continue

        # Also try to parse $title section for additional metadata
        self._parse_title_section(lines)

        # Try to parse $basis section if present (for basis set information)
        self._parse_basis_section(lines)

    def _parse_title_section(self, lines):
        """Parse $title section for molecule information."""
        title_start = None
        title_end = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.upper() == '$TITLE':
                title_start = i
            elif title_start is not None and line_stripped.upper() == '$END':
                title_end = i
                break

        if title_start is not None and title_end is not None:
            title_lines = lines[title_start + 1:title_end]
            title = ' '.join(line.strip() for line in title_lines if line.strip())
            if title:
                self.wfn.title = title

    def _parse_basis_section(self, lines):
        """Parse $basis section for basis set information."""
        basis_start = None
        basis_end = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.upper() == '$BASIS':
                basis_start = i
            elif basis_start is not None and line_stripped.upper() == '$END':
                basis_end = i
                break

        if basis_start is not None and basis_end is not None:
            # Extract basis set name (simplified)
            basis_lines = lines[basis_start + 1:basis_end]
            if basis_lines:
                # Look for basis set name in first few lines
                for line in basis_lines[:5]:
                    line = line.strip()
                    if line and not line.startswith('*') and not line.startswith('#'):
                        self.wfn.basis_set_name = line
                        break

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