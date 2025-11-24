"""
Parser for Gaussian input files (.gjf, .com).
Gaussian input format for quantum chemistry calculations.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import ELEMENT_NAMES
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class GJFLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse Gaussian input file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        self._parse_gjf(content)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_gjf(self, content: str):
        """Parse Gaussian input format."""
        lines = content.strip().split('\n')

        # Skip blank lines and comments
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                clean_lines.append(line)

        if len(clean_lines) < 2:
            raise ValueError("Gaussian input file is too short")

        # Extract title and method/basis
        self.wfn.title = clean_lines[0] if clean_lines[0] else "Gaussian Input"

        # Extract method and basis set from route section (if available)
        if len(clean_lines) > 1:
            method_basis = clean_lines[1]
            self.wfn.method = method_basis.split()[0] if method_basis else "Unknown"
            # Extract basis set if available
            if '/' in method_basis:
                self.wfn.basis_set_name = method_basis.split('/')[-1].split()[0]

        # Find coordinate section
        coord_start = 2
        for i in range(2, len(clean_lines)):
            line = clean_lines[i].lower()
            if any(keyword in line for keyword in ['charge', 'multiplicity', '0', '1']):
                coord_start = i + 1
                break

        # Parse charge and multiplicity
        if coord_start > 2 and coord_start <= len(clean_lines):
            try:
                charge_mult = clean_lines[coord_start - 1].split()
                if len(charge_mult) >= 2:
                    charge = int(charge_mult[0])
                    self.wfn.multiplicity = int(charge_mult[1])
                    # Adjust electron count
                    self.wfn.num_electrons = None  # Will be calculated from atoms
            except (ValueError, IndexError):
                pass

        # Parse atomic coordinates
        for line in clean_lines[coord_start:]:
            if not line or line.startswith('--'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0].title()
                    x = float(parts[1]) * ANGSTROM_TO_BOHR
                    y = float(parts[2]) * ANGSTROM_TO_BOHR
                    z = float(parts[3]) * ANGSTROM_TO_BOHR

                    # Get atomic number
                    if element in ELEMENT_NAMES:
                        atomic_num = ELEMENT_NAMES.index(element) + 1
                    else:
                        atomic_num = self._element_to_atomic_number(element)

                    self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num))
                except (ValueError, IndexError):
                    continue

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