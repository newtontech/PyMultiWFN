"""
Parser for XYZ coordinate files (.xyz, .XYZ).
XYZ format is a simple text format for molecular coordinates.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import ELEMENT_NAMES
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class XYZLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse XYZ file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_xyz(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_xyz(self, lines):
        """Parse XYZ format."""
        if len(lines) < 3:
            raise ValueError("XYZ file is too short - must have at least 3 lines")

        # First line: number of atoms
        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("First line of XYZ file must contain the number of atoms")

        # Second line: comment/title (optional)
        if len(lines) > 1:
            self.wfn.title = lines[1].strip()

        # Remaining lines: atomic coordinates
        atom_lines = lines[2:2 + num_atoms]

        for i, line in enumerate(atom_lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0].title()  # Capitalize first letter
                    x = float(parts[1]) * ANGSTROM_TO_BOHR
                    y = float(parts[2]) * ANGSTROM_TO_BOHR
                    z = float(parts[3]) * ANGSTROM_TO_BOHR

                    # Try to get atomic number
                    if element in ELEMENT_NAMES:
                        atomic_num = ELEMENT_NAMES.index(element) + 1
                    else:
                        # Try to parse from symbol (e.g., "C" -> 6)
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
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
        }
        return element_mapping.get(element.title(), 0)