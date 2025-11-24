"""
Parser for MDL Mol files (.mol, .sdf).
MDL Mol format is a chemical file format for storing molecular information.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import ELEMENT_NAMES
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class MOLLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse MOL/SDF file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_mol(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_mol(self, lines):
        """Parse MOL format."""
        if len(lines) < 4:
            raise ValueError("MOL file is too short")

        # Header block (3 lines)
        self.wfn.title = lines[0].strip() if lines else "MOL File"

        # Software info (line 2) and comment (line 3) - can be skipped
        if len(lines) >= 3:
            pass

        # Counts line (line 4) - contains number of atoms and bonds
        if len(lines) < 4:
            raise ValueError("MOL file missing counts line")

        counts_line = lines[3].strip()
        if len(counts_line) < 6:
            raise ValueError("Invalid counts line in MOL file")

        try:
            num_atoms = int(counts_line[0:3])
            num_bonds = int(counts_line[3:6])
        except ValueError:
            raise ValueError("Cannot parse atom and bond counts from MOL file")

        # Atom block (next num_atoms lines)
        atom_block_start = 4
        atom_block_end = atom_block_start + num_atoms

        if len(lines) < atom_block_end:
            raise ValueError(f"MOL file expects {num_atoms} atoms but only {len(lines)-4} lines found")

        for i in range(atom_block_start, atom_block_end):
            line = lines[i].strip()
            if len(line) < 34:
                continue

            try:
                x = float(line[0:10]) * ANGSTROM_TO_BOHR
                y = float(line[10:20]) * ANGSTROM_TO_BOHR
                z = float(line[20:30]) * ANGSTROM_TO_BOHR
                element_symbol = line[31:34].strip()

                # Convert element symbol to atomic number
                atomic_num = self._element_to_atomic_number(element_symbol)

                self.wfn.add_atom(element_symbol, atomic_num, x, y, z, float(atomic_num))
            except (ValueError, IndexError):
                continue

        # Bond block (next num_bonds lines) - currently ignored for wavefunction
        # Could be useful for molecular structure analysis later

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