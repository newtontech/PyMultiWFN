"""
Parser for Tripos Mol2 files (.mol2).
Mol2 format is a chemical file format developed by Tripos Inc.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class MOL2Loader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse Mol2 file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_mol2(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_mol2(self, lines):
        """Parse Mol2 format."""
        # Find @<TRIPOS> MOLECULE section
        molecule_found = False
        atoms_found = False

        for i, line in enumerate(lines):
            line = line.strip()

            if line == '@<TRIPOS>MOLECULE':
                molecule_found = True
                # Next line is molecule name/title
                if i + 1 < len(lines):
                    self.wfn.title = lines[i + 1].strip()
                continue

            if line == '@<TRIPOS>ATOM' and molecule_found:
                atoms_found = True
                # Parse atom section - skip the next line (header)
                atom_start = i + 2
                self._parse_atoms_section(lines, atom_start)
                break

        if not molecule_found:
            raise ValueError("No @<TRIPOS>MOLECULE section found in Mol2 file")
        if not atoms_found:
            raise ValueError("No @<TRIPOS>ATOM section found in Mol2 file")

    def _parse_atoms_section(self, lines, start_idx):
        """Parse the @<TRIPOS>ATOM section."""
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()

            # Stop at next @<TRIPOS> section or end of file
            if line.startswith('@<TRIPOS>') or not line:
                break

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                # Mol2 format: atom_id atom_name x y z atom_type [...]
                atom_id = int(parts[0])
                atom_name = parts[1]
                x = float(parts[2]) * ANGSTROM_TO_BOHR
                y = float(parts[3]) * ANGSTROM_TO_BOHR
                z = float(parts[4]) * ANGSTROM_TO_BOHR
                atom_type = parts[5]

                # Extract element symbol from atom name or type
                element = self._extract_element(atom_name, atom_type)
                atomic_num = self._element_to_atomic_number(element)

                self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num))
            except (ValueError, IndexError):
                continue

    def _extract_element(self, atom_name: str, atom_type: str) -> str:
        """Extract element symbol from atom name or type."""
        # Try to extract from atom name first (e.g., C1, O2, N3)
        for char in atom_name:
            if char.isalpha():
                element_candidate = char
                if len(atom_name) > 1 and atom_name[1].islower():
                    element_candidate = atom_name[:2]

                if self._element_to_atomic_number(element_candidate) > 0:
                    return element_candidate.title()

        # Try to extract from atom type
        for char in atom_type:
            if char.isalpha():
                element_candidate = char
                if len(atom_type) > 1 and atom_type[1].islower():
                    element_candidate = atom_type[:2]

                if self._element_to_atomic_number(element_candidate) > 0:
                    return element_candidate.title()

        # Default to carbon if can't determine
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