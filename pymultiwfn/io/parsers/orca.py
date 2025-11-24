"""
Parsers for ORCA input and output files.
ORCA is a quantum chemistry program for electronic structure calculation.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class ORCALoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse ORCA file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        # Determine file type by content
        if self._is_orca_input(content):
            self._parse_orca_input(content)
        else:
            self._parse_orca_output(content)

        self.wfn._infer_occupations()
        return self.wfn

    def _is_orca_input(self, content: str) -> bool:
        """Determine if this is an ORCA input file."""
        lines = content.strip().split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('!'):
                return True  # ORCA input files start with ! (method/basis specification)
            if line.upper().startswith('* XYZ') or line.upper().startswith('*'):
                return True  # ORCA coordinate specification
        return False

    def _parse_orca_input(self, content: str):
        """Parse ORCA input file format."""
        lines = content.strip().split('\n')

        # Extract title (first non-comment, non-! line)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('!') and not line.startswith('#'):
                if not line.startswith('*'):  # Not a coordinate block
                    self.wfn.title = line
                    break

        # Find coordinate block
        coord_block_found = False
        coord_lines = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Start of coordinate block
            if line_stripped.upper().startswith('* XYZ') or line_stripped.upper().startswith('*'):
                coord_block_found = True
                i += 1
                # Skip any title lines after *
                while i < len(lines) and not lines[i].strip().split():
                    i += 1
                # Get number of atoms if specified
                try:
                    num_atoms = int(lines[i].strip().split()[0])
                    i += 1
                except (ValueError, IndexError):
                    num_atoms = None

                # Read coordinate lines
                while i < len(lines):
                    coord_line = lines[i].strip()
                    i += 1

                    # End of coordinate block
                    if coord_line.upper().startswith('*'):
                        break

                    if coord_line:
                        coord_lines.append(coord_line)

                break

        if not coord_block_found:
            raise ValueError("No coordinate block found in ORCA input file")

        # Parse coordinate lines
        # ORCA format: element_symbol x y z [optional: charge]
        for coord_line in coord_lines:
            parts = coord_line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0].title()
                    x = float(parts[1]) * ANGSTROM_TO_BOHR  # ORCA uses Angstroms
                    y = float(parts[2]) * ANGSTROM_TO_BOHR
                    z = float(parts[3]) * ANGSTROM_TO_BOHR

                    # Optional charge parameter
                    if len(parts) >= 5:
                        try:
                            charge = float(parts[4])
                        except ValueError:
                            charge = 0.0
                    else:
                        charge = 0.0

                    atomic_num = self._element_to_atomic_number(element)
                    atom_info = {'charge': charge}

                    self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num), atom_info)
                except (ValueError, IndexError):
                    continue

    def _parse_orca_output(self, content: str):
        """Parse ORCA output file format."""
        lines = content.strip().split('\n')

        # Extract title from calculation setup
        for line in lines:
            if 'CALCULATION INPUT' in line.upper() or 'INPUT FILE' in line.upper():
                # Look for title in subsequent lines
                # This is simplified - ORCA output parsing can be quite complex
                break

        # Find Cartesian coordinates section
        coord_section_found = False

        for i, line in enumerate(lines):
            line_upper = line.upper()

            # Look for coordinate sections
            if any(keyword in line_upper for keyword in [
                'CARTESIAN COORDINATES (ANGSTROM)',
                'CARTESIAN COORDINATES (A.U.)',
                'COORDINATES IN ANGSTROM',
                'XYZ COORDINATES'
            ]):
                coord_section_found = True
                # Determine if coordinates are in Angstrom or Bohr
                use_bohr = 'A.U.' in line_upper or 'BOHR' in line_upper

                # Skip header lines
                i += 2
                while i < len(lines):
                    coord_line = lines[i].strip()
                    i += 1

                    # Check if we've reached the end of coordinate section
                    if not coord_line or coord_line.startswith('---') or coord_line.startswith('='):
                        continue

                    # Try to parse coordinate line
                    parts = coord_line.split()
                    if len(parts) >= 5:
                        try:
                            atom_index = int(parts[0])
                            element = parts[1].title()
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])

                            # Convert to Bohr if needed
                            if not use_bohr:
                                x *= ANGSTROM_TO_BOHR
                                y *= ANGSTROM_TO_BOHR
                                z *= ANGSTROM_TO_BOHR

                            atomic_num = self._element_to_atomic_number(element)
                            atom_info = {'orca_atom_index': atom_index}

                            self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num), atom_info)
                        except (ValueError, IndexError):
                            continue

                    # Stop at reasonable end of coordinate section
                    if i < len(lines) and lines[i].strip() and not lines[i][0].isdigit():
                        break

        if not coord_section_found:
            raise ValueError("No coordinate section found in ORCA output file")

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