"""
Parser for GAMESS input files (.gms, .dat).
GAMESS (General Atomic and Molecular Electronic Structure System) input format.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class GMSLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse GAMESS file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_gms(lines)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_gms(self, lines):
        """Parse GAMESS input format."""
        # Find the $DATA section which contains molecular information
        data_section_start = None
        for i, line in enumerate(lines):
            if line.strip().upper().startswith('$DATA'):
                data_section_start = i
                break

        if data_section_start is None:
            raise ValueError("No $DATA section found in GAMESS input file")

        # Parse the $DATA section
        line_idx = data_section_start + 1
        if line_idx < len(lines):
            # First line after $DATA is title
            self.wfn.title = lines[line_idx].strip()
            line_idx += 1

        if line_idx < len(lines):
            # Second line is symmetry information (skip)
            line_idx += 1

        if line_idx >= len(lines):
            raise ValueError("Incomplete $DATA section in GAMESS input file")

        # Parse atoms until we hit $END or end of file
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            line_idx += 1

            # Check for end of data section
            if line.upper().startswith('$END') or line.upper().startswith(' $END'):
                break

            if not line:
                continue

            # Parse atom line format:
            # Atomic_number Atomic_symbol x y z
            # or
            # Atomic_symbol Atomic_number x y z (some variations)
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                # Try different parsing strategies
                atomic_num = None
                element = None
                x, y, z = 0.0, 0.0, 0.0

                # Strategy 1: Atomic_number Atomic_symbol x y z
                try:
                    atomic_num = int(parts[0])
                    element = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                except ValueError:
                    # Strategy 2: Atomic_symbol Atomic_number x y z
                    element = parts[0]
                    atomic_num = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])

                # Validate element symbol
                if self._element_to_atomic_number(element) != atomic_num:
                    # If mismatch, try to correct based on atomic number
                    element = self._atomic_number_to_element(atomic_num)

                # Convert coordinates to Bohr (GAMESS coordinates are usually in Bohr)
                # but some versions use Angstroms - check context if available
                x_bohr = x  # Assume Bohr by default
                y_bohr = y
                z_bohr = z

                atom_info = {
                    'gms_atomic_number': atomic_num,
                    'gms_symbol': element
                }

                self.wfn.add_atom(element, atomic_num, x_bohr, y_bohr, z_bohr, float(atomic_num), atom_info)

            except (ValueError, IndexError):
                continue

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        element_mapping = {
            'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
            'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
            'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26,
            'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34,
            'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42,
            'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
            'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58,
            'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66,
            'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74,
            'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82,
            'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90,
            'PA': 91, 'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 'BK': 97, 'CF': 98,
            'ES': 99, 'FM': 100, 'MD': 101, 'NO': 102, 'LR': 103, 'RF': 104, 'DB': 105,
            'SG': 106, 'BH': 107, 'HS': 108, 'MT': 109, 'DS': 110, 'RG': 111, 'CN': 112,
            'NH': 113, 'FL': 114, 'MC': 115, 'LV': 116, 'TS': 117, 'OG': 118
        }
        return element_mapping.get(element.upper(), 0)

    def _atomic_number_to_element(self, atomic_num: int) -> str:
        """Convert atomic number to element symbol."""
        number_to_element = {
            1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'NE',
            11: 'NA', 12: 'MG', 13: 'AL', 14: 'SI', 15: 'P', 16: 'S', 17: 'CL', 18: 'AR',
            19: 'K', 20: 'CA', 21: 'SC', 22: 'TI', 23: 'V', 24: 'CR', 25: 'MN', 26: 'FE',
            27: 'CO', 28: 'NI', 29: 'CU', 30: 'ZN', 31: 'GA', 32: 'GE', 33: 'AS', 34: 'SE',
            35: 'BR', 36: 'KR', 37: 'RB', 38: 'SR', 39: 'Y', 40: 'ZR', 41: 'NB', 42: 'MO',
            43: 'TC', 44: 'RU', 45: 'RH', 46: 'PD', 47: 'AG', 48: 'CD', 49: 'IN', 50: 'SN',
            51: 'SB', 52: 'TE', 53: 'I', 54: 'XE', 55: 'CS', 56: 'BA', 57: 'LA', 58: 'CE',
            59: 'PR', 60: 'ND', 61: 'PM', 62: 'SM', 63: 'EU', 64: 'GD', 65: 'TB', 66: 'DY',
            67: 'HO', 68: 'ER', 69: 'TM', 70: 'YB', 71: 'LU', 72: 'HF', 73: 'TA', 74: 'W',
            75: 'RE', 76: 'OS', 77: 'IR', 78: 'PT', 79: 'AU', 80: 'HG', 81: 'TL', 82: 'PB',
            83: 'BI', 84: 'PO', 85: 'AT', 86: 'RN', 87: 'FR', 88: 'RA', 89: 'AC', 90: 'TH',
            91: 'PA', 92: 'U', 93: 'NP', 94: 'PU', 95: 'AM', 96: 'CM', 97: 'BK', 98: 'CF',
            99: 'ES', 100: 'FM', 101: 'MD', 102: 'NO', 103: 'LR', 104: 'RF', 105: 'DB',
            106: 'SG', 107: 'BH', 108: 'HS', 109: 'MT', 110: 'DS', 111: 'RG', 112: 'CN',
            113: 'NH', 114: 'FL', 115: 'MC', 116: 'LV', 117: 'TS', 118: 'OG'
        }
        return number_to_element.get(atomic_num, 'C').title()