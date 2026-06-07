"""
Parser for XYZ coordinate files (.xyz, .XYZ).
XYZ format is a simple text format for molecular coordinates.
"""

from pymultiwfn.core.constants import ANGSTROM_TO_BOHR
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import get_atomic_number


class XYZLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse XYZ file and return Wavefunction object."""
        with open(self.filename, "r") as f:
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
        atom_lines = lines[2 : 2 + num_atoms]

        for i, line in enumerate(atom_lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0].title()  # Capitalize first letter
                    x = float(parts[1]) * ANGSTROM_TO_BOHR
                    y = float(parts[2]) * ANGSTROM_TO_BOHR
                    z = float(parts[3]) * ANGSTROM_TO_BOHR

                    atomic_num = self._element_to_atomic_number(element)

                    self.wfn.add_atom(element, atomic_num, x, y, z, float(atomic_num))
                except (ValueError, IndexError):
                    continue

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        return get_atomic_number(element)
