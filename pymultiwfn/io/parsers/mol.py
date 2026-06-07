"""
Parser for MDL Mol files (.mol, .sdf).
MDL Mol format is a chemical file format for storing molecular information.
"""

from pymultiwfn.core.constants import ANGSTROM_TO_BOHR
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import get_atomic_number


class MOLLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse MOL/SDF file and return Wavefunction object."""
        with open(self.filename, "r") as f:
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
        except ValueError:
            raise ValueError("Cannot parse atom and bond counts from MOL file")

        # Atom block (next num_atoms lines)
        atom_block_start = 4
        atom_block_end = atom_block_start + num_atoms

        if len(lines) < atom_block_end:
            raise ValueError(
                f"MOL file expects {num_atoms} atoms but only {len(lines)-4} lines found"
            )

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

                self.wfn.add_atom(
                    element_symbol, atomic_num, x, y, z, float(atomic_num)
                )
            except (ValueError, IndexError):
                continue

        # Bond block (next num_bonds lines) - currently ignored for wavefunction
        # Could be useful for molecular structure analysis later

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        return get_atomic_number(element)
