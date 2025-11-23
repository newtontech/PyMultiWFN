import os
from typing import List
from pymultiwfn.core import Wavefunction, Atom

def load_wfn(filepath: str) -> Wavefunction:
    """Placeholder loader for .wfn files.

    This stub reads a simple custom format where the first line is the number of atoms
    followed by lines with element symbol and Cartesian coordinates (Angstrom).
    It returns a minimal Wavefunction instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist")

    wf = Wavefunction()
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return wf
        natoms = int(lines[0])
        from pymultiwfn.core.constants import ANGSTROM_TO_BOHR
        for i in range(1, min(natoms + 1, len(lines))):
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            element = parts[0]
            x, y, z = map(float, parts[1:4])
            x_bohr = x * ANGSTROM_TO_BOHR
            y_bohr = y * ANGSTROM_TO_BOHR
            z_bohr = z * ANGSTROM_TO_BOHR
            atom = Atom(element=element, index=0, x=x_bohr, y=y_bohr, z=z_bohr, charge=0.0)
            wf.atoms.append(atom)
        wf.num_electrons = len(wf.atoms)
    except Exception as e:
        print(f"Error parsing .wfn file: {e}")
    return wf
