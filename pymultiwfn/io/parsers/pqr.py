"""
Parser for PQR files (.pqr, .PQR).
PQR format is similar to PDB but includes partial charges and radii.
"""

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class PQRLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse PQR file and return Wavefunction object."""
        # PQR format is similar to PDB but with charge and radius information
        # For now, delegate to PDB parser with additional charge handling
        from .pdb import PDBLoader
        pdb_loader = PDBLoader(self.filename)
        wfn = pdb_loader.load()

        # TODO: Parse additional PQR-specific charge and radius information
        return wfn