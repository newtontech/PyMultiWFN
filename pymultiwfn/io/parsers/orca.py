"""
Parsers for ORCA input and output files.
"""

from pymultiwfn.core.data import Wavefunction

class ORCALoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse ORCA file and return Wavefunction object."""
        # TODO: Implement ORCA input/output parsing
        # Need to distinguish between .inp input files and .out output files
        raise NotImplementedError("ORCA parser not yet implemented")