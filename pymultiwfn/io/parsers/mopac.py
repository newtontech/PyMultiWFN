"""
Parser for MOPAC input files (.mop, .MOP).
"""

from pymultiwfn.core.data import Wavefunction

class MOPACLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse MOPAC file and return Wavefunction object."""
        # TODO: Implement MOPAC input parsing
        raise NotImplementedError("MOPAC parser not yet implemented")