"""
Parser for GROMACS coordinate files (.gro).
"""

from pymultiwfn.core.data import Wavefunction

class GROLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse GRO file and return Wavefunction object."""
        # TODO: Implement GRO file parsing for GROMACS coordinates
        raise NotImplementedError("GRO parser not yet implemented")