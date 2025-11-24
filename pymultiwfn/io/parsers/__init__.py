"""
Parsers for common wavefunction file formats.
Supports Gaussian formatted checkpoint (.fchk), Molden (.molden), WFN (.wfn),
WFX (.wfx), XYZ (.xyz), Cube (.cube), CP2K (.out), PDB (.pdb), PQR (.pqr),
and other quantum chemistry file formats.
"""

# Import core parsers for MVP
from .fchk import FchkLoader

# All available parsers
try:
    from .molden import MoldenLoader
except ImportError:
    MoldenLoader = None

try:
    from .wfn import WFNLoader
except ImportError:
    WFNLoader = None

try:
    from .wfx import WFXLoader
except ImportError:
    WFXLoader = None

try:
    from .xyz import XYZLoader
except ImportError:
    XYZLoader = None

try:
    from .cube import CubeLoader
except ImportError:
    CubeLoader = None

try:
    from .cp2k import CP2KLoader
except ImportError:
    CP2KLoader = None

try:
    from .pdb import PDBLoader
except ImportError:
    PDBLoader = None

try:
    from .pqr import PQRLoader
except ImportError:
    PQRLoader = None

__all__ = [
    "FchkLoader", "MoldenLoader", "WFNLoader", "WFXLoader",
    "XYZLoader", "CubeLoader", "CP2KLoader", "PDBLoader", "PQRLoader"
]
