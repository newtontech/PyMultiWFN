"""
Parsers for common wavefunction file formats.
Supports Gaussian formatted checkpoint (.fchk), Molden (.molden), WFN (.wfn),
and other quantum chemistry file formats.
"""

# Import core parsers for MVP
from .fchk import FchkLoader

# Optional imports (with fallback)
try:
    from .molden import MoldenLoader
    from .wfn import WFNLoader
    from .wfx import WFXLoader
    from .xyz import XYZLoader
    from .cube import CubeLoader
    from .cp2k import CP2KLoader
except ImportError:
    MoldenLoader = None
    WFNLoader = None
    WFXLoader = None
    XYZLoader = None
    CubeLoader = None
    CP2KLoader = None

__all__ = [
    "FchkLoader", "MoldenLoader", "WFNLoader", "WFXLoader",
    "XYZLoader", "CubeLoader", "CP2KLoader"
]
