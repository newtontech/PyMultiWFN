"""
Parsers for common wavefunction file formats.
Supports Gaussian formatted checkpoint (.fchk), Molden (.molden), WFN (.wfn),
and other quantum chemistry file formats.
"""

from .fchk import FchkLoader
from .molden import MoldenLoader
from .wfn import WFNLoader
try:
    from .cp2k import CP2KLoader
except ImportError:
    CP2KLoader = None

__all__ = [
    "FchkLoader", "MoldenLoader", "WFNLoader", "CP2KLoader"
]
