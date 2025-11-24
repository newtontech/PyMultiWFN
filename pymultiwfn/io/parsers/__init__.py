"""
Parsers for common wavefunction file formats.
Supports Gaussian formatted checkpoint (.fchk), Molden (.molden), WFN (.wfn),
and other quantum chemistry file formats.
"""

# Import all working parsers
from .fchk import FchkLoader
from .molden import MoldenLoader
from .wfn import WFNLoader
from .wfx import WFXLoader
from .mwfn import MWFNLoader
from .xyz import XYZLoader
from .pdb import PDBLoader
from .cube import CubeLoader
from .gjf import GJFLoader
from .mol import MOLLoader
from .mol2 import MOL2Loader
from .pqr import PQRLoader
from .gro import GROLoader
from .cif import CIFLoader
from .gms import GMSLoader
from .mopac import MOPACLoader
from .orca import ORCALoader
from .turbomole import VASPLoader
from .vasp import VASPLoader
from .dx import DXLoader

# Optional imports (with fallback)
try:
    from .cp2k import CP2KLoader
except ImportError:
    CP2KLoader = None

__all__ = [
    "FchkLoader", "MoldenLoader", "WFNLoader", "WFXLoader", "MWFNLoader",
    "XYZLoader", "PDBLoader", "CubeLoader", "GJFLoader",
    "MOLLoader", "MOL2Loader", "PQRLoader", "GROLoader", "CIFLoader",
    "GMSLoader", "MOPACLoader", "ORCALoader", "VASPLoader", "DXLoader",
    "CP2KLoader"
]
