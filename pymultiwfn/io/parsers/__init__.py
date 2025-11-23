"""
Parsers for common wavefunction file formats.
Currently supports Gaussian formatted checkpoint (.fchk).
"""

from .fchk import FchkLoader

__all__ = ["FchkLoader"]
