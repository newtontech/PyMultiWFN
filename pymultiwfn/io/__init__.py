"""
Enhanced IO module for PyMultiWFN with comprehensive file format support.

This module provides a unified interface for loading various quantum chemistry
file formats including Gaussian FCHK, WFN, Molden, and others.
"""

import os
import warnings
from typing import Dict, Any, Optional

from .parsers.fchk import FchkLoader
from .parsers.wfn import WFNLoader
from .parsers.molden import MoldenLoader
try:
    from .parsers.cp2k import CP2KLoader
except ImportError:
    CP2KLoader = None


# Registry of supported file formats and their loaders
FILE_FORMATS = {
    '.fchk': FchkLoader,
    '.fch': FchkLoader,
    '.wfn': WFNLoader,
    '.wfx': WFNLoader,
    '.molden': MoldenLoader,
    '.molden.input': MoldenLoader,
    '.molf': MoldenLoader,
    '.inp': MoldenLoader,  # ORCA molden.inp
}

# Add CP2K loader if available
if CP2KLoader:
    FILE_FORMATS['.cp2k'] = CP2KLoader


def load(filename: str, **kwargs) -> Any:
    """
    Generic loader that detects file type and returns a Wavefunction object.

    Args:
        filename: Path to the input file
        **kwargs: Additional keyword arguments passed to the loader

    Returns:
        Wavefunction: Loaded wavefunction object

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Check file extension
    _, ext = os.path.splitext(filename.lower())

    # Handle special cases
    if 'molden' in filename.lower() and ext not in FILE_FORMATS:
        loader = MoldenLoader(filename)
        return loader.load()

    if ext in FILE_FORMATS:
        loader_class = FILE_FORMATS[ext]
        loader = loader_class(filename, **kwargs)
        return loader.load()
    else:
        # Try to auto-detect format
        loader = _auto_detect_format(filename)
        if loader:
            return loader.load()
        else:
            raise NotImplementedError(
                f"File type '{ext}' for {filename} not yet supported. "
                f"Supported formats: {', '.join(FILE_FORMATS.keys())}"
            )


def _auto_detect_format(filename: str) -> Optional[Any]:
    """
    Attempt to auto-detect file format by examining file content.

    Args:
        filename: Path to the file

    Returns:
        Loader instance if format is detected, None otherwise
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            first_lines = [f.readline().strip() for _ in range(10)]

    # Check for Molden format
    for line in first_lines:
        if line.startswith('[Atoms]') or line.startswith('[GTO]') or line.startswith('[MO]'):
            return MoldenLoader(filename)

    # Check for WFN format (numeric header)
    if first_lines and all(part.replace('.', '').replace('-', '').isdigit()
                          for part in first_lines[0].split() if part):
        return WFNLoader(filename)

    # Check for FCHK format (typical Gaussian header)
    if first_lines and ('Gaussian' in first_lines[0] or 'Formated Checkpoint' in first_lines[0]):
        return FchkLoader(filename)

    return None


def get_supported_formats() -> Dict[str, str]:
    """
    Get a dictionary of supported file formats and their descriptions.

    Returns:
        Dictionary mapping file extensions to format descriptions
    """
    formats = {
        '.fchk': 'Gaussian Formatted Checkpoint file',
        '.fch': 'Gaussian Formatted Checkpoint file',
        '.wfn': 'Gaussian Wavefunction file',
        '.wfx': 'Gaussian Wavefunction file (extended)',
        '.molden': 'Molden visualization file',
        '.molden.input': 'Molden file (ORCA format)',
        '.molf': 'Molden file',
        '.inp': 'Molden input file (ORCA format)',
    }

    # Add CP2K format if loader is available
    if CP2KLoader:
        formats['.cp2k'] = 'CP2K output file'

    return formats


def validate_file(filename: str) -> bool:
    """
    Validate if a file can be loaded by the IO module.

    Args:
        filename: Path to the file

    Returns:
        True if file can be loaded, False otherwise
    """
    try:
        load(filename)
        return True
    except Exception:
        return False
