"""
Parser factory for automatically selecting the appropriate parser based on file extension.
"""

import os
from typing import Optional, Dict, Type, List
from pymultiwfn.core.data import Wavefunction

# Import all parser classes
from .fchk import FchkLoader
from .molden import MoldenLoader
from .wfn import WFNLoader
from .wfx import WFXLoader
from .mwfn import MWFNLoader
from .xyz import XYZLoader
from .pdb import PDBLoader
from .cube import CubeLoader
from .gjf import GJFLoader
from .cp2k import CP2KLoader
from .mol import MOLLoader
from .mol2 import MOL2Loader
from .pqr import PQRLoader
from .gro import GROLoader
from .cif import CIFLoader
from .gms import GMSLoader
from .mopac import MOPACLoader
from .orca import ORCALoader
from .turbomole import TurbomoleLoader
from .vasp import VASPLoader
from .dx import DXLoader

class ParserFactory:
    """Factory class for creating appropriate parser instances based on file extension."""

    # Mapping of file extensions to parser classes
    PARSERS: Dict[str, Type] = {
        # Wavefunction formats
        '.fchk': FchkLoader,
        '.fch': FchkLoader,
        '.molden': MoldenLoader,
        '.molf': MoldenLoader,
        '.wfn': WFNLoader,
        '.wfx': WFXLoader,
        '.mwfn': MWFNLoader,

        # Coordinate formats
        '.xyz': XYZLoader,
        '.pdb': PDBLoader,
        '.mol': MOLLoader,
        '.sdf': MOLLoader,
        '.mol2': MOL2Loader,
        '.pqr': PQRLoader,
        '.gro': GROLoader,
        '.cif': CIFLoader,

        # Input file formats
        '.gjf': GJFLoader,
        '.com': GJFLoader,
        '.gjf': GJFLoader,
        '.gms': GMSLoader,
        '.dat': GMSLoader,  # GAMESS files
        '.mop': MOPACLoader,
        '.inp': ORCALoader,  # ORCA input
        '.out': ORCALoader,  # ORCA output

        # Grid data formats
        '.cube': CubeLoader,
        '.cub': CubeLoader,
        '.dx': DXLoader,

        # Program-specific formats
        '.inp': CP2KLoader,
        '.restart': CP2KLoader,
        '.coord': TurbomoleLoader,  # Turbomole coordinate files
        'poscar': VASPLoader,
        'contcar': VASPLoader,
        'chgc': VASPLoader,  # VASP charge density
        'chg': VASPLoader,
        'elfc': VASPLoader,  # VASP ELF
        'locpot': VASPLoader,
    }

    @classmethod
    def get_parser(cls, filename: str, file_format: Optional[str] = None):
        """
        Get the appropriate parser for a given file.

        Parameters:
        -----------
        filename : str
            Path to the file to parse
        file_format : str, optional
            Explicit file format to use, bypassing extension detection

        Returns:
        --------
        Parser instance for the given file

        Raises:
        -------
        ValueError: If no suitable parser is found
        FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # If explicit format is specified, use it
        if file_format:
            file_format = file_format.lower()
            if file_format.startswith('.'):
                file_format = file_format[1:]
            extension = '.' + file_format
        else:
            # Determine format from file extension
            _, ext = os.path.splitext(filename.lower())
            extension = ext

        # Special handling for molden files which can have various extensions
        if 'molden' in filename.lower():
            extension = '.molden'

        # Get parser class
        if extension in cls.PARSERS:
            parser_class = cls.PARSERS[extension]
            return parser_class(filename)
        else:
            # Try content-based detection
            parser_class = cls._detect_from_content(filename)
            if parser_class:
                return parser_class(filename)

            # Try some alternative extensions
            alternative_extensions = {
                '.fch': '.fchk',
                '.molf': '.molden',
            }

            if extension in alternative_extensions:
                alt_ext = alternative_extensions[extension]
                if alt_ext in cls.PARSERS:
                    parser_class = cls.PARSERS[alt_ext]
                    return parser_class(filename)

            raise ValueError(f"No parser available for file extension '{extension}'. "
                           f"Supported formats: {', '.join(cls.PARSERS.keys())}")

    @classmethod
    def _detect_from_content(cls, filename: str) -> Optional[Type]:
        """
        Try to detect file format from file content.

        Parameters:
        -----------
        filename : str
            Path to the file to analyze

        Returns:
        --------
        Parser class if format is detected, None otherwise
        """
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(10)]

            content = '\n'.join(first_lines)

            # Check for Gaussian formatted checkpoint
            if 'Gaussian' in content and 'Formatted Checkpoint' in content:
                return FchkLoader

            # Check for Molden format
            if '[Molden Format]' in content or '[Atoms]' in content:
                return MoldenLoader

            # Check for WFN format
            if any(line and line.split()[0].isdigit() for line in first_lines if line):
                # Could be WFN or similar format
                try:
                    second_line = first_lines[1] if len(first_lines) > 1 else ""
                    if second_line and len(second_line.split()) >= 4:
                        # Likely WFN format (NMO NPRIMITIVES NELECTRONS MULTIPLICITY)
                        return WFNLoader
                except:
                    pass

            # Check for XYZ format
            if first_lines and first_lines[0].strip().isdigit():
                # First line is number of atoms - typical XYZ format
                return XYZLoader

            # Check for PDB format
            if any(line.startswith(('ATOM', 'HETATM', 'TITLE', 'HEADER')) for line in first_lines):
                return PDBLoader

            # Check for Cube format
            if len(first_lines) >= 2 and first_lines[1].strip().isdigit():
                try:
                    # Check if second line looks like atom count + origin
                    parts = first_lines[1].split()
                    if len(parts) >= 4:
                        return CubeLoader
                except:
                    pass

            # Check for DX format
            if any('object' in line and ('gridpositions' in line or 'field' in line) for line in first_lines):
                return DXLoader

            # Check for GAMESS format ($DATA section)
            if any('$DATA' in line.upper() for line in first_lines):
                return GMSLoader

            # Check for ORCA input format (! and * xyz)
            if any(line.startswith('!') for line in first_lines[:3]):
                return ORCALoader

            # Check for MOPAC format
            if any('XYZ' in line.upper() for line in first_lines[:1]):
                return MOPACLoader

            # Check for Turbomole format ($coord section)
            if any('$COORD' in line.upper() for line in first_lines):
                return TurbomoleLoader

            # Check for MOL format (counts line)
            if len(first_lines) >= 4:
                try:
                    counts_line = first_lines[3].strip()
                    if len(counts_line) >= 6 and counts_line[0:3].isdigit() and counts_line[3:6].isdigit():
                        return MOLLoader
                except:
                    pass

            # Check for MOL2 format (@<TRIPOS>)
            if any('@<TRIPOS>' in line for line in first_lines):
                return MOL2Loader

            # Check for CIF format (data_ or loop_)
            if any(line.startswith('data_') or line.startswith('loop_') for line in first_lines):
                return CIFLoader

            # Check for GRO format (second line is atom count)
            if len(first_lines) >= 2:
                try:
                    int(first_lines[1].strip())
                    # Additional check for GRO format (residue format)
                    if len(first_lines) >= 3 and len(first_lines[2].split()) >= 5:
                        return GROLoader
                except:
                    pass

        except Exception:
            # If we can't read the file or parse content, return None
            pass

        return None

    @classmethod
    def load_file(cls, filename: str, file_format: Optional[str] = None) -> Wavefunction:
        """
        Load a file and return a Wavefunction object.

        Parameters:
        -----------
        filename : str
            Path to the file to load
        file_format : str, optional
            Explicit file format to use

        Returns:
        --------
        Wavefunction object containing the parsed data
        """
        parser = cls.get_parser(filename, file_format)
        return parser.load()

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of supported file formats.

        Returns:
        --------
        List of supported file extensions
        """
        return list(cls.PARSERS.keys())

    @classmethod
    def register_parser(cls, extension: str, parser_class: Type):
        """
        Register a new parser class for a given file extension.

        Parameters:
        -----------
        extension : str
            File extension (e.g., '.myformat')
        parser_class : Type
            Parser class to register
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        cls.PARSERS[extension.lower()] = parser_class