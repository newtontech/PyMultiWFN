"""
Enhanced file manager for PyMultiWFN with comprehensive format support.

This module provides a high-level interface for managing file input/output operations
including loading, saving, and format conversion for various quantum chemistry file formats.
"""

import os
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.io import load, get_supported_formats, validate_file


class FileManager:
    """
    Enhanced file manager for PyMultiWFN.

    Provides unified interface for loading, saving, and converting various
    quantum chemistry file formats with comprehensive error handling
    and validation.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the file manager.

        Args:
            verbose: Whether to print verbose output during operations
        """
        self.verbose = verbose
        self.supported_formats = get_supported_formats()
        self.loaded_files: Dict[str, Wavefunction] = {}  # Cache for loaded files

    def load_wavefunction(self, file_path: str, use_cache: bool = True, **kwargs) -> Wavefunction:
        """
        Load wavefunction data from a specified file path.

        Args:
            file_path: Path to the input file
            use_cache: Whether to use cached data if file was previously loaded
            **kwargs: Additional keyword arguments passed to the loader

        Returns:
            Wavefunction: Loaded wavefunction object

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check cache first
        if use_cache and file_path in self.loaded_files:
            if self.verbose:
                print(f"Using cached wavefunction for {file_path}")
            return self.loaded_files[file_path]

        try:
            if self.verbose:
                print(f"Loading wavefunction from {file_path}")

            # Use the unified load function from io module
            wavefunction = load(file_path, **kwargs)

            # Cache the result
            if use_cache:
                self.loaded_files[file_path] = wavefunction

            if self.verbose:
                print(f"Successfully loaded wavefunction: {wavefunction.num_atoms} atoms, "
                      f"{wavefunction.num_basis} basis functions")

            return wavefunction

        except Exception as e:
            raise ValueError(f"Failed to load wavefunction from {file_path}: {str(e)}")

    def save_wavefunction(self, wavefunction: Wavefunction, file_path: str, format_hint: Optional[str] = None, **kwargs):
        """
        Save wavefunction data to a specified file path.

        Args:
            wavefunction: Wavefunction object to save
            file_path: Output file path
            format_hint: Optional hint for output format (e.g., 'fchk', 'molden')
            **kwargs: Additional keyword arguments for the writer

        Raises:
            NotImplementedError: If the output format is not yet supported
            ValueError: If the wavefunction data is incomplete
        """
        file_path = os.path.abspath(file_path)

        if self.verbose:
            print(f"Saving wavefunction to {file_path}")

        # Validate wavefunction data
        self._validate_wavefunction_for_saving(wavefunction)

        # Determine output format
        _, ext = os.path.splitext(file_path.lower())
        output_format = format_hint or ext[1:] if ext else None

        if not output_format:
            raise ValueError(f"Cannot determine output format for {file_path}")

        # Dispatch to appropriate writer
        if output_format in ['fchk', 'fch']:
            self._save_fchk(wavefunction, file_path, **kwargs)
        elif output_format in ['molden', 'molf']:
            self._save_molden(wavefunction, file_path, **kwargs)
        elif output_format in ['wfn', 'wfx']:
            self._save_wfn(wavefunction, file_path, **kwargs)
        else:
            raise NotImplementedError(f"Output format '{output_format}' is not yet supported")

        if self.verbose:
            print(f"Successfully saved wavefunction to {file_path}")

    def convert_file(self, input_path: str, output_path: str, **kwargs) -> Wavefunction:
        """
        Convert between different quantum chemistry file formats.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            **kwargs: Additional arguments for conversion

        Returns:
            Wavefunction: The loaded wavefunction object
        """
        if self.verbose:
            print(f"Converting {input_path} to {output_path}")

        # Load input file
        wavefunction = self.load_wavefunction(input_path, use_cache=False)

        # Save to output format
        self.save_wavefunction(wavefunction, output_path, **kwargs)

        return wavefunction

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file can be loaded by the file manager.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file is valid and can be loaded, False otherwise
        """
        return validate_file(file_path)

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a file without fully loading it.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return {"error": "File not found"}

        info = {
            "path": file_path,
            "size": os.path.getsize(file_path),
            "extension": os.path.splitext(file_path)[1].lower(),
            "exists": True,
            "readable": os.access(file_path, os.R_OK),
            "format": None,
            "description": None
        }

        # Determine format
        if info["extension"] in self.supported_formats:
            info["format"] = info["extension"]
            info["description"] = self.supported_formats[info["extension"]]
        else:
            # Try to auto-detect
            try:
                from pymultiwfn.io import _auto_detect_format
                loader = _auto_detect_format(file_path)
                if loader:
                    info["format"] = type(loader).__name__.replace('Loader', '').lower()
                    info["description"] = "Auto-detected format"
                else:
                    info["format"] = "unknown"
                    info["description"] = "Unknown format"
            except Exception:
                info["format"] = "unknown"
                info["description"] = "Cannot determine format"

        return info

    def clear_cache(self, file_path: Optional[str] = None):
        """
        Clear cached wavefunction data.

        Args:
            file_path: Specific file to remove from cache, or None to clear all
        """
        if file_path:
            file_path = os.path.abspath(file_path)
            self.loaded_files.pop(file_path, None)
            if self.verbose:
                print(f"Cleared cache for {file_path}")
        else:
            self.loaded_files.clear()
            if self.verbose:
                print("Cleared all cached wavefunction data")

    def _validate_wavefunction_for_saving(self, wavefunction: Wavefunction):
        """Validate that wavefunction has sufficient data for saving."""
        if not wavefunction.atoms:
            raise ValueError("Wavefunction has no atomic coordinates")

        if not wavefunction.shells:
            raise ValueError("Wavefunction has no basis function information")

        # Additional validation can be added here

    def _save_fchk(self, wavefunction: Wavefunction, file_path: str, **kwargs):
        """Save wavefunction in Gaussian FCHK format."""
        # Placeholder implementation
        raise NotImplementedError("FCHK output writer is not yet implemented")

    def _save_molden(self, wavefunction: Wavefunction, file_path: str, **kwargs):
        """Save wavefunction in Molden format."""
        # Placeholder implementation
        raise NotImplementedError("Molden output writer is not yet implemented")

    def _save_wfn(self, wavefunction: Wavefunction, file_path: str, **kwargs):
        """Save wavefunction in WFN format."""
        # Placeholder implementation
        raise NotImplementedError("WFN output writer is not yet implemented")

    def list_supported_formats(self) -> Dict[str, str]:
        """Get a dictionary of supported file formats."""
        return self.supported_formats.copy()
