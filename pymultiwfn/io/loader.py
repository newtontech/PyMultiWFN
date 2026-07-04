# pymultiwfn/io/loader.py

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.io.file_manager import FileManager

# Instantiate FileManager once to handle all loading requests
_file_manager = FileManager()


def load_wavefunction(file_path: str) -> Wavefunction:
    """
    Convenience function to load a wavefunction from a file using the FileManager.
    """
    return _file_manager.load_wavefunction(file_path)


# You can add other top-level loading functions here if needed,
# dispatching to the FileManager for different types of data.
