from .parsers.fchk import FchkLoader

def load(filename: str):
    """
    Generic loader that detects file type and returns a Wavefunction.
    Currently supports: .fchk
    """
    if filename.lower().endswith('.fchk') or filename.lower().endswith('.fch'):
        loader = FchkLoader(filename)
        return loader.load()
    else:
        raise NotImplementedError(f"File type for {filename} not yet supported.")
