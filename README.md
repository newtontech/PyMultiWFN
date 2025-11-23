# PyMultiWFN

**PyMultiWFN** is a modernization of the Multiwfn wavefunction analysis program, rewritten in Python for easier extension and packaging. The MVP keeps the core data structures, FCHK parsing, and density evaluation while avoiding any compilation step so it can ship as a pure-Python wheel to TestPyPI.

## What’s in the MVP
- Pure-Python package, `pip install`-able without compilation.
- `FchkLoader` to read Gaussian `.fchk` files into a `Wavefunction` object.
- Vectorized Gaussian basis evaluation and electron-density calculator.
- Minimal CLI (`pymultiwfn path/to/file.fchk`) for quick inspection.
- Optional Fortran wrappers documented in `pymultiwfn/math/fortran` (not required for install).

## Install (TestPyPI)
Publish a build to TestPyPI, then install:

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine upload --repository testpypi dist/*

# install from TestPyPI (replace VERSION)
python -m pip install -U --extra-index-url https://test.pypi.org/simple pymultiwfn==0.1.1
```

## Quick start
```python
import numpy as np
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density

wfn = load_wavefunction("molecule.fchk")
points = np.array([[0.0, 0.0, 0.0]])  # Bohr
density = calc_density(wfn, points)
print("ρ(0,0,0) =", density[0])
```

Command line:
```bash
pymultiwfn molecule.fchk   # prints header info and a density sanity check
```

## Package layout
```text
pymultiwfn/
├── core/         # Wavefunction, Atom/Shell containers, definitions & constants
├── io/           # Parsers (currently .fchk)
├── math/         # Basis and density evaluation; optional f2py stubs
├── analysis/     # Stubs for bonding/density/orbital analyses (extensible)
├── vis/          # GUI/plotting placeholders
├── utils/        # Helpers
└── config.py     # Runtime configuration singleton
```

## Development tips
- Keep changes additive; use NumPy vectorization for heavy math.
- Use `pymultiwfn/math/fortran/lebedev.pyf` if you need compiled grids, but it is optional for wheels.
- `consistency_verifier/` can be used to compare outputs with the original Multiwfn Fortran code.

## License & citation
- Code: MIT License (see `LICENSE`).
- Please cite the original Multiwfn papers when results rely on its algorithms:  
  Tian Lu, Feiwu Chen, *J. Comput. Chem.* **33**, 580–592 (2012)  
  Tian Lu, *J. Chem. Phys.* **161**, 082503 (2024)
