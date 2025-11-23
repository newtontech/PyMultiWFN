# Fortran Extensions for PyMultiWFN

This directory contains Fortran extensions that are wrapped using `f2py` for performance.

## 1. Lebedev Grid Generator

The file `Lebedev-Laikov.F` (from the original Multiwfn source) contains efficient routines for generating spherical integration grids.

### Prerequisites

*   A Fortran compiler (e.g., `gfortran`, `ifort`).
*   `numpy` installed in your Python environment.

### Compilation Instructions

1.  **Copy the Source File**:
    Copy `Lebedev-Laikov.F` from the original Multiwfn source directory to this directory (`pymultiwfn/math/fortran/`).

2.  **Generate/Verify Signature (Optional)**:
    If the provided `lebedev.pyf` is incorrect, you can regenerate it:
    ```bash
    python -m numpy.f2py Lebedev-Laikov.F -h lebedev.pyf -m lebedev --overwrite-signature
    ```
    *Note: You may need to adjust the `intent` attributes in the generated `.pyf` file manually.*

3.  **Compile**:
    Run the following command to build the extension:

    **Linux/macOS**:
    ```bash
    python -m numpy.f2py -c lebedev.pyf Lebedev-Laikov.F
    ```

    **Windows** (using MinGW or Intel Fortran):
    ```powershell
    python -m numpy.f2py -c lebedev.pyf Lebedev-Laikov.F --compiler=mingw32
    # Or for MSVC/Intel if configured:
    # python -m numpy.f2py -c lebedev.pyf Lebedev-Laikov.F
    ```

    This will generate a file named `lebedev.cpython-XY-platform.so` (Linux) or `.pyd` (Windows).

## 2. Usage in Python

Once compiled, you can import and use the module as follows:

```python
import numpy as np
from pymultiwfn.math.fortran import lebedev

def get_spherical_grid(n_points):
    """
    Generates a Lebedev spherical grid.
    """
    # Allocate arrays
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)
    w = np.zeros(n_points)
    
    # Call Fortran routine
    # Note: Function name might be lowercase
    lebedev.lebedevgen(n_points, x, y, z, w)
    
    return x, y, z, w
```
