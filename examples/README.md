# PyMultiWFN Examples

This directory contains example scripts demonstrating PyMultiWFN usage.

---

## Available Examples

### 1. `basic_usage.py`

**Purpose**: Fundamental PyMultiWFN operations

**Demonstrates**:
- Loading wavefunction files
- Accessing molecular properties
- Calculating electron density
- Computing bond orders

**Usage**:
```bash
python basic_usage.py molecule.wfn
```

**Output**:
- Molecular information (atoms, electrons, basis)
- Density at atomic positions
- Bond order analysis

---

### 2. `density_analysis.py`

**Purpose**: Electron density analysis and visualization

**Demonstrates**:
- Calculating density on 3D grids
- Finding density maxima/minima
- Radial density profiles
- Approximate electron counting

**Usage**:
```bash
python density_analysis.py molecule.wfn
```

**Output**:
- Density statistics on grid
- Top density maxima
- Radial density profile
- Integrated electron count

---

### 3. `bond_analysis.py`

**Purpose**: Comprehensive bond order analysis

**Demonstrates**:
- Mayer and Wiberg bond orders
- Bond type classification
- Connectivity analysis
- Bond order matrix visualization

**Usage**:
```bash
python bond_analysis.py molecule.wfn
```

**Output**:
- Bond order list with classification
- Mayer vs Wiberg comparison
- Connectivity map
- Bond order statistics

---

## Getting Test Data

These examples require wavefunction files (.wfn or .fch). You can:

1. **Generate from Gaussian**:
   ```bash
   # Add output=wfn to Gaussian input
   # Or use formchk to convert .chk to .fch
   formchk molecule.chk molecule.fch
   ```

2. **Use provided test data**:
   ```bash
   # Test files in tests/test_data/
   python basic_usage.py ../tests/test_data/H2_CCSD.wfn
   ```

3. **Download from databases**:
   - [ioChem-BD](https://www.iochem-bd.org/)
   - [NIST Computational Chemistry](https://cccbdb.nist.gov/)

---

## Example Output

### basic_usage.py
```
============================================================
PyMultiWFN Basic Usage Example
============================================================

1. Loading Wavefunction
------------------------------------------------------------
✓ Successfully loaded: molecule.wfn
  Title: Water molecule
  Method: B3LYP
  Basis: 6-31G*

2. Molecular Properties
------------------------------------------------------------
  Number of atoms: 3
  Number of electrons: 10.0
  Charge: 0
  Multiplicity: 1
  Basis functions: 13

  Atoms:
     0. O   (  0.0000,   0.0000,   0.1173)
     1. H   (  0.0000,   0.7572,  -0.4692)
     2. H   (  0.0000,  -0.7572,  -0.4692)
...
```

---

## Common Workflows

### Workflow 1: Quick Analysis

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order

# Load and analyze
wfn = load_wavefunction("molecule.wfn")
bonds = calculate_mayer_bond_order(wfn)

# Print bonds
for i in range(wfn.num_atoms):
    for j in range(i+1, wfn.num_atoms):
        if bonds['total'][i, j] > 0.5:
            print(f"Bond {i}-{j}: {bonds['total'][i, j]:.4f}")
```

### Workflow 2: Density Grid

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density
import numpy as np

wfn = load_wavefunction("molecule.wfn")

# Create grid
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
z = np.zeros(1)

coords = np.array([[xi, yi, 0.0] for xi in x for yi in y])
density = calc_density(wfn, coords)

# Reshape for visualization
density_grid = density.reshape(50, 50)
```

### Workflow 3: Compare Methods

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding.bondorder import (
    calculate_mayer_bond_order,
    calculate_wiberg_bond_order
)

wfn = load_wavefunction("molecule.wfn")

mayer = calculate_mayer_bond_order(wfn)
wiberg = calculate_wiberg_bond_order(wfn)

# Compare
for i in range(wfn.num_atoms):
    for j in range(i+1, wfn.num_atoms):
        m = mayer['total'][i, j]
        w = wiberg['total'][i, j]
        if max(m, w) > 0.5:
            print(f"Bond {i}-{j}: Mayer={m:.4f}, Wiberg={w:.4f}")
```

---

## Tips

1. **Performance**: Use `use_cache=True` (default) for repeated calculations
2. **Memory**: For large systems, process in chunks
3. **Visualization**: Export density grids to cube files for molecular viewers
4. **Validation**: Compare with Multiwfn for consistency

---

## Next Steps

- Read the [User Guide](../docs/user_guide.md) for detailed API documentation
- Check [tests/](../tests/) for more usage examples
- See [benchmark_performance.py](../benchmark_performance.py) for performance optimization

---

**Questions?** See [User Guide](../docs/user_guide.md) or open an issue on GitHub.
