# PyMultiWFN User Guide

**Version**: 0.1.0
**Last Updated**: 2026-02-21

A Python refactoring of Multiwfn for wavefunction analysis.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Common Tasks](#common-tasks)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

### What is PyMultiWFN?

PyMultiWFN is a Python library for quantum chemistry wavefunction analysis. It's a refactoring of the popular Multiwfn program, designed to be:

- **Pythonic**: Clean, readable Python API
- **Modular**: Easy to extend and customize
- **Tested**: Comprehensive test suite for reliability
- **Performant**: Optimized with NumPy and caching

### Features

- Electron density calculations
- Bond order analysis (Mayer, Wiberg, Mulliken)
- Population analysis (Mulliken, Hirshfeld, Becke)
- Orbital analysis
- Consistency validation with Multiwfn

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy (optional, for advanced features)

### Install from Source

```bash
git clone https://github.com/yourusername/PyMultiWFN.git
cd PyMultiWFN
pip install -e .
```

### Install Dependencies

```bash
pip install numpy scipy pytest
```

---

## Quick Start

### Loading a Wavefunction

```python
from pymultiwfn.io.loader import load_wavefunction

# Load from .wfn file
wfn = load_wavefunction("molecule.wfn")

# Load from .fch file
wfn = load_wavefunction("molecule.fch")

print(f"Loaded: {wfn.title}")
print(f"Atoms: {wfn.num_atoms}")
print(f"Electrons: {wfn.num_electrons}")
print(f"Basis functions: {wfn.num_basis}")
```

### Calculating Electron Density

```python
from pymultiwfn.math.density import calc_density
import numpy as np

# Define coordinates (N x 3 array)
coords = np.array([
    [0.0, 0.0, 0.0],  # Origin
    [1.0, 0.0, 0.0],  # 1 bohr along x
    [0.0, 1.0, 0.0],  # 1 bohr along y
])

# Calculate density
density = calc_density(wfn, coords)

print(f"Density values: {density}")
```

### Bond Order Analysis

```python
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order

# Calculate Mayer bond orders
bond_orders = calculate_mayer_bond_order(wfn)

# bond_orders is a dictionary with 'total', 'alpha', 'beta'
total_bond_order = bond_orders['total']

# Print bond order matrix
print("Bond Order Matrix:")
print(total_bond_order)

# Get specific bond order
atom_i = 0  # First atom
atom_j = 1  # Second atom
print(f"Bond order between atom {atom_i} and {atom_j}: {total_bond_order[atom_i, atom_j]}")
```

---

## Core Concepts

### Wavefunction Object

The `Wavefunction` object is the central data structure:

```python
class Wavefunction:
    # System information
    atoms: List[Atom]           # Atomic coordinates and types
    num_electrons: float        # Total number of electrons
    charge: int                 # Molecular charge
    multiplicity: int           # Spin multiplicity
    
    # Basis set information
    shells: List[Shell]         # Basis function shells
    num_basis: int              # Number of basis functions
    
    # Orbital information
    coefficients: np.ndarray    # MO coefficients (nmo x nbasis)
    energies: np.ndarray        # Orbital energies
    occupations: np.ndarray     # Orbital occupations
    
    # Matrices (optional)
    overlap_matrix: np.ndarray  # Overlap matrix S_uv
    Ptot: np.ndarray           # Total density matrix
```

### Atom Object

```python
class Atom:
    element: str       # Element symbol (e.g., "C", "H")
    index: int         # Atomic number
    x: float          # X coordinate (bohr)
    y: float          # Y coordinate (bohr)
    z: float          # Z coordinate (bohr)
    charge: float     # Nuclear charge
```

### Shell Object

```python
class Shell:
    type: int                  # Shell type (0=S, 1=P, 2=D, ...)
    center_idx: int            # Atom index
    exponents: np.ndarray      # Primitive exponents
    coefficients: np.ndarray   # Contraction coefficients
```

---

## Common Tasks

### Task 1: Analyze Molecular Properties

```python
from pymultiwfn.io.loader import load_wavefunction

wfn = load_wavefunction("water.wfn")

# Basic properties
print(f"Number of atoms: {wfn.num_atoms}")
print(f"Number of electrons: {wfn.num_electrons}")
print(f"Charge: {wfn.charge}")
print(f"Multiplicity: {wfn.multiplicity}")

# Atomic composition
for i, atom in enumerate(wfn.atoms):
    print(f"Atom {i}: {atom.element} at ({atom.x:.3f}, {atom.y:.3f}, {atom.z:.3f})")
```

### Task 2: Calculate Grid-Based Properties

```python
from pymultiwfn.math.density import calc_density
import numpy as np

# Create a 3D grid
n_points = 20
x = np.linspace(-3, 3, n_points)
y = np.linspace(-3, 3, n_points)
z = np.linspace(-3, 3, n_points)

# Generate grid points
grid = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])

# Calculate density on grid
densities = calc_density(wfn, grid)

# Find maximum density
max_density_idx = np.argmax(densities)
max_density = densities[max_density_idx]
max_point = grid[max_density_idx]

print(f"Maximum density: {max_density:.6f}")
print(f"At point: {max_point}")
```

### Task 3: Compare Bond Orders

```python
from pymultiwfn.analysis.bonding.bondorder import (
    calculate_mayer_bond_order,
    calculate_wiberg_bond_order
)

# Load wavefunction
wfn = load_wavefunction("ethane.wfn")

# Calculate different bond order methods
mayer = calculate_mayer_bond_order(wfn)
wiberg = calculate_wiberg_bond_order(wfn)

# Compare for specific bond
atom_i, atom_j = 0, 1
print(f"Mayer bond order:  {mayer['total'][atom_i, atom_j]:.4f}")
print(f"Wiberg bond order: {wiberg['total'][atom_i, atom_j]:.4f}")
```

### Task 4: Performance Optimization

```python
from pymultiwfn.math.density import calc_density, clear_density_cache, get_cache_stats

# Clear cache before critical calculations
clear_density_cache()

# Calculate density (caches density matrix)
coords = np.random.randn(1000, 3)
rho1 = calc_density(wfn, coords, use_cache=True)

# Subsequent calls are faster (cache hit)
rho2 = calc_density(wfn, coords, use_cache=True)

# Check cache statistics
stats = get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Max size: {stats['max_size']}")
```

---

## API Reference

### IO Module

#### `load_wavefunction(filepath)`

Load a wavefunction from file.

**Parameters:**
- `filepath` (str): Path to .wfn or .fch file

**Returns:**
- `Wavefunction`: Loaded wavefunction object

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is not supported

**Example:**
```python
wfn = load_wavefunction("molecule.wfn")
```

### Density Module

#### `calc_density(wfn, coords, use_cache=True)`

Calculate electron density at given coordinates.

**Parameters:**
- `wfn` (Wavefunction): Wavefunction object
- `coords` (np.ndarray): (N, 3) array of coordinates in bohr
- `use_cache` (bool): Whether to use density matrix caching (default: True)

**Returns:**
- `np.ndarray`: (N,) array of density values

**Example:**
```python
coords = np.array([[0.0, 0.0, 0.0]])
density = calc_density(wfn, coords)
```

#### `clear_density_cache()`

Clear the density matrix cache.

**Example:**
```python
clear_density_cache()
```

#### `get_cache_stats()`

Get cache statistics.

**Returns:**
- `dict`: Dictionary with 'cache_size', 'max_size', 'cache_keys'

**Example:**
```python
stats = get_cache_stats()
print(f"Cache: {stats['cache_size']}/{stats['max_size']}")
```

### Bond Order Module

#### `calculate_mayer_bond_order(wfn)`

Calculate Mayer bond orders.

**Parameters:**
- `wfn` (Wavefunction): Wavefunction object with density and overlap matrices

**Returns:**
- `dict`: Dictionary with 'total', 'alpha', 'beta' bond order matrices

**Raises:**
- `ValueError`: If overlap matrix or density matrix is missing

**Example:**
```python
bond_orders = calculate_mayer_bond_order(wfn)
print(bond_orders['total'])
```

#### `calculate_wiberg_bond_order(wfn)`

Calculate Wiberg bond orders (alias for Mayer for closed-shell).

**Parameters:**
- `wfn` (Wavefunction): Wavefunction object

**Returns:**
- `dict`: Bond order dictionary

---

## Examples

### Example 1: Water Molecule Analysis

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order
import numpy as np

# Load water molecule
wfn = load_wavefunction("water.wfn")

print("=== Water Molecule Analysis ===")
print(f"Formula: H2O")
print(f"Electrons: {wfn.num_electrons}")
print(f"Basis functions: {wfn.num_basis}")

# Bond orders
bond_orders = calculate_mayer_bond_order(wfn)
print(f"\nO-H bond orders:")
print(f"  O(0)-H(1): {bond_orders['total'][0, 1]:.4f}")
print(f"  O(0)-H(2): {bond_orders['total'][0, 2]:.4f}")

# Density at oxygen
oxygen_coord = np.array([[wfn.atoms[0].x, wfn.atoms[0].y, wfn.atoms[0].z]])
density_at_oxygen = calc_density(wfn, oxygen_coord)
print(f"\nDensity at oxygen nucleus: {density_at_oxygen[0]:.6f}")
```

### Example 2: Benzene Bond Order Analysis

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order

wfn = load_wavefunction("benzene.wfn")
bond_orders = calculate_mayer_bond_order(wfn)
bo = bond_orders['total']

print("=== Benzene Bond Orders ===")
# Benzene has alternating single/double bonds
for i in range(6):
    j = (i + 1) % 6  # Next carbon
    bond = bo[i, j]
    print(f"C({i})-C({j}): {bond:.4f}")
```

### Example 3: Density Visualization Grid

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density
import numpy as np

wfn = load_wavefunction("molecule.wfn")

# Create 2D grid in xy plane (z=0)
n_points = 50
x = np.linspace(-5, 5, n_points)
y = np.linspace(-5, 5, n_points)
z = np.zeros(n_points * n_points)

# Mesh grid
xx, yy = np.meshgrid(x, y)
coords = np.column_stack([xx.ravel(), yy.ravel(), z])

# Calculate density
density = calc_density(wfn, coords)
density_grid = density.reshape(n_points, n_points)

# Now you can visualize density_grid with matplotlib
# import matplotlib.pyplot as plt
# plt.contourf(xx, yy, density_grid)
# plt.colorbar()
# plt.show()
```

---

## Troubleshooting

### Common Errors

#### `ValueError: Overlap matrix is required`

**Problem**: Bond order calculation requires overlap matrix.

**Solution**: Make sure your wavefunction file contains overlap matrix data, or calculate it:

```python
# Check if overlap matrix exists
if wfn.overlap_matrix is None:
    print("Warning: Overlap matrix not available")
    # May need to recalculate or load from different file
```

#### `FileNotFoundError: molecule.wfn`

**Problem**: Wavefunction file not found.

**Solution**: Check file path and ensure file exists:

```python
import os
filepath = "molecule.wfn"
if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    print(f"Current directory: {os.getcwd()}")
```

#### `MemoryError` with large systems

**Problem**: Out of memory for large molecules.

**Solution**: Use chunked processing or disable caching:

```python
# Process in smaller chunks
n_points = 100000
chunk_size = 10000
coords = np.random.randn(n_points, 3)

densities = []
for i in range(0, n_points, chunk_size):
    chunk = coords[i:i+chunk_size]
    rho_chunk = calc_density(wfn, chunk, use_cache=True)
    densities.append(rho_chunk)

densities = np.concatenate(densities)
```

---

## FAQ

### Q: What file formats are supported?

**A**: Currently supported:
- `.wfn` - Gaussian wfn format
- `.fch` - Gaussian formatted checkpoint

Planned:
- `.molden` - Molden format
- `.wfx` - Extended wfn format

### Q: How do I cite PyMultiWFN?

**A**: If you use PyMultiWFN in your research, please cite both:
1. The original Multiwfn paper
2. PyMultiWFN repository

### Q: Can I use PyMultiWFN with other quantum chemistry programs?

**A**: Yes! As long as you can export to .wfn or .fch format. Most QC programs (Gaussian, ORCA, Q-Chem, etc.) support these formats.

### Q: How accurate is PyMultiWFN compared to Multiwfn?

**A**: PyMultiWFN is validated against Multiwfn reference values. See `tests/test_consistency*.py` for validation results. Generally agrees to 1e-6 or better.

### Q: Is PyMultiWFN faster than Multiwfn?

**A**: For some operations, yes (due to NumPy optimization and caching). For others, Multiwfn may be faster (Fortran optimization). Performance varies by task.

### Q: How can I contribute?

**A**: Contributions welcome! See CONTRIBUTING.md for guidelines. Areas needing help:
- More file format support
- Additional analysis methods
- Documentation improvements
- Performance optimization

---

## Getting Help

- **Documentation**: This guide and API reference
- **Issues**: GitHub Issues for bug reports
- **Examples**: See `examples/` directory
- **Tests**: See `tests/` for usage examples

---

## License

PyMultiWFN is released under the MIT License. See LICENSE file for details.

---

**Version**: 0.1.0
**Last Updated**: 2026-02-21
**Authors**: PyMultiWFN Development Team
