"""
Density grid analysis utilities.
Provides functions to generate a regular 3D grid, compute electron density on the grid,
and write the result to a simple Gaussian cube file.
"""

import numpy as np
from pymultiwfn.math.basis import evaluate_basis
from pymultiwfn.math.density import calc_density


def generate_grid(min_coords: np.ndarray, max_coords: np.ndarray, spacing: float) -> np.ndarray:
    """Generate a regular 3D grid of points.

    Parameters
    ----------
    min_coords : (3,) array_like
        Minimum x, y, z coordinates.
    max_coords : (3,) array_like
        Maximum x, y, z coordinates.
    spacing : float
        Grid spacing in the same units as the coordinates (Bohr).

    Returns
    -------
    points : (N, 3) ndarray
        Flattened array of grid points.
    """
    mins = np.asarray(min_coords, dtype=float)
    maxs = np.asarray(max_coords, dtype=float)
    # Create ranges for each axis
    xs = np.arange(mins[0], maxs[0] + spacing, spacing)
    ys = np.arange(mins[1], maxs[1] + spacing, spacing)
    zs = np.arange(mins[2], maxs[2] + spacing, spacing)
    # Meshgrid and flatten
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points


def compute_density_grid(wfn, grid_points: np.ndarray) -> np.ndarray:
    """Compute electron density on a set of grid points.

    Parameters
    ----------
    wfn : Wavefunction
        Wavefunction object containing basis set and coefficients.
    grid_points : (N, 3) ndarray
        Cartesian coordinates where density is evaluated.

    Returns
    -------
    rho : (N,) ndarray
        Electron density values at each grid point.
    """
    # Evaluate basis functions on the grid
    phi = evaluate_basis(wfn, grid_points)
    # Compute density using the density module
    rho = calc_density(wfn, grid_points)
    return rho


def write_cube(filename: str, wfn, grid_points: np.ndarray, density: np.ndarray) -> None:
    """Write a simple Gaussian cube file.

    Parameters
    ----------
    filename : str
        Output file path.
    wfn : Wavefunction
        Wavefunction (used for atom information).
    grid_points : (N, 3) ndarray
        Grid points (must be a regular grid).
    density : (N,) ndarray
        Density values corresponding to grid points.
    """
    # Determine grid dimensions from the points
    # Assuming points are ordered as produced by generate_grid (i,j,k loops)
    # Recover spacing and counts
    # Find unique coordinates along each axis
    xs = np.unique(grid_points[:, 0])
    ys = np.unique(grid_points[:, 1])
    zs = np.unique(grid_points[:, 2])
    nx, ny, nz = len(xs), len(ys), len(zs)
    spacing = xs[1] - xs[0] if nx > 1 else 0.0

    origin = xs[0], ys[0], zs[0]

    with open(filename, 'w') as f:
        f.write("PyMultiWFN cube file\n")
        f.write("OUTER LOOP: X, Y, Z\n")
        # Number of atoms, origin, and grid vectors
        f.write(f"{len(wfn.atoms):5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n")
        f.write(f"{nx:5d}{spacing:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        f.write(f"{ny:5d}{0.0:12.6f}{spacing:12.6f}{0.0:12.6f}\n")
        f.write(f"{nz:5d}{0.0:12.6f}{0.0:12.6f}{spacing:12.6f}\n")
        # Atom lines (element number, charge, coordinates)
        for atom in wfn.atoms:
            f.write(f"{atom.index:5d}{atom.charge:12.6f}{atom.x:12.6f}{atom.y:12.6f}{atom.z:12.6f}\n")
        # Write density values, 6 per line as per cube format
        values = density.reshape((nx, ny, nz), order='F')  # Fortran order matches generate_grid
        flat_vals = values.ravel(order='F')
        for i in range(0, len(flat_vals), 6):
            line_vals = flat_vals[i:i+6]
            f.write(''.join(f"{v:13.5e}" for v in line_vals) + '\n')
"
