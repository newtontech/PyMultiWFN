"""
General mathematical functions reimplemented from Fortran modules.
Based on functionalities from 0123dim.f90, Bspline.f90, DFTxclib.F, function.f90,
grid.f90, integral.f90, minpack.f90, O1.f90, sym.F, util.f90

This module provides pure Python implementations using NumPy and SciPy to replace
the original Fortran mathematical routines from Multiwfn.
"""

import numpy as np
import scipy.special
from scipy import integrate, optimize, interpolate
from typing import Tuple, Optional, Union, Callable
import warnings

# Physical constants and conversion factors
PI = np.pi
BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Hermite polynomial integration tables (roots and weights)
# Data from Multiwfn util.f90
_HERMITE_ROOTS = [
    [0.0],  # n=1
    [-0.7071067811865475, 0.7071067811865475],  # n=2
    [-1.2247448713915890, 0.0, 1.2247448713915890],  # n=3
    [-1.6506801238857846, -0.5246476232752903, 0.5246476232752903, 1.6506801238857846],  # n=4
    [-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],  # n=5
    [-2.3506049736744922, -1.3358490740136969, -0.4360774119276165, 0.4360774119276165,
     1.3358490740136969, 2.3506049736744922],  # n=6
    [-2.6519613568352335, -1.6735516287674714, -0.8162878828589647, 0.0, 0.8162878828589647,
     1.6735516287674714, 2.6519613568352335],  # n=7
    [-2.9306374202572440, -1.9816567566958429, -1.1571937124467802, -0.3811869902073221,
     0.3811869902073221, 1.1571937124467802, 1.9816567566958429, 2.9306374202572440],  # n=8
    [-3.1909932017815276, -2.2665805845318431, -1.4685532892166679, -0.7235510187528376, 0.0,
     0.7235510187528376, 1.4685532892166679, 2.2665805845318431, 3.1909932017815276],  # n=9
    [-3.4361591188377376, -2.5327316742327898, -1.7566836492998818, -1.0366108297895137,
     -0.3429013272237046, 0.3429013272237046, 1.0366108297895137, 1.7566836492998818,
     2.5327316742327898, 3.4361591188377376]  # n=10
]

_HERMITE_WEIGHTS = [
    [1.7724538509055160],  # n=1 (sqrt(pi))
    [0.8862269254527580, 0.8862269254527580],  # n=2
    [0.29540897515091934, 1.1816359006036774, 0.29540897515091934],  # n=3
    [0.08131283544724518, 0.8049140900055128, 0.8049140900055128, 0.08131283544724518],  # n=4
    [0.019953242059045913, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116,
     0.019953242059045913],  # n=5
    [0.004530009905508846, 0.15706732032285664, 0.7246295952243925, 0.7246295952243925,
     0.15706732032285664, 0.004530009905508846],  # n=6
    [0.0009717812450995191, 0.05451558281912703, 0.4256072526101278, 0.8102646175568073,
     0.4256072526101278, 0.05451558281912703, 0.0009717812450995191],  # n=7
    [0.00019960407221136762, 0.017077983007413475, 0.20780232581489180, 0.6611470125582413,
     0.6611470125582413, 0.20780232581489180, 0.017077983007413475, 0.00019960407221136762],  # n=8
    [3.960697726326438e-05, 0.004943624275536947, 0.08847452739437657, 0.4326515590025558,
     0.7202352156060510, 0.4326515590025558, 0.08847452739437657, 0.004943624275536947,
     3.960697726326438e-05],  # n=9
    [7.640432855232621e-06, 0.0013436457467812327, 0.03387439445548106, 0.24013861108231469,
     0.6108626337353258, 0.6108626337353258, 0.24013861108231469, 0.03387439445548106,
     0.0013436457467812327, 7.640432855232621e-06]  # n=10
]


def hermite_nodes_weights(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Hermite polynomial nodes and weights for Gaussian integration.

    Parameters
    ----------
    n_points : int
        Number of integration points (1-10 supported)

    Returns
    -------
    nodes : np.ndarray
        Integration nodes (roots of Hermite polynomial)
    weights : np.ndarray
        Integration weights

    Notes
    -----
    Based on the Rhm and Whm tables from Multiwfn's util.f90.
    """
    if n_points < 1 or n_points > 10:
        raise ValueError("Number of Hermite points must be between 1 and 10")

    nodes = np.array(_HERMITE_ROOTS[n_points - 1][:n_points])
    weights = np.array(_HERMITE_WEIGHTS[n_points - 1][:n_points])

    return nodes.copy(), weights.copy()


def hermite_integrate_1d(func: Callable, n_points: int = 10) -> float:
    """
    Perform 1D Hermite-Gauss integration.

    Parameters
    ----------
    func : Callable
        Function to integrate f(x) where x is in real space
    n_points : int
        Number of integration points (1-10)

    Returns
    -------
    float
        Integral value
    """
    nodes, weights = hermite_nodes_weights(n_points)
    return np.sum(weights * func(nodes))


def overlap_gaussian_primitive(exp1: float, center1: np.ndarray,
                              power1: np.ndarray, exp2: float,
                              center2: np.ndarray, power2: np.ndarray) -> float:
    """
    Calculate overlap integral between two unnormalized Gaussian primitive functions.

    S(x1,y1,z1)^l1x S(y1)^l1y S(z1)^l1z * S(x2,y2,z2)^l2x S(y2)^l2y S(z2)^l2z

    Parameters
    ----------
    exp1, exp2 : float
        Gaussian exponents
    center1, center2 : np.ndarray
        Atomic centers [x, y, z]
    power1, power2 : np.ndarray
        Angular momentum powers [lx, ly, lz]

    Returns
    -------
    float
        Overlap integral value

    Notes
    -----
    Based on doSintactual from Multiwfn's integral.f90.
    """
    # Ensure inputs are arrays
    center1 = np.asarray(center1)
    center2 = np.asarray(center2)
    power1 = np.asarray(power1, dtype=int)
    power2 = np.asarray(power2, dtype=int)

    # Combined exponent
    exp_sum = exp1 + exp2
    sqrt_exp_sum = np.sqrt(exp_sum)

    # Gaussian product center
    product_center = (exp1 * center1 + exp2 * center2) / exp_sum

    # Exponential factor
    r2 = np.sum((center1 - center2) ** 2)
    exp_factor = np.exp(-exp1 * exp2 * r2 / exp_sum)

    # Calculate integral for each dimension
    integral = 1.0

    for dim in range(3):  # x, y, z
        l1, l2 = power1[dim], power2[dim]
        c1, c2 = center1[dim], center2[dim]
        pc = product_center[dim]

        # Number of integration points needed
        n_points = int(np.ceil((l1 + l2 + 1) / 2.0))
        n_points = max(1, min(n_points, 10))  # Limit to available data

        nodes, weights = hermite_nodes_weights(n_points)

        # Transform nodes to integration points
        integration_points = nodes / sqrt_exp_sum + pc

        # Calculate dimension contribution
        dim_integral = 0.0
        for i, xi in enumerate(integration_points):
            term1 = (xi - c1) ** l1
            term2 = (xi - c2) ** l2
            dim_integral += weights[i] * term1 * term2

        integral *= dim_integral / sqrt_exp_sum

    return integral * exp_factor


def boys_function(n: int, t: float) -> float:
    """
    Calculate Boys function F_n(t).

    F_n(t) = ∫₀¹ u^(2n) * exp(-t*u²) du

    Parameters
    ----------
    n : int
        Order of Boys function (n >= 0)
    t : float
        Argument (t >= 0)

    Returns
    -------
    float
        Value of Boys function

    Notes
    -----
    Used in electron repulsion integral calculations.
    """
    if t == 0.0:
        return 1.0 / (2 * n + 1)
    elif t < 0:
        raise ValueError("Boys function argument t must be non-negative")
    elif n == 0:
        # F_0(t) = (1/2) * sqrt(pi/t) * erf(sqrt(t))
        return 0.5 * np.sqrt(np.pi / t) * scipy.special.erf(np.sqrt(t))
    else:
        # Recurrence relation: F_n(t) = (2n-1)F_{n-1}(t) - t^(2n-1) * exp(-t)) / (2t)
        return (2 * n - 1) * boys_function(n - 1, t) - t**(2 * n - 1) * np.exp(-t) / (2 * t)


def erfc_function(x: float) -> float:
    """
    Complementary error function.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        erfc(x) = 1 - erf(x)
    """
    return scipy.special.erfc(x)


def vector_angle(vec1: np.ndarray, vec2: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate angle between two vectors.

    Parameters
    ----------
    vec1, vec2 : np.ndarray
        Input vectors (3D)
    degrees : bool
        If True, return angle in degrees; if False, return in radians

    Returns
    -------
    float
        Angle between vectors

    Notes
    -----
    Based on vecang from Multiwfn's util.f90.
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate angle with zero-length vector")

    cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range

    angle = np.arccos(cos_theta)

    if degrees:
        return np.degrees(angle)
    else:
        return angle


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculate distance matrix between coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array with shape (n_points, n_dim)

    Returns
    -------
    np.ndarray
        Distance matrix with shape (n_points, n_points)
    """
    return np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)


def cross_product(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Calculate cross product of two 3D vectors.

    Parameters
    ----------
    vec1, vec2 : np.ndarray
        Input vectors (3D)

    Returns
    -------
    np.ndarray
        Cross product vector
    """
    return np.cross(vec1, vec2)


def dihedral_angle(coords: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate dihedral angle from four atomic coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array with shape (4, 3) containing coordinates of four atoms
    degrees : bool
        If True, return angle in degrees; if False, return in radians

    Returns
    -------
    float
        Dihedral angle

    Notes
    -----
    Calculates the improper torsion angle between planes defined by
    (0,1,2) and (1,2,3).
    """
    if coords.shape != (4, 3):
        raise ValueError("Coordinates must have shape (4, 3)")

    # Define the four points
    p0, p1, p2, p3 = coords

    # Calculate vectors
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    # Calculate normal vectors
    n1 = cross_product(b1, b2)
    n2 = cross_product(b2, b3)

    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    b2 = b2 / np.linalg.norm(b2)

    # Calculate angle
    m1 = cross_product(n1, b2)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    angle = np.arctan2(-y, x)

    if degrees:
        return np.degrees(angle)
    else:
        return angle


class BSplineInterpolator:
    """
    Multi-dimensional B-spline interpolation wrapper.

    Based on the B-spline implementation from Multiwfn's Bspline.f90,
    but implemented using SciPy's B-spline capabilities for better performance.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray,
                 order: int = 4, bc_type: str = 'natural'):
        """
        Initialize B-spline interpolator.

        Parameters
        ----------
        x_data : np.ndarray
            Input data points (1D or multi-dimensional)
        y_data : np.ndarray
            Function values at data points
        order : int
            Spline order (3=quadratic, 4=cubic, etc.)
        bc_type : str
            Boundary condition type
        """
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.order = order
        self.bc_type = bc_type

        # Handle different dimensions
        if self.x_data.ndim == 1:
            self._setup_1d_spline()
        elif self.x_data.ndim == 2:
            self._setup_2d_spline()
        elif self.x_data.ndim == 3:
            self._setup_3d_spline()
        else:
            raise ValueError(f"Unsupported dimension: {self.x_data.ndim}")

    def _setup_1d_spline(self):
        """Setup 1D B-spline interpolation."""
        self.spline = interpolate.CubicSpline(
            self.x_data, self.y_data,
            bc_type=self.bc_type
        )

    def _setup_2d_spline(self):
        """Setup 2D B-spline interpolation."""
        from scipy.interpolate import RectBivariateSpline

        x1, x2 = self.x_data.shape
        x1_grid = np.arange(x1)
        x2_grid = np.arange(x2)

        self.spline = RectBivariateSpline(
            x1_grid, x2_grid, self.y_data,
            kx=min(self.order, x1-1), ky=min(self.order, x2-1)
        )

    def _setup_3d_spline(self):
        """Setup 3D B-spline interpolation using RegularGridInterpolator."""
        x1, x2, x3 = self.x_data.shape
        grid = (np.arange(x1), np.arange(x2), np.arange(x3))

        self.spline = interpolate.RegularGridInterpolator(
            grid, self.y_data, method='cubic'
        )

    def __call__(self, *args) -> Union[float, np.ndarray]:
        """Evaluate the spline at given points."""
        if self.x_data.ndim == 1:
            return self.spline(args[0])
        elif self.x_data.ndim == 2:
            return self.spline(args[0], args[1])[0, 0]
        elif self.x_data.ndim == 3:
            return self.spline([args])[0]


def bspline_interpolate(x_points: np.ndarray, y_points: np.ndarray,
                       x_query: np.ndarray, order: int = 4) -> np.ndarray:
    """
    Perform B-spline interpolation of 1D data.

    Parameters
    ----------
    x_points : np.ndarray
        Input x coordinates (must be monotonically increasing)
    y_points : np.ndarray
        Input y values
    x_query : np.ndarray
        Points where to evaluate the interpolated function
    order : int
        Spline order (3=quadratic, 4=cubic, 5=quartic, 6=quintic)

    Returns
    -------
    np.ndarray
        Interpolated values at x_query points

    Notes
    -----
    Based on the B-spline routines from Multiwfn's Bspline.f90.
    """
    # Ensure input is properly sorted
    sort_idx = np.argsort(x_points)
    x_sorted = x_points[sort_idx]
    y_sorted = y_points[sort_idx]

    # Create spline
    if order == 3:  # Quadratic
        k = 2
    elif order == 4:  # Cubic
        k = 3
    elif order == 5:  # Quartic
        k = 4
    elif order == 6:  # Quintic
        k = 5
    else:
        raise ValueError(f"Unsupported spline order: {order}")

    # Use SciPy's UnivariateSpline for flexibility
    spline = interpolate.UnivariateSpline(x_sorted, y_sorted, k=k, s=0)

    return spline(x_query)


def linear_interpolation_3d(grid_data: np.ndarray,
                           x: float, y: float, z: float,
                           x_bounds: Tuple[float, float] = None,
                           y_bounds: Tuple[float, float] = None,
                           z_bounds: Tuple[float, float] = None) -> float:
    """
    Perform trilinear interpolation on 3D grid data.

    Parameters
    ----------
    grid_data : np.ndarray
        3D array of function values
    x, y, z : float
        Coordinates where to interpolate
    x_bounds, y_bounds, z_bounds : tuple
        Spatial bounds of the grid (min, max). If None, assumes unit grid.

    Returns
    -------
    float
        Interpolated value

    Notes
    -----
    Based on linintp3d function from Multiwfn.
    """
    nx, ny, nz = grid_data.shape

    # Set default bounds if not provided
    if x_bounds is None:
        x_bounds = (0, nx - 1)
    if y_bounds is None:
        y_bounds = (0, ny - 1)
    if z_bounds is None:
        z_bounds = (0, nz - 1)

    # Convert to grid indices
    x_norm = (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * (nx - 1)
    y_norm = (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * (ny - 1)
    z_norm = (z - z_bounds[0]) / (z_bounds[1] - z_bounds[0]) * (nz - 1)

    # Find neighboring grid points
    i0 = int(np.floor(x_norm))
    j0 = int(np.floor(y_norm))
    k0 = int(np.floor(z_norm))

    # Ensure indices are within bounds
    i0 = max(0, min(i0, nx - 2))
    j0 = max(0, min(j0, ny - 2))
    k0 = max(0, min(k0, nz - 2))

    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

    # Calculate interpolation weights
    dx = x_norm - i0
    dy = y_norm - j0
    dz = z_norm - k0

    # Get corner values
    v000 = grid_data[i0, j0, k0]
    v001 = grid_data[i0, j0, k1]
    v010 = grid_data[i0, j1, k0]
    v011 = grid_data[i0, j1, k1]
    v100 = grid_data[i1, j0, k0]
    v101 = grid_data[i1, j0, k1]
    v110 = grid_data[i1, j1, k0]
    v111 = grid_data[i1, j1, k1]

    # Trilinear interpolation
    v00 = v000 * (1 - dx) + v100 * dx
    v01 = v001 * (1 - dx) + v101 * dx
    v10 = v010 * (1 - dx) + v110 * dx
    v11 = v011 * (1 - dx) + v111 * dx

    v0 = v00 * (1 - dy) + v10 * dy
    v1 = v01 * (1 - dy) + v11 * dy

    result = v0 * (1 - dz) + v1 * dz

    return result


def find_bond_critical_points(density_grid: np.ndarray,
                             coords: np.ndarray,
                             threshold: float = 0.1) -> np.ndarray:
    """
    Find bond critical points in electron density using gradient analysis.

    Parameters
    ----------
    density_grid : np.ndarray
        3D electron density grid
    coords : np.ndarray
        Atomic coordinates in grid space
    threshold : float
        Threshold for critical point detection

    Returns
    -------
    np.ndarray
        Coordinates of bond critical points

    Notes
    -----
    Simplified implementation based on Multiwfn topology analysis.
    """
    # Calculate gradient using numpy
    grad_x, grad_y, grad_z = np.gradient(density_grid)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Find points where gradient is minimal (potential critical points)
    threshold_scaled = threshold * np.max(density_grid)
    critical_mask = (grad_magnitude < threshold_scaled) & (density_grid > threshold_scaled)

    # Get coordinates of critical points
    critical_points = np.argwhere(critical_mask)

    # Convert grid indices to real coordinates
    if len(critical_points) > 0:
        # Simple mapping - in practice, need grid transformation
        return critical_points * (coords.max(axis=0) - coords.min(axis=0)) / np.array(density_grid.shape) + coords.min(axis=0)
    else:
        return np.array([]).reshape(0, 3)


def atomic_radii_valence(electrons: int, period: int,
                        covalent: bool = True) -> float:
    """
    Estimate atomic radius based on electron count and period.

    Parameters
    ----------
    electrons : int
        Number of valence electrons
    period : int
        Period in periodic table (1-7)
    covalent : bool
        If True, return covalent radius; if False, van der Waals radius

    Returns
    -------
    float
        Estimated atomic radius in Angstroms

    Notes
    -----
    Simple empirical formula based on periodic trends.
    """
    if covalent:
        # Covalent radii approximation
        base_radii = np.array([0.31, 0.28, 1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.62])
        if period <= len(base_radii):
            base = base_radii[period - 1]
        else:
            base = 1.5  # Rough estimate for heavy elements

        # Adjust based on valence electrons
        if electrons <= 2:
            factor = 1.0
        elif electrons <= 8:
            factor = 1.0 + 0.05 * (electrons - 2)
        else:
            factor = 1.4  # Transition metals

        return base * factor
    else:
        # Van der Waals radii (approximately 1.5x covalent)
        return 1.5 * atomic_radii_valence(electrons, period, covalent=True)


def matrix_inverse_3x3(matrix: np.ndarray) -> np.ndarray:
    """
    Efficiently compute inverse of 3x3 matrix.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 matrix to invert

    Returns
    -------
    np.ndarray
        Inverted matrix

    Notes
    -----
    Optimized for 3x3 matrices commonly used in coordinate transformations.
    """
    if matrix.shape != (3, 3):
        raise ValueError("Matrix must be 3x3")

    det = np.linalg.det(matrix)
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular or nearly singular")

    return np.linalg.inv(matrix)


def sort_array(arr: np.ndarray, descending: bool = False) -> np.ndarray:
    """
    Sort array with option for descending order.

    Parameters
    ----------
    arr : np.ndarray
        Array to sort
    descending : bool
        If True, sort in descending order

    Returns
    -------
    np.ndarray
        Sorted array
    """
    if descending:
        return -np.sort(-arr)
    else:
        return np.sort(arr)
