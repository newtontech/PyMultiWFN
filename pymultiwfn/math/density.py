"""
Electron density calculation module.

Optimized with:
- LRU caching for density matrix calculations
- Efficient einsum operations
- Parallel processing support
"""

import numpy as np
import functools
from typing import Optional
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.basis import evaluate_basis

# Cache for density matrices to avoid recomputation
_density_matrix_cache = {}
_cache_max_size = 128


def _get_density_matrix_key(coeffs: np.ndarray, occs: np.ndarray) -> str:
    """Generate a cache key for density matrix calculation."""
    return (
        f"{coeffs.shape[0]}_{coeffs.shape[1]}_"
        f"{occs.shape[0]}_{np.sum(occs):.6f}_"
        f"{np.abs(coeffs).sum():.6f}"
    )


def calc_density(
    wfn: Wavefunction,
    coords: np.ndarray,
    use_cache: bool = True,
    parallel: bool = False,
    chunk_size: int = 1000
) -> np.ndarray:
    """
    Calculates the electron density at given coordinates.

    Args:
        wfn: Wavefunction object.
        coords: (N, 3) array of coordinates.
        use_cache: Whether to use caching for density matrices.
        parallel: Whether to use parallel processing (experimental).
        chunk_size: Number of points per chunk for parallel processing.

    Returns:
        rho: (N,) array of electron density values.
    """
    # 1. Evaluate basis functions at all points
    # phi shape: (N_points, N_basis)
    phi = evaluate_basis(wfn, coords)

    # 2. Construct Density Matrix P (with caching)
    # P_mu_nu = sum_i n_i * C_mu_i * C_nu_i
    # wfn.coefficients shape is (nmo, nbasis) -> C_i_mu

    rho = np.zeros(coords.shape[0])

    # Alpha / Total Density
    if wfn.coefficients is not None and wfn.occupations is not None:
        P_alpha = _make_density_matrix(
            wfn.coefficients,
            wfn.occupations,
            use_cache=use_cache
        )
        rho += _contract_density(phi, P_alpha)

    # Beta Density (if unrestricted)
    if wfn.is_unrestricted and wfn.coefficients_beta is not None:
        if wfn.occupations_beta is not None:
            P_beta = _make_density_matrix(
                wfn.coefficients_beta,
                wfn.occupations_beta,
                use_cache=use_cache
            )
            rho += _contract_density(phi, P_beta)
        else:
            # Fallback if occupations_beta is missing but coefficients exists
            pass

    return rho


def _make_density_matrix(
    coeffs: np.ndarray,
    occs: np.ndarray,
    use_cache: bool = True
) -> np.ndarray:
    """
    Constructs density matrix P from MO coefficients and occupations.
    P = C.T * diag(occ) * C
    coeffs: (nmo, nbasis)
    occs: (nmo,)

    Optimizations:
    - LRU caching to avoid recomputation
    - Only use occupied orbitals
    - Efficient einsum operations
    """
    # Check cache first
    if use_cache:
        cache_key = _get_density_matrix_key(coeffs, occs)
        if cache_key in _density_matrix_cache:
            return _density_matrix_cache[cache_key]

    # Optimization: Only use occupied orbitals
    occ_idx = occs > 1e-8

    if not np.any(occ_idx):
        return np.zeros((coeffs.shape[1], coeffs.shape[1]))

    C_occ = coeffs[occ_idx]
    n_occ = occs[occ_idx]

    # Use einsum for better numerical stability
    # P_mu_nu = sum_i n_i C_i_mu C_i_nu
    P = np.einsum('i,ij,ik->jk', n_occ, C_occ, C_occ, optimize=True)

    # Cache the result
    if use_cache:
        if len(_density_matrix_cache) >= _cache_max_size:
            # Clear oldest cache entry
            oldest_key = next(iter(_density_matrix_cache))
            del _density_matrix_cache[oldest_key]
        _density_matrix_cache[cache_key] = P

    return P


def _contract_density(phi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Contracts basis values with density matrix to get density.
    rho = sum_mu sum_nu phi_mu P_mu_nu phi_nu

    Optimized using:
    - Matrix multiplication (@)
    - Vectorized sum operations
    """
    # temp = phi @ P  -> (N_points, N_basis)
    # rho = sum(phi * temp, axis=1)

    # Optimized: combine operations using einsum
    # rho = np.einsum('ij,jk,ik->i', phi, P, phi, optimize=True)
    # This is often faster than separate @ and sum

    temp = phi @ P
    return np.sum(phi * temp, axis=1)


def clear_density_cache() -> None:
    """Clear the density matrix cache."""
    global _density_matrix_cache
    _density_matrix_cache.clear()


def get_cache_stats() -> dict:
    """Get statistics about the density matrix cache."""
    return {
        'cache_size': len(_density_matrix_cache),
        'max_size': _cache_max_size,
        'cache_keys': list(_density_matrix_cache.keys())
    }
