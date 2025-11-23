"""
Electron density calculation module.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.basis import evaluate_basis

def calc_density(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Calculates the electron density at given coordinates.
    
    Args:
        wfn: Wavefunction object.
        coords: (N, 3) array of coordinates.
        
    Returns:
        rho: (N,) array of electron density values.
    """
    # 1. Evaluate basis functions at all points
    # phi shape: (N_points, N_basis)
    phi = evaluate_basis(wfn, coords)
    
    # 2. Construct Density Matrix P
    # P_mu_nu = sum_i n_i * C_mu_i * C_nu_i
    # wfn.coefficients shape is (nmo, nbasis) -> C_i_mu
    
    rho = np.zeros(coords.shape[0])
    
    # Alpha / Total Density
    if wfn.coefficients is not None and wfn.occupations is not None:
        P_alpha = _make_density_matrix(wfn.coefficients, wfn.occupations)
        rho += _contract_density(phi, P_alpha)
        
    # Beta Density (if unrestricted)
    if wfn.is_unrestricted and wfn.coefficients_beta is not None:
        if wfn.occupations_beta is not None:
            P_beta = _make_density_matrix(wfn.coefficients_beta, wfn.occupations_beta)
            rho += _contract_density(phi, P_beta)
        else:
            # Fallback if occupations_beta is missing but coefficients exist (e.g. some FCHK)
            # Usually occupations are stored.
            pass
            
    return rho

def _make_density_matrix(coeffs: np.ndarray, occs: np.ndarray) -> np.ndarray:
    """
    Constructs density matrix P from MO coefficients and occupations.
    P = C.T * diag(occ) * C
    coeffs: (nmo, nbasis)
    occs: (nmo,)
    """
    # Multiply each row i of C by occ[i]
    # C_occ = coeffs * occs[:, np.newaxis]
    # P = C.T @ C_occ
    
    # Optimization: Only use occupied orbitals
    occ_idx = occs > 1e-8
    C_occ = coeffs[occ_idx]
    n_occ = occs[occ_idx]
    
    # P_mu_nu = sum_i n_i C_i_mu C_i_nu
    # P = (C_occ.T * n_occ) @ C_occ
    
    # C_occ.T shape: (nbasis, n_occupied)
    # n_occ shape: (n_occupied,)
    
    return (C_occ.T * n_occ) @ C_occ

def _contract_density(phi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Contracts basis values with density matrix to get density.
    rho = sum_mu sum_nu phi_mu P_mu_nu phi_nu
    """
    # temp = phi @ P  -> (N_points, N_basis)
    # rho = sum(phi * temp, axis=1)
    
    temp = phi @ P
    return np.sum(phi * temp, axis=1)
