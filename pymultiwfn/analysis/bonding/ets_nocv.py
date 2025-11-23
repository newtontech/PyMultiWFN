
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy.linalg import eigh, fractional_matrix_power

from pymultiwfn.core.data import Wavefunction

def _orthogonalize_orbitals(coeffs: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """
    Orthogonalizes orbitals using Lowdin orthogonalization: C_new = C * (C.T * S * C)^(-1/2)
    
    Args:
        coeffs: (N_basis, N_orb) matrix of orbitals.
        overlap: (N_basis, N_basis) overlap matrix.
        
    Returns:
        (N_basis, N_orb) orthogonalized coefficients.
    """
    # Calculate overlap between orbitals: S_orb = C.T * S * C
    s_orb = coeffs.T @ overlap @ coeffs
    
    # Calculate S_orb^(-1/2)
    # eigh for symmetric matrix
    evals, evecs = np.linalg.eigh(s_orb)
    
    # Filter small eigenvalues to avoid numerical instability
    mask = evals > 1e-8
    evals_inv_sqrt = np.zeros_like(evals)
    evals_inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    
    s_inv_sqrt = evecs @ np.diag(evals_inv_sqrt) @ evecs.T
    
    # C_new = C @ S_inv_sqrt
    return coeffs @ s_inv_sqrt

def calculate_ets_nocv(
    complex_wfn: Wavefunction,
    fragment_wfns: List[Wavefunction]
) -> Dict[str, Any]:
    """
    Performs Extended Transition State - Natural Orbitals for Chemical Valence (ETS-NOCV) analysis.
    
    Args:
        complex_wfn: Wavefunction of the complex.
        fragment_wfns: List of Wavefunctions for the fragments.
                       Assumes the basis set of the complex is the concatenation of fragment basis sets.
                       
    Returns:
        Dictionary containing:
        - "nocv_eigenvalues": (N_basis,) array (or tuple for unrestricted)
        - "nocv_orbitals": (N_basis, N_basis) array (or tuple)
        - "p_diff": Difference density matrix (or tuple)
    """
    
    # Basic consistency check
    total_frag_basis = sum(f.num_basis for f in fragment_wfns)
    if total_frag_basis != complex_wfn.num_basis:
        raise ValueError(f"Total basis functions of fragments ({total_frag_basis}) does not match complex ({complex_wfn.num_basis}).")
    
    overlap = complex_wfn.overlap_matrix
    if overlap is None:
        raise ValueError("Complex wavefunction must have overlap matrix.")
        
    is_unrestricted = complex_wfn.is_unrestricted
    
    # Helper to process one spin channel
    def process_spin(
        complex_coeffs: np.ndarray,
        complex_occs: np.ndarray,
        frag_coeffs_list: List[np.ndarray],
        frag_occs_list: List[np.ndarray],
        spin_factor: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # 1. Construct Frozen State Density P_frz
        # Collect all occupied fragment orbitals into a single matrix
        occ_orb_list = []
        current_basis_offset = 0
        
        for i, frag_wfn in enumerate(fragment_wfns):
            n_basis_frag = frag_wfn.num_basis
            coeffs = frag_coeffs_list[i] # (N_basis_frag, N_mo_frag) usually, but check data.py convention
            # data.py: coefficients is (nmo, nbasis)
            # We need (nbasis, nmo) for matrix math usually
            coeffs = coeffs.T 
            
            occs = frag_occs_list[i]
            
            # Select occupied orbitals
            # Threshold for occupation? Usually > 0.
            occ_indices = np.where(occs > 1e-6)[0]
            
            # Construct full-sized vectors for these occupied orbitals
            # They are localized on the fragment, so we pad with zeros
            for idx in occ_indices:
                full_vec = np.zeros(complex_wfn.num_basis)
                # Map fragment basis to complex basis
                full_vec[current_basis_offset : current_basis_offset + n_basis_frag] = coeffs[:, idx]
                occ_orb_list.append(full_vec)
            
            current_basis_offset += n_basis_frag
            
        # Matrix of occupied fragment orbitals (N_basis, N_occ_total)
        c_occ_frags = np.column_stack(occ_orb_list)
        
        # Orthogonalize these orbitals to get Frozen Orbitals
        c_frz = _orthogonalize_orbitals(c_occ_frags, overlap)
        
        # Calculate P_frz
        # P = sum_i occ_i * c_i * c_i.T
        # For frozen state, orbitals are fully occupied (1.0 or 2.0 depending on spin_factor)
        # spin_factor is 2.0 for Restricted (doubly occupied), 1.0 for Unrestricted
        # But wait, c_frz are spatial orbitals derived from occupied fragment orbitals.
        # If we treat them as occupied, they have occupation 'spin_factor'.
        p_frz = spin_factor * (c_frz @ c_frz.T)
        
        # 2. Calculate Complex Density P_complex
        # Using complex_wfn density matrix if available, or calculate from coeffs
        # We assume Palpha/Pbeta/Ptot are available or calculable
        # But here we are inside process_spin, so we construct it from arguments
        # complex_coeffs is (N_mo, N_basis) -> Transpose to (N_basis, N_mo)
        c_complex = complex_coeffs.T
        # P = C * diag(occ) * C.T
        p_complex = (c_complex * complex_occs) @ c_complex.T
        
        # 3. Difference Density
        p_diff = p_complex - p_frz
        
        # 4. NOCV
        # Solve generalized eigenvalue problem: P_diff * v = lambda * S * v
        # scipy.linalg.eigh(a, b) solves a*v = lambda*b*v
        evals, evecs = eigh(p_diff, overlap)
        
        # Sort by absolute magnitude of eigenvalues (standard for NOCV)
        # or just descending? Usually NOCVs come in pairs +/-.
        # Let's sort descending
        idx_sorted = np.argsort(evals)[::-1]
        evals = evals[idx_sorted]
        evecs = evecs[:, idx_sorted]
        
        return evals, evecs, p_diff

    # Execute for spins
    if is_unrestricted:
        # Alpha
        evals_a, evecs_a, p_diff_a = process_spin(
            complex_wfn.coefficients, complex_wfn.occupations,
            [f.coefficients for f in fragment_wfns],
            [f.occupations for f in fragment_wfns],
            1.0
        )
        # Beta
        evals_b, evecs_b, p_diff_b = process_spin(
            complex_wfn.coefficients_beta, complex_wfn.occupations_beta,
            [f.coefficients_beta for f in fragment_wfns],
            [f.occupations_beta for f in fragment_wfns],
            1.0
        )
        
        return {
            "alpha": {"eigenvalues": evals_a, "orbitals": evecs_a, "p_diff": p_diff_a},
            "beta": {"eigenvalues": evals_b, "orbitals": evecs_b, "p_diff": p_diff_b}
        }
    else:
        # Restricted
        # spin_factor = 2.0 because P_frz is constructed from spatial orbitals that are doubly occupied
        evals, evecs, p_diff = process_spin(
            complex_wfn.coefficients, complex_wfn.occupations,
            [f.coefficients for f in fragment_wfns],
            [f.occupations for f in fragment_wfns],
            2.0
        )
        
        return {
            "total": {"eigenvalues": evals, "orbitals": evecs, "p_diff": p_diff}
        }
