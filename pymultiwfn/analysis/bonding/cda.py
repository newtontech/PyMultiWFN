
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pymultiwfn.core.data import Wavefunction

def calculate_cda(
    complex_wfn: Wavefunction,
    fragments: List[Wavefunction],
    fragment_indices: Tuple[int, int] = (0, 1),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Performs Charge Decomposition Analysis (CDA).
    
    Args:
        complex_wfn: Wavefunction of the complex.
        fragments: List of Wavefunction objects for the fragments.
        fragment_indices: Indices of the two fragments in the 'fragments' list to analyze (0-based).
        verbose: Whether to print results to stdout.
        
    Returns:
        Dictionary containing CDA results:
        - 'd': Donation (from frag1 to frag2)
        - 'b': Back-donation (from frag2 to frag1)
        - 'r': Repulsive polarization
        - 'orbital_terms': List of dicts with per-orbital d, b, r terms.
    """
    
    # Basic validation
    if len(fragments) < 2:
        raise ValueError("At least two fragments are required for CDA.")
    
    frag1_idx, frag2_idx = fragment_indices
    frag1 = fragments[frag1_idx]
    frag2 = fragments[frag2_idx]
    
    # Check basis set consistency
    total_basis_fragments = sum(f.num_basis for f in fragments)
    if complex_wfn.num_basis != total_basis_fragments:
        raise ValueError(f"Number of basis functions in complex ({complex_wfn.num_basis}) "
                         f"does not match sum of fragments ({total_basis_fragments}).")

    # Ensure density matrices/occupations are available
    if complex_wfn.occupations is None:
        # This might happen if only coefficients are loaded. 
        # For now, assume occupations are critical.
        raise ValueError("Complex wavefunction must have occupation numbers.")

    # Construct C_FO_AO matrix (Transformation from Fragment Orbitals to AO basis)
    # This is a block diagonal matrix where diagonal blocks are MO coefficients of fragments.
    # Dimensions: (Total Basis) x (Total Basis) assuming n_mo = n_basis for fragments (or we use all MOs)
    
    c_fo_ao = np.zeros((complex_wfn.num_basis, complex_wfn.num_basis))
    
    current_basis_idx = 0
    fragment_mo_ranges = []
    
    for i, frag in enumerate(fragments):
        n_basis = frag.num_basis
        n_mo = frag.coefficients.shape[0] # Assuming coefficients are (N_MO, N_Basis)
        
        # We assume N_MO == N_Basis for the transformation to be square and invertible
        # If N_MO < N_Basis, we might need to handle it (e.g. linear dependence), but typically for full basis they match.
        if n_mo != n_basis:
             # If strictly required, we might need to pad or handle rectangular matrices.
             # For now, let's assume square for simplicity or take the min.
             pass

        # Copy fragment coefficients to the block
        # frag.coefficients is (N_MO, N_Basis). We want (N_Basis, N_MO) in the block if we consider columns as MOs.
        # But wait, standard matrix multiplication C_AO_MO * v_MO = v_AO.
        # So columns should be MOs.
        # frag.coefficients usually stores MOs as rows (PySCF/Multiwfn convention in Python often).
        # Let's check pymultiwfn convention. In `mayer.py`: `PS_total = wavefunction.Ptot @ wavefunction.overlap_matrix`
        # `Ptot` is density matrix.
        # `coefficients` in `data.py` is likely (N_MO, N_Basis).
        # So we need transpose to get (N_Basis, N_MO).
        
        c_frag_T = frag.coefficients.T
        c_fo_ao[current_basis_idx : current_basis_idx + n_basis, current_basis_idx : current_basis_idx + n_basis] = c_frag_T
        
        fragment_mo_ranges.append((current_basis_idx, current_basis_idx + n_basis))
        current_basis_idx += n_basis

    # Construct C_Complex_FO (Complex orbitals expressed in Fragment Orbital basis)
    # C_Complex_AO = C_FO_AO @ C_Complex_FO
    # => C_Complex_FO = inv(C_FO_AO) @ C_Complex_AO
    
    # complex_wfn.coefficients is (N_MO_Complex, N_Basis). Transpose to (N_Basis, N_MO_Complex).
    c_complex_ao_T = complex_wfn.coefficients.T
    
    try:
        c_fo_ao_inv = np.linalg.inv(c_fo_ao)
    except np.linalg.LinAlgError:
        raise ValueError("Fragment coefficient matrix is singular. Basis functions might be linearly dependent.")
        
    c_complex_fo = c_fo_ao_inv @ c_complex_ao_T
    # c_complex_fo is now (N_FO, N_MO_Complex). Rows are FOs, Columns are Complex MOs.
    
    # Calculate Overlap Matrix between Fragment Orbitals (S_FO)
    # S_FO = C_FO_AO.T @ S_AO @ C_FO_AO
    # But we need to handle the "intra-fragment overlap is zero" logic from CDA.f90 if we want to match it exactly.
    # CDA.f90: ovlpbasmatblk is off-diagonal block version of ovlpbasmat.
    # FOovlpmat = tmpmat^T @ ovlpbasmatblk @ tmpmat
    # And then diagonal elements set to 1.
    
    if complex_wfn.overlap_matrix is None:
        raise ValueError("Complex wavefunction must have overlap matrix.")
        
    s_ao = complex_wfn.overlap_matrix
    
    # Create block-masked S_AO (only inter-fragment blocks)
    s_ao_masked = s_ao.copy()
    
    # Zero out intra-fragment blocks
    curr_idx = 0
    for frag in fragments:
        n_b = frag.num_basis
        s_ao_masked[curr_idx : curr_idx + n_b, curr_idx : curr_idx + n_b] = 0.0
        curr_idx += n_b
        
    # Calculate S_FO (inter-fragment)
    # c_fo_ao is (N_Basis, N_FO)
    s_fo = c_fo_ao.T @ s_ao_masked @ c_fo_ao
    
    # Set diagonal elements to 1 (as per CDA.f90)
    np.fill_diagonal(s_fo, 1.0)
    
    # Now calculate d, b, r terms
    # Iterate over complex orbitals
    
    d_terms = np.zeros(complex_wfn.coefficients.shape[0])
    b_terms = np.zeros(complex_wfn.coefficients.shape[0])
    r_terms = np.zeros(complex_wfn.coefficients.shape[0])
    
    # Get ranges for the two fragments of interest
    range1 = fragment_mo_ranges[frag1_idx]
    range2 = fragment_mo_ranges[frag2_idx]
    
    # Reference occupation (2 for closed shell, 1 for open)
    # Assuming closed shell for now or inferring from complex
    ref_occ = 2.0 if not complex_wfn.is_unrestricted else 1.0
    
    orbital_data = []
    
    for i_orb in range(complex_wfn.coefficients.shape[0]):
        occ_complex = complex_wfn.occupations[i_orb]
        
        d_val = 0.0
        b_val = 0.0
        r_val = 0.0
        
        # Iterate over FOs of frag1
        for i_a in range(range1[0], range1[1]):
            # Iterate over FOs of frag2
            for i_b in range(range2[0], range2[1]):
                
                # Get occupations of fragment orbitals
                # We need to map i_a (index in total FOs) to index in frag1
                idx_in_frag1 = i_a - range1[0]
                idx_in_frag2 = i_b - range2[0]
                
                occ_a = fragments[frag1_idx].occupations[idx_in_frag1]
                occ_b = fragments[frag2_idx].occupations[idx_in_frag2]
                
                # occfac = (occ_A - occ_B) / refocc
                occfac = (occ_a - occ_b) / ref_occ
                
                # Coeffs of complex orbital i_orb in FO basis
                c_i_a = c_complex_fo[i_a, i_orb]
                c_i_b = c_complex_fo[i_b, i_orb]
                
                s_ab = s_fo[i_a, i_b]
                
                term = occ_complex * occfac * c_i_a * c_i_b * s_ab
                
                # r term uses 2 * min(occ_A, occ_B) / refocc
                # Note: CDA.f90 uses `2 * min(...)`. If refocc is 2, it cancels out?
                # CDA.f90 line 590: tmpval2=occCDA(iorb,0)*2*min(occCDA(iAidx,ifrag),occCDA(iBidx,jfrag))/refocc*coFO(iA,iorb)*coFO(iB,iorb)*FOovlpmat(iA,iB)
                r_term_val = occ_complex * (2.0 * min(occ_a, occ_b) / ref_occ) * c_i_a * c_i_b * s_ab
                
                if occfac > 0:
                    d_val += term
                elif occfac < 0:
                    b_val -= term # Minus because occfac is negative
                
                r_val += r_term_val
        
        d_terms[i_orb] = d_val
        b_terms[i_orb] = b_val
        r_terms[i_orb] = r_val
        
        orbital_data.append({
            "orbital_idx": i_orb,
            "occupation": occ_complex,
            "d": d_val,
            "b": b_val,
            "r": r_val
        })

    total_d = np.sum(d_terms)
    total_b = np.sum(b_terms)
    total_r = np.sum(r_terms)
    
    if verbose:
        print("Charge Decomposition Analysis (CDA)")
        print(f"Fragment 1: Index {frag1_idx}")
        print(f"Fragment 2: Index {frag2_idx}")
        print(f"{'Orbital':<8} {'Occ':<8} {'d':<12} {'b':<12} {'d-b':<12} {'r':<12}")
        for item in orbital_data:
            if abs(item['d']) > 1e-3 or abs(item['b']) > 1e-3 or abs(item['r']) > 1e-3:
                print(f"{item['orbital_idx']:<8} {item['occupation']:<8.4f} {item['d']:<12.6f} {item['b']:<12.6f} {item['d']-item['b']:<12.6f} {item['r']:<12.6f}")
        print("-" * 70)
        print(f"{'Sum':<8} {'':<8} {total_d:<12.6f} {total_b:<12.6f} {total_d-total_b:<12.6f} {total_r:<12.6f}")

    return {
        "d": total_d,
        "b": total_b,
        "r": total_r,
        "orbital_terms": orbital_data
    }
