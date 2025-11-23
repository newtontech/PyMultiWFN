
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations
from pymultiwfn.core.data import Wavefunction

def search_adndp_candidates(
    dm_nao: np.ndarray,
    atom_to_nao_indices: Dict[int, List[int]],
    search_atoms: List[int],
    n_centers: int,
    threshold: float
) -> List[Dict[str, Any]]:
    """
    Performs an exhaustive search for AdNDP candidate orbitals with a specific number of centers.
    
    Args:
        dm_nao: Density matrix in NAO basis (N_NAO x N_NAO).
        atom_to_nao_indices: Dictionary mapping atom index to list of NAO indices.
        search_atoms: List of atom indices to include in the search.
        n_centers: Number of centers (atoms) to search for (e.g., 2 for 2c-2e bonds).
        threshold: Occupation threshold to consider an orbital as a candidate.
                   (e.g., 1.7-1.9 for restricted 2e bonds).
                   
    Returns:
        List of dictionaries, each representing a candidate orbital:
        {
            "occupation": float,
            "vector": np.ndarray, # Coefficient vector in full NAO basis
            "atoms": List[int],   # Atom indices involved
            "eigenvalue_index": int # Index in the block diagonalization
        }
    """
    candidates = []
    
    # Iterate over all combinations of n_centers atoms from the search list
    for atom_comb in combinations(search_atoms, n_centers):
        atom_comb_list = list(atom_comb)
        
        # Collect all NAO indices for this group of atoms
        nao_indices = []
        for atom_idx in atom_comb_list:
            if atom_idx in atom_to_nao_indices:
                nao_indices.extend(atom_to_nao_indices[atom_idx])
        
        if not nao_indices:
            continue
            
        # Extract density matrix block
        # dm_block = dm_nao[np.ix_(nao_indices, nao_indices)]
        # Using np.ix_ for block extraction
        dm_block = dm_nao[np.ix_(nao_indices, nao_indices)]
        
        # Diagonalize
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(dm_block)
        
        # Check eigenvalues against threshold
        # Iterate in reverse (largest first)
        for i in range(len(eigvals) - 1, -1, -1):
            occ = eigvals[i]
            if occ >= threshold:
                # Found a candidate
                # Construct full vector
                full_vec = np.zeros(dm_nao.shape[0])
                # eigvecs[:, i] corresponds to the i-th eigenvalue
                local_vec = eigvecs[:, i]
                full_vec[nao_indices] = local_vec
                
                candidates.append({
                    "occupation": occ,
                    "vector": full_vec,
                    "atoms": atom_comb_list,
                    "eigenvalue_index": i
                })
            else:
                # Since sorted, if this one is below threshold, smaller ones are too
                break
                
    # Sort candidates by occupation (descending)
    candidates.sort(key=lambda x: x["occupation"], reverse=True)
    
    return candidates

def deplete_density(dm_nao: np.ndarray, orbital_vector: np.ndarray, occupation: float) -> np.ndarray:
    """
    Depletes the density of a selected orbital from the density matrix.
    P_new = P_old - occ * v * v.T
    
    Args:
        dm_nao: Current density matrix.
        orbital_vector: Normalized orbital vector (column vector).
        occupation: Occupation number to remove.
        
    Returns:
        Updated density matrix.
    """
    # Ensure vector is column vector for outer product
    v = orbital_vector.reshape(-1, 1)
    return dm_nao - occupation * (v @ v.T)

# TODO: Add a high-level driver function that manages the iterative process
# once we have a way to load/generate NAO density matrices.
