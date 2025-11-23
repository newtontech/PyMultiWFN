import numpy as np
from pymultiwfn.core.data import Wavefunction
from typing import Tuple, Optional

def calculate_mulliken_bond_order(
    wavefunction: Wavefunction, 
    overlap_matrix: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates the Mulliken bond order (Mulliken overlap population) matrix.

    Args:
        wavefunction: The Wavefunction object containing MO coefficients, occupations, etc.
        overlap_matrix: The overlap matrix (S_uv).

    Returns:
        A tuple (bnd_mattot, bnd_mata, bnd_matb)
        bnd_mattot: Total Mulliken bond order matrix.
        bnd_mata: Alpha Mulliken bond order matrix (None if restricted).
        bnd_matb: Beta Mulliken bond order matrix (None if restricted).
    """
    if wavefunction.Ptot is None or wavefunction.Palpha is None or wavefunction.Pbeta is None:
        wavefunction.calculate_density_matrices()

    num_atoms = wavefunction.num_atoms
    
    # Get mapping from atom index to its basis function indices
    atom_to_bfs_map = wavefunction.get_atomic_basis_indices()

    bnd_mattot = np.zeros((num_atoms, num_atoms))
    bnd_mata = None
    bnd_matb = None

    # --- Calculate total Mulliken bond order ---
    # In Fortran, this is PSmata = Sbas * Ptot (element-wise product)
    PS_tot_element_wise = wavefunction.Ptot * overlap_matrix

    for i in range(num_atoms):
        bfs_i = atom_to_bfs_map.get(i, []) # Get list of bfs for atom i
        if not bfs_i: # Skip if atom has no basis functions assigned
            continue

        for j in range(i + 1, num_atoms):
            bfs_j = atom_to_bfs_map.get(j, []) # Get list of bfs for atom j
            if not bfs_j: # Skip if atom has no basis functions assigned
                continue

            # Sum up elements of the population matrix belonging to basis functions
            # on atom i and atom j.
            bond_order_val = np.sum(PS_tot_element_wise[np.ix_(bfs_i, bfs_j)])
            
            bnd_mattot[i, j] = bond_order_val
            bnd_mattot[j, i] = bond_order_val # Symmetric matrix

    # Fortran code applies factor of 2 and then sums diagonals
    # bndmattot=2*(bndmattot+transpose(bndmattot))
    # forall (i=1:ncenter) bndmattot(i,i)=sum(bndmattot(i,:))
    # It seems to apply 2* to off-diagonals, and then sum for diagonals.
    # We can do this in two steps to be explicit.
    bnd_mattot_off_diag = bnd_mattot + bnd_mattot.T # Already symmetric from above loop, just make explicit
    bnd_mattot = 2 * bnd_mattot_off_diag
    for i in range(num_atoms):
        bnd_mattot[i, i] = np.sum(bnd_mattot[i, :])


    # --- Calculate Alpha and Beta Mulliken Bond Orders for Unrestricted Wavefunctions ---
    if wavefunction.is_unrestricted:
        bnd_mata = np.zeros((num_atoms, num_atoms))
        bnd_matb = np.zeros((num_atoms, num_atoms))

        PS_alpha_element_wise = wavefunction.Palpha * overlap_matrix
        PS_beta_element_wise = wavefunction.Pbeta * overlap_matrix

        for i in range(num_atoms):
            bfs_i = atom_to_bfs_map.get(i, [])
            if not bfs_i:
                continue

            for j in range(i + 1, num_atoms):
                bfs_j = atom_to_bfs_map.get(j, [])
                if not bfs_j:
                    continue

                bond_order_alpha = np.sum(PS_alpha_element_wise[np.ix_(bfs_i, bfs_j)])
                bond_order_beta = np.sum(PS_beta_element_wise[np.ix_(bfs_i, bfs_j)])
                
                bnd_mata[i, j] = bond_order_alpha
                bnd_mata[j, i] = bond_order_alpha
                bnd_matb[i, j] = bond_order_beta
                bnd_matb[j, i] = bond_order_beta
        
        bnd_mata_off_diag = bnd_mata + bnd_mata.T
        bnd_mata = 2 * bnd_mata_off_diag
        bnd_matb_off_diag = bnd_matb + bnd_matb.T
        bnd_matb = 2 * bnd_matb_off_diag

        # Set diagonal elements
        for i in range(num_atoms):
            bnd_mata[i, i] = np.sum(bnd_mata[i, :])
            bnd_matb[i, i] = np.sum(bnd_matb[i, :])
        
        # Total Mulliken bond order for unrestricted case is sum of alpha and beta
        bnd_mattot = bnd_mata + bnd_matb


    return bnd_mattot, bnd_mata, bnd_matb