
import numpy as np
from typing import Tuple, Optional

from pymultiwfn.core.data import Wavefunction

def calculate_mayer_bond_order(wavefunction: Wavefunction) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates the Mayer bond order matrix for a given wavefunction.

    Args:
        wavefunction: A Wavefunction object containing density matrices and overlap matrix.

    Returns:
        A tuple:
        - A numpy array representing the total Mayer bond order matrix.
        - A numpy array representing the alpha Mayer bond order matrix (if unrestricted), else None.
        - A numpy array representing the beta Mayer bond order matrix (if unrestricted), else None.
    """
    num_atoms = wavefunction.num_atoms
    num_basis = wavefunction.num_basis

    if wavefunction.overlap_matrix is None:
        raise ValueError("Overlap matrix (Sbas) is not available in the Wavefunction object.")
    if wavefunction.Ptot is None:
        wavefunction.calculate_density_matrices() # Ensure density matrices are calculated
    if wavefunction.Ptot is None: # Check again in case calculate_density_matrices failed
        raise ValueError("Total density matrix (Ptot) could not be calculated.")

    atomic_basis_indices = wavefunction.get_atomic_basis_indices()

    bond_order_matrix_total = np.zeros((num_atoms, num_atoms))
    bond_order_matrix_alpha = None
    bond_order_matrix_beta = None

    # Calculate P.S for total density
    PS_total = wavefunction.Ptot @ wavefunction.overlap_matrix

    # Calculate Mayer bond order for total density
    for i in range(num_atoms):
        basis_fns_i = atomic_basis_indices[i]
        for j in range(i + 1, num_atoms):
            basis_fns_j = atomic_basis_indices[j]
            
            # Extract relevant submatrices
            ps_ij = PS_total[np.ix_(basis_fns_i, basis_fns_j)]
            ps_ji = PS_total[np.ix_(basis_fns_j, basis_fns_i)]

            # Perform the sum for Mayer bond order
            accum = np.sum(ps_ij * ps_ji)
            bond_order_matrix_total[i, j] = accum
            bond_order_matrix_total[j, i] = accum
    
    # Fill diagonal elements: Mayer valence
    for i in range(num_atoms):
        bond_order_matrix_total[i, i] = np.sum(bond_order_matrix_total[i, :])


    # Handle unrestricted case
    if wavefunction.is_unrestricted:
        if wavefunction.Palpha is None or wavefunction.Pbeta is None:
             raise ValueError("Alpha and Beta density matrices are not available for unrestricted calculation.")

        bond_order_matrix_alpha = np.zeros((num_atoms, num_atoms))
        bond_order_matrix_beta = np.zeros((num_atoms, num_atoms))

        PS_alpha = wavefunction.Palpha @ wavefunction.overlap_matrix
        PS_beta = wavefunction.Pbeta @ wavefunction.overlap_matrix

        for i in range(num_atoms):
            basis_fns_i = atomic_basis_indices[i]
            for j in range(i + 1, num_atoms):
                basis_fns_j = atomic_basis_indices[j]

                ps_alpha_ij = PS_alpha[np.ix_(basis_fns_i, basis_fns_j)]
                ps_alpha_ji = PS_alpha[np.ix_(basis_fns_j, basis_fns_i)]
                accum_alpha = np.sum(ps_alpha_ij * ps_alpha_ji)
                bond_order_matrix_alpha[i, j] = accum_alpha
                bond_order_matrix_alpha[j, i] = accum_alpha

                ps_beta_ij = PS_beta[np.ix_(basis_fns_i, basis_fns_j)]
                ps_beta_ji = PS_beta[np.ix_(basis_fns_j, basis_fns_i)]
                accum_beta = np.sum(ps_beta_ij * ps_beta_ji)
                bond_order_matrix_beta[i, j] = accum_beta
                bond_order_matrix_beta[j, i] = accum_beta
        
        # Multiply by 2 as per Fortran code for alpha and beta contributions (seems to be a Multiwfn convention)
        # Note: The Fortran code has "bndmata=2*(bndmata+transpose(bndmata))" for alpha/beta contributions after filling.
        # This implies that the 'accum' already is 1/2 of the final value.
        # My current accum calculation for alpha/beta is the same as for total, so I need to account for this factor of 2.
        # However, the Fortran `calcMayerbndord` subroutine returns `bndmata(i,j)` as `accuma` and `bndmatb(i,j)` as `accumb`
        # and then `bndmata=2*(bndmata+transpose(bndmata))`. So the sum is effectively `2 * sum(PS * PS)`.
        # The total `bndmattot` for open-shell is *not* multiplied by 2 in the same way, implying `bndmattot` is Generalized Wiberg.
        # For simplicity and consistency with Fortran output interpretation:
        # For alpha/beta bond orders, the Fortran seems to apply a factor of 2.
        # Let's apply it here for alpha/beta to match Fortran output.

        bond_order_matrix_alpha = 2 * bond_order_matrix_alpha
        bond_order_matrix_beta = 2 * bond_order_matrix_beta

        # Fill diagonal elements for alpha and beta valences
        for i in range(num_atoms):
            bond_order_matrix_alpha[i, i] = np.sum(bond_order_matrix_alpha[i, :])
            bond_order_matrix_beta[i, i] = np.sum(bond_order_matrix_beta[i, :])

    return bond_order_matrix_total, bond_order_matrix_alpha, bond_order_matrix_beta
