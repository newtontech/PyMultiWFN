
import numpy as np
from typing import Tuple, Optional

from pymultiwfn.core.data import Wavefunction

def calculate_mulliken_bond_order(wavefunction: Wavefunction) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates the Mulliken bond order matrix for a given wavefunction.

    Args:
        wavefunction: A Wavefunction object containing density matrices and overlap matrix.

    Returns:
        A tuple:
        - A numpy array representing the total Mulliken bond order matrix.
        - A numpy array representing the alpha Mulliken bond order matrix (if unrestricted), else None.
        - A numpy array representing the beta Mulliken bond order matrix (if unrestricted), else None.
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

    # Calculate Mulliken bond order for total density
    # The Fortran code calculates only for upper triangle (j = i+1 to ncenter), then makes it symmetric and multiplies by 2.
    for i in range(num_atoms):
        basis_fns_i = atomic_basis_indices[i]
        for j in range(i + 1, num_atoms):
            basis_fns_j = atomic_basis_indices[j]
            
            # Extract relevant submatrices
            ptot_ij = wavefunction.Ptot[np.ix_(basis_fns_i, basis_fns_j)]
            sbas_ij = wavefunction.overlap_matrix[np.ix_(basis_fns_i, basis_fns_j)]

            # Perform the sum for Mulliken bond order
            accum_total = np.sum(ptot_ij * sbas_ij)
            bond_order_matrix_total[i, j] = accum_total
    
    # Make symmetric and multiply by 2
    bond_order_matrix_total = 2 * (bond_order_matrix_total + bond_order_matrix_total.T)

    # Fill diagonal elements: Sum of corresponding row elements
    for i in range(num_atoms):
        bond_order_matrix_total[i, i] = np.sum(bond_order_matrix_total[i, :])

    # Handle unrestricted case
    if wavefunction.is_unrestricted:
        if wavefunction.Palpha is None or wavefunction.Pbeta is None:
             raise ValueError("Alpha and Beta density matrices are not available for unrestricted calculation.")

        bond_order_matrix_alpha = np.zeros((num_atoms, num_atoms))
        bond_order_matrix_beta = np.zeros((num_atoms, num_atoms))

        for i in range(num_atoms):
            basis_fns_i = atomic_basis_indices[i]
            for j in range(i + 1, num_atoms):
                basis_fns_j = atomic_basis_indices[j]

                ptot_alpha_ij = wavefunction.Palpha[np.ix_(basis_fns_i, basis_fns_j)]
                ptot_beta_ij = wavefunction.Pbeta[np.ix_(basis_fns_i, basis_fns_j)]
                sbas_ij = wavefunction.overlap_matrix[np.ix_(basis_fns_i, basis_fns_j)]

                accum_alpha = np.sum(ptot_alpha_ij * sbas_ij)
                bond_order_matrix_alpha[i, j] = accum_alpha

                accum_beta = np.sum(ptot_beta_ij * sbas_ij)
                bond_order_matrix_beta[i, j] = accum_beta
        
        # Make symmetric and multiply by 2 for alpha and beta
        bond_order_matrix_alpha = 2 * (bond_order_matrix_alpha + bond_order_matrix_alpha.T)
        bond_order_matrix_beta = 2 * (bond_order_matrix_beta + bond_order_matrix_beta.T)

        # Fill diagonal elements for alpha and beta
        for i in range(num_atoms):
            bond_order_matrix_alpha[i, i] = np.sum(bond_order_matrix_alpha[i, :])
            bond_order_matrix_beta[i, i] = np.sum(bond_order_matrix_beta[i, :])

    return bond_order_matrix_total, bond_order_matrix_alpha, bond_order_matrix_beta
