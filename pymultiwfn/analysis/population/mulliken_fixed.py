import numpy as np
from pymultiwfn.core.data import Wavefunction
from typing import Tuple, Optional, Dict

def calculate_mulliken_population_and_charges(
    wavefunction: Wavefunction, 
    overlap_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates Mulliken atomic populations and charges.
    FIXED VERSION - Correctly sums only diagonal elements for each atom's block.
    
    Args:
        wavefunction: The Wavefunction object containing MO coefficients, occupations, etc.
        overlap_matrix: The overlap matrix (S_uv).
    
    Returns:
        A tuple:
        - total_atomic_populations (np.ndarray): Array of total Mulliken atomic populations.
        - total_atomic_charges (np.ndarray): Array of total Mulliken atomic charges.
        - alpha_atomic_populations (Optional[np.ndarray]): Alpha atomic populations (if unrestricted).
        - beta_atomic_populations (Optional[np.ndarray]): Beta atomic populations (if unrestricted).
        - spin_densities (Optional[np.ndarray]): Spin densities (alpha - beta populations, if unrestricted).
    """
    if wavefunction.Ptot is None or wavefunction.Palpha is None or wavefunction.Pbeta is None:
        wavefunction.calculate_density_matrices()

    num_atoms = wavefunction.num_atoms
    num_basis = wavefunction.num_basis
    
    atom_to_bfs_map = wavefunction.get_atomic_basis_indices()
    
    total_atomic_populations = np.zeros(num_atoms)
    alpha_atomic_populations = None
    beta_atomic_populations = None
    spin_densities = None

    # Calculate P_tot * S (element-wise product)
    PS_tot_element_wise = wavefunction.Ptot * overlap_matrix

    # Calculate total atomic populations
    # FIX: Only sum diagonal elements corresponding to atom i's basis functions
    for i in range(num_atoms):
        bfs_i = atom_to_bfs_map.get(i, [])
        if not bfs_i:
            continue
        
        # Extract diagonal elements for atom i's basis functions
        atom_diagonal = np.diag(PS_tot_element_wise)[bfs_i]
        total_atomic_populations[i] = np.sum(atom_diagonal)

    total_atomic_charges = np.array([atom.charge for atom in wavefunction.atoms]) - total_atomic_populations

    # Handle unrestricted case
    if wavefunction.is_unrestricted:
        alpha_atomic_populations = np.zeros(num_atoms)
        beta_atomic_populations = np.zeros(num_atoms)

        PS_alpha_element_wise = wavefunction.Palpha * overlap_matrix
        PS_beta_element_wise = wavefunction.Pbeta * overlap_matrix

        # FIX: Same fix for alpha and beta populations
        for i in range(num_atoms):
            bfs_i = atom_to_bfs_map.get(i, [])
            if not bfs_i:
                continue

            # Extract diagonal elements for alpha and beta
            atom_alpha_diagonal = np.diag(PS_alpha_element_wise)[bfs_i]
            atom_beta_diagonal = np.diag(PS_beta_element_wise)[bfs_i]

            alpha_atomic_populations[i] = np.sum(atom_alpha_diagonal)
            beta_atomic_populations[i] = np.sum(atom_beta_diagonal)

        spin_densities = alpha_atomic_populations - beta_atomic_populations

    return (total_atomic_populations, total_atomic_charges, 
            alpha_atomic_populations, beta_atomic_populations, spin_densities)
