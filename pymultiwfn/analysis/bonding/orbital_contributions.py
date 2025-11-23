
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from pymultiwfn.core.data import Wavefunction
from .mayer import calculate_mayer_bond_order # To reuse Mayer BO calculation

def calculate_orbital_mulliken_contribution(
    wavefunction: Wavefunction,
    atom1_idx: int,
    atom2_idx: int
) -> List[Dict[str, Any]]:
    """
    Decomposes Mulliken bond order between two atoms into orbital contributions.
    Translated from decompMullikenBO in bondorder.f90.

    Args:
        wavefunction: A Wavefunction object.
        atom1_idx: 0-based index of the first atom.
        atom2_idx: 0-based index of the second atom.

    Returns:
        A list of dictionaries, each containing 'orbital_idx', 'occupation', 'energy', 'contribution_alpha', 'contribution_beta', 'total_contribution'.
    """
    if wavefunction.coefficients is None or wavefunction.overlap_matrix is None or wavefunction.occupations is None:
        raise ValueError("Wavefunction coefficients, overlap matrix, or occupations are missing.")

    wavefunction.calculate_density_matrices() # Ensure density matrices and occupations are inferred

    atomic_basis_indices = wavefunction.get_atomic_basis_indices()
    basis_fns_1 = atomic_basis_indices[atom1_idx]
    basis_fns_2 = atomic_basis_indices[atom2_idx]

    results = []

    # Alpha orbitals
    for imo in range(wavefunction.coefficients.shape[0]):
        occ_alpha = wavefunction.occupations[imo] if wavefunction.occupations is not None else 0.0
        energy_alpha = wavefunction.energies[imo] if wavefunction.energies is not None else 0.0
        
        if occ_alpha < 1e-10: # Skip virtual orbitals or very low occupation
            continue

        # Fortran: accum = MOocc(irealmo) * ptmat(i,imo) * ptmat(j,imo) * Sbas(i,j)
        # summed over i in atom1_bfs, j in atom2_bfs
        # ptmat is CObasa (coefficients) or CObasb.
        
        contrib_alpha = 0.0
        # Submatrices of coefficients and overlap matrix for efficient calculation
        coeffs_1_mo = wavefunction.coefficients[imo, basis_fns_1]
        coeffs_2_mo = wavefunction.coefficients[imo, basis_fns_2]
        overlap_12 = wavefunction.overlap_matrix[np.ix_(basis_fns_1, basis_fns_2)]

        # Outer product of coefficients, then element-wise multiply with overlap, then sum
        contrib_alpha = occ_alpha * np.sum(np.outer(coeffs_1_mo, coeffs_2_mo) * overlap_12)
        
        result_entry = {
            "orbital_idx": imo,
            "occupation": occ_alpha,
            "energy": energy_alpha,
            "contribution_alpha": 2 * contrib_alpha, # Fortran multiplies by 2 for the output
            "contribution_beta": 0.0, # Will be filled for unrestricted if applicable
            "total_contribution": 2 * contrib_alpha
        }
        results.append(result_entry)

    # Beta orbitals for unrestricted cases
    if wavefunction.is_unrestricted and wavefunction.coefficients_beta is not None and wavefunction.occupations_beta is not None:
        for imo_beta in range(wavefunction.coefficients_beta.shape[0]):
            occ_beta = wavefunction.occupations_beta[imo_beta] if wavefunction.occupations_beta is not None else 0.0
            energy_beta = wavefunction.energies_beta[imo_beta] if wavefunction.energies_beta is not None else 0.0
            
            if occ_beta < 1e-10:
                continue
            
            contrib_beta = 0.0
            coeffs_1_mo_beta = wavefunction.coefficients_beta[imo_beta, basis_fns_1]
            coeffs_2_mo_beta = wavefunction.coefficients_beta[imo_beta, basis_fns_2]
            overlap_12 = wavefunction.overlap_matrix[np.ix_(basis_fns_1, basis_fns_2)]

            contrib_beta = occ_beta * np.sum(np.outer(coeffs_1_mo_beta, coeffs_2_mo_beta) * overlap_12)
            
            # Find existing entry or create a new one for this beta orbital contribution
            # For Mulliken, alpha and beta MOs are distinct, so they would be separate entries.
            result_entry = {
                "orbital_idx": imo_beta,
                "occupation": occ_beta,
                "energy": energy_beta,
                "contribution_alpha": 0.0,
                "contribution_beta": 2 * contrib_beta,
                "total_contribution": 2 * contrib_beta
            }
            results.append(result_entry)

    # Combine contributions for total if both alpha and beta are present for the *same* orbital index in restricted cases,
    # but for Mulliken decomposition, alpha and beta are generally distinct.
    # The Fortran code has separate accumulators `bndorda` and `bndordb` and sums them up at the end.
    # The current structure already handles separate entries.
    return results


def calculate_orbital_perturbed_mayer_bond_order(
    wavefunction: Wavefunction,
    atom1_idx: int,
    atom2_idx: int
) -> List[Dict[str, Any]]:
    """
    Calculates the orbital occupancy-perturbed Mayer bond order between two atoms.
    Translated from OrbPertMayer in bondorder.f90.

    Args:
        wavefunction: A Wavefunction object.
        atom1_idx: 0-based index of the first atom.
        atom2_idx: 0-based index of the second atom.

    Returns:
        A list of dictionaries, each containing 'orbital_idx', 'occupation', 'energy', 
        'bond_order_after_perturbation_alpha', 'bond_order_after_perturbation_beta', 
        'bond_order_after_perturbation_total', 'variance'.
    """
    if wavefunction.coefficients is None or wavefunction.overlap_matrix is None or wavefunction.occupations is None:
        raise ValueError("Wavefunction coefficients, overlap matrix, or occupations are missing.")

    wavefunction.calculate_density_matrices() # Ensure density matrices and occupations are inferred

    num_basis = wavefunction.num_basis
    atomic_basis_indices = wavefunction.get_atomic_basis_indices()

    # Calculate the unperturbed Mayer bond order (total, alpha, beta)
    unperturbed_mayer_total_matrix, unperturbed_mayer_alpha_matrix, unperturbed_mayer_beta_matrix = \
        calculate_mayer_bond_order(wavefunction)
    
    before_pert_total = unperturbed_mayer_total_matrix[atom1_idx, atom2_idx]
    before_pert_alpha = unperturbed_mayer_alpha_matrix[atom1_idx, atom2_idx] if unperturbed_mayer_alpha_matrix is not None else 0.0
    before_pert_beta = unperturbed_mayer_beta_matrix[atom1_idx, atom2_idx] if unperturbed_mayer_beta_matrix is not None else 0.0
    
    results = []

    # Iterate through all MOs and perturb the density matrix
    for imo in range(wavefunction.coefficients.shape[0]):
        occ = wavefunction.occupations[imo]
        energy = wavefunction.energies[imo]

        if occ < 1e-10 and not (wavefunction.is_unrestricted and wavefunction.occupations_beta is not None and imo < wavefunction.coefficients_beta.shape[0] and wavefunction.occupations_beta[imo] > 1e-10):
            # Skip if both alpha and beta occupations are negligible for this MO index
            continue

        perturbed_ptot = np.copy(wavefunction.Ptot)
        perturbed_palpha = np.copy(wavefunction.Palpha)
        perturbed_pbeta = np.copy(wavefunction.Pbeta)

        # Perturb the density matrices by removing the contribution of the current MO
        # Fortran: Ptottmp(ibas,jbas)=Ptottmp(ibas,jbas)-MOocc(imo)*CObasa(ibas,imo)*CObasa(jbas,imo)

        # For total (closed-shell like or mixed open-shell)
        # The perturbation logic in Fortran for total is only for closed shell.
        # For open-shell, it perturbs Palpha and Pbeta separately then sums up.
        if not wavefunction.is_unrestricted: # Closed-shell
            if occ > 1e-10:
                mo_coeffs = wavefunction.coefficients[imo, :]
                perturbed_ptot -= occ * np.outer(mo_coeffs, mo_coeffs) # This should be `occ` times 2 for a doubly occupied orbital in restricted calc.
                # The Fortran uses MOocc(imo) * CObasa(ibas,imo) * CObasa(jbas,imo)
                # For closed shell, MOocc is 2.0 for occupied, 0.0 for virtual.
                # So `occ * outer` is correct here for contribution removal.

        else: # Unrestricted
            # Determine if this MO is an alpha or beta MO based on its index relative to nbasis
            # This mapping might be tricky. Fortran `OrbPertMayer` has complex `wfntype` logic.
            # Let's simplify: if an MO index is for alpha, perturb alpha; if for beta, perturb beta.
            # Assuming MOs are indexed 0 to nbasis-1 for alpha, and nbasis to nbasis+num_beta_mos-1 for beta.

            # Alpha part
            if imo < wavefunction.coefficients.shape[0] and wavefunction.occupations[imo] > 1e-10:
                mo_coeffs_alpha = wavefunction.coefficients[imo, :]
                perturbed_palpha -= wavefunction.occupations[imo] * np.outer(mo_coeffs_alpha, mo_coeffs_alpha)
            
            # Beta part (if corresponds to a beta MO for the current `imo` context)
            if wavefunction.coefficients_beta is not None and wavefunction.occupations_beta is not None and \
               imo < wavefunction.coefficients_beta.shape[0] and wavefunction.occupations_beta[imo] > 1e-10:
                mo_coeffs_beta = wavefunction.coefficients_beta[imo, :]
                perturbed_pbeta -= wavefunction.occupations_beta[imo] * np.outer(mo_coeffs_beta, mo_coeffs_beta)

        # Recalculate Mayer bond order with perturbed density matrix
        # Create a dummy wavefunction object to pass to calculate_mayer_bond_order
        perturbed_wf = Wavefunction(
            atoms=wavefunction.atoms,
            num_electrons=wavefunction.num_electrons,
            charge=wavefunction.charge,
            multiplicity=wavefunction.multiplicity,
            shells=wavefunction.shells,
            num_basis=wavefunction.num_basis,
            num_primitives=wavefunction.num_primitives,
            coefficients=wavefunction.coefficients,
            energies=wavefunction.energies,
            occupations=wavefunction.occupations,
            is_unrestricted=wavefunction.is_unrestricted,
            coefficients_beta=wavefunction.coefficients_beta,
            energies_beta=wavefunction.energies_beta,
            occupations_beta=wavefunction.occupations_beta,
            Palpha=perturbed_palpha,
            Pbeta=perturbed_pbeta,
            Ptot=perturbed_ptot, # This needs to be consistent, if Palpha/Pbeta are perturbed, Ptot should be Palpha+Pbeta
            overlap_matrix=wavefunction.overlap_matrix
        )
        if wavefunction.is_unrestricted:
             perturbed_wf.Ptot = perturbed_palpha + perturbed_pbeta # Update Ptot if alpha/beta were perturbed

        perturbed_mayer_total_matrix, perturbed_mayer_alpha_matrix, perturbed_mayer_beta_matrix = \
            calculate_mayer_bond_order(perturbed_wf)

        after_pert_total = perturbed_mayer_total_matrix[atom1_idx, atom2_idx]
        after_pert_alpha = perturbed_mayer_alpha_matrix[atom1_idx, atom2_idx] if perturbed_mayer_alpha_matrix is not None else 0.0
        after_pert_beta = perturbed_mayer_beta_matrix[atom1_idx, atom2_idx] if perturbed_mayer_beta_matrix is not None else 0.0
        
        # Calculate variance
        variance_total = after_pert_total - before_pert_total
        variance_alpha = after_pert_alpha - before_pert_alpha
        variance_beta = after_pert_beta - before_pert_beta

        results.append({
            "orbital_idx": imo,
            "occupation": occ,
            "energy": energy,
            "bond_order_after_perturbation_alpha": after_pert_alpha if wavefunction.is_unrestricted else None,
            "bond_order_after_perturbation_beta": after_pert_beta if wavefunction.is_unrestricted else None,
            "bond_order_after_perturbation_total": after_pert_total,
            "variance_total": variance_total,
            "variance_alpha": variance_alpha if wavefunction.is_unrestricted else None,
            "variance_beta": variance_beta if wavefunction.is_unrestricted else None,
        })

    return results
