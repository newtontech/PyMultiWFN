
import numpy as np
from typing import List, Tuple, Optional
from itertools import permutations
import math

from pymultiwfn.core.data import Wavefunction

def _factorial(n: int) -> int:
    """Calculates n!"""
    return math.factorial(n)

def _reverse_list(lst: List) -> List:
    """Reverses a list."""
    return lst[::-1]

def calculate_multicenter_bond_order_do(
    ps_matrix: np.ndarray,
    atomic_basis_indices: dict,
    atom_indices: List[int]
) -> float:
    """
    Actual working horse for multi-center index calculation (tensor contraction).
    Translated from calcmultibndord_do in Multiwfn.f90.
    This uses the iterative matrix multiplication approach for efficiency.

    Args:
        ps_matrix: The PS matrix (P_uv * S_uv for Mulliken, or P @ S for Mayer/Wiberg).
        atomic_basis_indices: A dictionary mapping atom index (0-based) to a list of its basis function indices.
        atom_indices: A list of 0-based atomic indices in the order of connectivity.

    Returns:
        The calculated multi-center bond order value.
    """
    nbndcen = len(atom_indices)
    matdim = ps_matrix.shape[0] # nbasis or numNAO

    result = 0.0

    if nbndcen == 2:
        # Special case: Two atoms (Wiberg bond order)
        atom1_idx = atom_indices[0]
        atom2_idx = atom_indices[1]
        
        basis_fns_1 = atomic_basis_indices[atom1_idx]
        basis_fns_2 = atomic_basis_indices[atom2_idx]

        # In Fortran: sum_{ib} sum_{ia} PSmat(ia,ib) * PSmat(ib,ia)
        # In NumPy: sum of element-wise product of two submatrices
        sub_matrix_12 = ps_matrix[np.ix_(basis_fns_1, basis_fns_2)]
        sub_matrix_21 = ps_matrix[np.ix_(basis_fns_2, basis_fns_1)]
        result = np.sum(sub_matrix_12 * sub_matrix_21)
        return float(result)

    # General case for nbndcen > 2 using iterative tensor contraction
    # Corresponds to the 'mat1', 'mat2' approach in Fortran code

    # Initialize mat_a and mat_b (alternating matrices)
    mat_a = np.zeros((matdim, matdim))
    mat_b = np.zeros((matdim, matdim))
    
    # First contraction
    # iatm is the "rightmost" atom in the current contraction step
    # kbeg/kend are for the "leftmost" atom (first atom in the cycle)
    
    # Fortran:
    # iatm=nbndcen-1
    # katm=1
    # kbeg=basstart(cenind(katm))
    # kend=basend(cenind(katm))
    # icontract=1
    # ibeg=basstart(cenind(iatm))
    # iend=basend(cenind(iatm))
    # jatm=iatm+1
    # jbeg=basstart(cenind(jatm))
    # jend=basend(cenind(jatm))
    # mat2=PSmat (initialize)
    # mat1(ibas,kbas)=sum(PSmat(ibas,jbeg:jend)*mat2(jbeg:jend,kbas))

    # Python translation:
    # cenind in Fortran is atom_indices here (0-based)
    # Fortran atom indices are 1-based, Python are 0-based.
    # Fortran (iatm, jatm, katm) correspond to indices in atom_indices list.

    current_mat = ps_matrix # This will alternate between mat_a and mat_b conceptually
    previous_mat = ps_matrix # Used for the first iteration

    # Iterate nbndcen - 2 times (from 1 to nbndcen-2 in Fortran's icontract)
    for icontract_step in range(nbndcen - 1): # Python loop from 0 to nbndcen-2
        # `iatm` in Fortran means the (nbndcen - 1 - icontract_step)th atom in the cycle
        # `jatm` in Fortran means the (nbndcen - icontract_step)th atom in the cycle
        # `katm` in Fortran means the 0th atom in the cycle
        
        # Corresponds to `cenind(iatm)` (Fortran 1-based index)
        current_atom_idx_in_list = nbndcen - 1 - icontract_step 
        # Corresponds to `cenind(jatm)` (Fortran 1-based index)
        next_atom_idx_in_list = nbndcen - icontract_step
        # Corresponds to `cenind(katm)` (Fortran 1-based index)
        first_atom_idx_in_list = 0

        # Get actual atom numbers (0-based)
        current_atom_num = atom_indices[current_atom_idx_in_list]
        next_atom_num = atom_indices[next_atom_idx_in_list]
        first_atom_num = atom_indices[first_atom_idx_in_list]

        # Get basis function indices
        current_atom_bfs = atomic_basis_indices[current_atom_num]
        next_atom_bfs = atomic_basis_indices[next_atom_num]
        first_atom_bfs = atomic_basis_indices[first_atom_num]

        # Initialize mat_a or mat_b depending on the iteration
        target_mat = mat_a if (icontract_step % 2 == 0) else mat_b
        source_mat = previous_mat

        # Perform the contraction (generalized matrix multiplication with selected blocks)
        # Fortran: mat1(ibas,kbas)=sum(PSmat(ibas,jbeg:jend)*mat2(jbeg:jend,kbas))
        # This is essentially: target_mat[current_atom_bfs, first_atom_bfs] =
        # sum_{j_bfs in next_atom_bfs} PSmat[current_atom_bfs, j_bfs] * source_mat[j_bfs, first_atom_bfs]
        
        # Using np.einsum for generalized matrix multiplication over blocks:
        # result_ik = sum_j (A_ij * B_jk)
        # Here, A is ps_matrix (from current_atom to next_atom)
        # B is source_mat (from next_atom to first_atom)
        
        # We need to compute:
        # mat_new[i, k] = sum_j (ps_matrix[i, j] * previous_mat[j, k])
        # where i in current_atom_bfs, j in next_atom_bfs, k in first_atom_bfs

        # Creating temporary views for clearer einsum operation
        ps_block = ps_matrix[np.ix_(current_atom_bfs, next_atom_bfs)]
        source_block = source_mat[np.ix_(next_atom_bfs, first_atom_bfs)]

        # Perform the dot product for the relevant blocks
        contracted_block = ps_block @ source_block
        
        # Store the result in the corresponding block of the target matrix
        target_mat[np.ix_(current_atom_bfs, first_atom_bfs)] = contracted_block

        previous_mat = target_mat # Update for the next iteration

    # Final summation (Fortran: last loop for `result`)
    # Fortran:
    # do ibas=basstart(cenind(1)),basend(cenind(1))
    #     jbeg=basstart(cenind(2))
    #     jend=basend(cenind(2))
    #     if (icalc==1) then # means current_mat is mat2 (python target_mat from icontract_step % 2 == 0)
    #         result=result+sum(PSmat(ibas,jbeg:jend)*mat2(jbeg:jend,ibas))
    #     else if (icalc==2) then # means current_mat is mat1 (python target_mat from icontract_step % 2 == 1)
    #         result=result+sum(PSmat(ibas,jbeg:jend)*mat1(jbeg:jend,ibas))

    # Python translation:
    final_mat = previous_mat # The last target_mat is the final result of contractions

    # The last contraction involves `ps_matrix` (from first_atom to second_atom)
    # and `final_mat` (from second_atom to first_atom, result of previous contractions)
    
    first_atom_num = atom_indices[0]
    second_atom_num = atom_indices[1]
    
    first_atom_bfs = atomic_basis_indices[first_atom_num]
    second_atom_bfs = atomic_basis_indices[second_atom_num]

    # Calculate sum_{ia in atom_1} sum_{ib in atom_2} PSmat(ia,ib) * final_mat(ib,ia)
    ps_block_final = ps_matrix[np.ix_(first_atom_bfs, second_atom_bfs)]
    final_mat_block = final_mat[np.ix_(second_atom_bfs, first_atom_bfs)]
    
    result = np.sum(ps_block_final * final_mat_block)

    return float(result)


def calculate_multicenter_bond_order(
    wavefunction: Wavefunction,
    atom_indices: List[int],
    mcbo_type: int = 0, # 0: usual, 1: averaged (forward/reverse), 2: all permutations
    is_nao_basis: bool = False,
    density_matrix_type: str = "total" # "total", "alpha", "beta", "mixed"
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Calculates the multi-center bond order (MCBO) for a given set of atoms.
    Corresponds to 'multicenter' and 'multicenterNAO' in Multiwfn.

    Args:
        wavefunction: A Wavefunction object.
        atom_indices: A list of 0-based atomic indices for the multicenter bond.
                      The order is important for mcbo_type=0.
        mcbo_type:
            0: Standard MCBO (order-dependent).
            1: Averaged MCBO (average of forward and reverse order).
            2: Permutation-averaged MCBO (average over all possible permutations).
        is_nao_basis: If True, uses NAO density matrix. Otherwise, uses MO density matrix.
        density_matrix_type: Specifies which density matrix to use for MCBO calculation.
                             "total" (Ptot), "alpha" (Palpha), "beta" (Pbeta), "mixed" (Ptot for open-shell, Palpha+Pbeta).

    Returns:
        A tuple:
        - The calculated total MCBO value.
        - The alpha MCBO value (if unrestricted and applicable), else None.
        - The beta MCBO value (if unrestricted and applicable), else None.
    """
    if is_nao_basis:
        # For NAO basis, we expect DMNAO, DMNAOa, DMNAOb to be available
        # Need to ensure Wavefunction parser has loaded these.
        # This part requires a more robust NAO handling in Wavefunction class first.
        # For now, let's assume Ptot, Palpha, Pbeta are already NAO-based if is_nao_basis is True.
        # This is a simplification and needs to be properly addressed when NAO parsing is fully implemented.
        pass # Placeholder

    num_atoms_in_bond = len(atom_indices)
    
    atomic_basis_indices = wavefunction.get_atomic_basis_indices()

    # Prepare PS_matrix based on density_matrix_type and whether it's NAO basis
    ps_matrix_total = None
    ps_matrix_alpha = None
    ps_matrix_beta = None

    if is_nao_basis:
        # In this simplified version, assume DMNAO, DMNAOa, DMNAOb are directly used as PS_matrix
        # when is_nao_basis is true. This will need adjustment once proper NAO parsing is in place.
        # For current PyMultiWFN, Palpha/Pbeta/Ptot are MO-based.
        # A proper NAO conversion or loading is needed.
        # For now, this branch is incomplete without actual NAO density matrices in Wavefunction.
        raise NotImplementedError("NAO basis calculation is not yet fully implemented.")
    else:
        # Use MO basis and overlap matrix
        if wavefunction.overlap_matrix is None:
            raise ValueError("Overlap matrix (Sbas) is not available in the Wavefunction object.")
        if wavefunction.Ptot is None:
            wavefunction.calculate_density_matrices() # Ensure density matrices are calculated
        if wavefunction.Ptot is None:
            raise ValueError("Total density matrix (Ptot) could not be calculated.")

        ps_matrix_total = wavefunction.Ptot @ wavefunction.overlap_matrix

        if wavefunction.is_unrestricted:
            if wavefunction.Palpha is None or wavefunction.Pbeta is None:
                raise ValueError("Alpha and Beta density matrices are not available for unrestricted calculation.")
            ps_matrix_alpha = wavefunction.Palpha @ wavefunction.overlap_matrix
            ps_matrix_beta = wavefunction.Pbeta @ wavefunction.overlap_matrix
    
    calculated_mcbo_total = 0.0
    calculated_mcbo_alpha = None
    calculated_mcbo_beta = None

    if density_matrix_type == "total":
        # Calculate for total density matrix
        if mcbo_type == 0:
            calculated_mcbo_total = calculate_multicenter_bond_order_do(ps_matrix_total, atomic_basis_indices, atom_indices)
        elif mcbo_type == 1:
            forward_mcbo = calculate_multicenter_bond_order_do(ps_matrix_total, atomic_basis_indices, atom_indices)
            reversed_atom_indices = _reverse_list(atom_indices)
            reverse_mcbo = calculate_multicenter_bond_order_do(ps_matrix_total, atomic_basis_indices, reversed_atom_indices)
            calculated_mcbo_total = (forward_mcbo + reverse_mcbo) / 2.0
        elif mcbo_type == 2:
            all_permutations_mcbo = []
            for perm_indices_tuple in permutations(atom_indices):
                perm_indices = list(perm_indices_tuple)
                all_permutations_mcbo.append(calculate_multicenter_bond_order_do(ps_matrix_total, atomic_basis_indices, perm_indices))
            calculated_mcbo_total = np.sum(all_permutations_mcbo) / (2 * num_atoms_in_bond) # Fortran formula: sum / (2*nbndcen)
        else:
            raise ValueError(f"Unknown mcbo_type: {mcbo_type}")

        if wavefunction.is_unrestricted:
            # If density_matrix_type is "total" but it's unrestricted, Multiwfn also reports alpha/beta components.
            # In Fortran, for `multicenter`, `PSmattot` is used for `bndordmix`, and `PSmata`, `PSmatb` for `bndordalpha`, `bndordbeta`.
            # And `bndordalpha = accum * 2**(nbndcen-1)`.
            # Let's align with the Fortran `multicenter` output for open-shell.

            # Alpha component
            mcbo_alpha_raw = calculate_multicenter_bond_order_do(ps_matrix_alpha, atomic_basis_indices, atom_indices)
            calculated_mcbo_alpha = mcbo_alpha_raw * (2**(num_atoms_in_bond - 1)) # Apply factor

            # Beta component
            mcbo_beta_raw = calculate_multicenter_bond_order_do(ps_matrix_beta, atomic_basis_indices, atom_indices)
            calculated_mcbo_beta = mcbo_beta_raw * (2**(num_atoms_in_bond - 1)) # Apply factor

    elif density_matrix_type == "alpha" and wavefunction.is_unrestricted:
        if mcbo_type == 0:
            calculated_mcbo_alpha = calculate_multicenter_bond_order_do(ps_matrix_alpha, atomic_basis_indices, atom_indices) * (2**(num_atoms_in_bond - 1))
        elif mcbo_type == 1:
            forward_mcbo_alpha = calculate_multicenter_bond_order_do(ps_matrix_alpha, atomic_basis_indices, atom_indices)
            reversed_atom_indices = _reverse_list(atom_indices)
            reverse_mcbo_alpha = calculate_multicenter_bond_order_do(ps_matrix_alpha, atomic_basis_indices, reversed_atom_indices)
            calculated_mcbo_alpha = ((forward_mcbo_alpha + reverse_mcbo_alpha) / 2.0) * (2**(num_atoms_in_bond - 1))
        elif mcbo_type == 2:
            all_permutations_mcbo_alpha = []
            for perm_indices_tuple in permutations(atom_indices):
                perm_indices = list(perm_indices_tuple)
                all_permutations_mcbo_alpha.append(calculate_multicenter_bond_order_do(ps_matrix_alpha, atomic_basis_indices, perm_indices))
            calculated_mcbo_alpha = (np.sum(all_permutations_mcbo_alpha) / (2 * num_atoms_in_bond)) * (2**(num_atoms_in_bond - 1))
    
    elif density_matrix_type == "beta" and wavefunction.is_unrestricted:
        if mcbo_type == 0:
            calculated_mcbo_beta = calculate_multicenter_bond_order_do(ps_matrix_beta, atomic_basis_indices, atom_indices) * (2**(num_atoms_in_bond - 1))
        elif mcbo_type == 1:
            forward_mcbo_beta = calculate_multicenter_bond_order_do(ps_matrix_beta, atomic_basis_indices, atom_indices)
            reversed_atom_indices = _reverse_list(atom_indices)
            reverse_mcbo_beta = calculate_multicenter_bond_order_do(ps_matrix_beta, atomic_basis_indices, reversed_atom_indices)
            calculated_mcbo_beta = ((forward_mcbo_beta + reverse_mcbo_beta) / 2.0) * (2**(num_atoms_in_bond - 1))
        elif mcbo_type == 2:
            all_permutations_mcbo_beta = []
            for perm_indices_tuple in permutations(atom_indices):
                perm_indices = list(perm_indices_tuple)
                all_permutations_mcbo_beta.append(calculate_multicenter_bond_order_do(ps_matrix_beta, atomic_basis_indices, perm_indices))
            calculated_mcbo_beta = (np.sum(all_permutations_mcbo_beta) / (2 * num_atoms_in_bond)) * (2**(num_atoms_in_bond - 1))

    return calculated_mcbo_total, calculated_mcbo_alpha, calculated_mcbo_beta

# Integrate into __init__.py when ready.
# For now, this is just the multicenter.py content
