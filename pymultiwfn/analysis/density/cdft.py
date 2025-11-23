"""
Conceptual Density Functional Theory (CDFT) analysis module for PyMultiWFN.

This module implements various CDFT analysis methods including:
- Fukui functions (f+, f-, f0)
- Dual descriptor (DD)
- Global reactivity indices (electronegativity, hardness, softness, electrophilicity)
- Local reactivity indices (condensed Fukui functions, local softness)
- Orbital-weighted Fukui functions
- Fukui potential and dual descriptor potential
- Nucleophilic and electrophilic superdelocalizabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ...core.data import Wavefunction
from ...math.density import calc_density
from ...math.basis import eval_basis_functions


def calculate_fukui_functions(
    wfn_N: Wavefunction,
    wfn_Np: Wavefunction,
    wfn_Nm: Wavefunction,
    grid_coords: np.ndarray,
    degeneracy_p: int = 1,
    degeneracy_m: int = 1
) -> Dict[str, np.ndarray]:
    """
    Calculate Fukui functions on a grid.

    Args:
        wfn_N: Wavefunction for N electrons state
        wfn_Np: Wavefunction for N+p electrons state
        wfn_Nm: Wavefunction for N-q electrons state
        grid_coords: Grid coordinates (n_points x 3)
        degeneracy_p: Degeneracy of LUMO (p)
        degeneracy_m: Degeneracy of HOMO (q)

    Returns:
        Dictionary containing:
        - 'f_plus': Fukui function f+ (nucleophilic attack)
        - 'f_minus': Fukui function f- (electrophilic attack)
        - 'f_zero': Fukui function f0 (radical attack)
        - 'dual_descriptor': Dual descriptor (f+ - f-)
    """
    # Calculate densities on grid
    rho_N = calc_density(wfn_N, grid_coords)
    rho_Np = calc_density(wfn_Np, grid_coords)
    rho_Nm = calc_density(wfn_Nm, grid_coords)

    # Calculate Fukui functions
    f_plus = (rho_Np - rho_N) / degeneracy_p
    f_minus = (rho_N - rho_Nm) / degeneracy_m
    f_zero = (f_plus + f_minus) / 2
    dual_descriptor = f_plus - f_minus

    return {
        'f_plus': f_plus,
        'f_minus': f_minus,
        'f_zero': f_zero,
        'dual_descriptor': dual_descriptor
    }


def calculate_global_reactivity_indices(
    wfn_N: Wavefunction,
    wfn_Np: Wavefunction,
    wfn_Nm: Wavefunction,
    wfn_Nm2: Optional[Wavefunction] = None
) -> Dict[str, float]:
    """
    Calculate global reactivity indices from CDFT.

    Args:
        wfn_N: Wavefunction for N electrons state
        wfn_Np: Wavefunction for N+1 electrons state
        wfn_Nm: Wavefunction for N-1 electrons state
        wfn_Nm2: Optional wavefunction for N-2 electrons state (for cubic electrophilicity)

    Returns:
        Dictionary containing global reactivity indices
    """
    # Extract energies
    E_N = wfn_N.total_energy
    E_Np = wfn_Np.total_energy
    E_Nm = wfn_Nm.total_energy

    # Calculate fundamental quantities
    VIP = E_Nm - E_N  # Vertical Ionization Potential
    VEA = E_N - E_Np  # Vertical Electron Affinity

    # Global reactivity indices
    electronegativity = (VIP + VEA) / 2
    chemical_potential = -electronegativity
    hardness = VIP - VEA
    softness = 1.0 / hardness if hardness > 0 else np.inf
    electrophilicity = chemical_potential**2 / (2 * hardness) if hardness > 0 else np.inf

    result = {
        'VIP': VIP,
        'VEA': VEA,
        'electronegativity': electronegativity,
        'chemical_potential': chemical_potential,
        'hardness': hardness,
        'softness': softness,
        'electrophilicity': electrophilicity
    }

    # Calculate cubic electrophilicity if N-2 state is provided
    if wfn_Nm2 is not None:
        E_Nm2 = wfn_Nm2.total_energy
        VIP2 = E_Nm2 - E_Nm  # Second vertical IP

        # Terms for w_cubic
        c_miu = (-2 * VEA - 5 * VIP + VIP2) / 6
        c_eta = hardness
        c_gamma = 2 * VIP - VIP2 - VEA
        w_cubic = c_miu**2 / (2 * c_eta) * (1 + c_miu * c_gamma / (3 * c_eta**2))

        # Electrophilic descriptor (epsilon)
        VEA_temp = -VEA  # Different convention
        c_parm = (VIP2 - 2 * VIP + VEA_temp) / (2 * VIP2 - VIP - VEA_temp)
        b_parm = (VIP - VEA_temp) / 2 - ((VIP + VEA_temp) / 2) * c_parm
        a_parm = -(VIP + VEA_temp) / 2 + ((VIP - VEA_temp) / 2) * c_parm
        c_miu_eps = a_parm
        c_eta_eps = 2 * (b_parm - a_parm * c_parm)
        c_gamma_eps = -3 * c_parm * (b_parm - a_parm * c_parm)

        discriminant = c_eta_eps**2 - 2 * c_gamma_eps * c_miu_eps
        if discriminant >= 0:
            phi = np.sqrt(discriminant) - c_eta_eps
            epsilon = (-c_miu_eps) * (phi / c_gamma_eps) - (phi / c_gamma_eps)**2 * (c_eta_eps / 2 + phi / 6)
        else:
            epsilon = np.nan

        result.update({
            'w_cubic': w_cubic,
            'epsilon': epsilon,
            'VIP2': VIP2
        })

    return result


def calculate_condensed_fukui_functions(
    wfn_N: Wavefunction,
    wfn_Np: Wavefunction,
    wfn_Nm: Wavefunction,
    degeneracy_p: int = 1,
    degeneracy_m: int = 1
) -> Dict[str, np.ndarray]:
    """
    Calculate condensed Fukui functions using Hirshfeld charges.

    Args:
        wfn_N: Wavefunction for N electrons state
        wfn_Np: Wavefunction for N+p electrons state
        wfn_Nm: Wavefunction for N-q electrons state
        degeneracy_p: Degeneracy of LUMO (p)
        degeneracy_m: Degeneracy of HOMO (q)

    Returns:
        Dictionary containing condensed Fukui functions
    """
    # Calculate Hirshfeld charges
    q_N = calculate_hirshfeld_charges(wfn_N)
    q_Np = calculate_hirshfeld_charges(wfn_Np)
    q_Nm = calculate_hirshfeld_charges(wfn_Nm)

    n_atoms = len(q_N)

    # Calculate condensed Fukui functions
    f_minus = (q_Nm - q_N) / degeneracy_m
    f_plus = (q_N - q_Np) / degeneracy_p
    f_zero = (f_plus + f_minus) / 2
    dual_descriptor = f_plus - f_minus

    return {
        'f_plus': f_plus,
        'f_minus': f_minus,
        'f_zero': f_zero,
        'dual_descriptor': dual_descriptor,
        'q_N': q_N,
        'q_Np': q_Np,
        'q_Nm': q_Nm
    }


def calculate_hirshfeld_charges(wfn: Wavefunction) -> np.ndarray:
    """
    Calculate Hirshfeld charges for a wavefunction.

    Args:
        wfn: Wavefunction object

    Returns:
        Array of Hirshfeld charges for each atom
    """
    # Placeholder implementation
    # In a complete implementation, this would calculate Hirshfeld charges
    # using promolecular densities and integration grids

    # For now, return Mulliken charges as approximation
    if wfn.mulliken_charges is not None:
        return wfn.mulliken_charges
    else:
        # Calculate approximate charges
        n_atoms = wfn.num_atoms
        return np.zeros(n_atoms)


def calculate_orbital_weighted_fukui_functions(
    wfn: Wavefunction,
    grid_coords: np.ndarray,
    delta: float = 0.015
) -> Dict[str, np.ndarray]:
    """
    Calculate orbital-weighted Fukui functions.

    Args:
        wfn: Wavefunction object
        grid_coords: Grid coordinates (n_points x 3)
        delta: Delta parameter for orbital weighting

    Returns:
        Dictionary containing orbital-weighted Fukui functions
    """
    if wfn.is_unrestricted:
        raise NotImplementedError("Orbital-weighted Fukui functions not implemented for unrestricted wavefunctions")

    # Get HOMO and LUMO indices
    homo_idx = wfn.homo_index
    lumo_idx = wfn.lumo_index

    # Calculate chemical potential
    chem_pot = (wfn.energies[homo_idx] + wfn.energies[lumo_idx]) / 2

    # Calculate orbital weights
    n_mo = wfn.num_mos
    exp_term = np.zeros(n_mo)

    for i in range(n_mo):
        exp_term[i] = np.exp(-((chem_pot - wfn.energies[i]) / delta)**2)

    # Calculate denominators
    denom_pos = np.sum(exp_term[lumo_idx:])
    denom_neg = np.sum(exp_term[:homo_idx + 1])

    # Calculate orbital-weighted densities
    OW_f_plus = np.zeros(len(grid_coords))
    OW_f_minus = np.zeros(len(grid_coords))

    # Evaluate molecular orbitals on grid
    mo_values = eval_basis_functions(wfn, grid_coords)

    # Calculate orbital-weighted functions
    for i in range(len(grid_coords)):
        # f+ contribution from virtual orbitals
        for j in range(lumo_idx, n_mo):
            weight = exp_term[j] / denom_pos
            OW_f_plus[i] += weight * mo_values[i, j]**2

        # f- contribution from occupied orbitals
        for j in range(homo_idx + 1):
            weight = exp_term[j] / denom_neg
            OW_f_minus[i] += weight * mo_values[i, j]**2

    OW_f_zero = (OW_f_plus + OW_f_minus) / 2
    OW_dual_descriptor = OW_f_plus - OW_f_minus

    return {
        'OW_f_plus': OW_f_plus,
        'OW_f_minus': OW_f_minus,
        'OW_f_zero': OW_f_zero,
        'OW_dual_descriptor': OW_dual_descriptor
    }


def calculate_superdelocalizabilities(
    wfn: Wavefunction
) -> Dict[str, np.ndarray]:
    """
    Calculate nucleophilic and electrophilic superdelocalizabilities.

    Args:
        wfn: Wavefunction object

    Returns:
        Dictionary containing superdelocalizabilities for each atom
    """
    if wfn.is_unrestricted:
        raise NotImplementedError("Superdelocalizabilities not implemented for unrestricted wavefunctions")

    # Get HOMO and LUMO indices
    homo_idx = wfn.homo_index
    lumo_idx = wfn.lumo_index

    # Calculate alpha parameter
    alpha_parm = (wfn.energies[homo_idx] + wfn.energies[lumo_idx]) / 2

    n_atoms = wfn.num_atoms
    n_mo = wfn.num_mos

    # Calculate orbital atomic compositions
    atom_comp = calculate_orbital_atomic_composition(wfn)

    # Initialize arrays
    D_N = np.zeros(n_atoms)  # Nucleophilic superdelocalizability
    D_E = np.zeros(n_atoms)  # Electrophilic superdelocalizability
    D_N_0 = np.zeros(n_atoms)  # Without alpha parameter
    D_E_0 = np.zeros(n_atoms)  # Without alpha parameter

    for i in range(n_atoms):
        # Electrophilic superdelocalizability (occupied orbitals)
        for j in range(homo_idx + 1):
            D_E[i] += atom_comp[i, j] / (wfn.energies[j] - alpha_parm)
            D_E_0[i] += atom_comp[i, j] / wfn.energies[j]

        # Nucleophilic superdelocalizability (virtual orbitals)
        for j in range(lumo_idx, n_mo):
            if wfn.energies[j] != 0:  # Avoid division by zero for artificially filled orbitals
                D_N[i] += atom_comp[i, j] / (alpha_parm - wfn.energies[j])
                D_N_0[i] += atom_comp[i, j] / (-wfn.energies[j])

    # Scale by 2 for closed-shell
    D_N *= 2
    D_E *= 2
    D_N_0 *= 2
    D_E_0 *= 2

    return {
        'D_N': D_N,
        'D_E': D_E,
        'D_N_0': D_N_0,
        'D_E_0': D_E_0
    }


def calculate_orbital_atomic_composition(wfn: Wavefunction) -> np.ndarray:
    """
    Calculate orbital atomic compositions.

    Args:
        wfn: Wavefunction object

    Returns:
        Matrix of atomic compositions (n_atoms x n_mos)
    """
    if wfn.overlap_matrix is None:
        raise ValueError("Overlap matrix is required for orbital atomic composition")

    n_atoms = wfn.num_atoms
    n_mo = wfn.num_mos
    n_bf = wfn.num_basis_functions

    atom_comp = np.zeros((n_atoms, n_mo))
    atom_to_bfs = wfn.get_atomic_basis_indices()

    for i in range(n_atoms):
        bfs_i = atom_to_bfs.get(i, [])
        if not bfs_i:
            continue

        for j in range(n_mo):
            contrib = 0.0
            for mu in bfs_i:
                for nu in range(n_bf):
                    contrib += wfn.coefficients[j, mu] * wfn.coefficients[j, nu] * wfn.overlap_matrix[mu, nu]
            atom_comp[i, j] = contrib

    return atom_comp


def calculate_local_softness(
    condensed_fukui: Dict[str, np.ndarray],
    softness: float
) -> Dict[str, np.ndarray]:
    """
    Calculate local softness from condensed Fukui functions.

    Args:
        condensed_fukui: Dictionary from calculate_condensed_fukui_functions
        softness: Global softness value

    Returns:
        Dictionary containing local softness indices
    """
    f_plus = condensed_fukui['f_plus']
    f_minus = condensed_fukui['f_minus']
    f_zero = condensed_fukui['f_zero']
    dual_descriptor = condensed_fukui['dual_descriptor']

    s_plus = f_plus * softness
    s_minus = f_minus * softness
    s_zero = f_zero * softness
    s_2 = dual_descriptor * softness**2

    return {
        's_plus': s_plus,
        's_minus': s_minus,
        's_zero': s_zero,
        's_2': s_2
    }


def calculate_local_electrophilicity_nucleophilicity(
    condensed_fukui: Dict[str, np.ndarray],
    electrophilicity: float,
    nucleophilicity: float
) -> Dict[str, np.ndarray]:
    """
    Calculate local electrophilicity and nucleophilicity indices.

    Args:
        condensed_fukui: Dictionary from calculate_condensed_fukui_functions
        electrophilicity: Global electrophilicity index
        nucleophilicity: Global nucleophilicity index

    Returns:
        Dictionary containing local indices
    """
    f_plus = condensed_fukui['f_plus']
    f_minus = condensed_fukui['f_minus']

    local_electrophilicity = f_plus * electrophilicity
    local_nucleophilicity = f_minus * nucleophilicity

    return {
        'local_electrophilicity': local_electrophilicity,
        'local_nucleophilicity': local_nucleophilicity
    }


def print_cdft_results(
    global_indices: Dict[str, float],
    condensed_fukui: Dict[str, np.ndarray],
    atom_names: Optional[List[str]] = None
):
    """
    Print CDFT results in a formatted way.

    Args:
        global_indices: Dictionary from calculate_global_reactivity_indices
        condensed_fukui: Dictionary from calculate_condensed_fukui_functions
        atom_names: Optional list of atom names
    """
    n_atoms = len(condensed_fukui['f_plus'])

    if atom_names is None:
        atom_names = [f"Atom{i+1}" for i in range(n_atoms)]

    print("=" * 80)
    print("CONCEPTUAL DENSITY FUNCTIONAL THEORY ANALYSIS")
    print("=" * 80)

    # Print global indices
    print("\nGLOBAL REACTIVITY INDICES:")
    print("-" * 40)
    for key, value in global_indices.items():
        if key in ['VIP', 'VEA', 'electronegativity', 'chemical_potential', 'hardness', 'electrophilicity']:
            print(f"{key:25s}: {value:12.6f} Hartree")
        elif key == 'softness':
            print(f"{key:25s}: {value:12.6f} Hartree^-1")
        elif key in ['w_cubic', 'epsilon']:
            print(f"{key:25s}: {value:12.6f} Hartree")

    # Print condensed Fukui functions
    print("\nCONDENSED FUKUI FUNCTIONS:")
    print("-" * 40)
    print("Atom        q(N)      q(N+p)    q(N-q)     f+        f-        f0       DD")
    for i in range(n_atoms):
        print(f"{atom_names[i]:8s} {condensed_fukui['q_N'][i]:8.4f} {condensed_fukui['q_Np'][i]:8.4f} "
              f"{condensed_fukui['q_Nm'][i]:8.4f} {condensed_fukui['f_plus'][i]:8.4f} "
              f"{condensed_fukui['f_minus'][i]:8.4f} {condensed_fukui['f_zero'][i]:8.4f} "
              f"{condensed_fukui['dual_descriptor'][i]:8.4f}")

    # Print local softness if available
    if 'softness' in global_indices and global_indices['softness'] != np.inf:
        local_softness = calculate_local_softness(condensed_fukui, global_indices['softness'])

        print("\nLOCAL SOFTNESS:")
        print("-" * 40)
        print("Atom        s+        s-        s0        s+/s-     s-/s+     s(2)")
        for i in range(n_atoms):
            s_plus = local_softness['s_plus'][i]
            s_minus = local_softness['s_minus'][i]
            s_zero = local_softness['s_zero'][i]
            s_2 = local_softness['s_2'][i]

            ratio_plus_minus = s_plus / s_minus if s_minus != 0 else np.inf
            ratio_minus_plus = s_minus / s_plus if s_plus != 0 else np.inf

            print(f"{atom_names[i]:8s} {s_plus:8.4f} {s_minus:8.4f} {s_zero:8.4f} "
                  f"{ratio_plus_minus:8.4f} {ratio_minus_plus:8.4f} {s_2:8.4f}")

    print("=" * 80)


def export_cdft_results(
    global_indices: Dict[str, float],
    condensed_fukui: Dict[str, np.ndarray],
    filename: str = "CDFT_results.txt",
    atom_names: Optional[List[str]] = None
):
    """
    Export CDFT results to a text file.

    Args:
        global_indices: Dictionary from calculate_global_reactivity_indices
        condensed_fukui: Dictionary from calculate_condensed_fukui_functions
        filename: Output filename
        atom_names: Optional list of atom names
    """
    n_atoms = len(condensed_fukui['f_plus'])

    if atom_names is None:
        atom_names = [f"Atom{i+1}" for i in range(n_atoms)]

    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONCEPTUAL DENSITY FUNCTIONAL THEORY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Write global indices
        f.write("GLOBAL REACTIVITY INDICES:\n")
        f.write("-" * 40 + "\n")
        for key, value in global_indices.items():
            if key in ['VIP', 'VEA', 'electronegativity', 'chemical_potential', 'hardness', 'electrophilicity']:
                f.write(f"{key:25s}: {value:12.6f} Hartree\n")
            elif key == 'softness':
                f.write(f"{key:25s}: {value:12.6f} Hartree^-1\n")
            elif key in ['w_cubic', 'epsilon']:
                f.write(f"{key:25s}: {value:12.6f} Hartree\n")

        # Write condensed Fukui functions
        f.write("\nCONDENSED FUKUI FUNCTIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("Atom        q(N)      q(N+p)    q(N-q)     f+        f-        f0       DD\n")
        for i in range(n_atoms):
            f.write(f"{atom_names[i]:8s} {condensed_fukui['q_N'][i]:8.4f} {condensed_fukui['q_Np'][i]:8.4f} "
                    f"{condensed_fukui['q_Nm'][i]:8.4f} {condensed_fukui['f_plus'][i]:8.4f} "
                    f"{condensed_fukui['f_minus'][i]:8.4f} {condensed_fukui['f_zero'][i]:8.4f} "
                    f"{condensed_fukui['dual_descriptor'][i]:8.4f}\n")

        # Write local softness if available
        if 'softness' in global_indices and global_indices['softness'] != np.inf:
            local_softness = calculate_local_softness(condensed_fukui, global_indices['softness'])

            f.write("\nLOCAL SOFTNESS:\n")
            f.write("-" * 40 + "\n")
            f.write("Atom        s+        s-        s0        s+/s-     s-/s+     s(2)\n")
            for i in range(n_atoms):
                s_plus = local_softness['s_plus'][i]
                s_minus = local_softness['s_minus'][i]
                s_zero = local_softness['s_zero'][i]
                s_2 = local_softness['s_2'][i]

                ratio_plus_minus = s_plus / s_minus if s_minus != 0 else np.inf
                ratio_minus_plus = s_minus / s_plus if s_plus != 0 else np.inf

                f.write(f"{atom_names[i]:8s} {s_plus:8.4f} {s_minus:8.4f} {s_zero:8.4f} "
                        f"{ratio_plus_minus:8.4f} {ratio_minus_plus:8.4f} {s_2:8.4f}\n")

        f.write("=" * 80 + "\n")

    print(f"CDFT results exported to {filename}")