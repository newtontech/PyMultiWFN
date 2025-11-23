
import numpy as np
from typing import List, Dict, Tuple, Optional
from pymultiwfn.core.data import Wavefunction

# UFF Parameters
# Well depth D (kcal/mol)
UFF_A = np.array([
    0.0, # Dummy for 0 index
    0.044, 0.056, # 1-2 (H, He)
    0.025, 0.085, 0.18, 0.105, 0.069, 0.06, 0.05, 0.042, # 3-10 (Li-Ne)
    0.03, 0.111, 0.505, 0.402, 0.305, 0.274, 0.227, 0.185, # 11-18 (Na-Ar)
    0.035, 0.238, 0.019, 0.017, 0.016, 0.015, 0.013, 0.013, 0.014, 0.015, 0.005, 0.124, # 19-30 (K-Zn)
    0.415, 0.379, 0.309, 0.291, 0.251, 0.22, # 31-36 (Ga-Kr)
    0.04, 0.235, 0.072, 0.069, 0.059, 0.056, 0.048, 0.056, 0.053, 0.048, 0.036, 0.228, # 37-48 (Rb-Cd)
    0.599, 0.567, 0.449, 0.398, 0.339, 0.332, # 49-54 (In-Xe)
    0.045, 0.364, 0.017, 0.013, 0.010, 0.010, 0.009, 0.008, 0.008, 0.009, 0.007, 0.007, 0.007, 0.007, 0.006, 0.228, 0.041, # 55-71 (Cs-Lu)
    0.072, 0.081, 0.067, 0.066, 0.037, 0.073, 0.080, 0.039, 0.385, 0.680, 0.663, 0.518, 0.325, 0.284, 0.248, # 72-86 (Hf-Rn)
    0.050, 0.404, 0.033, 0.026, 0.022, 0.022, 0.019, 0.016, 0.014, 0.013, 0.013, 0.013, 0.012, 0.012, 0.011, 0.011, 0.011 # 87-103 (Fr-Lr)
])

# vdW distance x (Angstrom)
UFF_B = np.array([
    0.0, # Dummy
    2.886, 2.362, # 1-2
    2.451, 2.745, 4.083, 3.851, 3.660, 3.500, 3.364, 3.243, # 3-10
    2.983, 3.021, 4.499, 4.295, 4.147, 4.035, 3.947, 3.868, # 11-18
    3.812, 3.399, 3.295, 3.175, 3.144, 3.023, 2.961, 2.912, 2.872, 2.834, 3.495, 2.763, # 19-30
    4.383, 4.280, 4.230, 4.205, 4.189, 4.141, # 31-36
    4.114, 3.641, 3.345, 3.124, 3.165, 3.052, 2.998, 2.963, 2.929, 2.899, 3.148, 2.848, # 37-48
    4.463, 4.392, 4.420, 4.470, 4.50, 4.404, # 49-54
    4.517, 3.703, 3.522, 3.556, 3.606, 3.575, 3.547, 3.520, 3.493, 3.368, 3.451, 3.428, 3.409, 3.391, 3.374, 3.355, 3.640, # 55-71
    3.141, 3.170, 3.069, 2.954, 3.120, 2.840, 2.754, 3.293, 2.705, 4.347, 4.297, 4.370, 4.709, 4.750, 4.765, # 72-86
    4.90, 3.677, 3.478, 3.396, 3.424, 3.395, 3.424, 3.424, 3.381, 3.326, 3.339, 3.313, 3.299, 3.286, 3.274, 3.248, 3.236 # 87-103
])

# AMBER99 Parameters (Subset for illustration, full list is long)
# In a real implementation, these should be loaded from a data file or a complete dictionary.
# For now, I will implement UFF fully as it covers all elements.
# AMBER/GAFF require atom types which are not standard in Wavefunction object.
# I will implement UFF mode primarily.

def calculate_eda_ff(
    wavefunction: Wavefunction,
    fragments: List[List[int]],
    ff_model: str = "UFF",
    ele_model: str = "1/r"
) -> Dict[str, np.ndarray]:
    """
    Performs Energy Decomposition Analysis based on Forcefield (EDA-FF).
    
    Args:
        wavefunction: Wavefunction object containing atoms and charges.
        fragments: List of lists, where each inner list contains 0-based atom indices for a fragment.
        ff_model: "UFF" (currently only UFF is fully supported without atom types).
        ele_model: "1/r" (Coulomb) or "1/r^2".
        
    Returns:
        Dictionary containing interaction matrices:
        - "electrostatic": (n_frag, n_frag) matrix (kJ/mol)
        - "repulsion": (n_frag, n_frag) matrix (kJ/mol)
        - "dispersion": (n_frag, n_frag) matrix (kJ/mol)
        - "total": (n_frag, n_frag) matrix (kJ/mol)
    """
    n_frag = len(fragments)
    n_atoms = wavefunction.num_atoms
    
    ele_mat = np.zeros((n_frag, n_frag))
    rep_mat = np.zeros((n_frag, n_frag))
    disp_mat = np.zeros((n_frag, n_frag))
    
    # Constants
    au2kj = 2625.5  # Hartree to kJ/mol
    cal2j = 4.184
    
    # Assign parameters
    parm_a = np.zeros(n_atoms) # Well depth D
    parm_b = np.zeros(n_atoms) # vdW distance x
    
    for i, atom in enumerate(wavefunction.atoms):
        z = atom.index
        if ff_model == "UFF":
            if z < len(UFF_A):
                parm_a[i] = UFF_A[z]
                parm_b[i] = UFF_B[z]
            else:
                # Default or error
                parm_a[i] = 0.0
                parm_b[i] = 0.0
        else:
            raise NotImplementedError(f"Forcefield model {ff_model} not implemented.")

    # Calculate interactions
    for i_frag in range(n_frag):
        for j_frag in range(i_frag + 1, n_frag):
            atoms_i = fragments[i_frag]
            atoms_j = fragments[j_frag]
            
            ele_sum = 0.0
            rep_sum = 0.0
            disp_sum = 0.0
            
            for idx_i in atoms_i:
                for idx_j in atoms_j:
                    atom_i = wavefunction.atoms[idx_i]
                    atom_j = wavefunction.atoms[idx_j]
                    
                    # Distance in Angstrom
                    # Wavefunction coords are usually Bohr.
                    # Multiwfn code uses `atomdist(iatm,jatm,1)` which returns Angstrom if 3rd arg is 1?
                    # Let's assume Wavefunction stores Bohr and convert.
                    # 1 Bohr = 0.529177 Angstrom
                    dist_vec = atom_i.coord - atom_j.coord
                    dist_bohr = np.linalg.norm(dist_vec)
                    dist_ang = dist_bohr * 0.529177210903
                    
                    if dist_ang < 1e-3:
                        continue # Avoid singularity
                    
                    # Electrostatic
                    # Charges in Wavefunction are in atomic units (e)
                    # Energy in Hartree if 1/r (Bohr).
                    # But Multiwfn converts to kJ/mol.
                    # Formula: q1*q2 / r
                    # If r is Angstrom, E = (q1*q2 / (r/0.529)) * 2625.5 = q1*q2/r * 1389.35
                    # Multiwfn code: eleval=a(iatm)%charge*a(jatm)%charge/atomdist(iatm,jatm,1) * au2kJ
                    # Wait, atomdist(...,1) returns Angstrom?
                    # In `util.f90` (not shown but inferred), `atomdist` usually returns Angstrom or Bohr.
                    # If Multiwfn multiplies by `au2kJ` (2625.5), then the term `q*q/r` must be in Hartree.
                    # So `r` should be in Bohr.
                    # But `atomdist(...,1)` usually implies Angstrom in Multiwfn conventions (0 for Bohr, 1 for Angstrom).
                    # Let's check `calcAAele`:
                    # eleval=a(iatm)%charge*a(jatm)%charge/atomdist(iatm,jatm,1)
                    # eleval=eleval*au2kJ
                    # If `atomdist` is Angstrom, then `q*q/r_ang` is NOT Hartree.
                    # 1 Hartree = e^2 / Bohr.
                    # q*q / r_ang = q*q / (r_bohr * 0.529) = (q*q/r_bohr) * (1/0.529) = E_hartree * 1.889
                    # So if `atomdist` is Angstrom, the result is in "Angstrom-Hartree"?
                    # Actually, standard conversion: E(kJ/mol) = 1389.35 * q1 * q2 / r(Angstrom).
                    # au2kJ is 2625.5.
                    # 2625.5 / 0.529177 = 4961.
                    # Maybe `atomdist` returns Bohr?
                    # Let's look at `calcAAvdW`: `tmpval=(Xij/atomdistA(iatm,jatm,0))**6`. `atomdistA` likely returns Angstrom.
                    # `calcAAele` uses `atomdist`.
                    # I will use the standard physics formula:
                    # E_ele (Hartree) = q1 * q2 / r_bohr
                    # E_ele (kJ/mol) = E_ele (Hartree) * 2625.5
                    
                    e_ele_hartree = 0.0
                    if ele_model == "1/r":
                        e_ele_hartree = (atom_i.charge * atom_j.charge) / dist_bohr
                    elif ele_model == "1/r^2":
                        # Multiwfn supports this non-standard model
                        # In code: q*q / r^2.
                        # If r is Bohr, unit is Hartree/Bohr?
                        # Multiwfn code: `eleval=.../atomdist(...)**2`.
                        # I'll stick to 1/r for standard physics unless specified.
                        e_ele_hartree = (atom_i.charge * atom_j.charge) / (dist_bohr**2)
                    
                    ele_sum += e_ele_hartree * au2kj
                    
                    # vdW (UFF)
                    # D_ij = sqrt(D_i * D_j) (kcal/mol)
                    # X_ij = sqrt(X_i * X_j) (Angstrom)
                    # E_vdw = D_ij * [ (X_ij/r)^12 - 2*(X_ij/r)^6 ]
                    # Note: Multiwfn code:
                    # Dij=dsqrt(parmA(iatm)*parmA(jatm))*cal2J !Well depth in kJ/mol
                    # Xij=dsqrt(parmB(iatm)*parmB(jatm)) !vdW distance
                    # tmpval=(Xij/atomdistA(iatm,jatm,0))**6
                    # repval=Dij*tmpval**2
                    # dispval=-2*Dij*tmpval
                    
                    d_i = parm_a[idx_i] # kcal/mol
                    d_j = parm_a[idx_j]
                    x_i = parm_b[idx_i] # Angstrom
                    x_j = parm_b[idx_j]
                    
                    d_ij = np.sqrt(d_i * d_j) * cal2j # Convert to kJ/mol
                    x_ij = np.sqrt(x_i * x_j)
                    
                    ratio = x_ij / dist_ang
                    term6 = ratio**6
                    term12 = term6**2
                    
                    rep_sum += d_ij * term12
                    disp_sum += -2.0 * d_ij * term6
            
            ele_mat[i_frag, j_frag] = ele_sum
            ele_mat[j_frag, i_frag] = ele_sum
            
            rep_mat[i_frag, j_frag] = rep_sum
            rep_mat[j_frag, i_frag] = rep_sum
            
            disp_mat[i_frag, j_frag] = disp_sum
            disp_mat[j_frag, i_frag] = disp_sum
            
    total_mat = ele_mat + rep_mat + disp_mat
    
    return {
        "electrostatic": ele_mat,
        "repulsion": rep_mat,
        "dispersion": disp_mat,
        "total": total_mat
    }
