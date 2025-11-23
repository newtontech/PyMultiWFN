
import os
import sys
import numpy as np
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.analysis.bonding import (
    calculate_mayer_bond_order,
    calculate_mulliken_bond_order,
    calculate_multicenter_bond_order,
    calculate_homa,
    calculate_eda_ff
)

def main():
    print("========================================")
    print("      PyMultiWFN MVP Demonstration      ")
    print("========================================")

    # Path to test file
    test_file = os.path.join("consistency_verifier", "examples", "H2O.fch")
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        return

    print(f"Loading wavefunction from: {test_file}")
    try:
        wfn = load_wavefunction(test_file)
        print("Wavefunction loaded successfully.")
        print(f"System: {wfn.num_atoms} atoms, {wfn.num_electrons} electrons, {wfn.num_basis} basis functions.")
    except Exception as e:
        print(f"Failed to load wavefunction: {e}")
        return

    # Ensure density matrices are calculated
    if wfn.Ptot is None:
        print("Calculating density matrices...")
        wfn.calculate_density_matrices()

    # 1. Mayer Bond Order
    print("\n--- Mayer Bond Order Analysis ---")
    try:
        mayer_tot, _, _ = calculate_mayer_bond_order(wfn)
        print("Mayer Bond Order Matrix (Total):")
        print(np.array_str(mayer_tot, precision=4, suppress_small=True))
    except Exception as e:
        print(f"Error in Mayer analysis: {e}")

    # 2. Mulliken Bond Order
    print("\n--- Mulliken Bond Order Analysis ---")
    try:
        mulliken_tot, _, _ = calculate_mulliken_bond_order(wfn)
        print("Mulliken Bond Order Matrix (Total):")
        print(np.array_str(mulliken_tot, precision=4, suppress_small=True))
    except Exception as e:
        print(f"Error in Mulliken analysis: {e}")

    # 3. Multicenter Bond Order (3-center for H-O-H)
    print("\n--- Multicenter Bond Order Analysis (H-O-H) ---")
    try:
        # Atoms are likely O, H, H (indices 0, 1, 2)
        # Let's check atom elements
        atom_elements = [a.element for a in wfn.atoms]
        print(f"Atoms: {atom_elements}")
        
        # Assuming O is index 0, H are 1 and 2
        indices = [0, 1, 2]
        mcbo, _, _ = calculate_multicenter_bond_order(wfn, indices)
        print(f"3-Center Bond Order for atoms {indices}: {mcbo:.6f}")
    except Exception as e:
        print(f"Error in Multicenter analysis: {e}")

    # 4. HOMA (Not applicable to H2O but testing execution)
    print("\n--- HOMA Analysis (Testing execution) ---")
    try:
        # HOMA requires a ring. H2O is not a ring. 
        # But we can pass indices to see if it runs (it might raise error due to missing params for O-H bonds if not in DB)
        # HOMA params usually for C-C, C-N etc.
        # Let's try to catch the expected error or see result if params exist (unlikely for O-H)
        indices = [0, 1, 2]
        homa_val = calculate_homa(wfn, indices)
        print(f"HOMA Index: {homa_val:.6f}")
    except Exception as e:
        print(f"HOMA Analysis skipped/failed (Expected for H2O): {e}")

    # 5. EDA-FF
    print("\n--- EDA-FF Analysis ---")
    try:
        # Define fragments: O (0) and H,H (1, 2)
        fragments = [[0], [1, 2]]
        print(f"Fragments: {fragments}")
        eda_results = calculate_eda_ff(wfn, fragments, ff_model="UFF")
        
        print("Electrostatic Interaction Matrix (kJ/mol):")
        print(np.array_str(eda_results["electrostatic"], precision=2))
        
        print("Repulsion Interaction Matrix (kJ/mol):")
        print(np.array_str(eda_results["repulsion"], precision=2))
        
        print("Dispersion Interaction Matrix (kJ/mol):")
        print(np.array_str(eda_results["dispersion"], precision=2))
        
        print("Total Interaction Matrix (kJ/mol):")
        print(np.array_str(eda_results["total"], precision=2))
        
    except Exception as e:
        print(f"Error in EDA-FF analysis: {e}")

    print("\n========================================")
    print("      MVP Demonstration Completed       ")
    print("========================================")

if __name__ == "__main__":
    main()
