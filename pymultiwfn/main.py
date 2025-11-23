import sys
import os
import argparse
import numpy as np
from pymultiwfn.config import Config
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density

# Global variable to hold the current wavefunction
current_wfn = None

def main():
    global current_wfn
    parser = argparse.ArgumentParser(description="PyMultiWFN: A Python refactoring of Multiwfn")
    parser.add_argument("filename", nargs="?", help="Input file path")
    args = parser.parse_args()

    print_splash()

    if args.filename:
        process_file(args.filename)
    else:
        # Interactive mode (simplified)
        filename = input("Input file path: ")
        if filename:
            process_file(filename)
        else:
            print("No file specified. Exiting.")
            return

    if current_wfn:
        main_menu()

def print_splash():
    print(" Multiwfn -- A Multifunctional Wavefunction Analyzer")
    print(" (c) Tian Lu, 2025")
    print(" Refactored in Python by PyMultiWFN Team")
    print("")

def process_file(filepath):
    global current_wfn
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    try:
        print(f"Loading {filepath}...")
        current_wfn = load_wavefunction(filepath)
        print(f"Loaded wavefunction with {current_wfn.num_atoms} atoms.")
        print(f"Title: {current_wfn.title}")
        print(f"Method: {current_wfn.method} / {current_wfn.basis_set_name}")
    except Exception as e:
        print(f"Error loading file: {e}")
        current_wfn = None

def show_detailed_info():
    global current_wfn
    if current_wfn is None:
        print("No wavefunction loaded.")
        return

    wfn = current_wfn
    print("\n--- Detailed Information ---")
    print(f"Title: {wfn.title}")
    print(f"Method: {wfn.method} / {wfn.basis_set_name}")
    print(f"Atoms: {wfn.num_atoms}")
    print(f"Electrons: {wfn.num_electrons}")
    print(f"Basis functions: {wfn.num_basis}")
    print(f"Charge: {wfn.charge}")
    print(f"Multiplicity: {wfn.multiplicity}")
    
    if wfn.occupations is not None:
        print("Occupations: Loaded/Inferred")
    else:
        print("Occupations: Not available")

    # Density check at the first atom
    if wfn.num_atoms > 0:
        atom0 = wfn.atoms[0]
        coords = np.array([atom0.coord]) # Shape (1, 3)
        print(f"\nCalculating density at atom 1 ({atom0.element}) position: {coords[0]}")
        try:
            rho = calc_density(wfn, coords)
            print(f"Electron Density: {rho[0]:.6f}")
            if rho[0] > 1.0:
                print("Result looks reasonable (high density at nucleus).")
            else:
                print("Result looks suspicious (low density at nucleus).")
        except Exception as e:
            print(f"Error calculating density: {e}")

def main_menu():
    while True:
        print("---------------------------------------------------------")
        print(" Main function menu:")
        print(" 0 Show molecular structure and view orbitals (Not Implemented)")
        print(" 1 Show detailed information")
        print(" ...")
        print(" q Exit")
        
        choice = input("Input command: ")
        if choice == 'q':
            break
        elif choice == '1':
            show_detailed_info()
        elif choice == '0':
             print("Function 0 is not yet implemented.")
        else:
            print("Invalid command")

if __name__ == "__main__":
    main()
