#!/usr/bin/env python3
"""
Script to fix F841 violations (unused variables) in PyMultiWFN.
This script removes or marks unused variables as intentionally unused.
"""

import re
from pathlib import Path

# Define fixes for each file
fixes = {
    "pymultiwfn/analysis/bonding/bondorder.py": [
        (533, "fuzzy_pops", "remove"),  # Remove unused variable
    ],
    "pymultiwfn/analysis/bonding/cda.py": [
        (34, "frag1", "remove"),
        (35, "frag2", "remove"),
    ],
    "pymultiwfn/analysis/bonding/mayer.py": [
        (21, "num_basis", "remove"),
    ],
    "pymultiwfn/analysis/bonding/multicenter.py": [
        (85, "current_mat", "mark"),  # Mark as intentionally unused
    ],
    "pymultiwfn/analysis/bonding/orbital_contributions.py": [
        (157, "num_basis", "remove"),
        (158, "atomic_basis_indices", "remove"),
    ],
    "pymultiwfn/analysis/density/cdft.py": [
        (171, "n_atoms", "remove"),
    ],
    "pymultiwfn/analysis/density_grid.py": [
        (59, "phi", "remove"),  # Already computed but not used
    ],
    "pymultiwfn/analysis/population/fuzzy_atoms.py": [
        (273, "n_atoms", "remove"),
        (274, "n_points", "remove"),
        (422, "n_atoms", "remove"),
    ],
    "pymultiwfn/analysis/population/mulliken.py": [
        (39, "num_basis", "remove"),
    ],
    "pymultiwfn/analysis/population/mulliken_fixed.py": [
        (39, "num_basis", "remove"),
    ],
    "pymultiwfn/analysis/spectrum/excitations.py": [
        (146, "energy_au", "remove"),
        (402, "dipole_x", "remove"),
        (403, "dipole_y", "remove"),
        (404, "dipole_z", "remove"),
    ],
    "pymultiwfn/analysis/surface/examples.py": [
        (307, "water_surface", "remove"),
        (308, "hirshfeld_surface", "remove"),
    ],
}

def fix_file(filepath: str, line_fixes: list):
    """Apply fixes to a single file."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    with open(path, 'r') as f:
        lines = f.readlines()

    # Sort fixes by line number in reverse order to avoid offset issues
    line_fixes.sort(key=lambda x: x[0], reverse=True)

    for line_num, var_name, action in line_fixes:
        idx = line_num - 1  # Convert to 0-indexed
        if idx >= len(lines):
            print(f"Line {line_num} out of range in {filepath}")
            continue

        line = lines[idx]

        if action == "remove":
            # Check if line contains the variable assignment
            if var_name in line and '=' in line:
                # Remove the entire line
                lines.pop(idx)
                print(f"Removed line {line_num} in {filepath}: {line.strip()}")
        elif action == "mark":
            # Rename variable to mark as intentionally unused
            if var_name in line:
                lines[idx] = line.replace(var_name, f"_{var_name}")
                print(f"Marked unused variable in {filepath}:{line_num}")

    with open(path, 'w') as f:
        f.writelines(lines)

    print(f"Fixed {filepath}")

def main():
    print("Fixing F841 violations...")
    for filepath, line_fixes in fixes.items():
        fix_file(filepath, line_fixes)
    print("\nDone! Run flake8 to verify fixes.")

if __name__ == "__main__":
    main()
