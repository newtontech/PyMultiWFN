#!/usr/bin/env python3
"""
Script to fix remaining F841 violations (unused variables) in PyMultiWFN.
This script marks unused variables as intentionally unused or removes them.
"""

import re
from pathlib import Path

# Define fixes for each file
fixes = {
    "pymultiwfn/analysis/bonding/multicenter.py": [
        # Already marked, but flake8 still complains
        (85, "_current_mat", "remove"),  # Remove commented placeholder
    ],
    "pymultiwfn/analysis/spectrum/excitations.py": [
        (400, "nbasis", "remove"),
    ],
    "pymultiwfn/analysis/surface/examples.py": [
        (307, "mapped_surface", "remove"),
        (308, "comparison_results", "remove"),
    ],
    "pymultiwfn/analysis/topology/topology.py": [
        (84, "gradients", "remove"),
    ],
    "pymultiwfn/io/parsers/cif.py": [
        (258, "alpha_rad", "remove"),
        (259, "beta_rad", "remove"),
        (260, "gamma_rad", "remove"),
    ],
    "pymultiwfn/io/parsers/dx.py": [
        (63, "points", "remove"),  # Empty list, can remove
    ],
    "pymultiwfn/io/parsers/gjf.py": [
        (64, "charge", "remove"),
    ],
    "pymultiwfn/io/parsers/mol.py": [
        (47, "num_bonds", "remove"),
    ],
    "pymultiwfn/io/parsers/mol2.py": [
        (68, "atom_id", "remove"),
    ],
    "pymultiwfn/io/parsers/molden.py": [
        (140, "index", "remove"),
    ],
    "pymultiwfn/io/parsers/orca.py": [
        (75, "num_atoms", "remove"),
    ],
    "pymultiwfn/math/gradient.py": [
        (159, "radial_s", "remove"),
        (496, "P_lap", "remove"),
    ],
    "pymultiwfn/vis/display.py": [
        (192, "scatter", "remove"),  # Return value not needed
    ],
    "pymultiwfn/vis/molecular.py": [
        (431, "im", "remove"),  # Return value not needed
        (445, "text", "remove"),  # Return value not needed
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

    fixed_count = 0
    for line_num, var_name, action in line_fixes:
        idx = line_num - 1  # Convert to 0-indexed
        if idx >= len(lines):
            print(f"Line {line_num} out of range in {filepath}")
            continue

        line = lines[idx]

        if action == "remove":
            # Check if line contains the variable assignment
            if var_name in line and ('=' in line or 'scatter' in line or 'imshow' in line or 'text' in line):
                # Remove the entire line
                removed_line = lines.pop(idx)
                print(f"✓ Removed line {line_num} in {filepath}")
                fixed_count += 1
            else:
                print(f"⚠ Line {line_num} doesn't match expected pattern in {filepath}")

    with open(path, 'w') as f:
        f.writelines(lines)

    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} violations in {filepath}\n")
    return fixed_count

def main():
    print("Fixing remaining F841 violations...\n")
    total_fixed = 0
    for filepath, line_fixes in fixes.items():
        fixed = fix_file(filepath, line_fixes)
        if fixed:
            total_fixed += fixed

    print(f"\n{'='*60}")
    print(f"Total violations fixed: {total_fixed}")
    print(f"{'='*60}\n")
    print("Run flake8 to verify fixes:")
    print("  flake8 pymultiwfn/ --max-line-length=88 --extend-ignore=E203,W503 --count")

if __name__ == "__main__":
    main()
