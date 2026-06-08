#!/usr/bin/env python3
"""Check C2H4 orbital energies."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/yhm/software/PyMultiWFN')

from pymultiwfn.io import load

# Test with C2H4 wavefunction
wfn_path = Path("tests/test_data/C2H4_HF.wfn")
wfn = load(str(wfn_path))

print("C2H4 Wavefunction info:")
print(f"  Number of orbitals: {len(wfn.energies)}")
print(f"  Number of electrons: {wfn.num_electrons}")

print(f"\nOrbital energies (first 15):")
for i in range(min(15, len(wfn.energies))):
    print(f"  MO {i:2d}: {wfn.energies[i]:8.4f} Ha  occ={wfn.occupations[i]:.2f}")

print(f"\nOrbital energies (around HOMO):")
n_electrons = int(wfn.num_electrons)
homo_guess = n_electrons // 2 - 1  # For restricted calculation

for i in range(max(0, homo_guess-3), min(len(wfn.energies), homo_guess+5)):
    print(f"  MO {i:2d}: {wfn.energies[i]:8.4f} Ha  occ={wfn.occupations[i]:.2f}")
