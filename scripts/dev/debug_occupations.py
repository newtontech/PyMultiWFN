#!/usr/bin/env python3
"""Debug orbital occupations."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/yhm/software/PyMultiWFN')

from pymultiwfn.io import load

# Test with H2 wavefunction
wfn_path = Path("consistency_verifier/examples/H2_CCSD.wfn")
wfn = load(str(wfn_path))

print("Wavefunction info:")
print(f"  Number of orbitals: {len(wfn.energies)}")
print(f"  Number of electrons: {wfn.num_electrons}")
print(f"  Number of atoms: {len(wfn.atoms)}")

print(f"\nOrbital energies (first 10):")
for i in range(min(10, len(wfn.energies))):
    print(f"  MO {i:2d}: {wfn.energies[i]:8.4f} Ha  occ={wfn.occupations[i]:.2f}")

print(f"\nOrbital energies (last 10):")
for i in range(max(0, len(wfn.energies)-10), len(wfn.energies)):
    print(f"  MO {i:2d}: {wfn.energies[i]:8.4f} Ha  occ={wfn.occupations[i]:.2f}")

print(f"\nOccupation statistics:")
occupied = np.sum(wfn.occupations > 0)
print(f"  Number of occupied orbitals: {occupied}")
print(f"  Total occupation: {np.sum(wfn.occupations):.2f}")
print(f"  Expected electrons: {wfn.num_electrons}")
