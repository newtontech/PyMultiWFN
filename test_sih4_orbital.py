#!/usr/bin/env python3
"""Test orbital energy with SiH4.fch file."""

import sys
from pathlib import Path

sys.path.insert(0, '/home/yhm/software/PyMultiWFN')

from pymultiwfn.io import load
from pymultiwfn.orbitals import OrbitalsAnalyzer

# Test with SiH4 formatted checkpoint file
fch_path = Path("consistency_verifier/examples/SiH4_C6/SiH4.fch")
if not fch_path.exists():
    print(f"Error: Test file {fch_path} not found")
    sys.exit(1)

print(f"Loading wavefunction from {fch_path}...")
wfn = load(str(fch_path))

print(f"\nWavefunction info:")
print(f"  Number of orbitals: {len(wfn.energies)}")
print(f"  Number of electrons: {wfn.num_electrons}")

print(f"\nFirst 10 orbital energies:")
for i in range(min(10, len(wfn.energies))):
    print(f"  MO {i:2d}: {wfn.energies[i]:8.4f} Ha  occ={wfn.occupations[i]:.2f}")

print(f"\nCreating OrbitalsAnalyzer...")
analyzer = OrbitalsAnalyzer(wfn)

print(f"\nOrbital Energy Analysis Results:")
print(f"  HOMO index: {analyzer.homo_index}")
print(f"  HOMO energy: {analyzer.homo_energy:.6f} Hartree")
print(f"  LUMO index: {analyzer.lumo_index}")
print(f"  LUMO energy: {analyzer.lumo_energy:.6f} Hartree")
print(f"  HOMO-LUMO gap: {analyzer.gap:.6f} Hartree ({analyzer.gap * 27.2114:.2f} eV)")
print(f"  Fermi level: {analyzer.fermi_level:.6f} Hartree")

print(f"\n✅ Test passed!")
