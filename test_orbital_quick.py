#!/usr/bin/env python3
"""Quick test of orbital energy analysis functionality."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/yhm/software/PyMultiWFN')

from pymultiwfn.io import load
from pymultiwfn.orbitals import OrbitalsAnalyzer

# Test with H2 wavefunction
wfn_path = Path("consistency_verifier/examples/H2_CCSD.wfn")
if not wfn_path.exists():
    print(f"Error: Test file {wfn_path} not found")
    sys.exit(1)

print(f"Loading wavefunction from {wfn_path}...")
wfn = load(str(wfn_path))

print(f"\nCreating OrbitalsAnalyzer...")
analyzer = OrbitalsAnalyzer(wfn)

print(f"\nOrbital Energy Analysis Results:")
print(f"  Number of orbitals: {len(analyzer.alpha_energies)}")
print(f"  HOMO index: {analyzer.homo_index}")
print(f"  HOMO energy: {analyzer.homo_energy:.6f} Hartree")
print(f"  LUMO index: {analyzer.lumo_index}")
print(f"  LUMO energy: {analyzer.lumo_energy:.6f} Hartree")
print(f"  HOMO-LUMO gap: {analyzer.gap:.6f} Hartree")
print(f"  Fermi level: {analyzer.fermi_level:.6f} Hartree")

# Test energy diagram
diagram = analyzer.get_energy_diagram(n_orbitals=5)
print(f"\nEnergy Diagram (5 orbitals around Fermi level):")
for i, (energy, occ) in enumerate(zip(diagram['energies'], diagram['occupations'])):
    idx = diagram['indices'][i]
    label = "HOMO" if idx == analyzer.homo_index else "LUMO" if idx == analyzer.lumo_index else ""
    print(f"  MO {idx:2d}: {energy:8.4f} Ha  occ={occ:.2f} {label}")

print(f"\n✅ All basic tests passed!")
