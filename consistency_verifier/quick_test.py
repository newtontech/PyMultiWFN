#!/usr/bin/env python3
"""
Quick consistency test for PyMultiWFN
Run this script to quickly verify that PyMultiWFN matches Multiwfn results
"""

import os
import sys
import subprocess

# Set Multiwfn path
os.environ['MULTIWFN_BIN'] = '/home/yhm/software/PyMultiWFN/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn'

# Test file
TEST_FILE = '/home/yhm/software/PyMultiWFN/consistency_verifier/examples/H2_CCSD.wfn'

print("=" * 60)
print("PyMultiWFN Consistency Quick Test")
print("=" * 60)
print(f"Test file: {TEST_FILE}")
print(f"Multiwfn: {os.environ.get('MULTIWFN_BIN')}")
print("")

# Test 1: Load with PyMultiWFN
print("[1/3] Loading with PyMultiWFN...")
try:
    from pymultiwfn.io.file_manager import FileManager
    fm = FileManager()
    wfn = fm.load_wavefunction(TEST_FILE)
    py_electrons = wfn.num_electrons
    py_charge = wfn.charge
    py_atoms = len(wfn.atoms)
    print(f"  ✓ Electrons: {py_electrons}")
    print(f"  ✓ Charge: {py_charge}")
    print(f"  ✓ Atoms: {py_atoms}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Run with Multiwfn
print("\n[2/3] Running with Multiwfn...")
try:
    result = subprocess.run(
        [os.environ['MULTIWFN_BIN'], TEST_FILE],
        input='18\n1\nq\n',
        capture_output=True,
        text=True,
        timeout=10
    )
    output = result.stdout
    
    # Parse electrons from output
    for line in output.split('\n'):
        if 'Total/Alpha/Beta electrons:' in line:
            parts = line.split(':')
            if len(parts) > 1:
                mw_electrons = float(parts[1].strip())
                print(f"  ✓ Electrons: {mw_electrons}")
                break
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Compare
print("\n[3/3] Comparing results...")
if abs(py_electrons - mw_electrons) < 0.001:
    print(f"  ✓ Electrons match: {py_electrons} == {mw_electrons}")
    print("\n✅ CONSISTENCY CHECK PASSED")
    print("=" * 60)
    sys.exit(0)
else:
    print(f"  ✗ Electrons mismatch: {py_electrons} != {mw_electrons}")
    print(f"    Difference: {abs(py_electrons - mw_electrons)}")
    print("\n❌ CONSISTENCY CHECK FAILED")
    print("=" * 60)
    sys.exit(1)
