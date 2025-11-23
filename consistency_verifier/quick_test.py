#!/usr/bin/env python3
"""
Quick test script for specific PyMultiWFN functions.
Useful for rapid testing during development.
"""

import os
import sys
import argparse
from pathlib import Path
from verifier import ConsistencyVerifier


def test_basic_function(multiwfn_exe: str, test_file: str):
    """Test basic information function."""
    print(f"Testing basic information on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["0", "0"]  # Show info, then exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Basic information test passed")
        return True
    else:
        print("❌ FAIL: Basic information test failed")
        if result["error_ref"]:
            print(f"Multiwfn error: {result['error_ref']}")
        if result["error_py"]:
            print(f"PyMultiWFN error: {result['error_py']}")
        return False


def test_density_analysis(multiwfn_exe: str, test_file: str):
    """Test electron density analysis."""
    print(f"Testing electron density analysis on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["1", "0", "0"]  # Density, critical points, exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Density analysis test passed")
        return True
    else:
        print("❌ FAIL: Density analysis test failed")
        return False


def test_orbital_info(multiwfn_exe: str, test_file: str):
    """Test molecular orbital information."""
    print(f"Testing orbital information on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["5", "0"]  # Orbital info, exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Orbital information test passed")
        return True
    else:
        print("❌ FAIL: Orbital information test failed")
        return False


def test_pop_analysis(multiwfn_exe: str, test_file: str):
    """Test population analysis."""
    print(f"Testing population analysis on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["7", "0"]  # Mulliken population, exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Population analysis test passed")
        return True
    else:
        print("❌ FAIL: Population analysis test failed")
        return False


def test_dipole_moment(multiwfn_exe: str, test_file: str):
    """Test dipole moment calculation."""
    print(f"Testing dipole moment on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["11", "0"]  # Dipole moment, exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Dipole moment test passed")
        return True
    else:
        print("❌ FAIL: Dipole moment test failed")
        return False


def test_bandgap(multiwfn_exe: str, test_file: str):
    """Test band gap calculation."""
    print(f"Testing band gap on {os.path.basename(test_file)}")

    verifier = ConsistencyVerifier(multiwfn_exe)
    commands = ["17", "0"]  # Band gap, exit

    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("✅ PASS: Band gap test passed")
        return True
    else:
        print("❌ FAIL: Band gap test failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick PyMultiWFN function tester")
    parser.add_argument("--multiwfn", help="Path to Multiwfn executable")
    parser.add_argument("--file", help="Test file path")
    parser.add_argument("--function", choices=[
        "basic", "density", "orbital", "population", "dipole", "bandgap", "all"
    ], default="all", help="Function to test")

    args = parser.parse_args()

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    multiwfn_exe = args.multiwfn or os.path.join(project_root, "Multiwfn_3.8_dev_bin_Win64", "Multiwfn.exe")

    # Find a suitable test file if not specified
    if args.file:
        test_file = args.file
    else:
        # Look for a simple test file
        examples_dir = os.path.join(project_root, "Multiwfn_3.8_dev_bin_Win64", "examples")
        simple_files = [
            "H2O_m3ub3lyp.wfn",
            "C2H5F.wfn",
            "CH3CONH2.fch",
            "LiF.wfn"
        ]

        test_file = None
        for filename in simple_files:
            potential_file = os.path.join(examples_dir, filename)
            if os.path.exists(potential_file):
                test_file = potential_file
                break

        if not test_file:
            print("No suitable test file found. Please specify --file")
            return 1

    if not os.path.exists(multiwfn_exe):
        print(f"Multiwfn executable not found: {multiwfn_exe}")
        return 1

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return 1

    print(f"Quick Test Runner")
    print(f"Multiwfn: {multiwfn_exe}")
    print(f"Test file: {test_file}")
    print(f"Function: {args.function}")
    print("-" * 50)

    # Test functions
    tests = {
        "basic": lambda: test_basic_function(multiwfn_exe, test_file),
        "density": lambda: test_density_analysis(multiwfn_exe, test_file),
        "orbital": lambda: test_orbital_info(multiwfn_exe, test_file),
        "population": lambda: test_pop_analysis(multiwfn_exe, test_file),
        "dipole": lambda: test_dipole_moment(multiwfn_exe, test_file),
        "bandgap": lambda: test_bandgap(multiwfn_exe, test_file)
    }

    passed = 0
    total = 0

    if args.function == "all":
        for name, test_func in tests.items():
            total += 1
            if test_func():
                passed += 1
            print()
    else:
        total += 1
        if args.function in tests:
            if tests[args.function]():
                passed += 1
        else:
            print(f"Unknown function: {args.function}")
            return 1

    print("SUMMARY:")
    print(f"Passed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())