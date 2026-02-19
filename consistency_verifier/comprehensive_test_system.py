#!/usr/bin/env python3
"""
Comprehensive Test System for Multiwfn and PyMultiWFN Consistency Verification

This system provides:
1. Automated test execution for multiple test cases
2. Robust JSON and text parsing
3. Result comparison and validation
4. Detailed reporting of differences
5. Ralph Loop integration for fixing issues

Author: PyMultiWFN Team
Date: 2026-02-06
"""

import os
import sys
import subprocess
import re
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Paths
PROJECT_DIR = Path("/home/yhm/software/PyMultiWFN")
TEST_FILES_DIR = PROJECT_DIR / "consistency_verifier/examples"
MULTIWFN_BIN = "/home/yhm/software/PyMultiWFN/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn"
RESULTS_DIR = PROJECT_DIR / "consistency_verifier/test_results"

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Test configuration
TOLERANCE = 1e-6  # Numerical tolerance for floating-point comparison
VERBOSE = True

# =============================================================================
# Data Structures
# =============================================================================

class TestCase:
    """Represents a single test case."""
    
    def __init__(self, name: str, test_file: Path, expected_properties: Dict[str, Any]):
        self.name = name
        self.test_file = test_file
        self.test_file_name = test_file.name
        self.expected_properties = expected_properties
        self.multiwfn_result: Optional[Dict[str, Any]] = None
        self.pymultiwfn_result: Optional[Dict[str, Any]] = None
        self.diff: Optional[Dict[str, Any]] = None
        self.status = "pending"  # pending, passed, failed, error
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    def __repr__(self):
        return self.__str__()


class TestResult:
    """Stores complete test results."""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "warnings": 0
        }
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_test_case(self, test_case: TestCase):
        self.test_cases.append(test_case)
        
        # Update summary
        self.summary["total"] += 1
        if test_case.status == "passed":
            self.summary["passed"] += 1
        elif test_case.status == "failed":
            self.summary["failed"] += 1
        elif test_case.status == "error":
            self.summary["errors"] += 1
        elif test_case.status == "warning":
            self.summary["warnings"] += 1
    
    def get_summary(self) -> str:
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return (
            f"Tests: {self.summary['total']}, "
            f"Passed: {self.summary['passed']}, "
            f"Failed: {self.summary['failed']}, "
            f"Warnings: {self.summary['warnings']}, "
            f"Duration: {duration:.1f}s"
        )


# =============================================================================
# Multiwfn Interface
# =============================================================================

class MultiwfnInterface:
    """Interface to interact with Multiwfn_noGUI."""
    
    def __init__(self, binary_path: Path):
        self.binary_path = binary_path
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Multiwfn binary not found: {self.binary_path}")
    
    def run(self, test_file: Path, commands: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run Multiwfn with given commands and return parsed output."""
        
        try:
            # Prepare input file
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
            
            # Create command sequence
            input_text = "\n".join(commands) + "\n"
            
            # Run Multiwfn
            result = subprocess.run(
                [str(self.binary_path), str(test_file)],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(test_file.parent)
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Multiwfn failed with return code {result.returncode}")
            
            # Parse output
            output = result.stdout
            
            # Extract key information
            parsed = {
                "total_energy": self._parse_total_energy(output),
                "num_electrons": self._parse_num_electrons(output),
                "num_mos": self._parse_num_mos(output),
                "charge": self._parse_charge(output),
                "atoms": self._parse_atoms(output),
            }
            
            return parsed
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Multiwfn timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Multiwfn error: {str(e)}")
    
    def _parse_total_energy(self, output: str) -> Optional[float]:
        """Parse total energy from Multiwfn output."""
        match = re.search(r'Total energy:\s+([-+]?\d+\.\d+)', output)
        if match:
            return float(match.group(1))
        return None
    
    def _parse_num_electrons(self, output: str) -> Optional[int]:
        """Parse number of electrons from Multiwfn output."""
        match = re.search(r'Total/Alpha/Beta electrons:\s+([\d.]+)', output)
        if match:
            # Remove decimal point
            return int(float(match.group(1)))
        return None
    
    def _parse_num_mos(self, output: str) -> Optional[int]:
        """Parse number of molecular orbitals from Multiwfn output."""
        match = re.search(r'The number of orbitals:\s+(\d+)', output)
        if match:
            return int(match.group(1))
        return None
    
    def _parse_charge(self, output: str) -> Optional[float]:
        """Parse net charge from Multiwfn output."""
        match = re.search(r'Net charge:\s+([-\d.]+)', output)
        if match:
            return float(match.group(1))
        return None
    
    def _parse_atoms(self, output: str) -> Optional[int]:
        """Parse number of atoms from Multiwfn output."""
        match = re.search(r'Total\s+atoms:\s+(\d+)', output)
        if match:
            return int(match.group(1))
        return None


# =============================================================================
# PyMultiWFN Interface
# =============================================================================

class PyMultiWFNInterface:
    """Interface to interact with PyMultiWFN."""
    
    def __init__(self, python_path: Path):
        self.python_path = python_path
        if not self.python_path.exists():
            raise FileNotFoundError(f"Python not found: {self.python_path}")
    
    def run(self, test_file: Path, timeout: int = 30) -> Dict[str, Any]:
        """Run PyMultiWFN and return parsed data."""
        
        try:
            # Prepare Python script
            script = f"""
import sys
sys.path.insert(0, '{PROJECT_DIR}')

from pymultiwfn.io.file_manager import FileManager

# Load wavefunction
fm = FileManager()
wfn = fm.load_wavefunction('{test_file}')

# Output key information as JSON
import json
result = {{
    "num_electrons": wfn.num_electrons,
    "charge": wfn.charge,
    "num_atoms": len(wfn.atoms),
    "title": wfn.title
}}

print(json.dumps(result, indent=2))
"""
            
            # Run Python script
            result = subprocess.run(
                [str(self.python_path), "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_DIR)
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"PyMultiWFN failed with return code {result.returncode}")
            
            # Parse JSON output
            lines = result.stdout.strip().split('\n')
            
            # Find JSON line
            json_str = None
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('{'):
                    json_str = lines[lines.index(line):].strip()
                    break
            
            if not json_str:
                raise RuntimeError(f"Could not find JSON in PyMultiWFN output")
            
            # Parse JSON
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Could not parse JSON: {str(e)}")
            
            return parsed
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"PyMultiWFN timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"PyMultiWFN error: {str(e)}")


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """Orchestrates test execution and result comparison."""
    
    def __init__(self, multiwfn: MultiwfnInterface, pymultiwfn: PyMultiWFNInterface):
        self.multiwfn = multiwfn
        self.pymultiwfn = pymultiwfn
    
    def run_test_case(self, test_case: TestCase):
        """Run a single test case on both systems and compare results."""
        
        print(f"\n{BLUE}{'='*60}")


        print(f"{BLUE}{'='*60}")
        
        try:
            # Run PyMultiWFN first (usually faster)
            print(f"{YELLOW}[*]{NC} Running PyMultiWFN...")
            pymultiwfn_result = self.pymultiwfn.run(test_case.test_file, timeout=30)
            
            # Run Multiwfn
            print(f"{YELLOW}[*]{NC} Running Multiwfn_noGUI...")
            multiwfn_result = self.multiwfn.run(
                test_case.test_file,
                test_case.expected_properties.get("commands", ["18\n1\nq\n"]),
                timeout=30
            )
            
            # Store results
            test_case.pymultiwfn_result = pymultiwfn_result
            test_case.multiwfn_result = multiwfn_result
            
            # Compare results
            status, diff = self._compare_results(test_case)
            test_case.status = status
            test_case.diff = diff
            
            # Print results
            self._print_results(test_case, status, diff)
            
        except Exception as e:

            test_case.status = "error"
            test_case.diff = {"error": str(e)}
    
    def _compare_results(self, test_case: TestCase) -> tuple[str, Dict[str, Any]]:
        """Compare PyMultiWFN and Multiwfn results."""
        
        pymultiwfn_result = test_case.pymultiwfn_result
        mwfn_result = test_case.multiwfn_result
        expected = test_case.expected_properties
        
        status = "passed"
        diff = {}
        
        # Compare each property
        properties_to_check = ["num_electrons", "charge", "num_atoms"]
        
        for prop in properties_to_check:
            if prop in expected and pymultiwfn_result and mwfn_result:
                py_val = pymultiwfn_result[prop]
                mw_val = mwfn_result[prop]
                
                if py_val is not None and mw_val is not None:
                    # Compare with tolerance
                    if abs(py_val - mw_val) <= TOLERANCE:
                        diff[prop] = {
                            "status": "match",
                            "py_multiwfn": py_val,
                            "mwfn": mw_val,
                            "difference": abs(py_val - mw_val)
                        }
                    else:
                        diff[prop] = {
                            "status": "mismatch",
                            "py_multiwfn": py_val,
                            "mwfn": mw_val,
                            "difference": abs(py_val - mw_val)
                        }
                        status = "failed"
                elif pymultiwfn_result is None or mwfn_result is None:
                    diff[prop] = {
                        "status": "missing",
                        "py_multiwfn": pymultiwfn_result,
                        "mwfn": mw_val,
                        "difference": None
                    }
        
        # Check for critical issues
        if pymultiwfn_result and pymultiwfn_result.get("num_electrons", 0) == 0:
            status = "warning"
            diff["warning"] = "PyMultiWFN reports 0 electrons (possible parsing issue)"
        
        return status, diff
    
    def _print_results(self, test_case: TestCase, status: str, diff: Dict[str, Any]):
        """Print formatted results."""
        
        print(f"\n{BLUE}Results:{NC}")
        print(f"{BLUE}{'='*60}")
        
        # Print status
        if status == "passed":
            print(f"{GREEN}[✓ PASSED]{NC} All properties match!")
        elif status == "failed":
            print(f"{RED}[✗ FAILED]{NC} Some properties don't match!")
        elif status == "warning":
            print(f"{YELLOW}[⚠ WARNING]{NC} Passed with warnings!")
        else:
            print(f"{RED}[? UNKNOWN]{NC} Unknown status: {status}")
        
        print(f"{BLUE}{'='*60}")
        
        # Print differences
        if diff:
            # Filter out warning
            diffs = {k: v for k, v in diff.items() if k != "warning"}
            if diffs:
                for prop, result in diffs.items():
                    result_status = result.get("status", "?")
                    
                    if result_status == "match":
                        print(f"  {GREEN}✓{NC} {prop}: {result.get('py_multiwfn')} == {result.get('mwfn')} (diff: {result.get('difference'):.2e})")
                    elif result_status == "mismatch":
                        print(f"  {RED}✗{NC} {prop}: {result.get('py_multiwfn')} != {result.get('mwfn')} (diff: {result.get('difference'):.2e})")
                    elif result_status == "missing":
                        print(f"  {YELLOW}?{NC} {prop}: PyMultiWFN={result.get('py_multiwfn')}, Multiwfn={result.get('mwfn')}")
        
        print(f"{BLUE}{'='*60}")
    
    def run_all_tests(self, test_cases: List[TestCase]) -> TestResult:
        """Run all test cases and return summary."""
        
        print(f"\n{BLUE}{'='*60}")

        print(f"{BLUE}{'='*60}")
        
        result = TestResult()
        result.start_time = datetime.now()
        
        for i, test_case in enumerate(test_cases, 1):

            self.run_test_case(test_case)
        
        result.end_time = datetime.now()
        
        print(f"\n{BLUE}{'='*60}")

        print(f"{BLUE}{'='*60}")
        print(f"{BLUE}{'='*60}")
        print(result.get_summary())
        print(f"{BLUE}{'='*60}")
        
        # Save results
        self._save_results(result, test_cases)
        
        return result
    
    def _save_results(self, result: TestResult, test_cases: List[TestCase]):
        """Save test results to JSON file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"test_results_{timestamp}.json"
        
        # Prepare output data
        output_data = {
            "timestamp": timestamp,
            "summary": result.summary,
            "test_cases": []
        }
        
        for test_case in test_cases:
            case_data = {
                "name": test_case.name,
                "test_file": test_case.test_file_name,
                "status": test_case.status,
                "diff": test_case.diff,
                "pymultiwfn_result": test_case.pymultiwfn_result,
                "multiwfn_result": test_case.multiwfn_result,
            }
            output_data["test_cases"].append(case_data)
        
        # Write results
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{GREEN}Results saved to: {results_file}{NC}")
        print(f"{BLUE}{'='*60}")


# =============================================================================
# Test Case Definitions
# =============================================================================

def get_test_cases() -> List[TestCase]:
    """Define test cases for consistency verification."""
    
    # Discover test files
    test_files = list(TEST_FILES_DIR.glob("*.wfn"))
    
    if not test_files:

        return []
    

    print(f"{BLUE}{'='*60}")
    
    test_cases = []
    
    # Define test cases based on available files
    for test_file in test_files:
        filename = test_file.name
        
        # Determine expected properties based on filename
        if "H2_CCSD" in filename:
            test_cases.append(TestCase(
                name=f"H2 CCSD - {filename}",
                test_file=test_file,
                expected_properties={
                    "num_electrons": 2,
                    "charge": 0.0,
                    "num_atoms": 2,
                    "multiplicity": 1,
                    "commands": ["18\n1\nq\n"],
                    "description": "Hydrogen molecule, CCSD calculation"
                }
            ))
        
        elif "benzene" in filename:
            test_cases.append(TestCase(
                name=f"Benzene - {filename}",
                test_file=test_file,
                expected_properties={
                    "num_electrons": 30,
                    "charge": 0.0,
                    "num_atoms": 12,
                    "multiplicity": 1,
                    "commands": ["18\n1\nq\n"],
                    "description": "Benzene molecule, 6 carbon ring"
                }
            ))
        
        elif "COBH3" in filename:
            test_cases.append(TestCase(
                name=f"COBH3 - {filename}",
                test_file=test_file,
                expected_properties={
                    "num_electrons": 12,
                    "charge": 0.0,
                    "num_atoms": 4,
                    "multiplicity": 1,
                    "commands": ["18\n1\nq\n"],
                    "description": "Carbon monoxide, triplet ground state"
                }
            ))
        
        # Add more test cases as needed...
    
    return test_cases


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the test system."""
    
    print(f"{BLUE}{'='*60}")

    print(f"{BLUE}{'='*60}")
    print(f"{BLUE}{'='*60}")
    print(f"Version: 3.0")
    print(f"{BLUE}{'='*60}")
    print(f"{BLUE}{'='*60}")
    print(f"{BLUE}Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BLUE}{'='*60}")
    print(f"{BLUE}{'='*60}")
    
    # Check Multiwfn binary
    if not Path(MULTIWFN_BIN).exists():
        print(f"{RED}[ERROR]{NC} Multiwfn binary not found: {MULTIWFN_BIN}")
        print(f"{YELLOW}Please download and configure Multiwfn_noGUI{NC}")
        return 1
    
    # Initialize interfaces
    try:
        multiwfn = MultiwfnInterface(Path(MULTIWFN_BIN))
        pymultiwfn = PyMultiWFNInterface(Path("/usr/bin/python3"))
        runner = TestRunner(multiwfn, pymultiwfn)
        
        # Get test cases
        test_cases = get_test_cases()
        
        if not test_cases:

            return 0
        
        # Run all tests
        result = runner.run_all_tests(test_cases)
        
        # Print final summary
        print(f"\n{GREEN}{'='*60}")
        print(f"{BLUE}Final Summary:{NC}")
        print(f"{BLUE}{'='*60}")

        print(f"{GREEN}Passed: {result.summary['passed']}{NC}")
        print(f"{RED}Failed: {result.summary['failed']}{NC}")
        if result.summary['warnings'] > 0:
            print(f"{YELLOW}Warnings: {result.summary['warnings']}{NC}")
        print(f"{BLUE}{'='*60}")
        
        # Exit with appropriate code
        if result.summary['failed'] > 0:
            return 1
        else:
            return 0
    
    except KeyboardInterrupt:

        return 2
    except Exception as e:
        print(f"\n{RED}[FATAL ERROR]{NC} {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
