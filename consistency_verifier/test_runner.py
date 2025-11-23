#!/usr/bin/env python3
"""
Comprehensive test suite for PyMultiWFN consistency verification.
Runs tests on all available test cases from Multiwfn examples.
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import multiprocessing as mp

from verifier import ConsistencyVerifier


class TestSuite:
    def __init__(self, multiwfn_exe: str, examples_dir: str):
        self.multiwfn_exe = multiwfn_exe
        self.examples_dir = examples_dir
        self.test_results = []
        self.start_time = datetime.now()

        # Test configurations for different file types and functions
        self.test_configs = {
            # Basic molecular properties
            "basic_info": {
                "commands": ["0"],  # Show basic information
                "description": "Basic molecular information"
            },
            "electron_density": {
                "commands": ["1", "0"],  # Electron density at critical points
                "description": "Electron density analysis"
            },
            "orbital_info": {
                "commands": ["5"],  # Orbital information
                "description": "Molecular orbital information"
            },
            "mulliken_pop": {
                "commands": ["7"],  # Mulliken population analysis
                "description": "Mulliken population analysis"
            },
            "dipole_moment": {
                "commands": ["11"],  # Dipole moment
                "description": "Dipole moment calculation"
            },
            "esp_analysis": {
                "commands": ["12", "0"],  # Electrostatic potential
                "description": "Electrostatic potential analysis"
            },
            "bandgap": {
                "commands": ["17"],  # HOMO-LUMO gap
                "description": "Band gap analysis"
            }
        }

        # Advanced tests for specific file types
        self.advanced_configs = {
            "fchk_specific": {
                "commands": ["6"],  # Density matrix analysis (for .fch files)
                "description": "Density matrix analysis",
                "file_types": [".fch", ".fchk"]
            },
            "td_dft": {
                "commands": ["18"],  # Excited state analysis
                "description": "TD-DFT analysis",
                "special_files": ["acetic_acid_TDDFT.out"]
            },
            "nci_analysis": {
                "commands": ["20", "0", "0.03", "0.5"],  # NCI analysis
                "description": "Non-covalent interaction analysis"
            }
        }

    def find_test_files(self) -> List[Tuple[str, str]]:
        """Find all test files in the examples directory."""
        test_files = []

        # Support file extensions
        extensions = ["*.wfn", "*.fch", "*.fchk", "*.molden", "*.wfx", "*.gjf"]

        for ext in extensions:
            pattern = os.path.join(self.examples_dir, "**", ext)
            files = glob.glob(pattern, recursive=True)
            for file in files:
                # Skip directories and special files
                if os.path.isfile(file) and not any(skip in file.lower() for skip in ['template', 'example', 'readme']):
                    test_files.append((file, ext))

        return test_files

    def categorize_file(self, filepath: str) -> str:
        """Categorize the test file for appropriate test selection."""
        filename = os.path.basename(filepath).lower()

        if any(atom in filename for atom in ['h ', 'he', 'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne']):
            return "atom"
        elif "h2o" in filename:
            return "water"
        elif "benzene" in filename or "phenol" in filename:
            return "aromatic"
        elif any(mol in filename for mol in ['li', 'na', 'k', 'ca', 'mg']):
            return "metallic"
        elif "td" in filename or "excit" in filename:
            return "excited"
        elif any(dimer in filename for dimer in ['dimer', 'complex']):
            return "complex"
        else:
            return "general"

    def get_tests_for_file(self, filepath: str, category: str) -> List[Dict[str, Any]]:
        """Get appropriate tests for a given file."""
        file_ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath).lower()

        tests = []

        # Basic tests for all files
        tests.append(self.test_configs["basic_info"])

        # Add file-type specific tests
        if file_ext in ['.wfn', '.fch', '.fchk']:
            tests.extend([
                self.test_configs["electron_density"],
                self.test_configs["orbital_info"],
                self.test_configs["mulliken_pop"],
                self.test_configs["dipole_moment"],
                self.test_configs["bandgap"]
            ])

        if file_ext in ['.fch', '.fchk']:
            tests.append(self.test_configs["esp_analysis"])

        # Special handling for excited states
        if category == "excited" or "tddft" in filename:
            if "tddft" in filename:
                tests.append(self.advanced_configs["td_dft"])

        # Skip some advanced tests for atoms
        if category != "atom":
            tests.append(self.advanced_configs["nci_analysis"])

        return tests

    def run_single_test(self, filepath: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        verifier = ConsistencyVerifier(self.multiwfn_exe)

        # Add exit command to all command sequences
        commands = test_config["commands"] + ["0"]  # Exit

        result = verifier.verify(filepath, commands)

        return {
            "file": os.path.basename(filepath),
            "filepath": filepath,
            "test_name": test_config["description"],
            "commands": commands,
            "result": result,
            "success": result["match"] and len(result["error_ref"]) == 0 and len(result["error_py"]) == 0
        }

    def run_tests_for_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Run all appropriate tests for a single file."""
        category = self.categorize_file(filepath)
        tests = self.get_tests_for_file(filepath, category)

        file_results = []
        for test_config in tests:
            print(f"Testing {os.path.basename(filepath)} - {test_config['description']}")
            try:
                result = self.run_single_test(filepath, test_config)
                file_results.append(result)
            except Exception as e:
                file_results.append({
                    "file": os.path.basename(filepath),
                    "filepath": filepath,
                    "test_name": test_config["description"],
                    "commands": test_config["commands"],
                    "result": {"error": str(e)},
                    "success": False
                })

        return file_results

    def run_all_tests(self, parallel: bool = True, max_workers: int = None) -> Dict[str, Any]:
        """Run tests on all found files."""
        test_files = self.find_test_files()

        if not test_files:
            print("No test files found!")
            return {"success": False, "message": "No test files found"}

        print(f"Found {len(test_files)} test files")

        if max_workers is None:
            max_workers = min(4, mp.cpu_count())

        if parallel and len(test_files) > 1:
            print(f"Running tests in parallel with {max_workers} workers...")
            with mp.Pool(max_workers) as pool:
                all_results = pool.map(self.run_tests_for_file, [f[0] for f in test_files])
        else:
            print("Running tests sequentially...")
            all_results = []
            for filepath, _ in test_files:
                results = self.run_tests_for_file(filepath)
                all_results.append(results)

        # Flatten results
        self.test_results = [result for file_results in all_results for result in file_results]

        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get("success", False))
        failed_tests = total_tests - successful_tests

        # Group results by file
        by_file = {}
        for result in self.test_results:
            filename = result["file"]
            if filename not in by_file:
                by_file[filename] = {"total": 0, "passed": 0, "failed": 0, "tests": []}

            by_file[filename]["total"] += 1
            by_file[filename]["tests"].append(result)

            if result.get("success", False):
                by_file[filename]["passed"] += 1
            else:
                by_file[filename]["failed"] += 1

        # Group failures by type
        failure_types = {}
        for result in self.test_results:
            if not result.get("success", False):
                test_name = result.get("test_name", "Unknown")
                if test_name not in failure_types:
                    failure_types[test_name] = 0
                failure_types[test_name] += 1

        summary = {
            "total_files": len(by_file),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            "by_file": by_file,
            "failure_types": failure_types,
            "execution_time": (datetime.now() - self.start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

        return summary

    def save_report(self, summary: Dict[str, Any], output_dir: str = None):
        """Save detailed test report."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "test_reports")

        os.makedirs(output_dir, exist_ok=True)

        # Save summary as JSON
        summary_file = os.path.join(output_dir, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save human-readable report
        report_file = os.path.join(output_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self._write_human_readable_report(summary, report_file)

        # Save detailed results
        details_file = os.path.join(output_dir, f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        print(f"Reports saved to {output_dir}:")
        print(f"  - Summary: {os.path.basename(summary_file)}")
        print(f"  - Report: {os.path.basename(report_file)}")
        print(f"  - Details: {os.path.basename(details_file)}")

        return summary_file, report_file, details_file

    def _write_human_readable_report(self, summary: Dict[str, Any], report_file: str):
        """Write a human-readable report."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PyMultiWFN Consistency Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {summary['timestamp']}\n")
            f.write(f"Execution Time: {summary['execution_time']:.2f} seconds\n\n")

            f.write("OVERALL SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Files Tested: {summary['total_files']}\n")
            f.write(f"Total Tests Run: {summary['total_tests']}\n")
            f.write(f"Successful Tests: {summary['successful_tests']}\n")
            f.write(f"Failed Tests: {summary['failed_tests']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n\n")

            if summary['failure_types']:
                f.write("FAILURE BREAKDOWN:\n")
                f.write("-" * 20 + "\n")
                for failure_type, count in summary['failure_types'].items():
                    f.write(f"{failure_type}: {count} failures\n")
                f.write("\n")

            f.write("DETAILED RESULTS BY FILE:\n")
            f.write("-" * 30 + "\n")

            for filename, file_data in summary['by_file'].items():
                f.write(f"\n{filename}:\n")
                f.write(f"  Total: {file_data['total']}, Passed: {file_data['passed']}, Failed: {file_data['failed']}\n")

                if file_data['failed'] > 0:
                    f.write("  Failed tests:\n")
                    for test in file_data['tests']:
                        if not test.get('success', False):
                            f.write(f"    - {test.get('test_name', 'Unknown')}\n")
                            if 'error' in test.get('result', {}):
                                f.write(f"      Error: {test['result']['error']}\n")


def main():
    """Main test runner."""
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    multiwfn_exe = os.path.join(project_root, "Multiwfn_3.8_dev_bin_Win64", "Multiwfn.exe")
    examples_dir = os.path.join(project_root, "Multiwfn_3.8_dev_bin_Win64", "examples")

    # Check if Multiwfn executable exists
    if not os.path.exists(multiwfn_exe):
        print(f"Multiwfn executable not found at: {multiwfn_exe}")
        print("Please update the path or ensure Multiwfn is properly installed.")
        return 1

    if not os.path.exists(examples_dir):
        print(f"Examples directory not found at: {examples_dir}")
        return 1

    print("PyMultiWFN Consistency Test Suite")
    print("=" * 50)
    print(f"Multiwfn: {multiwfn_exe}")
    print(f"Examples: {examples_dir}")
    print()

    # Create and run test suite
    suite = TestSuite(multiwfn_exe, examples_dir)

    try:
        summary = suite.run_all_tests(parallel=True)

        # Print immediate summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY:")
        print(f"Files: {summary['total_files']}")
        print(f"Tests: {summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Time: {summary['execution_time']:.1f}s")

        # Save reports
        suite.save_report(summary)

        return 0 if summary['failed_tests'] == 0 else 1

    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())