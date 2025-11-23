import os
import subprocess
import sys
import difflib
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

class ConsistencyVerifier:
    def __init__(self, multiwfn_bin: str, pymultiwfn_cmd: str = "pymultiwfn"):
        """
        Initialize the verifier.
        
        Args:
            multiwfn_bin: Path to the Multiwfn executable.
            pymultiwfn_cmd: Command to run PyMultiWFN (default: "pymultiwfn").
        """
        self.multiwfn_bin = multiwfn_bin
        self.pymultiwfn_cmd = pymultiwfn_cmd

    def run_multiwfn(self, input_file: str, commands: List[str]) -> Tuple[str, str]:
        """
        Run Multiwfn with the given input file and commands.
        
        Args:
            input_file: Path to the input file (e.g., .wfn, .fch).
            commands: List of strings representing keystrokes/inputs.
            
        Returns:
            Tuple of (stdout, stderr).
        """
        if not os.path.exists(self.multiwfn_bin):
            return "", f"Error: Multiwfn binary not found at {self.multiwfn_bin}"

        # Prepare input string
        input_str = "\n".join(commands) + "\n"
        
        # Multiwfn usually takes the input file as an argument or prompts for it.
        # If passed as argument: Multiwfn file.wfn
        cmd = [self.multiwfn_bin, input_file]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(input_file) if os.path.dirname(input_file) else None
            )
            stdout, stderr = process.communicate(input=input_str, timeout=60)
            return stdout, stderr
        except Exception as e:
            return "", str(e)

    def run_pymultiwfn(self, input_file: str, commands: List[str]) -> Tuple[str, str]:
        """
        Run PyMultiWFN with the given input file and commands.
        """
        # Assuming PyMultiWFN accepts the file as an argument and reads commands from stdin
        # similar to Multiwfn for compatibility testing.
        cmd = [sys.executable, "-m", "pymultiwfn.main", input_file]
        
        input_str = "\n".join(commands) + "\n"
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(input_file) if os.path.dirname(input_file) else None
            )
            stdout, stderr = process.communicate(input=input_str, timeout=60)
            return stdout, stderr
        except Exception as e:
            return "", str(e)

    def clean_output(self, output: str) -> str:
        """
        Clean and normalize output for comparison.
        Removes version info, timestamps, and normalizes whitespace.
        """
        lines = output.splitlines()
        cleaned_lines = []

        for line in lines:
            # Skip lines that typically vary between versions
            skip_patterns = [
                r'Multiwfn\s+\d+\.\d+',  # Version number
                r'Compiled on.*',  # Compilation date
                r'Time elapsed.*',  # Timing information
                r'Written by.*',  # Author info
                r'Reference:.*',  # Reference lines
                r'^\s*$',  # Empty lines
            ]

            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break

            if not should_skip:
                # Normalize whitespace and scientific notation
                cleaned_line = re.sub(r'\s+', ' ', line.strip())
                # Normalize scientific notation (e.g., 1.23E-01 vs 1.23e-01)
                cleaned_line = re.sub(r'(\d+\.\d+)[eE]([+-]?\d+)', r'\1e\2', cleaned_line)
                # Normalize signs in scientific notation
                cleaned_line = re.sub(r'(\d+\.\d+)e\+?(\d+)', r'\1e+\2', cleaned_line)
                cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)

    def extract_numeric_values(self, output: str) -> Dict[str, List[float]]:
        """
        Extract numeric values from output for tolerance-based comparison.
        """
        # Patterns for common numeric outputs
        patterns = {
            'energies': r'[-+]?\d+\.\d+(?:[eE][+-]?\d+)?\s*a\.?u\.?',
            'coordinates': r'[-+]?\d+\.\d+(?:[eE][+-]?\d+)?',
            'charges': r'[-+]?\d+\.\d+(?:[eE][+-]?\d+)?',
            'dipole': r'[-+]?\d+\.\d+(?:[eE][+-]?\d+)?',
            'populations': r'[-+]?\d+\.\d+(?:[eE][+-]?\d+)?'
        }

        extracted = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                extracted[name] = [float(m) for m in matches]

        return extracted

    def compare_numeric_arrays(self, arr1: List[float], arr2: List[float],
                             rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Compare two arrays of numerical values with tolerance.
        """
        if len(arr1) != len(arr2):
            return False

        arr1_np = np.array(arr1)
        arr2_np = np.array(arr2)

        return np.allclose(arr1_np, arr2_np, rtol=rtol, atol=atol)

    def verify(self, input_file: str, commands: List[str], diff_mode: str = "unified",
               tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Run both and compare outputs with intelligent comparison.
        """
        print(f"Running Multiwfn on {os.path.basename(input_file)}...")
        out_ref, err_ref = self.run_multiwfn(input_file, commands)

        print(f"Running PyMultiWFN on {os.path.basename(input_file)}...")
        out_py, err_py = self.run_pymultiwfn(input_file, commands)

        # Check for execution errors
        if err_ref:
            return {
                "match": False,
                "error_type": "multiwfn_error",
                "error_ref": err_ref,
                "error_py": err_py,
                "output_ref": out_ref,
                "output_py": out_py
            }

        if err_py:
            return {
                "match": False,
                "error_type": "pymultiwfn_error",
                "error_ref": err_ref,
                "error_py": err_py,
                "output_ref": out_ref,
                "output_py": out_py
            }

        # Clean outputs for comparison
        clean_ref = self.clean_output(out_ref)
        clean_py = self.clean_output(out_py)

        # Primary string comparison
        exact_match = (clean_ref == clean_py)

        # Extract and compare numeric values
        numeric_ref = self.extract_numeric_values(clean_ref)
        numeric_py = self.extract_numeric_values(clean_py)

        numeric_matches = {}
        for key in numeric_ref:
            if key in numeric_py:
                numeric_matches[key] = self.compare_numeric_arrays(
                    numeric_ref[key], numeric_py[key], rtol=tolerance
                )

        # Determine overall match
        all_numeric_match = all(numeric_matches.values()) if numeric_matches else True
        overall_match = exact_match or (all_numeric_match and len(numeric_matches) > 0)

        # Generate diff for non-matching outputs
        diff = ""
        if not overall_match:
            ref_lines = clean_ref.splitlines()
            py_lines = clean_py.splitlines()

            if diff_mode == "unified":
                diff = "\n".join(difflib.unified_diff(
                    ref_lines, py_lines,
                    fromfile="Multiwfn (cleaned)", tofile="PyMultiWFN (cleaned)",
                    lineterm=""
                ))
            elif diff_mode == "ndiff":
                diff = "\n".join(difflib.ndiff(ref_lines, py_lines))

        # Detailed comparison report
        comparison_report = {
            "exact_match": exact_match,
            "numeric_match": all_numeric_match,
            "numeric_comparisons": numeric_matches,
            "ref_numeric_count": {k: len(v) for k, v in numeric_ref.items()},
            "py_numeric_count": {k: len(v) for k, v in numeric_py.items()},
            "ref_output_length": len(clean_ref),
            "py_output_length": len(clean_py)
        }

        return {
            "match": overall_match,
            "comparison_type": "numeric_tolerance" if not exact_match and all_numeric_match else "exact",
            "diff": diff,
            "output_ref": out_ref,
            "output_py": out_py,
            "clean_output_ref": clean_ref,
            "clean_output_py": clean_py,
            "error_ref": err_ref,
            "error_py": err_py,
            "comparison_report": comparison_report
        }
