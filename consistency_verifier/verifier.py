import os
import subprocess
import sys
import difflib
from pathlib import Path
from typing import List, Optional, Tuple

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

    def verify(self, input_file: str, commands: List[str], diff_mode: str = "unified") -> dict:
        """
        Run both and compare outputs.
        """
        print(f"Running Multiwfn on {input_file}...")
        out_ref, err_ref = self.run_multiwfn(input_file, commands)
        
        print(f"Running PyMultiWFN on {input_file}...")
        out_py, err_py = self.run_pymultiwfn(input_file, commands)
        
        # Simple string comparison
        match = (out_ref == out_py)
        
        diff = ""
        if not match:
            ref_lines = out_ref.splitlines()
            py_lines = out_py.splitlines()
            
            if diff_mode == "unified":
                diff = "\n".join(difflib.unified_diff(
                    ref_lines, py_lines, 
                    fromfile="Multiwfn", tofile="PyMultiWFN", 
                    lineterm=""
                ))
            elif diff_mode == "ndiff":
                diff = "\n".join(difflib.ndiff(ref_lines, py_lines))
        
        return {
            "match": match,
            "diff": diff,
            "output_ref": out_ref,
            "output_py": out_py,
            "error_ref": err_ref,
            "error_py": err_py
        }
