# Consistency Verification

This project includes a tool to verify the consistency between `PyMultiWFN` and the original `Multiwfn`.

## Prerequisites

1.  **Multiwfn Executable**: You need a working `Multiwfn` executable.
    *   If you are on Windows, ensure `Multiwfn.exe` is accessible.
    *   If you are on Linux/WSL, ensure the `Multiwfn` binary is compiled and accessible.
2.  **PyMultiWFN**: Installed via pip (e.g., `pip install -e .`).

## Usage

1.  Set the `MULTIWFN_BIN` environment variable to the path of your Multiwfn executable.
    *   Windows (PowerShell): `$env:MULTIWFN_BIN = "C:\path\to\Multiwfn.exe"`
    *   Linux/Bash: `export MULTIWFN_BIN=/path/to/Multiwfn`
    
    Alternatively, you can edit `run_verification.py` directly.

2.  Run the verification script:
    ```bash
    python run_verification.py
    ```

## How it works

The `ConsistencyVerifier` class (in `consistency_verifier/verifier.py`) runs both programs with the same input file and keystrokes. It captures the standard output and compares them line by line.

## Adding Test Cases

To add more test cases, modify `run_verification.py` to include different input files and command sequences.
