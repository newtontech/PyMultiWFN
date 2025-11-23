# PyMultiWFN Consistency Verifier

A comprehensive testing suite to ensure PyMultiWFN outputs are consistent with the original Multiwfn program.

## Overview

This testing framework automatically runs both Multiwfn and PyMultiWFN on a variety of test cases and compares their outputs using intelligent matching algorithms that account for:

- Exact string matching for identical outputs
- Tolerance-based numeric comparison for floating-point values
- Output normalization to handle version differences, formatting variations, and irrelevant metadata

## Files Structure

```
consistency_verifier/
├── verifier.py          # Core verification logic
├── test_runner.py       # Comprehensive test suite
├── quick_test.py        # Quick function testing
├── run_tests.bat        # Windows batch runner
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── test_reports/       # Generated test reports
```

## Requirements

- Python 3.8+
- NumPy
- Multiwfn 3.8 executable
- PyMultiWFN installed in development mode

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure PyMultiWFN is installed:
   ```bash
   pip install -e ..
   ```

3. Download and extract Multiwfn 3.8 to `../Multiwfn_3.8_dev_bin_Win64/`

## Usage

### Quick Testing

Test specific functions quickly:

```bash
# Quick test of all functions on a default file
python quick_test.py

# Test specific function
python quick_test.py --function density

# Test with custom file
python quick_test.py --file path/to/test.wfn
```

### Comprehensive Testing

Run full test suite on all examples:

```bash
# Run comprehensive tests
python test_runner.py

# Run sequentially (slower but easier debugging)
python test_runner.py --parallel 0
```

### Windows Batch Runner

Use the provided batch script for easy execution:

```cmd
# Quick tests
run_tests.bat quick

# Full comprehensive tests
run_tests.bat

# Sequential execution
run_tests.bat sequential
```

## Test Coverage

The framework tests the following Multiwfn functions:

- **Basic Information** (Function 0): Molecular data, atom counts, basis sets
- **Electron Density** (Function 1): Critical points, density analysis
- **Molecular Orbitals** (Function 5): Orbital energies, coefficients
- **Population Analysis** (Function 7): Mulliken charges, populations
- **Dipole Moment** (Function 11): Molecular dipole calculation
- **Electrostatic Potential** (Function 12): ESP mapping and analysis
- **Band Gap** (Function 17): HOMO-LUMO gap analysis
- **NCI Analysis** (Function 20): Non-covalent interaction analysis

## Supported File Formats

- `.wfn` - Wavefunction files
- `.fch` / `.fchk` - Gaussian formatted checkpoint files
- `.molden` - Molden format files
- `.wfx` - Extended wavefunction files

## Output Reports

The testing framework generates three types of reports:

1. **Summary Report** (`test_summary_*.json`): Overall statistics and pass/fail rates
2. **Human-Readable Report** (`test_report_*.txt`): Formatted text report
3. **Detailed Results** (`detailed_results_*.json`): Complete comparison data for all tests

Reports are saved in the `test_reports/` directory with timestamps.

## Understanding Test Results

### Success Criteria

A test passes if:
- Both programs execute without errors, AND
- Outputs match exactly, OR
- Numerical values match within specified tolerance (default: 1e-5)

### Report Sections

- **Overall Summary**: Pass/fail statistics, success rate
- **Failure Breakdown**: Types of failures and their frequency
- **Detailed Results**: Per-file and per-function analysis

### Comparison Types

- **Exact Match**: String comparison after normalization
- **Numeric Tolerance**: Numerical values match within tolerance when exact strings differ

## Troubleshooting

### Common Issues

1. **Multiwfn not found**: Ensure Multiwfn.exe is in the correct location
2. **PyMultiWFN import errors**: Install PyMultiWFN in development mode
3. **Permission errors**: Run with appropriate file permissions
4. **Timeout errors**: Increase timeout values in the verifier

### Debug Mode

For debugging, run tests sequentially to see detailed output:

```python
# In test_runner.py, set parallel=False
summary = suite.run_all_tests(parallel=False)
```

## Contributing

To add new test cases:

1. Add test files to the Multiwfn examples directory
2. Define new test configurations in `test_configs` or `advanced_configs`
3. Update file categorization logic if needed
4. Add appropriate patterns for output parsing

## Output Normalization

The framework automatically handles:

- Version number differences
- Compilation date variations
- Scientific notation formatting (1.23E-01 vs 1.23e-01)
- Whitespace normalization
- Timing information removal
- Reference text removal

## Configuration Options

Key parameters can be adjusted in the code:

- `tolerance`: Numeric comparison tolerance (default: 1e-5)
- `timeout`: Process timeout in seconds (default: 60)
- `max_workers`: Parallel execution workers
- `skip_patterns`: Output lines to ignore during comparison




  "Building upon the existing consistency_verifier/README.md, the first step is to establish the basic execution  
  of both the original Multiwfn program and the pymultiwfn library on a single test file.                         




1. Create a placeholder `verifier.py` in C:\Users\yanha\Downloads\PyMultiWFN\consistency_verifier\ if it       
   doesn't already exist.
2. Implement a function within `verifier.py` (e.g., run_multiwfn_and_pymultiwfn) that takes a file path to an  
   example input file (e.g., from C:\Users\yanha\Downloads\PyMultiWFN\Multiwfn_3.8_dev_bin_Win64\examples\) as 
   an argument.
3. Within this function, execute `Multiwfn.exe`:(or corresponding file in linux or mac OS)
      * Determine the correct command-line arguments to run Multiwfn.exe non-interactively on the input file to  
      generate a specific output (e.g., Function 0: Molecular data). You may need to consult Multiwfn
      documentation or experiment.
      * Capture its standard output and standard error.
4. Execute `pymultiwfn`:
      * Identify the equivalent pymultiwfn function or module that processes the same input file and performs the
      same analysis as Multiwfn in the previous step.
      * Capture its output (e.g., returned data structure or printed output).
5. For initial verification, print both raw outputs (Multiwfn and pymultiwfn) to the console for a selected    
   example file.

Goal: To confirm that both programs can be invoked programmatically and their raw outputs can be captured for a simple case.