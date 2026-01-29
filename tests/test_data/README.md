# Test Data Directory

This directory contains test data files for PyMultiWFN testing.

## Structure

- `wfn/`: WFN format wavefunction files
- `fchk/`: Gaussian fchk format files
- `molden/`: Molden format files

## Adding Test Data

When adding new test data files:

1. Use small, minimal examples when possible
2. Prefer files with permissive licenses or generated from public domain data
3. Document the source and properties of the data in a README in each subdirectory
4. Keep file sizes small to maintain fast test execution

## Notes

- Large files should be compressed (.gz) to save space
- Test data files are not packaged with the distribution
