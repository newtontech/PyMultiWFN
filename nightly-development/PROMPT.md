# PyMultiWFN Ralph Loop Development

You are working on the PyMultiWFN project, implementing overlap matrix calculation.

## Goal
Implement `calculate_overlap_matrix()` function to compute overlap matrix from basis set information.

## Context
- WFN files do not contain overlap matrix
- Parser currently sets overlap matrix to identity
- This causes inaccurate bond order and population calculations

## Resources
- `pymultiwfn/io/parsers/wfn.py` - WFN parser
- `pymultiwfn/integrals.py` - Integral calculation functions
- `tests/test_bonding.py` - Bonding tests

## Implementation Plan
See `IMPLEMENTATION_PLAN.md`

## Your Role
- Coder: Implement the feature
- Verifier: Validate the implementation

## Workflow
1. Read IMPLEMENTATION_PLAN.md for current task
2. Implement/test/verify one step
3. Update IMPLEMENTATION_PLAN.md with progress
4. Run tests to verify
5. If tests pass, commit and continue
6. If tests fail, fix and retry
