# Scientific Validation Framework

## Reference Set

The CI validation framework currently includes five compact diatomic references:

- H2
- LiH
- N2
- O2
- F2

These are represented as deterministic synthetic wavefunctions in
`tests/validation/test_reference_molecule_framework.py`.

## Validated Behavior

- Mayer bond-order matrix shape and symmetry.
- Expected atom-pair bond order for each reference molecule.
- Valence diagonal consistency through the shared bond-order helper.

## Why Synthetic References

Several legacy tests depend on example WFN files outside the repository. The new
validation tests avoid external paths so they can run consistently in local
checks and GitHub Actions.

## Expansion Criteria

Future reference molecules should include a source calculation, expected values,
tolerance rationale, and at least one regression test that can run without
network access.
