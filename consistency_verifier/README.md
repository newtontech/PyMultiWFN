# PyMultiWFN Consistency Verifier

This package-local tool compares PyMultiWFN against the retained Multiwfn 3.8
noGUI binary. It is intended for repository development, CI, and nightly parity
checks; it is not part of the installable `pymultiwfn` Python package.

## Oracle Boundary

- Oracle binary: `Multiwfn_3.8_bin_Linux_noGUI/Multiwfn`
- Oracle examples: `Multiwfn_3.8_bin_Linux_noGUI/examples/`
- Upstream source: `Multiwfn_3.8_dev_src_Linux_2025-Nov-23/`

The upstream source directory is for algorithm inspection and migration review.
The verifier's behavior standard comes from the retained noGUI binary output.

## Usage

```bash
python -m consistency_verifier run --suite smoke
python -m consistency_verifier run --suite pr
python -m consistency_verifier run --suite full
```

To use a different executable:

```bash
MULTIWFN_BIN=/path/to/Multiwfn python -m consistency_verifier run --suite smoke
```

The shell wrappers delegate to the same entry point:

```bash
consistency_verifier/run_tests.sh smoke
consistency_verifier/run_tests.sh pr
consistency_verifier/run_tests.sh full
```

## Suites

- `smoke`: fast metadata parity over a small H2/H2O/benzene set.
- `pr`: includes `smoke` plus representative density, gradient, bond-order,
  and orbital observations for PR triage.
- `full`: includes `smoke` and `pr` plus larger retained reference assets for
  nightly expansion.

## Manifests

Cases live in `consistency_verifier/cases/*.json`. Each case records:

- input file path, resolved from the repository root
- scripted Multiwfn menu commands
- PyMultiWFN observations to collect
- comparison fields, comparison kind, and tolerances
- optional regex extractors for additional oracle fields

Generated reports and transcripts are written to
`consistency_verifier/results/<run-id>/` and are intentionally ignored by git.
