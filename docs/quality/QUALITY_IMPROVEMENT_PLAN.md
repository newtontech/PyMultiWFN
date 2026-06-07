# Quality Improvement Plan

This plan tracks the remaining quality issues opened for PyMultiWFN and ties
each area to a concrete implementation step.

## Testability

- Added focused regression tests under `tests/quality/` for parser selection,
  element lookup, overlap type mapping, bond-order helpers, and the analysis base
  contract.
- Added `tests/validation/` reference-molecule checks so scientific invariants
  can run in CI without external data files.

## DRY

- Centralized element-symbol lookup in `pymultiwfn.core.definitions`.
- Reused one atom-pair iterator for Mayer and Mulliken bond-order loops.
- Introduced `BaseWavefunctionAnalysis` to standardize analyzer construction.

## Logging

- Replaced parser and core warning/debug `print` calls with module loggers.
- Debug output now stays silent by default and can be enabled by test or caller
  logging configuration.

## Security

- Removed unused placeholder modules that presented unsupported behavior as
  importable APIs.
- Replaced bare exception handling in parser detection with scoped exceptions and
  debug diagnostics.

## Configuration

- Kept package-level configuration minimal.
- Removed unsupported VASP grid formats from the parser factory so the advertised
  supported-format list matches implemented behavior.

## Next Checks

- Keep Gemini review optional until its required credentials are configured.
- Keep `test-success`, `docs`, and `quality` as the required branch-protection
  checks for `main`.
