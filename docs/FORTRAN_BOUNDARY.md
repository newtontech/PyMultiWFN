# Fortran Boundary

PyMultiWFN is presented as a Python-first package. Performance-sensitive routines may use Fortran-backed kernels, but public users should interact through Python modules and the CLI.

## Rules

- Keep public APIs in Python.
- Keep Fortran source under `pymultiwfn/math/fortran/` or another clearly named implementation directory.
- Add Python tests that verify numerical behavior and tolerances.
- Document build requirements before enabling a compiled extension by default.
- Prefer pure-Python fallback behavior when a compiled kernel is unavailable.

## Acceptance checks

- `pip install -e .[dev]` works from a clean checkout.
- `pytest` passes without requiring private data.
- Any compiled path has a matching pure-Python or fixture-based correctness test.
