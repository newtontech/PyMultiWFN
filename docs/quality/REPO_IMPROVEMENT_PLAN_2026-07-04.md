# PyMultiWFN Repository Improvement Plan

Date: 2026-07-04

## Audit Scope

This plan is based on the current checkout at `/Users/yhm/Desktop/code/PyMultiWFN`.
It preserves the established project boundary:

- `pymultiwfn/` is the installable Python package.
- `Multiwfn_3.8_dev_src_Linux_2025-Nov-23/` is reference-only source.
- `Multiwfn_3.8_bin_Linux_noGUI/` and its zip are retained oracle/reference assets.
- Worktrees should live under `.worktrees/`; `.gitignore` now ignores that directory.

No dependency additions are required for the first improvement pass.

## Current Proof

Commands run with the existing `.venv/bin/python` on macOS:

- `python -m pytest -q --no-cov --maxfail=20`: 537 tests collected; 509 passed, 28 skipped, 3 warnings in 34.82s.
- `python -m pytest tests/integration/test_file_loading.py tests/unit/test_io_loader.py --runintegration -q --no-cov`: 5 passed.
- `python -m black --check pymultiwfn tests`: passed; 163 files would be left unchanged.
- `python -m isort --check-only pymultiwfn tests`: passed.
- `python -m flake8 pymultiwfn tests --select=E9,F63,F7,F82`: passed.
- `python -m flake8 pymultiwfn tests --exit-zero | wc -l`: 122 advisory style/complexity findings.
- `git diff --check`: passed.
- `python -m build`: successfully built `pymultiwfn-0.1.2.tar.gz` and `pymultiwfn-0.1.2-py3-none-any.whl`.
- Wheel inspection: 0 retained `Multiwfn_` reference assets included; runtime dependencies come from `pyproject.toml`.
- `python -m consistency_verifier run --suite smoke --skip-oracle-if-unavailable`: 0 passed, 0 failed, 0 errors, 3 skipped, because the retained oracle binary is Linux-only on this host.

## Sprint 1 Results

Completed in this pass:

- Added `.worktrees/` to `.gitignore`.
- Removed active stale absolute fixture paths and pointed integration tests at tracked `tests/test_data` files.
- Ran a mechanical Black/isort cleanup across `pymultiwfn/` and `tests/`, then made those checks blocking in local pre-commit and the code-quality workflow.
- Kept mypy advisory in CI and labeled it that way.
- Added an explicit `python -m build` step before wheel inspection in the git-hooks workflow.
- Consolidated packaging metadata so `pyproject.toml` is authoritative and `setup.cfg` only carries tool config.
- Replaced hidden identity overlap behavior with explicit fallback gating in `Wavefunction.calculate_overlap_matrix()`.
- Routed supported Cartesian primitive overlaps through a direct Gaussian moment formula instead of the broken recurrence path.
- Gated the N2 reference bonding smoke test when it encounters pure spherical-D shell type `-2`, which is not yet implemented by the Cartesian overlap engine.

## Main Diagnosis

The repository is locally green at the Python test layer, and formatting/import checks now have real gates. The remaining risk is scientific confidence: Linux oracle parity has not run on this host, unsupported spherical shells are skipped, and several advertised analysis APIs still contain placeholder behavior.

Remaining issues:

- Broad flake8 remains advisory with 122 style/complexity findings.
- Important computational paths still use placeholders, identity overlap matrices, or zero-valued outputs.
- Pure spherical D/F overlap support is not implemented; the N2 FCHK reference fixture skips for shell type `-2`.
- Linux Multiwfn oracle parity still needs to run on Linux CI or a Linux host.
- Some skips still reflect missing historical fixture layouts such as `test_data/fchk/sample.fchk`.

## Priority 0: Make Quality Signals Truthful

Goal: make "green" mean "the checks we care about actually ran".

Tasks:

1. Update README and architecture status to reflect current verified state. Done in sprint 1.
   - Replace stale test counts and "0 violations" claims with command-derived status.
   - Mark scientific parity as partial until oracle runs pass on Linux.

2. Fix stale test-data paths. Mostly done in sprint 1.
   - Replace `tests/analysis/test_bonding.py`'s hard-coded `/home/yhm/software/PyMultiWFN/consistency_verifier/examples` fixture with the shared repo-root fixtures or `tests/test_data`.
   - Convert root-level `tests/test_data/*.wfn` into the structure expected by integration tests, or update tests to use the actual tracked structure.
   - Remove fallback paths under `/home/yhm/software/PyMultiWFN` from active tests.

3. Make quality CI honest. Done for Black/isort; mypy remains advisory by design.
   - Remove `|| true` and `continue-on-error` from Black/isort once the initial formatting pass lands.
   - Decide whether mypy should be advisory or blocking; reflect that in workflow names and badges.
   - Keep the existing critical flake8 gate as a fast safety check.

4. Make the GitHub Actions wheel check explicit. Done in sprint 1.
   - The `git-hooks` workflow currently checks `dist/*.whl` after pre-push hooks. Add an explicit `python -m build` step before inspecting the wheel so the check does not rely on hook side effects.

5. Consolidate packaging metadata. Done in sprint 1.
   - Treat `pyproject.toml` as the source of package metadata.
   - Either remove packaging sections from `setup.cfg` or keep `setup.cfg` only for tool config.
   - Keep GUI/visualization dependencies optional unless they are required for the base package.

Exit criteria:

- Full local pytest still passes.
- No unexpected skips caused by stale absolute paths.
- Black and isort pass.
- README status matches actual commands.
- `python -m build` produces a wheel that excludes Multiwfn reference assets.

## Priority 1: Replace Placeholders on Core Scientific Paths

Goal: make the package trustworthy for the small set of advertised workflows.

Tasks:

1. Implement real overlap matrix calculation in `Wavefunction.calculate_overlap_matrix()`. Partially done in sprint 1.
   - The method now delegates to the overlap integral engine for shell-backed wavefunctions.
   - Identity overlap requires `allow_identity_fallback=True` and emits a warning.
   - Reuse `pymultiwfn/integrals/overlap.py` where possible instead of adding a new math path.
   - Add tests that fail if non-orthogonal basis cases silently fall back to identity.

2. Reconcile WFN overlap behavior.
   - WFN parser comments currently treat identity overlap as acceptable for WFN orthonormal MO handling, but bond/population analysis needs a consistent AO-overlap contract.
   - Document when identity is mathematically intentional versus a fallback.
   - Emit explicit warnings or structured metadata when fallback overlap is used.

3. Finish basis and derivative coverage for advertised analyses.
   - `pymultiwfn/math/basis.py` still lacks G/H and spherical/cartesian support.
   - `pymultiwfn/math/gradient.py` still lacks F/G/H and spherical harmonic gradients.
   - Add reference tests using retained Multiwfn examples for at least S/P/D/F and one spherical case before expanding higher angular momentum.

4. Replace or demote placeholder analyzers.
   - Topology analysis currently returns zero gradients/Hessians and empty critical points.
   - Generic population analysis returns dummy charges and bond orders.
   - Hyperpolarizability and LDOS return zero placeholders.
   - Either implement minimal real behavior or move these APIs behind explicit "experimental/not implemented" errors so users cannot mistake dummy outputs for science.

Exit criteria:

- No advertised public analysis API returns dummy values without an explicit experimental marker.
- Non-trivial overlap and density reference cases compare against retained Multiwfn output within declared tolerances.
- The README feature list only advertises workflows covered by tests or oracle parity cases.

## Priority 2: Strengthen Multiwfn Parity

Goal: make the `consistency_verifier` the decision surface for scientific correctness.

Tasks:

1. Keep macOS skip mode, but require Linux CI oracle smoke on schedule and manually.
2. Add a small PR-level parity suite that runs on Linux with the retained noGUI binary.
3. Store expected observations in manifests, not test comments.
4. Add one case each for density, gradient, overlap/bond order, and orbital metadata.
5. Ensure verifier reports are uploaded and summarized in CI.

Exit criteria:

- `python -m consistency_verifier run --suite smoke` passes on Linux.
- `--skip-oracle-if-unavailable` is used only for local non-Linux development.
- Every new scientific feature has either unit tests plus a parity case, or an explicit reason why parity is not available.

## Priority 3: Documentation and Website

Goal: make docs useful to users and not just archives.

Tasks:

1. Split docs into current user docs versus historical development records.
2. Make `docs/index.html` or the GitHub Pages app show install, supported formats, examples, and limitations.
3. Replace status/badge claims with generated or recently verified status snippets.
4. Add a "Known Limitations" page listing unsupported file writers, fallback overlap behavior, platform-specific oracle behavior, and optional GUI dependencies.

Exit criteria:

- Docs workflow actually builds or validates the docs artifact.
- Example scripts compile and at least one smoke example runs.
- User-facing docs clearly distinguish implemented, experimental, and planned features.

## Priority 4: Cleanup and Maintainability

Goal: reduce maintenance drag without changing scientific behavior.

Tasks:

1. Remove or archive debug scripts that are no longer part of supported workflows.
2. Remove backup files such as `mulliken.py.backup` from the package tree or move them to docs/history if they are still useful.
3. Normalize formatting with Black/isort in one mechanical PR.
4. Add module ownership labels in docs: core, IO, density, bonding, population, visualization, verifier.
5. Convert broad print-based APIs to structured return values plus logging.

Exit criteria:

- Package tree contains only importable supported modules.
- Tool output is quiet enough for CI triage.
- New work has a clear module boundary and verification path.

## Recommended First Sprint

Do these in order:

1. Fix `.worktrees/` ignore rule and stale fixture paths.
2. Update README status from current proof.
3. Run Black/isort as a mechanical cleanup, then enforce them in CI.
4. Implement real overlap matrix calculation or explicitly gate identity fallback.
5. Add one Linux oracle parity case for the overlap/bond-order path. Still pending; local macOS smoke skips the Linux-only oracle.

This sprint does not need new dependencies.
