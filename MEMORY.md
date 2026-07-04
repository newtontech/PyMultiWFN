# MEMORY

- 2026-07-04 repo improvement audit: current proof, findings, and recommended sprint are in [docs/quality/REPO_IMPROVEMENT_PLAN_2026-07-04.md](docs/quality/REPO_IMPROVEMENT_PLAN_2026-07-04.md).
- 2026-07-04 sprint execution: first-pass results and remaining gaps are in [docs/quality/REPO_IMPROVEMENT_PLAN_2026-07-04.md#sprint-1-results](docs/quality/REPO_IMPROVEMENT_PLAN_2026-07-04.md#sprint-1-results).
- Key local result: default pytest 509 passed, 28 skipped, 3 warnings; integration opt-in 5 passed; Black/isort, critical flake8, `git diff --check`, `python -m build`, and wheel asset inspection passed.
- What worked: explicit overlap fallback gating, direct Cartesian overlap moments, stale fixture-path repair, blocking Black/isort gates, and `.worktrees/` ignore.
- What remains: Linux oracle unavailable locally, N2 pure spherical-D shell type `-2` skips, 122 advisory flake8 findings, and placeholder scientific APIs still need follow-up.
