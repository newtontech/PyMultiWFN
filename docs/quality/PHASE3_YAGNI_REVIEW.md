# Phase 3 YAGNI Review

## Decision

Phase 3 advanced bond-order work should remain deferred until there is a
validated user workflow and reference data for each proposed method.

## Evidence

- Existing open issues prioritized correctness, maintainability, tests, and CI
  reliability over new analysis breadth.
- Several modules contained placeholder APIs without working behavior. This pass
  removed the unreferenced placeholders and kept implemented parsers/analyzers
  available.
- The new reference validation tests create a safer baseline for deciding which
  advanced methods are worth implementing next.

## Roadmap

1. Stabilize Mayer, Mulliken, fuzzy, and delocalization analysis behavior with
   reference tests.
2. Add user-facing advanced methods only after their inputs, outputs, and
   validation references are documented.
3. Keep experimental code out of default imports until it has a passing test
   matrix and documented limitations.
