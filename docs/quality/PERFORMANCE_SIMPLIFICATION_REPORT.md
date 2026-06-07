# Performance And Simplification Report

## Scope

This pass reviewed parser selection, element mapping, overlap type conversion,
and bond-order loops because they were repeatedly called out in the open KISS and
maintainability issues.

## Changes

- Removed repeated element lookup tables from high-use parsers and routed them
  through `get_atomic_number`.
- Replaced repeated atom-pair loops in Mayer and Mulliken bond-order calculations
  with a shared helper.
- Replaced the long `_type_to_lmn` branch chain with a constant lookup table.
- Removed empty placeholder files and unsupported VASP grid parser classes.

## Complexity Impact

The implementation removes more lines than it adds in the touched areas. The
largest reduction comes from parser element lookup tables and placeholder code,
while the bond-order refactor reduces four nested-loop copies to one shared pair
iterator.

## Risk

The changes preserve existing public loader and analyzer names where behavior was
implemented. Unsupported placeholders were removed only after verifying there
were no internal references.
