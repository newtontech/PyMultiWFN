# IMPLEMENTATION_PLAN.md - Overlap Matrix Debug & Validation

**Status**: In Progress (Debugging Phase)
**Last Updated**: 2026-02-19 11:27

---

## Overview

Debug overlap matrix calculation and fix bond order tests. Overlap matrix calculation is implemented but tests are still failing.

---

## Implementation Steps

### Step 1: Study Existing Code (STATUS: DONE) ✅
- [x] Review `pymultiwfn/integrals.py` for overlap functions
- [x] Understand `overlap_gaussian_primitive()` signature
- [x] Analyze WFN parser structure
- [x] Identify basis set data structure

**Learned**:
- `overlap_gaussian_primitive()` exists and works
- WFN parser has shells with exponents and coefficients
- Need to aggregate primitives into basis functions

---

### Step 2: Implement calculate_overlap_matrix() (STATUS: DONE) ✅
- [x] Create function in `pymultiwfn/integrals.py`
- [x] Implement basis function index mapping
- [x] Calculate overlap for each primitive pair
- [x] Aggregate to basis function level
- [x] Add documentation and type hints

**Completed**:
- Function implemented in `pymultiwfn/integrals/overlap.py`
- Fixed SP shell coefficients handling
- Fixed numpy array hashability issue
- Fixed basis function count mismatch

---

### Step 3: Update WFN Parser (STATUS: DONE) ✅
- [x] Import `calculate_overlap_matrix()` in wfn.py
- [x] Replace `np.eye(num_basis)` with actual calculation
- [x] Add optional caching (performance)
- [x] Add error handling and fallback to identity matrix

**Completed**:
- Modified `pymultiwfn/io/parsers/wfn.py`
- Added try-except for overlap matrix calculation
- Fallback to identity matrix if calculation fails

---

### Step 4: Debug Overlap Matrix (STATUS: IN PROGRESS) 🔄
- [ ] Create debug script to inspect overlap matrix
- [ ] Verify symmetry: S == S.T
- [ ] Verify diagonal elements: S[i,i] > 0
- [ ] Verify integration: trace(S) ≈ num_electrons
- [ ] Compare with Multiwfn results

**Current Issue**:
- Mayer vs Wiberg test still failing
- Difference: 8.18% max relative difference
- Actual: [[0.005008, 0.005008], [0.005008, 0.005008]]
- Desired: [[0.00463, 0.00463], [0.00463, 0.00463]]

**Possible Causes**:
1. Overlap matrix calculation incorrect
2. Basis function indices mismatch with MO coefficients
3. MO coefficient indexing problem

---

### Step 5: Fix Basis Function Indexing (STATUS: PENDING)
- [ ] Investigate WFN file MO coefficients format
- [ ] Understand which basis functions are used
- [ ] Adjust overlap matrix to match MO coefficients
- [ ] Or adjust MO coefficients to match all basis functions

**Problem**:
- `wfn.num_basis` = 34 (from MO coefficients)
- Actual basis functions = 48 (from shells)
- Mismatch causes indexing errors

---

### Step 6: Write Unit Tests (STATUS: PENDING)
- [ ] Test simple case (H2 with STO-3G)
- [ ] Test medium case (C2H2 with larger basis)
- [ ] Test symmetry (S == S.T)
- [ ] Test diagonal elements (S[i,i] > 0)
- [ ] Test integration (should be close to 1)

---

### Step 7: Fix Bonding Tests (STATUS: PENDING)
- [ ] Re-run `test_mayer_vs_wiberg`
- [ ] Re-run `test_bond_orders_in_range[h2]`
- [ ] Re-run `test_bond_orders_in_range[c2h2]`
- [ ] Verify all tests pass

**Current Status**:
- ❌ `test_mayer_vs_wiberg` FAILED
- Max absolute difference: 0.00037875
- Max relative difference: 8.18%

---

### Step 8: Code Review and Documentation (STATUS: PENDING)
- [ ] Verifier reviews code
- [ ] Add docstrings
- [ ] Add type hints
- [ ] Update user documentation

---

### Step 9: Performance Optimization (STATUS: PENDING)
- [ ] Profile the function
- [ ] Implement symmetry optimization (only compute upper triangle)
- [ ] Consider parallelization
- [ ] Add caching

---

## Current Focus

**Step 4**: Debug overlap matrix calculation

**Sub-tasks**:
1. Create debug script to inspect overlap matrix properties
2. Verify symmetry, positivity, and integration
3. Compare with Multiwfn results

**Next Milestone**: All bonding tests passing

**Target Date**: 2026-02-19 11:57 (30 minutes from now)

---

## Progress Log

### 2026-02-19 09:27
- Started Ralph Loop
- Created PROMPT.md, AGENTS.md, IMPLEMENTATION_PLAN.md
- Step 1 completed (study existing code)

### 2026-02-19 10:27
- Step 2 completed (calculate_overlap_matrix implemented)
- Step 3 completed (WFN parser updated)
- Fixed multiple bugs (SP shell, hashability, basis count)
- Git commit: "feat: integrate calculate_overlap_matrix() into WFN parser"
- Tests still failing, need debugging

### 2026-02-19 11:27
- Updated IMPLEMENTATION_PLAN.md
- Focus: Debug overlap matrix calculation
- Next: Create debug script and verify matrix properties

---

## Testing Status

**Failing Tests**:
- `tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg` FAILED
  - Expected: Mayer == Wiberg for closed-shell systems
  - Actual: Max relative difference = 8.18%

**Test Command**:
```bash
pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
```

---

## Git Status

**Last Commit**: 2026-02-19 10:32
- Commit: "feat: integrate calculate_overlap_matrix() into WFN parser"
- Files modified:
  - `pymultiwfn/integrals/__init__.py`
  - `pymultiwfn/integrals/overlap.py`
  - `pymultiwfn/io/parsers/wfn.py`
