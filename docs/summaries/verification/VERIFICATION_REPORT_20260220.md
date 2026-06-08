# PyMultiWFN Bonding Analysis Verification Report

**Date:** 2026-02-20 01:31 GMT+8
**Verifier:** PyMultiWFN verifier agent
**Status:** ⚠️ **TESTS FAILED - CRITICAL BUG FOUND**

---

## Summary

Found **2 critical test failures** out of 44 tests due to **incorrect Mayer bond order calculation formula** in `bondorder.py` and `multicenter.py` modules.

---

## Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.2
collected 44 items

42 passed, 2 failed, 3 warnings, 4 rerun in 4.81s
```

### Failed Tests

1. **TestMulticenterBondOrder::test_multicenter_two_center**
   - Expected: MCBO should match Mayer BO (~1.0)
   - Actual: MCBO = 6.511, Mayer BO = 1.000
   - Error: Relative difference > 20% threshold

2. **TestIntegration::test_mayer_vs_wiberg**
   - Expected: Wiberg should equal Mayer (rtol=1e-10)
   - Actual: Mayer = 1.000161, Wiberg = 6.511136
   - Error: 84.6% relative difference

---

## Root Cause Analysis

### The Bug

Three different implementations of Mayer bond order calculation use **different formulas**:

#### 1. `mayer.py` (CORRECT)
```python
accum = np.trace(ps_ij @ ps_ji)  # Line 48
```
- Uses trace of matrix product
- Returns: **1.000161** for H2 (correct)

#### 2. `bondorder.py` (INCORRECT)
```python
accum = np.sum(ps_ij * ps_ji)  # Line 67
```
- Uses element-wise product sum
- Returns: **6.511136** for H2 (incorrect)

#### 3. `multicenter.py` (INCORRECT)
```python
result = np.sum(sub_matrix_12 * sub_matrix_21)  # Line 55
```
- Uses element-wise product sum
- Returns: **6.511136** for H2 (incorrect)

### Mathematical Analysis

For Mayer bond order:
```
BO_AB = sum_{mu in A} sum_{nu in B} (PS)_mu,nu * (PS)_nu,mu
```

Where `ps_ij = PS[np.ix_(bfs_i, bfs_j)]` and `ps_ji = PS[np.ix_(bfs_j, bfs_i)]`.

**Correct formula:**
```python
trace(ps_ij @ ps_ji) = sum_{i,k} (ps_ij @ ps_ji)[i,k] when i=k
                     = sum_{i,j,k} ps_ij[i,j] * ps_ji[j,k] when i=k
                     = sum_{i,j} ps_ij[i,j] * ps_ji[j,i] ✓
```

**Incorrect formula:**
```python
sum(ps_ij * ps_ji) = sum_{i,j} ps_ij[i,j] * ps_ji[i,j] ✗
                    # Note: ps_ji[i,j] vs ps_ji[j,i]
```

The error is that the incorrect formula uses `ps_ji[i,j]` instead of `ps_ji[j,i]`.

### Example Demonstration

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = A.T  # ps_ji is transpose of ps_ij

# Correct
trace_A_B = np.trace(A @ B)  # = 30.0

# Incorrect
sum_A_B = np.sum(A * B)  # = 29.0

# Difference = 1.0 (3.3% for this small example)
# For large matrices, the difference can be significant
```

---

## Code Review

### 1. Overlap Matrix Calculation

✅ **PASSED** - Overlap matrix is correctly computed and NOT falling back to identity matrix:
- Warning: "Standard overlap calculation failed... Using WFN-format-specific method."
- Result: Non-identity overlap matrix with proper symmetry
- Diagonal elements: [0.0099, 0.1712, 1.5778, 10.5865, 59.8169, ...]
- Off-diagonal: S[0,1] = 0.0229 (non-zero, indicating overlap)

### 2. Dimension Matching

✅ **PASSED** - All matrix dimensions are consistent:
- `num_basis = 34`
- `overlap_matrix.shape = (34, 34)`
- `density_matrix.shape = (34, 34)`
- Atomic basis indices properly mapped

### 3. Boundary Case Handling

✅ **PASSED** - Edge cases are well-handled:
- Empty wavefunction → Returns empty matrix
- Single atom → Returns 1x1 matrix
- Non-bonded atoms → Returns near-zero bond order
- Zero density → Returns zero bond order
- Missing overlap matrix → Raises proper ValueError

### 4. Code Style and Comments

✅ **PASSED** - Code quality is good:
- Clear docstrings with Args/Returns
- Type hints consistently used
- Vectorized operations for efficiency
- Good variable naming

---

## Mathematical Consistency Check

### Symmetry Test

✅ **PASSED** - Bond order matrices are symmetric:
```python
np.allclose(bond_matrix, bond_matrix.T, rtol=1e-10)  # True
```

### Diagonal Elements (Mayer Valence)

✅ **PASSED** - Diagonal equals sum of off-diagonal elements:
```python
for i in range(n_atoms):
    diagonal_val = bond_matrix[i, i]
    mayer_valence = np.sum(bond_matrix[i, :]) - diagonal_val
    # diagonal_val == mayer_valence
```

### Unrestricted Wavefunctions

✅ **PASSED** - Alpha + Beta = Total:
```python
bnd_total ≈ bnd_alpha + bnd_beta  # Within numerical tolerance
```

---

## Verification Standards

### ✓ All bonding tests must pass

**FAILED** - 2 out of 44 tests failed due to critical bug.

### ✓ Overlap matrix must not be identity matrix

**PASSED** - Overlap matrix is non-identity with correct properties.

### ✓ Bond order calculations must be in reasonable range

**MIXED** - Results depend on which function is called:
- `mayer.calculate_mayer_bond_order()`: Returns ~1.0 for H2 ✓
- `bondorder.calculate_mayer_bond_order()`: Returns ~6.5 for H2 ✗
- `bondorder.calculate_wiberg_bond_order()`: Returns ~6.5 for H2 ✗
- `multicenter.calculate_multicenter_bond_order()`: Returns ~6.5 for H2 ✗

---

## Required Fixes

### 1. Fix `pymultiwfn/analysis/bonding/bondorder.py`

**Line 67:** Change from:
```python
accum = np.sum(ps_ij * ps_ji)
```

To:
```python
accum = np.trace(ps_ij @ ps_ji)
```

**Line 68:** Change from:
```python
# Vectorized calculation: sum of element-wise products
```

To:
```python
# Vectorized calculation: trace of matrix product
# BO_ij = trace(PS_ij @ PS_ji) = sum_{mu in A} sum_{nu in B} (PS)_mu,nu * (PS)_nu,mu
```

**Line 127:** Same fix for alpha channel:
```python
accum_alpha = np.trace(ps_alpha_ij @ ps_alpha_ji)  # Not: np.sum(...)
```

**Line 133:** Same fix for beta channel:
```python
accum_beta = np.trace(ps_beta_ij @ ps_beta_ji)  # Not: np.sum(...)
```

### 2. Fix `pymultiwfn/analysis/bonding/multicenter.py`

**Line 55:** Change from:
```python
result = np.sum(sub_matrix_12 * sub_matrix_21)
```

To:
```python
result = np.trace(sub_matrix_12 @ sub_matrix_21)
```

**Line 50-52:** Update comment:
```python
# In Fortran: sum_{ib} sum_{ia} PSmat(ia,ib) * PSmat(ib,ia)
# In NumPy: trace of matrix product
sub_matrix_12 = ps_matrix[np.ix_(basis_fns_1, basis_fns_2)]
sub_matrix_21 = ps_matrix[np.ix_(basis_fns_2, basis_fns_1)]
result = np.trace(sub_matrix_12 @ sub_matrix_21)
```

### 3. Remove Duplicate `calculate_mayer_bond_order` from `bondorder.py`

The `bondorder.py` file contains its own implementation of `calculate_mayer_bond_order` which is incorrect and duplicates the one in `mayer.py`.

**Recommended:** Remove the duplicate and import from `mayer` module:
```python
from .mayer import calculate_mayer_bond_order  # Line 16-136 can be removed
```

Or at least make it call the correct implementation.

---

## Impact Assessment

### Severity: **HIGH**

1. **Wrong Results:** All bond order calculations using `bondorder.py` or `multicenter.py` are incorrect
2. **User Confusion:** Three different functions give different results for the same input
3. **Data Integrity:** Published results using these calculations would be incorrect

### Affected Modules:

- ❌ `pymultiwfn/analysis/bonding/bondorder.py`
  - `calculate_mayer_bond_order()`
  - `calculate_wiberg_bond_order()` (calls the broken mayer function)
  - `calculate_mulliken_bond_order()` (might be affected)

- ❌ `pymultiwfn/analysis/bonding/multicenter.py`
  - `calculate_multicenter_bond_order()`

- ✅ `pymultiwfn/analysis/bonding/mayer.py`
  - `calculate_mayer_bond_order()` (CORRECT)

---

## Test Validation After Fix

After applying the fixes, run:
```bash
pytest tests/analysis/test_bonding.py -v
```

Expected result:
```
44 passed, 0 failed, 3 warnings in ~5s
```

---

## Recommendations

### Immediate Actions:

1. **Fix the formula** in both `bondorder.py` and `multicenter.py`
2. **Remove duplicate** `calculate_mayer_bond_order` from `bondorder.py`
3. **Add regression tests** to catch this type of bug in the future
4. **Run full test suite** to ensure no other modules are affected

### Long-term Improvements:

1. **Code consolidation:** Have one canonical implementation of Mayer bond order
2. **Math verification:** Add unit tests that verify mathematical properties (e.g., trace formula vs element-wise sum)
3. **Documentation:** Document the correct formula clearly with mathematical notation
4. **Cross-validation:** Test against reference values from Multiwfn Fortran code

---

## Conclusion

**VERIFICATION FAILED**

The bonding analysis module has a critical bug in the Mayer bond order calculation formula. While 42 out of 44 tests pass, the 2 failures reveal a fundamental mathematical error that affects the correctness of bond order calculations.

**Key Finding:** The formula `np.sum(ps_ij * ps_ji)` incorrectly computes Mayer bond order. The correct formula is `np.trace(ps_ij @ ps_ji)`.

**Status:** Awaiting coder agent to fix the reported issues before final verification.

---

## Appendix: Complete Test Output

```
[See full test output in repository logs]
```

---

**Report generated by:** PyMultiWFN verifier agent
**Next step:** Report to main agent with findings
