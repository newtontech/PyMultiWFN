# PyMultiWFN Ralph Loop Status

**Session**: 2026-02-19 15:27 (Hourly Development)
**Mode**: Dual-Agent (Coder + Verifier)
**Status**: Running

---

## Session Setup

### Agents
1. **Coder Agent** - Implements features and fixes bugs
   - Tool: Claude Code (claude_glm - GLM-4.7)
   - Focus: Code implementation, debugging, testing
   - Workflow: Read → Implement → Test → Commit

2. **Verifier Agent** - Validates code quality
   - Tool: Claude Code (claude_glm - GLM-4.7)
   - Focus: Code review, test validation, quality checks
   - Workflow: Review → Test → Approve/Reject

### Workflow
```
Iteration N:
  1. Coder implements task
  2. Coder runs tests
  3. Verifier reviews code
  4. Verifier runs full test suite
  5. If approved: commit, move to next task
  6. If rejected: coder fixes in next iteration
```

### Configuration
- **Max Iterations**: 24
- **Timeout per iteration**: 15 minutes (coder) + 10 minutes (verifier)
- **Project Path**: `~/software/PyMultiWFN`
- **Log Directory**: `nightly-development/logs/`

---

## Current Task

**Priority**: 🔴 CRITICAL - Fix basis function count mismatch

**Issue**:
- WFN parser has 48 basis functions (from shells)
- MO coefficients only use 34 basis functions
- This causes 14-function mismatch in overlap matrix indexing
- Results in 8.18% error in bond order calculations

**Test Status**:
- ❌ `test_mayer_vs_wiberg` - FAILING (8.18% error)
- ❌ `test_bond_orders_in_range[h2]` - FAILING
- ❌ `test_bond_orders_in_range[c2h2]` - FAILING

**Expected Outcome**:
- All bonding tests pass
- Overlap matrix correctly sized (34x34 to match MO coefficients)
- Bond orders calculated accurately

---

## Implementation Plan Progress

### Step 1: Study Existing Code ✅ DONE
- [x] Review `pymultiwfn/integrals.py`
- [x] Understand `overlap_gaussian_primitive()` signature
- [x] Analyze WFN parser structure
- [x] Identify basis set data structure

### Step 2: Implement calculate_overlap_matrix() ✅ DONE
- [x] Create function in `pymultiwfn/integrals/overlap.py`
- [x] Implement basis function index mapping
- [x] Calculate overlap for each primitive pair
- [x] Aggregate to basis function level
- [x] Add documentation and type hints

### Step 3: Update WFN Parser ✅ DONE
- [x] Import `calculate_overlap_matrix()` in wfn.py
- [x] Replace identity matrix with actual calculation
- [x] Add optional caching
- [x] Add error handling and fallback

### Step 4: Debug Overlap Matrix 🔄 IN PROGRESS
**Current Focus**: Fix basis function count mismatch

**Immediate Tasks**:
1. ✅ Create debug script to inspect overlap matrix
2. ✅ Verify symmetry: S == S.T
3. ✅ Verify diagonal elements: S[i,i] > 0
4. ✅ Verify integration: trace(S) ≈ num_electrons
5. 🔴 **URGENT**: Investigate WFN file MO coefficients format
6. 🔴 **URGENT**: Map 34 MO coefficient indices to 48 shell-based indices
7. 🔴 **URGENT**: Calculate correct 34x34 overlap matrix

### Step 5: Fix Basis Function Indexing ⏳ PENDING
- [ ] Investigate WFN file MO coefficients format
- [ ] Understand which basis functions are used
- [ ] Adjust overlap matrix to match MO coefficients

### Step 6: Write Unit Tests ⏳ PENDING
- [ ] Test simple case (H2 with STO-3G)
- [ ] Test medium case (C2H2 with larger basis)
- [ ] Test symmetry and diagonal properties

### Step 7: Fix Bonding Tests ⏳ PENDING
- [ ] Re-run `test_mayer_vs_wiberg`
- [ ] Re-run `test_bond_orders_in_range[h2]`
- [ ] Re-run `test_bond_orders_in_range[c2h2]`
- [ ] Verify all tests pass

### Step 8: Code Review and Documentation ⏳ PENDING
- [ ] Verifier reviews code
- [ ] Add docstrings
- [ ] Add type hints
- [ ] Update user documentation

### Step 9: Performance Optimization ⏳ PENDING
- [ ] Profile the function
- [ ] Implement symmetry optimization
- [ ] Consider parallelization
- [ ] Add caching

---

## Recent Commits

```
f32cec31 docs: add Ralph Loop work summaries (04:27-10:27)
b58e4404 feat: integrate calculate_overlap_matrix() into WFN parser
b6d22008 docs: add Ralph Loop work summary for 09:27 execution
07966fd5 fix: resolve WFN parser basis function mismatch and correlation test
dc72a8de fix: resolve multiple test failures in bonding analysis tests
```

---

## Log Files

**Current Session Log**:
```
nightly-development/logs/dual_agent_ralph_20260219_152937.log
```

**Previous Sessions**:
```
nightly-development/logs/dual_agent_ralph_20260219_152842.log
nightly-development/logs/ralph_*.log
```

---

## Next Milestones

1. ✅ Coder agent implements basis function index fix
2. ✅ Verifier agent validates the fix
3. ✅ All bonding tests pass
4. ✅ Git commit: "fix: resolve basis function count mismatch in overlap matrix"
5. ⏳ Move to Step 5: Write comprehensive unit tests
6. ⏳ Optimize performance (Step 9)

---

## Expected Deliverables (This Hour)

- **1-2功能模块**: Basis function index mapping + Overlap matrix resizing
- **2-5个测试通过**: All bonding tests should pass
- **3-8次git commit**: Small atomic commits for each fix
- **测试覆盖率提升**: Add unit tests for overlap matrix
- **代码质量改善**: Code review and refactoring

---

## Notes

- The dual-agent loop will run for up to 24 iterations (~4 hours)
- Each iteration consists of: coder → verifier → test → (approve/fix)
- Progress is tracked in IMPLEMENTATION_PLAN.md
- All commits use conventional commit format
- Test commands are defined in AGENTS.md

---

**Last Updated**: 2026-02-19 15:30 CST
