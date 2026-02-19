#!/bin/bash
# PyMultiWFN Ralph Loop - Dual Agent Collaboration Mode
# Coder + Verifier iterative development

set -e

PROJECT_DIR="$HOME/software/PyMultiWFN"
LOG_DIR="$PROJECT_DIR/logs"
WORK_DIR="$PROJECT_DIR/nightly-development"
PROMPT="$WORK_DIR/PROMPT.md"
AGENTS="$WORK_DIR/AGENTS.md"
PLAN="$WORK_DIR/IMPLEMENTATION_PLAN.md"

mkdir -p "$LOG_DIR" "$WORK_DIR"

echo "========================================="
echo "PyMultiWFN Ralph Loop - Dual Agent Mode"
echo "========================================="
echo "Project: $PROJECT_DIR"
echo "Work dir: $WORK_DIR"
echo "========================================="
echo ""

# Create PROMPT.md if not exists
if [ ! -f "$PROMPT" ]; then
    cat > "$PROMPT" << 'EOF'
# PyMultiWFN Ralph Loop Development

You are working on the PyMultiWFN project, implementing overlap matrix calculation.

## Goal
Implement `calculate_overlap_matrix()` function to compute overlap matrix from basis set information.

## Context
- WFN files do not contain overlap matrix
- Parser currently sets overlap matrix to identity
- This causes inaccurate bond order and population calculations

## Resources
- `pymultiwfn/io/parsers/wfn.py` - WFN parser
- `pymultiwfn/integrals.py` - Integral calculation functions
- `tests/test_bonding.py` - Bonding tests

## Implementation Plan
See `IMPLEMENTATION_PLAN.md`

## Your Role
- Coder: Implement the feature
- Verifier: Validate the implementation

## Workflow
1. Read IMPLEMENTATION_PLAN.md for current task
2. Implement/test/verify one step
3. Update IMPLEMENTATION_PLAN.md with progress
4. Run tests to verify
5. If tests pass, commit and continue
6. If tests fail, fix and retry
EOF
fi

# Create AGENTS.md if not exists
if [ ! -f "$AGENTS" ]; then
    cat > "$AGENTS" << 'EOF'
# AGENTS.md - PyMultiWFN Ralph Loop

## Test Commands (Backpressure)

```bash
# Run bonding tests
pytest tests/analysis/test_bonding.py -v

# Run with coverage
pytest tests/analysis/test_bonding.py --cov=pymultiwfn --cov-report=html

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
```

## Project Structure

```
pymultiwfn/
├── io/
│   └── parsers/
│       └── wfn.py           # WFN parser (needs overlap matrix)
├── integrals.py             # Integral calculation functions
├── analysis/
│   ├── bonding.py           # Bond order calculations
│   └── population.py        # Population analysis
tests/
├── analysis/
│   └── test_bonding.py      # Bonding tests
```

## Key Functions

### overlap_gaussian_primitive()
Calculate overlap between two Gaussian primitives.

Parameters:
- exp1, exp2: Exponents
- coord1, coord2: Coordinates (x, y, z)
- l1, m1, n1: Angular momentum for primitive 1
- l2, m2, n2: Angular momentum for primitive 2

Returns: Overlap integral value

### WFN Parser
Located in `pymultiwfn/io/parsers/wfn.py`

Current issue: Sets `wavefunction.overlap = np.eye(num_basis)`

Need: Calculate actual overlap matrix from basis set info

## Failing Tests (Before Implementation)

```
tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg
- Issue: Wiberg != Mayer (8.2% difference)
- Cause: Overlap matrix is identity

tests/analysis/test_bonding.py::TestParameterized::test_bond_orders_in_range[h2_wavefunction]
- Issue: H-H bond order ~0.005 vs expected ~1.0
- Cause: Overlap matrix is identity

tests/analysis/test_bonding.py::TestParameterized::test_bond_orders_in_range[c2h2_wavefunction]
- Issue: C≡C bond order 3.697 vs expected 2.5-3.5
- Cause: Overlap matrix is identity
```

## Success Criteria

1. ✅ `calculate_overlap_matrix()` function implemented
2. ✅ WFN parser uses calculated overlap matrix
3. ✅ All bonding tests pass
4. ✅ Test coverage >= 80%
5. ✅ Code follows PEP 8
6. ✅ Documentation complete

## Git Commit Guidelines

- Small, atomic commits
- Use conventional commits:
  - `feat:` - new feature
  - `fix:` - bug fix
  - `test:` - test updates
  - `docs:` - documentation
  - `refactor:` - code refactoring

## Collaboration Protocol

1. Coder implements one step
2. Coder runs tests
3. Verifier reviews code and tests
4. If OK → commit, move to next step
5. If NOT OK → Coder fixes, repeat
EOF
fi

# Create IMPLEMENTATION_PLAN.md if not exists
if [ ! -f "$PLAN" ]; then
    cat > "$PLAN" << 'EOF'
# IMPLEMENTATION_PLAN.md - Overlap Matrix Calculation

**Status**: In Progress
**Last Updated**: 2026-02-19 09:27

---

## Overview

Implement overlap matrix calculation from basis set information to fix inaccurate bond order and population calculations.

---

## Implementation Steps

### Step 1: Study Existing Code (STATUS: DONE)
- [x] Review `pymultiwfn/integrals.py` for overlap functions
- [x] Understand `overlap_gaussian_primitive()` signature
- [x] Analyze WFN parser structure
- [x] Identify basis set data structure

**Learned**:
- `overlap_gaussian_primitive()` exists and works
- WFN parser has shells with exponents and coefficients
- Need to aggregate primitives into basis functions

---

### Step 2: Implement calculate_overlap_matrix() (STATUS: IN PROGRESS)
- [ ] Create function in `pymultiwfn/integrals.py`
- [ ] Implement basis function index mapping
- [ ] Calculate overlap for each primitive pair
- [ ] Aggregate to basis function level
- [ ] Add documentation and type hints

**Signature**:
```python
def calculate_overlap_matrix(wfn: Wavefunction) -> np.ndarray:
    """
    Calculate overlap matrix from basis set information.

    Parameters
    ----------
    wfn : Wavefunction
        Wavefunction object with basis set information

    Returns
    -------
    np.ndarray
        Overlap matrix (num_basis x num_basis)
    """
```

**Algorithm**:
1. Initialize num_basis x num_basis matrix
2. For each basis function i:
   a. Determine shell and primitive range
   b. For each basis function j:
      c. Determine shell and primitive range
      d. For each primitive p in i, q in j:
         e. Calculate overlap(p, q)
         f. Accumulate: S[i,j] += coeff[p] * coeff[q] * overlap(p,q)
3. Return symmetric matrix

---

### Step 3: Update WFN Parser (STATUS: PENDING)
- [ ] Import `calculate_overlap_matrix()` in wfn.py
- [ ] Replace `np.eye(num_basis)` with actual calculation
- [ ] Add optional caching (performance)
- [ ] Test with existing wavefunction files

---

### Step 4: Write Unit Tests (STATUS: PENDING)
- [ ] Test simple case (H2 with STO-3G)
- [ ] Test medium case (C2H2 with larger basis)
- [ ] Test symmetry (S == S.T)
- [ ] Test diagonal elements (S[i,i] > 0)
- [ ] Test integration (should be close to 1)

---

### Step 5: Fix Bonding Tests (STATUS: PENDING)
- [ ] Re-run `test_mayer_vs_wiberg`
- [ ] Re-run `test_bond_orders_in_range[h2]`
- [ ] Re-run `test_bond_orders_in_range[c2h2]`
- [ ] Verify all tests pass

---

### Step 6: Code Review and Documentation (STATUS: PENDING)
- [ ] Verifier reviews code
- [ ] Add docstrings
- [ ] Add type hints
- [ ] Update user documentation

---

### Step 7: Performance Optimization (STATUS: PENDING)
- [ ] Profile the function
- [ ] Implement symmetry optimization (only compute upper triangle)
- [ ] Consider parallelization
- [ ] Add caching

---

## Current Focus

**Step 2**: Implement `calculate_overlap_matrix()` function

**Next Milestone**: All bonding tests passing

**Target Date**: 2026-02-19 10:27 (1 hour from start)

---

## Progress Log

### 2026-02-19 09:27
- Started Ralph Loop
- Created PROMPT.md, AGENTS.md, IMPLEMENTATION_PLAN.md
- Step 1 completed (study existing code)
- Moving to Step 2 (implementation)

EOF
fi

# Create Ralph loop script
cat > "$WORK_DIR/ralph_loop.sh" << 'EOFSCRIPT'
#!/bin/bash
# Ralph Loop - Coder + Verifier iteration

set -e

PROJECT_DIR="$HOME/software/PyMultiWFN"
WORK_DIR="$PROJECT_DIR/nightly-development"
PROMPT="$WORK_DIR/PROMPT.md"
AGENTS="$WORK_DIR/AGENTS.md"
PLAN="$WORK_DIR/IMPLEMENTATION_PLAN.md"
ITERATION=0
MAX_ITERATIONS=10

cd "$PROJECT_DIR"

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================="
    echo "Ralph Loop Iteration $ITERATION"
    echo "========================================="
    date

    # Read current task from plan
    CURRENT_TASK=$(grep "STATUS: IN PROGRESS" "$PLAN" | head -1 | sed 's/### //' | sed 's/ (STATUS:.*//' | sed 's/Step [0-9]*: //')

    if [ -z "$CURRENT_TASK" ]; then
        echo "✅ No task in progress. All steps completed!"
        break
    fi

    echo "Current task: $CURRENT_TASK"
    echo ""

    # Load context files
    CONTEXT=$(cat "$PROMPT" "$AGENTS" "$PLAN")

    # Coder task
    CODER_TASK="$CONTEXT

You are the CODER agent.

Current task: $CURRENT_TASK

Your mission:
1. Read the implementation plan for the current step
2. Implement the code for this step
3. Run the tests (pytest tests/analysis/test_bonding.py -v)
4. If tests fail, fix the code and retry
5. If tests pass, update IMPLEMENTATION_PLAN.md:
   - Change current step STATUS: COMPLETED
   - Change next step STATUS: IN PROGRESS
   - Add entry to Progress Log

IMPORTANT:
- Small, incremental changes
- Run tests after each change
- Commit when a step is complete
- If you get stuck, add a note to the plan"

    echo "Phase 1: Coder agent..."
    export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
    export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
    export ANTHROPIC_MODEL=GLM-4.7

    echo "$CODER_TASK" | claude -

    # Check if step completed
    if grep -q "STATUS: IN PROGRESS" "$PLAN"; then
        echo ""
        echo "⚠️ Coder did not complete the step. Retrying..."
        continue
    fi

    # Verifier task
    VERIFIER_TASK="$CONTEXT

You are the VERIFIER agent.

Current step: $CURRENT_TASK

Your mission:
1. Review the code changes made by the coder
2. Run tests (pytest tests/analysis/test_bonding.py -v)
3. Run coverage (pytest tests/analysis/test_bonding.py --cov=pymultiwfn)
4. Check code quality:
   - PEP 8 compliance
   - Documentation
   - Type hints
   - Edge cases
5. If issues found, provide detailed feedback in IMPLEMENTATION_PLAN.md
6. If all good, mark as VERIFIED in the plan

IMPORTANT:
- Be thorough but constructive
- Focus on correctness and quality
- If code needs work, be specific about what needs fixing"

    echo ""
    echo "Phase 2: Verifier agent..."
    echo "$VERIFIER_TASK" | claude -

    # Check if verified
    if grep -q "VERIFIED" "$PLAN"; then
        echo ""
        echo "✅ Step verified! Proceeding to next step."
    else
        echo ""
        echo "⚠️ Verifier found issues. Coder will fix in next iteration."
    fi
done

echo ""
echo "========================================="
echo "Ralph Loop Completed"
echo "========================================="
echo "Total iterations: $ITERATION"
date
echo ""
echo "Final test results:"
pytest tests/analysis/test_bonding.py -v
EOFSCRIPT

chmod +x "$WORK_DIR/ralph_loop.sh"

echo "✅ Ralph Loop initialized"
echo ""
echo "Files created:"
echo "  - $PROMPT"
echo "  - $AGENTS"
echo "  - $PLAN"
echo "  - $WORK_DIR/ralph_loop.sh"
echo ""
echo "Starting Ralph Loop..."

# Start the loop in background
nohup bash "$WORK_DIR/ralph_loop.sh" > "$LOG_DIR/ralph_loop_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
LOOP_PID=$!

echo "Ralph Loop started (PID: $LOOP_PID)"
echo "Log: $LOG_DIR/ralph_loop_*.log"
echo ""
echo "Monitoring progress (first 10 lines)..."
sleep 5
tail -10 "$LOG_DIR"/ralph_loop_*.log 2>/dev/null || echo "Log file not ready yet..."

echo ""
echo "========================================="
echo "Ralph Loop is now running in background"
echo "To monitor: tail -f $LOG_DIR/ralph_loop_*.log"
echo "To stop: kill $LOOP_PID"
echo "========================================="
