#!/bin/env bash
# PyMultiWFN Dual-Agent Ralph Loop
# Coder implements features, Verifier validates code

set -euo pipefail

# Configuration
WORKSPACE="/home/yhm/software/PyMultiWFN"
LOG_DIR="$WORKSPACE/nightly-development/logs"
LOG_FILE="$LOG_DIR/dual_agent_ralph_$(date +%Y%m%d_%H%M%S).log"
MAX_ITERATIONS=24
MAX_ITERATIONS_PER_TASK=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup
mkdir -p "$LOG_DIR"
cd "$WORKSPACE"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Helper: Get current task from implementation plan
get_current_task() {
    grep -A 5 "Step 4:" "$WORKSPACE/nightly-development/IMPLEMENTATION_PLAN.md" | grep -E "^\s*-\s" | head -1
}

# Helper: Check if all tasks are complete
check_completion() {
    if grep -q "STATUS: COMPLETE" "$WORKSPACE/nightly-development/IMPLEMENTATION_PLAN.md" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Helper: Run tests
run_tests() {
    log "${BLUE}Running tests...${NC}"
    pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v --tb=short | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

# Main loop
log "${GREEN}=== PyMultiWFN Dual-Agent Ralph Loop Started ===${NC}"
log "Time: $(date)"
log "Max iterations: $MAX_ITERATIONS"
log "Log file: $LOG_FILE"
log ""

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    log "${BLUE}=== Iteration $iteration/$MAX_ITERATIONS ===${NC}"
    log ""

    # Check for completion
    if check_completion; then
        log "${GREEN}All tasks completed!${NC}"
        break
    fi

    # Get current task
    current_task=$(get_current_task)
    log "${YELLOW}Current Task:${NC} $current_task"
    log ""

    # Phase 1: Coder Agent - Implement Feature
    log "${BLUE}=== Phase 1: Coder Agent ===${NC}"
    log "${BLUE}Task: Implement/fix code based on IMPLEMENTATION_PLAN.md${NC}"
    log ""

    coder_task=$(cat <<'EOF'
You are the Coder Agent for PyMultiWFN Ralph Loop development.

Context:
- Read IMPLEMENTATION_PLAN.md for the current task
- Read nightly-development/PROMPT.md for project context
- Read AGENTS.md for testing and project structure

Your role:
1. Read and understand the current task from IMPLEMENTATION_PLAN.md
2. Investigate the relevant code (don't assume things are missing)
3. Implement the required changes
4. Run the test command: pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
5. If tests pass, update IMPLEMENTATION_PLAN.md and commit
6. If tests fail, debug and fix

Current Focus:
Debug overlap matrix calculation to fix bond order tests.

Key Issue:
- Overlap matrix calculation exists but tests still fail
- Basis function count mismatch: 34 (MO) vs 48 (shells)
- Max relative difference: 8.18%

Instructions:
- Small, atomic changes
- Run tests after each change
- Document what you did in IMPLEMENTATION_PLAN.md
- Commit with conventional commit messages

Stop when:
- All bonding tests pass, or
- You need verifier review
EOF
)

    # Spawn coder agent
    log "${BLUE}Spawning Coder Agent...${NC}"
    coder_output=$(sessions_spawn \
        --task "$coder_task" \
        --label "PyMultiWFN-Coder-Iter$iteration" \
        --agentId coding-agent \
        --timeoutSeconds 900 \
        2>&1)

    echo "$coder_output" | tee -a "$LOG_FILE"

    # Check if coder agent failed
    if echo "$coder_output" | grep -q "error\|Error\|ERROR" || ! run_tests; then
        log "${YELLOW}Coder completed with issues. Running verifier...${NC}"
    else
        log "${GREEN}Coder completed successfully! Running verifier...${NC}"
    fi
    log ""

    # Phase 2: Verifier Agent - Validate Code
    log "${BLUE}=== Phase 2: Verifier Agent ===${NC}"
    log "${BLUE}Task: Review code and validate tests${NC}"
    log ""

    verifier_task=$(cat <<'EOF'
You are the Verifier Agent for PyMultiWFN Ralph Loop development.

Context:
- Review the changes made by the Coder Agent
- Read IMPLEMENTATION_PLAN.md for current status
- Read AGENTS.md for testing requirements

Your role:
1. Review the code changes (git diff HEAD~1)
2. Run comprehensive tests: pytest tests/analysis/test_bonding.py -v
3. Check code quality (PEP 8, type hints, docstrings)
4. Verify implementation matches requirements
5. Check if implementation_plan.md is updated correctly

Validation Criteria:
- All bonding tests pass (especially test_mayer_vs_wiberg)
- Code follows best practices
- Documentation is complete
- No regressions in other tests

Outcome:
- If valid: Mark as APPROVED in IMPLEMENTATION_PLAN.md
- If invalid: Specify what needs to be fixed

Stop when:
- Code is validated and tests pass, or
- You have specific feedback for coder
EOF
)

    # Spawn verifier agent
    log "${BLUE}Spawning Verifier Agent...${NC}"
    verifier_output=$(sessions_spawn \
        --task "$verifier_task" \
        --label "PyMultiWFN-Verifier-Iter$iteration" \
        --agentId coding-agent \
        --timeoutSeconds 600 \
        2>&1)

    echo "$verifier_output" | tee -a "$LOG_FILE"

    # Check verifier outcome
    if echo "$verifier_output" | grep -qi "approved\|valid\|success"; then
        log "${GREEN}✅ Verifier approved changes!${NC}"
        log "${GREEN}Moving to next task...${NC}"
        echo ""
    else
        log "${YELLOW}⚠️  Verifier requested changes. Coder will address in next iteration.${NC}"
        echo ""
    fi

    # Check for overall completion
    if run_tests; then
        log "${GREEN}✅ All tests passing!${NC}"
        break
    fi

    # Small delay between iterations
    sleep 5
done

# Summary
log ""
log "${GREEN}=== Ralph Loop Summary ===${NC}"
log "Total iterations: $iteration"
log "Log file: $LOG_FILE"
log "Git status:" | tee -a "$LOG_FILE"
git status --short | tee -a "$LOG_FILE"
log ""
log "Recent commits:" | tee -a "$LOG_FILE"
git log --oneline -5 | tee -a "$LOG_FILE"
log ""

# Final test run
log "${BLUE}Final test run...${NC}"
pytest tests/analysis/test_bonding.py -v --tb=short | tee -a "$LOG_FILE"

log ""
log "${GREEN}=== PyMultiWFN Dual-Agent Ralph Loop Completed ===${NC}"
log "Time: $(date)"
