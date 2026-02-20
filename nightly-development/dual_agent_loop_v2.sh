#!/bin/env bash
# PyMultiWFN Dual-Agent Ralph Loop v2
# Coder implements features, Verifier validates code

set -euo pipefail

# Configuration
WORKSPACE="/home/yhm/software/PyMultiWFN"
LOG_DIR="$WORKSPACE/nightly-development/logs"
LOG_FILE="$LOG_DIR/dual_agent_ralph_$(date +%Y%m%d_%H%M%S).log"
MAX_ITERATIONS=24

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

# Helper: Check if all tests pass
check_tests() {
    pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v --tb=line 2>&1 | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

# Main loop
log "${GREEN}=== PyMultiWFN Dual-Agent Ralph Loop v2 Started ===${NC}"
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
    if grep -q "STATUS: COMPLETE" "$WORKSPACE/nightly-development/IMPLEMENTATION_PLAN.md" 2>/dev/null; then
        log "${GREEN}All tasks completed!${NC}"
        break
    fi

    # Phase 1: Coder Agent
    log "${BLUE}=== Phase 1: Coder Agent ===${NC}"
    log "${BLUE}Task: Implement/fix code${NC}"
    log ""

    coder_start=$(date +%s)
    ./nightly-development/coder_agent.sh 2>&1 | tee -a "$LOG_FILE"
    coder_end=$(date +%s)
    coder_duration=$((coder_end - coder_start))

    log "${BLUE}Coder completed in ${coder_duration}s${NC}"
    log ""

    # Phase 2: Verifier Agent
    log "${BLUE}=== Phase 2: Verifier Agent ===${NC}"
    log "${BLUE}Task: Validate code and tests${NC}"
    log ""

    verifier_start=$(date +%s)
    ./nightly-development/verifier_agent.sh 2>&1 | tee -a "$LOG_FILE"
    verifier_end=$(date +%s)
    verifier_duration=$((verifier_end - verifier_start))

    log "${BLUE}Verifier completed in ${verifier_duration}s${NC}"
    log ""

    # Check if all tests pass
    log "${BLUE}Checking test status...${NC}"
    if check_tests; then
        log "${GREEN}✅ All tests passing!${NC}"
        log "${GREEN}Development cycle complete!${NC}"
        break
    else
        log "${YELLOW}⚠️  Tests still failing. Continuing to next iteration...${NC}"
    fi

    log ""
    log "=== Summary ==="
    log "Coder time: ${coder_duration}s"
    log "Verifier time: ${verifier_duration}s"
    log "Total time: $((coder_duration + verifier_duration))s"
    log ""

    # Small delay between iterations
    sleep 10
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
