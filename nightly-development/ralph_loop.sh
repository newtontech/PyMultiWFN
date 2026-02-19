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
