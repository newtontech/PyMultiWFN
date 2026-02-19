#!/bin/bash
# Ralph Loop - Coder + Verifier iteration (fixed)

set -e

PROJECT_DIR="$HOME/software/PyMultiWFN"
WORK_DIR="$PROJECT_DIR/nightly-development"
PROMPT="$WORK_DIR/PROMPT.md"
AGENTS="$WORK_DIR/AGENTS.md"
PLAN="$WORK_DIR/IMPLEMENTATION_PLAN.md"
ITERATION=0
MAX_ITERATIONS=10

# Set environment for Claude CLI
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
export ANTHROPIC_MODEL=GLM-4.7

cd "$PROJECT_DIR"

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================="
    echo "Ralph Loop Iteration $ITERATION"
    echo "========================================="
    date

    # Read current task from plan (more robust parsing)
    CURRENT_STEP_LINE=$(grep "STATUS: IN PROGRESS" "$PLAN" | head -1)
    CURRENT_TASK=$(echo "$CURRENT_STEP_LINE" | sed 's/^### //' | sed 's/ (STATUS:.*//')

    if [ -z "$CURRENT_TASK" ]; then
        echo "✅ No task in progress. All steps completed!"
        break
    fi

    echo "Current step: $CURRENT_TASK"
    echo "Full line: $CURRENT_STEP_LINE"
    echo ""

    # Coder task
    cat > /tmp/coder_task_$$.txt << EOF
Current Step: $CURRENT_TASK

Full Step Line: $CURRENT_STEP_LINE

You are the CODER agent in PyMultiWFN Ralph Loop.

Your mission:
1. Read the implementation plan in $PLAN
2. Implement the code for this step (small, incremental changes)
3. Run the tests immediately after each change:
   pytest tests/analysis/test_bonding.py -v --tb=short
4. If tests fail, fix the code and retry
5. If tests pass, update IMPLEMENTATION_PLAN.md:
   - Change current step STATUS from IN PROGRESS to COMPLETED
   - Change next step STATUS from PENDING to IN PROGRESS
   - Add entry to Progress Log with date
6. Commit the changes:
   git add .
   git commit -m "feat: $(echo $CURRENT_TASK | head -c 50)"

IMPORTANT:
- Small, incremental changes (one function at a time)
- Run tests after each change
- Commit when a step is complete
- If you get stuck, add a note to the plan
- Use Python type hints and docstrings
- Follow PEP 8 style

Plan file: $PLAN
EOF

    echo "Phase 1: Coder agent..."

    cat /tmp/coder_task_$$.txt | claude -p -

    # Check if step completed
    if grep -q "STATUS: IN PROGRESS" "$PLAN"; then
        echo ""
        echo "⚠️ Coder did not complete the step. Retrying..."
        continue
    fi

    # Verifier task
    cat > /tmp/verifier_task_$$.txt << EOF
Previous Step: $CURRENT_TASK

You are the VERIFIER agent in PyMultiWFN Ralph Loop.

Your mission:
1. Review the code changes made by the coder
2. Check git log to see what was changed:
   git log -1 --stat
3. Run tests:
   pytest tests/analysis/test_bonding.py -v --tb=short
4. Run coverage:
   pytest tests/analysis/test_bonding.py --cov=pymultiwfn --cov-report=term-missing
5. Check code quality:
   - PEP 8 compliance
   - Documentation (docstrings)
   - Type hints
   - Edge cases
6. If issues found, provide detailed feedback and add to $PLAN
7. If all good, add 'VERIFIED' to the step in the plan

IMPORTANT:
- Be thorough but constructive
- Focus on correctness and quality
- If code needs work, be specific about what needs fixing
EOF

    echo ""
    echo "Phase 2: Verifier agent..."

    cat /tmp/verifier_task_$$.txt | claude -p -

    # Clean up
    rm -f /tmp/coder_task_$$.txt /tmp/verifier_task_$$.txt

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
echo ""
echo "Git history:"
git log --oneline -5
