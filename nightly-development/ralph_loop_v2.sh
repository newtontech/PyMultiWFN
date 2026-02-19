#!/bin/bash
# Ralph Loop - Coder + Verifier iteration (with proper env vars)

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

    # Read current task from plan
    CURRENT_TASK=$(grep "STATUS: IN PROGRESS" "$PLAN" | head -1 | sed 's/### //' | sed 's/ (STATUS:.*//' | sed 's/Step [0-9]*: //')

    if [ -z "$CURRENT_TASK" ]; then
        echo "✅ No task in progress. All steps completed!"
        break
    fi

    echo "Current task: $CURRENT_TASK"
    echo ""

    # Coder task
    CODER_TASK="$CURRENT_TASK

You are the CODER agent in PyMultiWFN Ralph Loop.

Your mission:
1. Read the implementation plan in $PLAN
2. Implement the code for this step (small, incremental changes)
3. Run the tests immediately after each change:
   pytest tests/analysis/test_bonding.py -v --tb=short
4. If tests fail, fix the code and retry
5. If tests pass, update IMPLEMENTATION_PLAN.md:
   - Change current step STATUS: COMPLETED
   - Change next step STATUS: IN PROGRESS
   - Add entry to Progress Log with date
6. Commit the changes:
   git add .
   git commit -m "feat: <step description>"

IMPORTANT:
- Small, incremental changes (one function at a time)
- Run tests after each change
- Commit when a step is complete
- If you get stuck, add a note to the plan
- Use Python type hints and docstrings
- Follow PEP 8 style"

    echo "Phase 1: Coder agent..."

    echo "$CODER_TASK" | claude -p -

    # Check if step completed
    if grep -q "STATUS: IN PROGRESS" "$PLAN"; then
        echo ""
        echo "⚠️ Coder did not complete the step. Retrying..."
        continue
    fi

    # Verifier task
    VERIFIER_TASK="You are the VERIFIER agent in PyMultiWFN Ralph Loop.

Previous step: $CURRENT_TASK

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
- If code needs work, be specific about what needs fixing"

    echo ""
    echo "Phase 2: Verifier agent..."

    echo "$VERIFIER_TASK" | claude -p -

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
