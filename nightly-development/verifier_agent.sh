#!/bin/env bash
# Verifier Agent - Validate code for PyMultiWFN

set -euo pipefail

WORKSPACE="/home/yhm/software/PyMultiWFN"
cd "$WORKSPACE"

# Read the task from stdin or use default
TASK="${1:-$(cat <<'EOF'
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
- If valid: Report "APPROVED" at the end
- If invalid: Specify what needs to be fixed

Stop when:
- Code is validated and tests pass, or
- You have specific feedback for coder
EOF
)}"

# Set up environment for claude_glm
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
export ANTHROPIC_MODEL=GLM-4.7

echo "=== Verifier Agent Started ==="
echo "Time: $(date)"
echo "Workspace: $WORKSPACE"
echo ""

# Run claude with the task
claude -p "$TASK"

echo ""
echo "=== Verifier Agent Completed ==="
echo "Time: $(date)"
