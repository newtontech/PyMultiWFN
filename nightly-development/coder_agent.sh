#!/bin/env bash
# Coder Agent - Implement features for PyMultiWFN

set -euo pipefail

WORKSPACE="/home/yhm/software/PyMultiWFN"
cd "$WORKSPACE"

# Read the task from stdin or use default
TASK="${1:-$(cat <<'EOF'
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
- You have completed the current task
EOF
)}"

# Set up environment for claude_glm
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
export ANTHROPIC_MODEL=GLM-4.7

echo "=== Coder Agent Started ==="
echo "Time: $(date)"
echo "Workspace: $WORKSPACE"
echo ""

# Run claude with the task
claude -p "$TASK"

echo ""
echo "=== Coder Agent Completed ==="
echo "Time: $(date)"
