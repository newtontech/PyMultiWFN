#!/usr/bin/env bash
# PyMultiWFN Hourly Development - Direct Coder Execution

PROJECT_DIR="$HOME/software/PyMultiWFN"
CODER_TASK_FILE="$PROJECT_DIR/CODER_TASK.md"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PyMultiWFN Hourly Development${NC}"
echo -e "${BLUE}Time: $(date '+%Y-%m-%d %H:%M')${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$PROJECT_DIR"

# Set up environment
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN=$CLAUDE_GLM_API_KEY
export ANTHROPIC_MODEL=GLM-4.7

echo -e "${GREEN}Starting Coder Agent...${NC}"
echo ""

# Read coder task and run claude
cat "$CODER_TASK_FILE" | claude

echo ""
echo -e "${GREEN}Coder Agent completed.${NC}"
echo -e "${BLUE}========================================${NC}"
