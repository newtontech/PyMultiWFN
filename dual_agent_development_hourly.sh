#!/usr/bin/env bash
set -euo pipefail

# PyMultiWFN Dual-Agent Ralph Loop - Hourly Development
# This script implements a coder + verifier agent collaboration loop

PROJECT_DIR="$HOME/software/PyMultiWFN"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/hourly_development_$(date +%Y%m%d_%H%M).log"
CODER_TASK_FILE="$PROJECT_DIR/CODER_TASK.md"
VERIFIER_TASK_FILE="$PROJECT_DIR/VERIFIER_TASK.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

# Function to run coder agent
run_coder_agent() {
    log "=== STARTING CODER AGENT ==="

    cd "$PROJECT_DIR"

    # Set up environment
    export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
    export ANTHROPIC_AUTH_TOKEN=$CLAUDE_GLM_API_KEY
    export ANTHROPIC_MODEL=GLM-4.7

    # Read coder task
    if [ ! -f "$CODER_TASK_FILE" ]; then
        log_error "CODER_TASK.md not found!"
        return 1
    fi

    log "Reading coder task from $CODER_TASK_FILE"
    local coder_task=$(cat "$CODER_TASK_FILE")

    # Run coder agent
    log "Running coder agent..."
    claude -p "You are the Coder Agent for PyMultiWFN Ralph Loop.

Project Directory: $PROJECT_DIR

Your task:
$coder_task

Please work on this task and report back when done.

Important:
- Read the files mentioned in the task
- Make the necessary modifications
- Run tests to verify your changes
- Git commit your changes when tests pass
- Report your progress and any issues encountered" 2>&1 | tee -a "$LOG_FILE"

    log_success "=== CODER AGENT COMPLETED ==="
}

# Function to run verifier agent
run_verifier_agent() {
    log "=== STARTING VERIFIER AGENT ==="

    cd "$PROJECT_DIR"

    # Set up environment
    export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
    export ANTHROPIC_AUTH_TOKEN=$CLAUDE_GLM_API_KEY
    export ANTHROPIC_MODEL=GLM-4.7

    # Create verifier task
    cat > "$VERIFIER_TASK_FILE" << 'EOF'
你是 PyMultiWFN Ralph Loop 的 Verifier Agent。

**任务：** 验证 Coder Agent 的工作

**验证步骤：**
1. **检查 Git 变更**
   - 运行 git status 查看修改的文件
   - 运行 git diff 查看具体的代码变更
   - 确认变更符合任务要求

2. **运行完整测试套件**
   - 运行 pytest tests/ -v
   - 检查是否有失败的测试
   - 如果有失败的测试，分析失败原因

3. **检查测试覆盖率**
   - 运行 pytest --cov=pymultiwfn --cov-report=term-missing
   - 检查覆盖率是否达到 70%+
   - 识别缺失测试的模块

4. **代码审查**
   - 审查修改的代码质量
   - 检查代码风格（PEP 8）
   - 检查是否有潜在问题或边界情况
   - 验证代码是否符合最佳实践

5. **生成验证报告**
   - 总结测试结果（通过/失败）
   - 报告覆盖率情况
   - 列出发现的问题（如果有）
   - 提供改进建议（如果有）

**验证结果：**
- 如果验证通过（所有测试通过，覆盖率 >= 70%）：标记为成功
- 如果验证失败（有测试失败或覆盖率不足）：返回给 coder 修复

**输出要求：**
- Git 变更摘要
- 测试结果详情
- 覆盖率报告
- 代码审查意见
- 验证结论（通过/失败）和下一步建议
EOF

    log "Reading verifier task from $VERIFIER_TASK_FILE"
    local verifier_task=$(cat "$VERIFIER_TASK_FILE")

    # Run verifier agent
    log "Running verifier agent..."
    claude -p "You are the Verifier Agent for PyMultiWFN Ralph Loop.

Project Directory: $PROJECT_DIR

Your task:
$verifier_task

Please perform verification and report back.

Important:
- Check git changes carefully
- Run full test suite
- Verify test coverage
- Review code quality
- Provide detailed verification report" 2>&1 | tee -a "$LOG_FILE"

    log_success "=== VERIFIER AGENT COMPLETED ==="
}

# Function to check if verification passed
check_verification_status() {
    log "Checking verification status..."

    # Check if verifier marked it as successful
    if grep -q "验证通过\|VERIFICATION PASSED\|All tests passed" "$LOG_FILE"; then
        log_success "Verification passed!"
        return 0
    else
        log_warning "Verification failed or inconclusive. Need to re-run coder agent."
        return 1
    fi
}

# Main loop
MAX_ITERATIONS=3
iteration=0

while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    log "=========================================="
    log "ITERATION $iteration / $MAX_ITERATIONS"
    log "=========================================="

    # Run coder agent
    run_coder_agent
    coder_status=$?

    if [ $coder_status -ne 0 ]; then
        log_error "Coder agent failed with status $coder_status"
        break
    fi

    # Run verifier agent
    run_verifier_agent
    verifier_status=$?

    if [ $verifier_status -ne 0 ]; then
        log_error "Verifier agent failed with status $verifier_status"
        break
    fi

    # Check if verification passed
    if check_verification_status; then
        log_success "✅ Verification passed! Task completed successfully."
        break
    fi

    log "Continuing to next iteration..."
    sleep 2
done

if [ $iteration -eq $MAX_ITERATIONS ]; then
    log_warning "Maximum iterations reached. Task may not be fully completed."
fi

log "=========================================="
log "Dual-Agent Ralph Loop Completed"
log "=========================================="
log "Log file: $LOG_FILE"

# Create summary
cat > "$PROJECT_DIR/HOURLY_DEVELOPMENT_SUMMARY_$(date +%Y%m%d_%H%M).md" << EOF
# PyMultiWFN Hourly Development Summary
**Date**: $(date '+%Y-%m-%d %H:%M')
**Session**: Hourly Developer (Offset 27m)
**Mode**: Dual-Agent Ralph Loop (Coder + Verifier)

---

## 📊 Session Overview

### Session Status
- **Duration**: ~$(ps -p $$ -o etime=)
- **Agent Mode**: Coder + Verifier (Dual-Agent)
- **Iterations**: $iteration
- **Log File**: $LOG_FILE

---

## 📋 Task Description

### Original Task (from CODER_TASK.md)
$(head -20 "$CODER_TASK_FILE")

---

## ✅ Completed Work

See log file for details: $LOG_FILE

---

## 📊 Test Results

Run the following command to check test results:
\`\`\`bash
cd $PROJECT_DIR
pytest tests/ --cov=pymultiwfn --cov-report=term-missing
\`\`\`

---

## 📝 Recent Commits

\`\`\`bash
cd $PROJECT_DIR
git log --oneline -5
\`\`\`

---

## 🎯 Next Steps

1. Review verification results in log file
2. Address any issues found
3. Continue with next task in IMPLEMENTATION_PLAN.md

---

**End of Summary**
**Prepared by**: PyMultiWFN Hourly Developer (Dual-Agent Ralph Loop)
**Date**: $(date '+%Y-%m-%d %H:%M GMT+8')
EOF

log_success "Summary created: $PROJECT_DIR/HOURLY_DEVELOPMENT_SUMMARY_$(date +%Y%m%d_%H%M).md"

exit 0
