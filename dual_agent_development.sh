#!/bin/bash
# PyMultiWFN Dual Agent Development Script
# Coder + Verifier collaboration mode

set -e

PROJECT_DIR="$HOME/software/PyMultiWFN"
LOG_DIR="$PROJECT_DIR/logs"
CODER_LOG="$LOG_DIR/coder_$(date +%Y%m%d_%H%M%S).log"
VERIFIER_LOG="$LOG_DIR/verifier_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="$LOG_DIR/agent_status.txt"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "PyMultiWFN Dual Agent Development"
echo "========================================="
echo "Project: $PROJECT_DIR"
echo "Coder log: $CODER_LOG"
echo "Verifier log: $VERIFIER_LOG"
echo "Status file: $STATUS_FILE"
echo "========================================="
echo ""

# Initialize status file
echo "STARTED: $(date)" > "$STATUS_FILE"
echo "STATUS: running" >> "$STATUS_FILE"

# Task for coder agent
CODER_TASK="你是 PyMultiWFN 的 coder agent。

任务：实现 overlap matrix 计算功能

背景：
- WFN 文件格式不包含 overlap matrix
- Parser 目前将 overlap matrix 设置为 identity
- 导致 bond order 和 population 计算不准确
- 现有代码有 overlap_gaussian_primitive 函数可以复用

要求：
1. 实现 calculate_overlap_matrix(wfn) 函数
2. 从基组信息计算 overlap matrix
3. 在 WFN parser 中调用它，替换 identity 矩阵
4. 添加必要的文档和注释
5. 编写单元测试

项目路径：$PROJECT_DIR

参考文件：
- pymultiwfn/io/parsers/wfn.py (WFN parser)
- pymultiwfn/integrals.py (积分计算函数)
- tests/test_bonding.py (bonding 测试)

注意：
- 使用 pytest 驱动开发（测试先行）
- 小步快跑，每步都要测试
- 遵循 PEP 8 规范
- 每次修改后运行测试：pytest tests/analysis/test_bonding.py -v

完成后将 'CODED: overlap_matrix' 写入 $STATUS_FILE"

# Task for verifier agent
VERIFIER_TASK="你是 PyMultiWFN 的 verifier agent。

任务：验证 overlap matrix 计算代码质量

背景：
- Coder agent 正在实现 overlap matrix 计算功能
- 需要验证代码的正确性、性能和可维护性

要求：
1. 检查代码是否正确计算 overlap matrix
2. 运行测试并验证测试结果
3. 进行代码审查（代码质量、可读性、性能）
4. 检查与原版 Multiwfn 的一致性（如果有）
5. 验证边界情况和错误处理
6. 提供反馈和改进建议

项目路径：$PROJECT_DIR

参考文件：
- pymultiwfn/io/parsers/wfn.py (WFN parser)
- pymultiwfn/integrals.py (积分计算函数)
- tests/test_bonding.py (bonding 测试)

注意：
- 运行测试：pytest tests/analysis/test_bonding.py -v
- 运行覆盖率：pytest tests/analysis/test_bonding.py --cov=pymultiwfn
- 如果发现问题，提供详细的错误报告
- 如果验证通过，将 'VERIFIED: overlap_matrix' 写入 $STATUS_FILE

完成后将 'VERIFIED: overlap_matrix' 写入 $STATUS_FILE"

# Function to start coder agent
start_coder() {
    echo "Starting coder agent..."
    cd "$PROJECT_DIR"
    export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
    export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
    export ANTHROPIC_MODEL=GLM-4.7

    echo "$CODER_TASK" | claude - > "$CODER_LOG" 2>&1

    # Check status
    if grep -q "CODED: overlap_matrix" "$STATUS_FILE"; then
        echo "✅ Coder agent completed successfully"
        return 0
    else
        echo "❌ Coder agent did not mark completion"
        return 1
    fi
}

# Function to start verifier agent
start_verifier() {
    echo "Starting verifier agent..."
    cd "$PROJECT_DIR"
    export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
    export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
    export ANTHROPIC_MODEL=GLM-4.7

    echo "$VERIFIER_TASK" | claude - > "$VERIFIER_LOG" 2>&1

    # Check status
    if grep -q "VERIFIED: overlap_matrix" "$STATUS_FILE"; then
        echo "✅ Verifier agent completed successfully"
        return 0
    else
        echo "❌ Verifier agent did not mark completion"
        return 1
    fi
}

# Main execution loop
echo "Phase 1: Coder agent implementing overlap matrix..."
start_coder
CODER_STATUS=$?

echo ""
echo "Phase 2: Verifier agent checking code..."
start_verifier
VERIFIER_STATUS=$?

echo ""
echo "========================================="
echo "Development Session Summary"
echo "========================================="
echo "Coder status: $([ $CODER_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')"
echo "Verifier status: $([ $VERIFIER_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')"
echo "Coder log: $CODER_LOG"
echo "Verifier log: $VERIFIER_LOG"
echo "========================================="

# Final status
if [ $CODER_STATUS -eq 0 ] && [ $VERIFIER_STATUS -eq 0 ]; then
    echo "✅ Dual agent development completed successfully"
    echo "STATUS: completed" >> "$STATUS_FILE"
    exit 0
else
    echo "❌ Dual agent development failed"
    echo "STATUS: failed" >> "$STATUS_FILE"
    exit 1
fi
