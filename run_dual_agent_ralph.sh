#!/bin/env bash
set -euo pipefail

# PyMultiWFN Ralph Loop - Dual Agent Collaboration
# Coder + Verifier working together to fix overlap matrix

export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN=$CLAUDE_GLM_API_KEY
export ANTHROPIC_MODEL=GLM-4.7

LOG_DIR="nightly-development/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/dual_agent_ralph_$TIMESTAMP.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "PyMultiWFN Ralph Loop - Dual Agent" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd ~/software/PyMultiWFN

# Initialize iteration
ITERATION=1
MAX_ITERATIONS=8
TEST_PASSED=0

while [ $ITERATION -le $MAX_ITERATIONS ] && [ $TEST_PASSED -eq 0 ]; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Iteration $ITERATION/$MAX_ITERATIONS" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # ==================== Coder Agent ====================
    echo "" | tee -a "$LOG_FILE"
    echo ">>> CODER AGENT: Implementing fix..." | tee -a "$LOG_FILE"
    echo ">>> CODER AGENT: Implementing fix..."

    claude << 'EOF' | tee -a "$LOG_FILE"
你是 PyMultiWFN coder agent。任务：修复 overlap matrix 计算问题。

当前问题：
- 测试失败：test_mayer_vs_wiberg, test_bond_orders_in_range[h2], test_bond_orders_in_range[c2h2]
- 原因：overlap matrix 回退到 identity matrix
- 警告："Extracted 163 basis functions, but num_basis is 193. Using identity matrix as fallback."

修复目标：
1. 调查 WFN parser 中 basis function 数量的不匹配问题
2. 确保 overlap matrix 维度与 MO coefficients 维度匹配
3. 实现正确的 overlap matrix 计算（不使用 identity fallback）

项目路径：~/software/PyMultiWFN

相关文件：
- pymultiwfn/io/parsers/wfn.py (WFN loader)
- pymultiwfn/integrals/overlap.py (overlap matrix 计算)
- pymultiwfn/debug_overlap_matrix.py (debug 脚本)

请：
1. 先运行 debug 脚本了解问题：python3 pymultiwfn/debug_overlap_matrix.py
2. 分析 WFN 文件格式，找出为什么 basis function 数量不匹配
3. 修复 overlap matrix 计算逻辑
4. 运行测试验证修复：pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
5. 更新 IMPLEMENTATION_PLAN.md 记录进度

不要跳过测试验证。每次修改后立即测试。
EOF

    # ==================== Verifier Agent ====================
    echo "" | tee -a "$LOG_FILE"
    echo ">>> VERIFIER AGENT: Reviewing and validating..." | tee -a "$LOG_FILE"
    echo ">>> VERIFIER AGENT: Reviewing and validating..."

    claude << 'EOF' | tee -a "$LOG_FILE"
你是 PyMultiWFN verifier agent。任务：验证代码质量和测试结果。

验证任务：
1. 代码审查：
   - 检查 overlap matrix 计算逻辑是否正确
   - 检查维度匹配是否合理
   - 检查是否有边界情况处理
   - 检查代码风格和注释

2. 测试验证：
   - 运行完整测试套件：pytest tests/analysis/test_bonding.py -v
   - 检查是否所有测试都通过
   - 如果测试失败，提供详细的失败原因和修复建议

3. 一致性检查：
   - 验证 overlap matrix 的数学性质（对称性、正定性等）

项目路径：~/software/PyMultiWFN

验证标准：
- 所有 bonding 测试必须通过
- overlap matrix 不能回退到 identity matrix
- 键级计算必须在合理范围内

请完成验证后，报告测试结果。如果所有测试通过，说"VERIFICATION_PASSED"。如果失败，提供详细的修复建议。
EOF

    # Check if verification passed
    if grep -q "VERIFICATION_PASSED" "$LOG_FILE"; then
        echo "" | tee -a "$LOG_FILE"
        echo ">>> VERIFICATION PASSED! Committing changes..." | tee -a "$LOG_FILE"
        echo ">>> VERIFICATION PASSED! Committing changes..."

        # Git commit
        git add -A
        git commit -m "fix: resolve overlap matrix calculation in dual-agent iteration $ITERATION" | tee -a "$LOG_FILE"

        TEST_PASSED=1
    else
        echo "" | tee -a "$LOG_FILE"
        echo ">>> VERIFICATION FAILED. Continuing to next iteration..." | tee -a "$LOG_FILE"
        echo ">>> VERIFICATION FAILED. Continuing to next iteration..."

        ITERATION=$((ITERATION + 1))
    fi
done

# Summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Summary" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

if [ $TEST_PASSED -eq 1 ]; then
    echo "✅ Success! All tests passed in $ITERATION iterations." | tee -a "$LOG_FILE"
    echo "✅ Success! All tests passed in $ITERATION iterations."
else
    echo "❌ Failed to complete after $MAX_ITERATIONS iterations." | tee -a "$LOG_FILE"
    echo "❌ Failed to complete after $MAX_ITERATIONS iterations."
    echo "Check the log file for details: $LOG_FILE"
fi

echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Ended at: $(date)" | tee -a "$LOG_FILE"
