#!/usr/bin/env bash
# PyMultiWFN Hourly Ralph Loop - Dual Agents (Coder + Verifier)
# Runs every hour at XX:27

set -euo pipefail

# Project path
PROJECT_DIR="$HOME/software/PyMultiWFN"
cd "$PROJECT_DIR" || exit 1

# Timestamp
TIMESTAMP=$(date "+%Y-%m-%d_%H%M")
LOG_FILE="logs/ralph_loop_hourly_${TIMESTAMP}.log"
WORK_SUMMARY="WORK_SUMMARY_${TIMESTAMP}.md"

# Create logs directory
mkdir -p logs

echo "=====================================" | tee "$LOG_FILE"
echo "PyMultiWFN Hourly Ralph Loop" | tee -a "$LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"

# Setup environment
source "$HOME/.bashrc" || true
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN="$CLAUDE_GLM_API_KEY"
export ANTHROPIC_MODEL=GLM-4.7

# ============================================
# CODER AGENT - Implement new features
# ============================================
echo "" | tee -a "$LOG_FILE"
echo "=== CODER AGENT: Creating debug script ===" | tee -a "$LOG_FILE"

CODER_TASK="You are the Coder Agent for PyMultiWFN project.

**Task**: Create a debug script to check overlap matrix properties.

**Project path**: $PROJECT_DIR

**Current problem**:
- Mayer vs Wiberg test still failing
- Max relative difference = 8.18%
- Overlap matrix calculation is implemented but may be incorrect

**Create this file**: pymultiwfn/debug_overlap_matrix.py

**Requirements**:
1. Load a WFN file (e.g., tests/data/H2.wfn)
2. Calculate overlap matrix using calculate_overlap_matrix()
3. Check these properties:
   - Symmetry: S == S.T (tolerance: 1e-10)
   - Positivity: all diagonal elements S[i,i] > 0
   - Integration: trace(S) should be close to number of electrons
   - Range: S[i,j] should be in [0, 1]
4. Print detailed statistics:
   - Matrix size
   - Max/min values
   - Symmetry check result
   - Diagonal element check result
   - Trace value
   - First 5x5 submatrix (for visualization)

**Reference files**:
- pymultiwfn/integrals/overlap.py - overlap matrix calculation
- pymultiwfn/io/parsers/wfn.py - WFN parser
- tests/data/H2.wfn - test data

**Expected output**:
\`\`\`
Overlap Matrix Debug Report
================================
Matrix size: 34x34
Max value: 0.9999
Min value: 0.0001
Symmetry: ✅ PASS (max diff = 1.2e-12)
Diagonal elements: ✅ PASS (all > 0)
Trace: 2.0001 (expected: ~2 electrons)

First 5x5 submatrix:
[[0.9999 0.1234 0.0567 ...]
 [0.1234 0.9998 0.2345 ...]
 [0.0567 0.2345 0.9997 ...]
 ...
]
\`\`\`

**Notes**:
1. Use try-except to handle errors
2. Use numpy for matrix operations
3. Add detailed comments
4. Make the script runnable: python pymultiwfn/debug_overlap_matrix.py

**Completion**: Report '✅ Created debug_overlap_matrix.py script' when done."

# Run Coder Agent
echo "Running Coder Agent..." | tee -a "$LOG_FILE"
CODER_OUTPUT=$(claude -p "$CODER_TASK" 2>&1 || true)
echo "$CODER_OUTPUT" | tee -a "$LOG_FILE"

# Check if coder succeeded
if ! echo "$CODER_OUTPUT" | grep -q "✅ Created debug_overlap_matrix.py script"; then
    echo "❌ Coder Agent failed!" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "=== SUMMARY ===" | tee -a "$LOG_FILE"
    echo "Status: FAILED" | tee -a "$LOG_FILE"
    echo "Error: Coder Agent did not complete successfully" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Coder Agent completed successfully" | tee -a "$LOG_FILE"

# ============================================
# VERIFIER AGENT - Run tests and verify
# ============================================
echo "" | tee -a "$LOG_FILE"
echo "=== VERIFIER AGENT: Running tests ===" | tee -a "$LOG_FILE"

VERIFIER_TASK="You are the Verifier Agent for PyMultiWFN project.

**Task**: Verify the debug script and run tests.

**Project path**: $PROJECT_DIR

**Steps**:
1. Check if pymultiwfn/debug_overlap_matrix.py exists
2. Run the debug script: python pymultiwfn/debug_overlap_matrix.py
3. Analyze the output and report:
   - Are all checks passing?
   - Is the overlap matrix symmetric?
   - Are diagonal elements positive?
   - Is trace reasonable?
   - Any anomalies or issues?
4. If issues found, suggest fixes
5. Run the main test: pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
6. Report test results

**Completion**: Report test results and any issues found."

# Run Verifier Agent
echo "Running Verifier Agent..." | tee -a "$LOG_FILE"
VERIFIER_OUTPUT=$(claude -p "$VERIFIER_TASK" 2>&1 || true)
echo "$VERIFIER_OUTPUT" | tee -a "$LOG_FILE"

# Check if verifier found issues
if echo "$VERIFIER_OUTPUT" | grep -q "❌" || echo "$VERIFIER_OUTPUT" | grep -q "FAILED"; then
    echo "⚠️ Verifier found issues!" | tee -a "$LOG_FILE"

    # If issues found, ask coder to fix
    echo "" | tee -a "$LOG_FILE"
    echo "=== CODER AGENT (Round 2): Fixing issues ===" | tee -a "$LOG_FILE"

    FIX_TASK="You are the Coder Agent for PyMultiWFN project.

**Fix the issues reported by Verifier**:

$VERIFIER_OUTPUT

**Project path**: $PROJECT_DIR

**Action**:
1. Read the verifier's report
2. Fix the issues found
3. Run tests to verify fixes
4. Git commit if successful

**Completion**: Report '✅ Fixed all issues' when done."

    FIX_OUTPUT=$(claude -p "$FIX_TASK" 2>&1 || true)
    echo "$FIX_OUTPUT" | tee -a "$LOG_FILE"

    if ! echo "$FIX_OUTPUT" | grep -q "✅ Fixed all issues"; then
        echo "❌ Failed to fix issues!" | tee -a "$LOG_FILE"
    fi
else
    echo "✅ Verifier: All checks passed!" | tee -a "$LOG_FILE"
fi

# ============================================
# FINAL SUMMARY
# ============================================
echo "" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"
echo "RALPH LOOP SUMMARY" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Duration: $SECONDS seconds" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Files created/modified:" | tee -a "$LOG_FILE"
git status --short | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Git commits:" | tee -a "$LOG_FILE"
git log --oneline -5 | tee -a "$LOG_FILE"

# Create work summary
cat > "$WORK_SUMMARY" << EOF
# PyMultiWFN Ralph Loop 开发报告

**日期**: $(date '+%Y-%m-%d')
**运行时间**: $(date '+%H:%M'GMT+8)
**开发者**: Coder + Verifier 双 Agents

---

## 🎯 本次开发目标

创建 debug 脚本检查 overlap matrix 的性质，修复 Mayer vs Wiberg 测试失败问题。

---

## 📊 执行情况

### 1. Coder Agent 工作进展

创建的文件：
- \`pymultiwfn/debug_overlap_matrix.py\`

功能：
- 加载 WFN 文件并计算 overlap matrix
- 检查对称性、正定性、积分
- 打印详细统计信息

---

### 2. Verifier Agent 验证结果

测试命令：
\`\`\`bash
python pymultiwfn/debug_overlap_matrix.py
pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
\`\`\`

结果：
待报告...

---

## 🔧 技术挑战

### 1. Overlap Matrix 性质验证 ⚠️

需要检查的性质：
- 对称性：S == S.T
- 正定性：S[i,i] > 0
- 积分：trace(S) ≈ num_electrons
- 范围：S[i,j] ∈ [0, 1]

---

## 📝 下一步计划

1. 根据 debug 脚本输出分析 overlap matrix 问题
2. 修复计算错误或基函数索引问题
3. 验证所有测试通过
4. Git commit 修改

---

## 📈 代码统计

**修改的文件**:
待统计...

---

## 🏆 总结

本次开发：
- ✅ 创建了 overlap matrix debug 脚本
- ⏳ 待验证测试结果
- ⏳ 待分析问题原因

---

**开发者**: PyMultiWFN Ralph Loop (Coder + Verifier)
**运行时长**: ~1 分钟
**状态**: 🔄 进行中
EOF

echo "" | tee -a "$LOG_FILE"
echo "✅ Work summary created: $WORK_SUMMARY" | tee -a "$LOG_FILE"

echo "=====================================" | tee -a "$LOG_FILE"
echo "RALPH LOOP COMPLETED" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"

# Send notification (optional)
if command -v openclaw &> /dev/null; then
    # Use OpenClaw message tool to send notification
    echo "Notification would be sent here"
fi

exit 0
