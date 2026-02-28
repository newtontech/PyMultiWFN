你是 PyMultiWFN Ralph Loop 的 Coder Agent。

**当前任务：Issue 2 - 代码质量改进（完成剩余工作）**

**进度回顾**：
- ✅ Black 格式化完成（163 files）
- ✅ 未使用导入删除完成（142 violations fixed）
- ⚠️ 剩余 73 个 F841 违规（未使用的变量）
- 📊 Issue 2 总体进度：60%

**本次任务（16:45）**：

### 目标：修复剩余 73 个 F841 违规

**违规类型**：F841 - Local variable assigned but never used

**主要分布**：
- analysis/bonding/*.py - 10 个
- analysis/density/*.py - 2 个
- analysis/population/*.py - 4 个
- analysis/spectrum/*.py - 4 个
- analysis/surface/*.py - 2 个
- io/parsers/*.py - 8 个
- 其他模块 - 43 个

**执行策略**：
1. 检查每个未使用的变量
2. 确定是否可以安全删除
3. 如果是调试代码，添加 `_` 前缀标记为有意未使用
4. 如果是未来功能的占位符，添加注释说明
5. 如果确实不需要，删除赋值

**执行步骤**：
```bash
cd ~/software/PyMultiWFN
source .venv/bin/activate

# 1. 列出所有 F841 违规
flake8 pymultiwfn/ --select=F841 --show-source

# 2. 逐个修复（手动或脚本辅助）

# 3. 验证修复
flake8 pymultiwfn/ --max-line-length=88 --extend-ignore=E203,W503 --count

# 4. 运行测试
pytest tests/ -v --tb=short -x

# 5. Git commit
git add -A
git commit -m "fix: remove unused variables (F841 violations)"
```

**成功标准**：
- [ ] Flake8 违规从 73 减少到 0（或 < 10）
- [ ] 所有测试通过
- [ ] Issue 2 完成度达到 80%+
- [ ] Git commit

**重要**：
- 每修复一批文件后运行测试
- 不要删除可能有用的调试变量（用 `_` 前缀标记）
- 小步快跑，频繁 commit

开始工作吧！
