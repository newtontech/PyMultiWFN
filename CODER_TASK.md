你是 PyMultiWFN Ralph Loop 的 Coder Agent。

**当前任务：继续 Issue 2 - 代码质量改进 + 选择下一个高优先级任务**

**上次进展（10:27）**：
- ✅ 修复了 pymultiwfn/analysis/density_grid.py 的语法错误
- ✅ 运行了 black . 格式化所有代码（160 个文件被修改）
- ⏳ 但还没有 commit
- ⏳ 还没有完成类型注解和 docstring

**本次任务（11:27）**：

### 第一部分：完成上次未完成的工作
1. **提交 black 格式化结果**
   - git add -A
   - git commit -m "style: apply black formatting to all Python files"

### 第二部分：继续 Issue 2 - 代码质量改进
2. **运行 flake8 检查代码风格**
   - source .venv/bin/activate
   - flake8 pymultiwfn/ --max-line-length=88 --extend-ignore=E203,W503
   - 记录发现的代码风格问题

3. **为核心模块添加类型注解（优先级：high）**
   - pymultiwfn/core/data.py - Wavefunction 数据结构
   - pymultiwfn/core/definitions.py - 核心定义
   - pymultiwfn/io/parsers/*.py - IO 模块

4. **为核心模块添加 docstring（如果缺失）**
   - 检查核心模块的 docstring 覆盖率
   - 为缺失 docstring 的函数添加 Google-style docstring

5. **运行测试验证**
   - pytest tests/ -v --tb=short
   - 确保没有破坏功能

6. **Git commit 每个改进**

### 第三部分：如果时间允许，开始新的高优先级任务
7. **Issue 3 - 性能优化（high priority）**
   - 优化电子密度计算
   - 实现并行化（使用 numba 或 multiprocessing）

**执行要求：**
1. 工作目录：~/software/PyMultiWFN
2. 环境变量：使用 Python 虚拟环境（.venv）
3. 思考模式：medium
4. 每个子任务完成后立即验证并 commit

**输出要求：**
- Git commit 记录
- Flake8 检查报告
- 类型注解添加进度
- 测试验证结果
- 下一步计划

开始工作吧！
