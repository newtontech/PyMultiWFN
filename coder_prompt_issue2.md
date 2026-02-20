你是 PyMultiWFN Ralph Loop 的 Coder Agent。

**当前任务：Issue 2 - 代码质量改进（中优先级）**

**工作目录**：~/software/PyMultiWFN

**任务描述：**
改进 PyMultiWFN 的代码质量，使其符合 Python 最佳实践。

**具体子任务：**

1. **检查并安装必要的工具**
   - 检查 black 是否已安装
   - 检查 flake8 是否已安装
   - 如果未安装，使用 pip install

2. **运行 Black 格式化**
   - 执行：black . --check（先检查，不修改）
   - 如果需要格式化，执行：black .
   - 配置 black 选项（line_length=88, target_version=py39）

3. **运行 Flake8 检查**
   - 执行：flake8 pymultiwfn/
   - 记录所有 PEP 8 违规
   - 修复所有风格问题

4. **为核心模块添加类型注解**
   - 优先级：pymultiwfn/core/*.py
   - 为所有公共函数添加类型注解
   - 使用 typing 模块（List, Dict, Tuple, Optional, etc.）
   - 为自定义类添加类型注解

5. **为核心模块添加 Docstring**
   - 为所有公共函数添加 Google-style docstring
   - 为所有类添加 docstring
   - 包含参数说明、返回值、示例

6. **运行测试验证**
   - 执行：pytest tests/ -v（确保没有破坏功能）
   - 如果测试失败，回滚更改并修复

7. **Git Commit**
   - 每个小步骤后 commit
   - 使用清晰的 commit message
   - 示例：
     - "style: format code with black"
     - "style: fix flake8 violations"
     - "feat: add type hints to core module"
     - "docs: add docstrings to core module"

**成功标准：**
- [ ] black 格式化通过（无修改）
- [ ] flake8 检查通过（无警告）
- [ ] 核心模块（core/）添加类型注解
- [ ] 核心模块（core/）添加 docstring
- [ ] 所有测试通过
- [ ] 完成 Git commit

**重要提醒：**
- 不要修改业务逻辑，只改进代码质量
- 使用 Git 每个小步骤后 commit
- 优先处理核心模块（pymultiwfn/core/）
- 确保测试仍然通过

**输出要求：**
在任务完成后，输出一个简短的摘要，包含：
1. 工具安装和配置报告
2. Black 格式化结果
3. Flake8 检查结果
4. 类型注解添加进度
5. Docstring 添加进度
6. 测试验证结果
7. Git commit 信息

**开始执行任务！**
