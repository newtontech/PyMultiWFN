# PyMultiWFN 测试框架使用指南

> 本文是中文扩展指南，保留更详细的本地开发命令和历史性能说明。
> 标准测试入口和分类规则以 [`TESTING.md`](TESTING.md) 为准。

## 📋 概述

PyMultiWFN 使用 pytest 作为测试框架，配置了以下高级功能：

- **并行测试**（pytest-xdist）- 多核加速
- **覆盖率报告**（pytest-cov）- 代码覆盖度分析
- **测试超时**（pytest-timeout）- 防止长时间运行
- **重试机制**（pytest-rerunfailures）- 处理不稳定的测试

## 🚀 快速开始

### 运行所有测试

```bash
# 使用虚拟环境中的 pytest
.venv/bin/python -m pytest
```

### 运行特定测试文件

```bash
.venv/bin/python -m pytest tests/math/test_density.py
```

### 运行特定测试函数

```bash
.venv/bin/python -m pytest tests/math/test_density.py::TestMakeDensityMatrix::test_single_occupied_orbital
```

## ⚙️ 高级用法

### 并行测试（多核加速）

```bash
# 自动检测 CPU 核心数
.venv/bin/python -m pytest -n auto

# 指定 worker 数量
.venv/bin/python -m pytest -n 4

# 结合覆盖率
.venv/bin/python -m pytest -n auto --cov=pymultiwfn
```

**性能对比：**
- 单线程：~5.2s（46 tests）
- 2 workers：~2.8s（46 tests）
- 4 workers：~1.4s（46 tests）

### 覆盖率报告

```bash
# 终端报告（显示缺失行）
.venv/bin/python -m pytest --cov=pymultiwfn --cov-report=term-missing

# HTML 报告（可视化）
.venv/bin/python -m pytest --cov=pymultiwfn --cov-report=html
# 查看报告：firefox htmlcov/index.html

# XML 报告（CI/CD）
.venv/bin/python -m pytest --cov=pymultiwfn --cov-report=xml:coverage.xml
```

**覆盖率目标：**
- 核心模块（density, gradient）：> 90%
- 数学模块：> 80%
- I/O 模块：> 70%
- 可视化模块：> 50%

### 测试超时

```bash
# 使用默认超时（10 分钟）
.venv/bin/python -m pytest --timeout=600

# 设置特定超时
.venv/bin/python -m pytest --timeout=300  # 5 分钟
```

### 重试不稳定测试

```bash
# 失败后重试 2 次
.venv/bin/python -m pytest --reruns 2

# 仅重试特定异常
.venv/bin/python -m pytest --reruns 2 --only-rerun AssertionError

# 延迟重试
.venv/bin/python -m pytest --reruns 2 --reruns-delay 5
```

## 🏷️ 测试标记（Markers）

### 定义的标记

- `@pytest.mark.unit` - 单元测试（快速，隔离）
- `@pytest.mark.integration` - 集成测试（需要外部资源）
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.requires_data` - 需要测试数据文件
- `@pytest.mark.benchmark` - 性能基准测试
- `@pytest.mark.expensive` - 需要大量计算资源

### 使用标记

```bash
# 只运行单元测试
.venv/bin/python -m pytest -m unit

# 跳过慢速测试
.venv/bin/python -m pytest -m "not slow"

# 运行所有集成测试
.venv/bin/python -m pytest -m integration

# 组合标记
.venv/bin/python -m pytest -m "unit and not slow"
```

## 📊 测试输出

### 详细输出

```bash
# 显示局部变量
.venv/bin/python -m pytest -l

# 显示所有测试摘要
.venv/bin/python -m pytest -ra

# 更详细的回溯信息
.venv/bin/python -m pytest -vv
```

### 失败时停止

```bash
# 第一个失败后停止
.venv/bin/python -m pytest -x

# N 个失败后停止
.venv/bin/python -m pytest --maxfail=3
```

## 🔍 调试测试

### 进入调试器

```bash
# 失败时进入 PDB
.venv/bin/python -m pytest --pdb

# 第一个失败时进入
.venv/bin/python -m pytest -x --pdb

# 使用 ipdb（如果已安装）
.venv/bin/python -m pytest --pdbcls=IPython.terminal.debugger:TerminalPdb
```

### 打印输出

```bash
# 捕获并显示打印输出
.venv/bin/python -m pytest -s

# 捕获并显示日志
.venv/bin/python -m pytest --log-cli-level=INFO
```

## 🎯 常用命令组合

### 快速验证（跳过慢速测试）

```bash
.venv/bin/python -m pytest -m "not slow" -n auto --tb=short
```

### 完整测试套件（带覆盖率）

```bash
.venv/bin/python -m pytest -n auto --cov=pymultiwfn --cov-report=html
```

### CI/CD 管道（快速 + XML 覆盖率）

```bash
.venv/bin/python -m pytest -n auto --cov=pymultiwfn --cov-report=xml --junitxml=test-results.xml
```

### 开发模式（快速反馈）

```bash
.venv/bin/python -m pytest -f --tb=short -q
# -f: 失败后立即停止
# -q: 安静模式
# --tb=short: 简短的回溯
```

## 📁 测试结构

```
tests/
├── unit/           # 单元测试
│   ├── core/
│   ├── io/
│   └── math/
├── integration/    # 集成测试
│   ├── analysis/
│   └── workflows/
├── fixtures/       # 测试数据
│   ├── wfn/
│   ├── cube/
│   └── xyz/
├── conftest.py     # 共享 fixtures
└── pytest.ini      # pytest 配置
```

## 🧪 编写测试

### 基本测试

```python
import pytest
from pymultiwfn.math.density import calculate_density

def test_simple_calculation():
    """简单计算测试"""
    result = calculate_density(...)
    assert result == expected_value
```

### 使用标记

```python
@pytest.mark.unit
def test_fast_unit_test():
    """快速单元测试"""
    pass

@pytest.mark.slow
def test_slow_integration_test():
    """慢速集成测试"""
    pass
```

### 使用 Fixtures

```python
def test_with_fixture(sample_wavefunction):
    """使用共享 fixture"""
    result = calculate_density(sample_wavefunction)
    assert result is not None
```

### 参数化测试

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parameterized(input, expected):
    """参数化测试"""
    assert input * 2 == expected
```

## 🔧 配置文件

### pytest.ini

主配置文件，定义了：
- 测试发现规则
- 标记定义
- 基本选项

### pyproject.toml

项目配置文件，包含：
- 依赖管理
- 覆盖率配置
- 超时配置

### conftest.py

共享 fixtures 和插件配置：
- 测试数据路径
- Wavefunction 对象
- 随机数生成器
- 临时输出目录

## 📈 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
      - name: Run tests
        run: |
          python -m pytest -n auto --cov=pymultiwfn --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## 🎓 最佳实践

1. **测试隔离**：每个测试应该独立运行
2. **快速反馈**：单元测试应该在 < 1 秒内完成
3. **命名清晰**：测试名称应该描述被测试的功能
4. **一个断言**：每个测试一个主要的断言
5. **使用标记**：正确分类测试（unit, integration, slow）
6. **覆盖率目标**：核心代码 > 90%，整体 > 70%
7. **并行测试**：使用 `-n auto` 加速测试套件

## 🆘 故障排除

### 并行测试失败

```bash
# 单线程运行以调试
.venv/bin/python -m pytest -n 0

# 减少 worker 数量
.venv/bin/python -m pytest -n 1
```

### 覆盖率不准确

```bash
# 清理缓存
rm -rf .coverage htmlcov

# 重新生成覆盖率
.venv/bin/python -m pytest --cov=pymultiwfn --cov-report=html
```

### 测试超时

```bash
# 增加超时时间
.venv/bin/python -m pytest --timeout=1200

# 跳过特定测试
.venv/bin/python -m pytest -k "not test_slow_function"
```

## 📚 参考资源

- [pytest 官方文档](https://docs.pytest.org/)
- [pytest-xdist 文档](https://pytest-xdist.readthedocs.io/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [PyMultiWFN 架构文档](../architecture_plan.md)

---

**最后更新：** 2026-02-18
**维护者：** PyMultiWFN Team
