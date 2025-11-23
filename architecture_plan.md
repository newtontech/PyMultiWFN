# PyMultiWFN 架构重构计划

## 1. 现有项目结构分析 (Reconnaissance)

### 1.1 文件类型与分布
当前目录 `@Multiwfn_3.8_dev_src_Linux_2025-Nov-23` 包含 Multiwfn 的源代码，主要是 Fortran 90/95 代码。

*   **核心源码 (.f90, .F)**: 约 50 个文件。
    *   `Multiwfn.f90`: 主程序入口，包含主菜单循环。
    *   `define.f90`: 定义了所有的全局变量、数据结构（Derived Types）和常数。这是项目的“状态中心”。
    *   `fileIO.f90`: 处理各种文件格式（.fch, .wfn, .molden 等）的读取。
    *   `function.f90`: 核心波函数与实空间函数计算模块。
*   **功能模块**:
    *   `topology.f90`: 电子密度拓扑分析 (AIM)。
    *   `population.f90`: 种群分析 (Hirshfeld, VDD, etc.)。
    *   `bondorder.f90`: 键级分析 (Mayer, Wiberg, etc.)。
    *   `basin.f90`: 盆分析。
    *   `spectrum.f90`: 光谱绘制。
    *   `grid.f90`: 网格生成。
*   **图形界面**: `GUI.f90`, `plot.f90`, `dislin_d.f90` (Dislin 图形库接口)。
*   **数学/辅助**: `minpack.f90` (优化算法), `integral.f90` (积分), `util.f90` (通用工具), `sub.f90` (子程序集)。
*   **构建系统**: `Makefile`。

### 1.2 核心模块划分推测
根据文件名和内容分析，Multiwfn 的核心模块可以划分为：

1.  **数据核心 (State Core)**: `define.f90`。存储原子坐标、基组信息、系数矩阵、密度矩阵等全局状态。
2.  **输入输出 (IO)**: `fileIO.f90`, `fparser.f90`。负责将外部文件解析为内部数据结构。
3.  **计算引擎 (Compute Engine)**: `function.f90`, `integral.f90`, `grid.f90`。负责计算波函数值、电子密度、静电势等实空间属性。
4.  **分析任务 (Analysis Tasks)**: `topology.f90`, `population.f90` 等。依赖计算引擎进行特定的化学分析。
5.  **可视化 (Visualization)**: `plot.f90`, `GUI.f90`。负责结果的图形化展示。

### 1.3 迁移到 Python 的难点与对策

*   **全局变量 (Common Blocks/Modules)**:
    *   *难点*: `define.f90` 中使用了大量的 `module` 级变量（如 `a`, `b`, `CO`），这在 Fortran 中是典型的做法，但在 Python 中会导致代码难以维护和测试。
    *   *对策*: 使用面向对象编程 (OOP)。创建一个 `Wavefunction` 类来封装所有与当前体系相关的状态。
*   **计算性能 (Performance)**:
    *   *难点*: Multiwfn 涉及大量的循环计算（如在数百万个格点上计算电子密度），纯 Python 循环极慢。
    *   *对策*:
        *   **Vectorization**: 尽可能使用 NumPy 进行矩阵运算和广播操作。
        *   **JIT Compilation**: 使用 `Numba` 对核心计算函数（如基组求值）进行即时编译。
        *   **C/Fortran Extensions**: 对于极度复杂的积分代码（如 `Lebedev-Laikov.F`），可以使用 `f2py` 直接调用原有的 Fortran 代码，或者寻找现有的 Python 科学计算库（如 `PySCF`, `Libcint`）替代。
*   **图形界面 (GUI)**:
    *   *难点*: 原项目深度绑定 Dislin 库。
    *   *对策*: 使用 `Matplotlib` 进行静态绘图，使用 `PyQt` 或 `Tkinter` 构建交互式 GUI，或者开发 Web 界面。

---

## 2. Python 架构设计 (Architecture Proposal)

### 2.1 目录树结构

建议采用模块化、分层的目录结构：

```text
pymultiwfn/
├── __init__.py
├── config.py               # 配置管理 (Singleton)
├── core/                   # 核心数据结构
│   ├── __init__.py
│   ├── data.py             # 定义 Atom, BasisSet, Wavefunction 类
│   ├── constants.py        # 物理常数 (对应 define.f90)
│   └── state.py            # 全局状态管理 (如果必须)
├── io/                     # 输入输出模块
│   ├── __init__.py
│   ├── loader.py           # 文件加载分发器
│   ├── parsers/            # 各种格式解析器
│   │   ├── fch.py
│   │   ├── molden.py
│   │   ├── wfn.py
│   │   └── xyz.py
│   └── writer.py
├── math/                   # 数学与计算引擎
│   ├── __init__.py
│   ├── basis.py            # 基组求值 (GTF evaluation) - 性能关键点
│   ├── grid.py             # 网格生成 (Lebedev, Uniform)
│   ├── integration.py      # 数值积分
│   └── linalg.py           # 线性代数辅助
├── analysis/               # 分析功能模块
│   ├── __init__.py
│   ├── topology/           # 对应 topology.f90
│   │   ├── cp_search.py
│   │   └── paths.py
│   ├── population/         # 对应 population.f90
│   ├── bonding/            # 对应 bondorder.f90
│   └── spectrum/           # 对应 spectrum.f90
├── vis/                    # 可视化
│   ├── __init__.py
│   ├── plotter.py          # Matplotlib 封装
│   └── gui/                # GUI 相关代码
└── utils/                  # 工具函数
    ├── __init__.py
    └── logger.py
```

### 2.2 全局变量的处理设计

在 Fortran 中，`define.f90` 充当了全局数据库。在 Python 中，我们应避免使用全局变量，而是通过**传递对象**来共享数据。

**推荐模式：Data Class + Context Object**

1.  **`Wavefunction` 类 (核心数据容器)**:
    使用 Python 的 `@dataclass` 定义核心数据结构。这个对象将包含原子坐标、基组信息、系数矩阵等。

    ```python
    from dataclasses import dataclass
    import numpy as np
    from typing import List

    @dataclass
    class Atom:
        element: str
        x: float
        y: float
        z: float
        charge: float

    @dataclass
    class Wavefunction:
        atoms: List[Atom]
        basis_set: list  # 具体的基组结构
        coefficients: np.ndarray # MO 系数矩阵
        energies: np.ndarray     # MO 能量
        # ... 其他属性
    ```

2.  **配置管理 (Singleton)**:
    对于 `settings.ini` 中的配置，可以使用单例模式或模块级变量。

    ```python
    # config.py
    class Config:
        _instance = None
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
                cls._instance.load_settings()
            return cls._instance
        
        def load_settings(self):
            # 读取 settings.ini
            self.nthreads = 4
            self.output_format = "png"
    ```

### 2.3 计算密集型部分的技术选型

Multiwfn 的核心瓶颈在于**实空间函数求值**（例如：在 1,000,000 个格点上计算电子密度）。

**方案 A: 纯 Python + NumPy (适用于向量化操作)**
对于矩阵运算（如密度矩阵构建、简单的格点操作），NumPy 足够快。
*   *例子*: 计算所有原子的距离矩阵。

**方案 B: Numba (JIT) (推荐用于循环)**
对于无法简单向量化的复杂循环（如计算每个格点上的基组函数值），使用 `numba.jit`。
*   *优势*: 保持 Python 语法，性能接近 C/Fortran。
*   *场景*: `function.f90` 中的 `calccub_den` (计算立方格点密度)。

```python
from numba import jit

@jit(nopython=True)
def evaluate_gtf(x, y, z, exps, coefs, centers):
    # 高性能的基组求值循环
    pass
```

**方案 C: 混合编程 (f2py / ctypes)**
如果某些 Fortran 代码（如 `Lebedev-Laikov.F` 球面格点生成）极其复杂且难以重写，可以直接编译为共享库并由 Python 调用。
*   *建议*: 初期重构时，对于非核心瓶颈，优先用 Python/NumPy 重写以提高可读性。对于核心数学库（如积分），可以考虑调用 `PySCF` 的 C 库 (`libcint`)，避免重复造轮子。

**总结建议**:
1.  **IO 和 逻辑控制**: 纯 Python。
2.  **矩阵运算**: NumPy。
3.  **格点函数求值 (最耗时)**: Numba。
4.  **复杂积分**: 考虑复用现有的 Python 量子化学库 (PySCF) 或保留 Fortran 核心。
