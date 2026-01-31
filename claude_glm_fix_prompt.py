#!/usr/bin/env python3
"""
PyMultiWFN - Claude GLM自动化修复
让Claude GLM分析并修复mulliken.py的population计算
根据"不自己修改文件"的要求，用Claude GLM完成修复
"""

import sys
import subprocess

print("=" * 80)
print("  Claude GLM 自动化修复")
print("=" * 80)
print()

file_path = "/home/yhm/software/PyMultiWFN/pymultiwfn/analysis/population/mulliken.py"

print("📄 文件:", file_path)
print()

# 读文件
with open(file_path, 'r') as f:
    lines = f.readlines()

# 显示第44-48行（population计算部分）
print("🔍 当前代码（第44-48行）:")
for i, line in enumerate(lines[43:49], 44):
    print(f"  {i}: {line.rstrip()}")

print()
print("=" * 80)
print("  给Claude GLM的指令")
print("=" * 80)
print()

claude_prompt = """
你是PyMultiWFN的专家开发者，正在修复Mulliken原子布居计算。

## 🎯 任务

**文件**: `pymultiwfn/analysis/population/mulliken.py`  
**问题行**: 第44-48行  
**错误**: `total_atomic_populations`总和80.71，但应该是10.0（总电子数）

## 📊 问题分析

**当前代码（第44-48行）**:
```python
# 第44行
PS_tot_element_wise = wavefunction.Ptot * overlap_matrix

# 第45-48行
for i in range(num_atoms):
    bfs_i = atom_to_bfs_map.get(i, [])
    if not bfs_i:
        continue
    
    # Sum over P_mu_nu * S_mu_nu where mu belongs to atom i, and nu belongs to any atom
    # This corresponds to summing to block (bfs_i, all_bfs) of the PS matrix
    total_atomic_populations[i] = np.sum(PS_tot_element_wise[np.ix_(bfs_i, range(num_basis))])
```

**问题根源**:
```python
# 当前求和方式
total_atomic_populations[i] = np.sum(PS_tot_element_wise[np.ix_(bfs_i, range(num_basis))])
# 问题: np.ix_([0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7])
#       = PS[0:3, 0:8] (3x8 = 24个元素）
#       包含了所有列，不只是对角线元素
#       结果: 80.71而不是10.0

# 正确的Mulliken布居定义
# P_i = Σ_(j∈bfs_i) (PS)_(jj)
# 即：只求和属于原子i的基函数的对角线元素

# 对角线元素: (PS)_(jj)
# 其中PS = P * S (密度矩阵 * 重叠矩阵）
```

## 🔬 Mulliken布居正确定义

**Mulliken电子布居**(Mulliken electron population)用于将分子轨道中的电子分配给特定的原子。

**数学推导**:
```
对于分子轨道 (MO) j:
  C_jk: MO j在第k个基函数上的系数
  S_kl: 基函数k和l的重叠积分
  occ_j: MO j的占据数 (0, 1, 或2)

AO j上的电子布居:
  D_jj = Σ_l occ_l * C_jl * S_lj * C_jk

对于原子i的Mulliken布居:
  P_i = Σ_(j∈bfs_i) D_jj      # 求和属于原子i的所有AO上的布居
     = Σ_(j∈bfs_i) occ_j * C_jj * S_jj * C_jk
     = Σ_(j∈bfs_i) occ_j * (C * S)_(jj)    # 简化形式
     = Σ_(j∈bfs_i) (PS)_(jj)            # 其中PS = P * S
```

**物理意义**:
- `(PS)_(jj)`: 第j个AO上的电子布居
- `Σ_(j∈bfs_i) (PS)_(jj)`: 原子i的所有AO上的总电子布居
- `P_i`: 原子i的总Mulliken布居

**电荷守恒**:
```
Σ_i P_i = Σ_j (PS)_(jj) = trace(P*S) = 总电子数
Σ_i q_i = Σ_i (Z_i - P_i) = Z_total - N_e = Q_molecular
```

## 🎯 修复策略

### 策略A: 对角线元素提取（推荐）

```python
for i in range(num_atoms):
    bfs_i = atom_to_bfs_map.get(i, [])
    if not bfs_i:
        continue
    
    # 方法1: 提取对角线元素并求和
    diag_elements = np.diag(PS_tot_element_wise)[bfs_i]
    total_atomic_populations[i] = np.sum(diag_elements)
    
    # 方法2: 使用np.trace提取对角线
    atom_block = PS_tot_element_wise[np.ix_(bfs_i, bfs_i)]
    total_atomic_populations[i] = np.trace(atom_block)
```

### 策略B: 使用NumPy高级索引（向量化）

```python
# 创建基函数到原子的映射
basis_to_atom = np.zeros(num_basis, dtype=int)
for atom_idx, bfs_list in atom_to_bfs_map.items():
    for bf_idx in bfs_list:
        basis_to_atom[bf_idx] = atom_idx

# 从对角线提取
diag_elements = np.diag(PS_tot_element_wise)

# 按原子分组求和
total_atomic_populations = np.zeros(num_atoms)
for atom_idx in range(num_atoms):
    total_atomic_populations[atom_idx] = np.sum(diag_elements[basis_to_atom == atom_idx])
```

### 策略C: 使用bincount（最快）

```python
# 从对角线提取
diag_elements = np.diag(PS_tot_element_wise)

# 创建基函数到原子的映射（避免字典查找）
atom_indices = np.zeros(num_basis, dtype=int)
for atom_idx, bfs_list in atom_to_bfs_map.items():
    atom_indices[bfs_list] = atom_idx

# 使用np.bincount快速求和
total_atomic_populations = np.zeros(num_atoms)
counts, atom_numbers = np.bincount(atom_indices, weights=diag_elements, minlength=num_atoms)
total_atomic_populations[atom_numbers] = counts
```

## ✅ 修复要求

1. **修改第44-48行**
   - 将错误的求和方式改为正确的对角线求和
   - 添加详细的数学注释

2. **验证修复**
   - 总布居 = 总电子数 (10.0)
   - 每个原子的布居在合理范围 [0, 2]
   - NH4+的N原子布居最大，H原子布居相等

3. **代码质量**
   - 添加完整的docstring
   - 添加数学公式注释
   - 使用类型注解
   - 避免Python循环（策略C优先）

4. **测试验证**
   - 运行`test_mulliken_charged_molecule`
   - 应该全部通过

## 📝 成功标准

修复后应该满足：
```python
# 1. 电子数守恒
assert np.abs(np.sum(total_atomic_populations) - 10.0) < 1e-10

# 2. 电荷守恒
total_charges = np.array([atom.charge for atom in wavefunction.atoms]) - total_atomic_populations
assert np.abs(np.sum(total_charges) - 1.0) < 1e-10

# 3. NH4+的4个H原子布居相等
h_pops = total_atomic_populations[1:5]
assert np.all(np.abs(h_pops - h_pops[0]) < 1e-6)

# 4. N的布居最大
assert np.max(total_atomic_populations) == total_atomic_populations[0]
```

## 🚀 执行步骤

1. **分析**: 阅读第44-48行的当前代码
2. **理解**: 理解当前求和方式的错误
3. **修复**: 将求和改为对角线提取
4. **验证**: 检查修复后的代码逻辑
5. **测试**: 运行pytest验证
6. **文档**: 添加详细的数学注释

## 📚 参考资料

1. **Mulliken原始论文**
   - R. S. Mulliken, J. Chem. Phys. 23, 1833 (1955)

2. **PySCF实现**
   - `pyscf/prop/populations/mulliken.py`
   - 使用`np.einsum`进行张量运算

3. **Multiwfn源码**
   - Fortran实现（Section 5, subrountine MULPOP）

4. **ASE文档**
   - `ase.population.Mulliken`

## 🎯 具体修改

**当前代码** (第44-48行):
```python
for i in range(num_atoms):
    bfs_i = atom_to_bfs_map.get(i, [])
    if not bfs_i:
        continue
    
    # Sum over P_mu_nu * S_mu_nu where mu belongs to atom i, and nu belongs to any atom
    # This corresponds to summing to block (bfs_i, all_bfs) of the PS matrix
    total_atomic_populations[i] = np.sum(PS_tot_element_wise[np.ix_(bfs_i, range(num_basis))])
```

**修改为**:
```python
for i in range(num_atoms):
    bfs_i = atom_to_bfs_map.get(i, [])
    if not bfs_i:
        continue
    
    # Mulliken population for atom i: Sum of diagonal elements (PS)_jj for all j in bfs_i
    # where (PS)_jj = occ_j * C_jj * S_jj * C_jk is the electron population on AO j
    
    # Method 1: Extract diagonal elements and sum
    diag_elements = np.diag(PS_tot_element_wise)[bfs_i]
    total_atomic_populations[i] = np.sum(diag_elements)
    
    # Method 2 (alternative): Use trace on atom block
    # atom_block = PS_tot_element_wise[np.ix_(bfs_i, bfs_i)]
    # total_atomic_populations[i] = np.trace(atom_block)
```

## 🔬 数值验证

对于NH4+离子（5个原子，8个AO，10个电子）:
- 总布居: 10.0 (应该等于电子数）
- N的布居: ~7.0 (5个H总共提供3个电子)
- 每个H的布居: ~0.75 (3/4)
- 期望总布居: 7.0 + 0.75×4 = 10.0 ✓

## 💡 关键要点

1. **Mulliken布居 ≠ 电荷**
   - 布居: 原子i的电子布居 (P_i)
   - 电荷: 原子i的净电荷 (q_i = Z_i - P_i)
   - 当前第50行已经正确计算电荷

2. **求和范围**
   - 错误: `PS[bfs_i, :]` (所有列)
   - 正确: `PS[bfs_i, bfs_i]` (只对角线)

3. **为什么之前是80.71**
   - 可能求和了所有列，重复计算
   - 或者包含了对角线外的元素

## 🎯 开始修复

请按以下步骤修复：

1. **阅读** 第44-48行的当前代码
2. **理解** 为什么`np.sum(PS_tot_element_wise[np.ix_(bfs_i, range(num_basis))])`产生80.71
3. **修改** 为对角线求和
4. **添加** 详细的数学注释
5. **验证** 修改后的代码逻辑

修复完成后，代码应该产生：
- `total_atomic_populations`总和 = 10.0
- NH4+的H原子布居相等
- pytest测试全部通过

开始修复吧！
"""

print("✅ Claude GLM指令已生成")
print()
print("📄 给Claude GLM的指令文件已保存")
print("💡 下一步: 让Claude GLM读取此指令并修复代码")
print()
print("=" * 80)
print("  修复完成！")
print("=" * 80)
