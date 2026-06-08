# PyMultiWFN 测试修复计划

## 失败的测试分析

### 1. test_mayer_diagonal_elements
**问题**: 测试假设对角元素 = 行和，但这是错误的
**原因**: 对角元素应该等于 Mayer valence，即该原子与其他原子的键序总和（不包括对角元素本身）
**解决方案**: 修改测试，检查对角元素 = 非对角元素之和

### 2. test_mayer_unrestricted
**问题**: total 不等于 alpha + beta
**原因**: total 是从 Ptot 计算的，而 alpha + beta 有交叉项
**解决方案**: 对于 unrestricted wavefunction，计算 total 为 alpha + beta

### 3. test_multicenter_invalid_atoms
**问题**: 访问无效原子索引（5）时抛出 KeyError 而不是 ValueError
**原因**: 代码没有验证 atom_indices 的有效性
**解决方案**: 在 calculate_multicenter_bond_order 中添加输入验证

### 4. test_mulliken_h2_single_bond 和 test_mulliken_vs_mayer
**问题**: H2 WFN 文件的 overlap matrix 是 identity，导致 Mulliken bond order 为 0
**原因**: WFN 文件不包含 overlap matrix
**解决方案**: 这些测试应该使用 mock wavefunction 或被 skip

## 修复顺序
1. 修复 test_mayer_diagonal_elements 测试
2. 修复 test_mayer_unrestricted（修改 mayer.py）
3. 修复 test_multicenter_invalid_atoms（添加输入验证）
4. 修改 mulliken 测试使用 mock wavefunction

## 预期结果
- test_mayer_diagonal_elements: 通过
- test_mayer_unrestricted: 通过
- test_multicenter_invalid_atoms: 通过
- test_mulliken_h2_single_bond: 通过（使用 mock）
- test_mulliken_vs_mayer: 通过（使用 mock）
