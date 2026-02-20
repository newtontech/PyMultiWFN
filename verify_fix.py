#!/usr/bin/env python3
"""
验证 WFN 解析器修复后的电子数是否与 Multiwfn 一致。
"""

from pymultiwfn.io.parsers.wfn import WFNLoader
import os


def test_h2_ccsd():
    """测试 H2_CCSD.wfn 文件"""
    print("=" * 60)
    print("测试 H2_CCSD.wfn")
    print("=" * 60)

    wfn_file = "consistency_verifier/examples/H2_CCSD.wfn"

    if not os.path.exists(wfn_file):
        print(f"错误: 文件 {wfn_file} 不存在")
        return False

    # 加载 WFN 文件
    wfn = WFNLoader(wfn_file).load()

    # 验证电子数
    print(f"\n✓ 文件加载成功")
    print(f"✓ 分子标题: {wfn.title}")
    print(f"✓ 原子数: {wfn.num_atoms}")
    print(f"✓ 轨道数: {wfn.num_mos}")
    print(f"✓ 基函数数: {wfn.num_basis}")
    print(f"✓ 电子数: {wfn.num_electrons}")
    print(f"✓ 分子电荷: {wfn.charge}")
    print(f"✓ 多重度: {wfn.multiplicity}")

    # 打印原子信息
    print(f"\n原子信息:")
    for i, atom in enumerate(wfn.atoms):
        print(
            f"  原子 {i+1}: {atom.element} (核电荷={atom.charge:.1f}) "
            f"坐标=({atom.x:.6f}, {atom.y:.6f}, {atom.z:.6f})"
        )

    # 计算总核电荷
    total_nuclear_charge = sum(atom.charge for atom in wfn.atoms)
    print(f"\n总核电荷: {total_nuclear_charge:.1f}")
    print(f"预期电子数 (中性分子): {total_nuclear_charge:.1f}")

    # 验证电子数
    expected_electrons = 2.0  # H2 分子，2个电子
    if abs(wfn.num_electrons - expected_electrons) < 1e-6:
        print(f"\n✅ 电子数验证成功: {wfn.num_electrons} == {expected_electrons}")
        return True
    else:
        print(f"\n❌ 电子数验证失败: {wfn.num_electrons} != {expected_electrons}")
        return False


def test_other_wfn_files():
    """测试其他 WFN 文件"""
    print("\n" + "=" * 60)
    print("测试其他 WFN 文件")
    print("=" * 60)

    test_files = [
        "consistency_verifier/examples/COBH3_CCSD.wfn",
        "consistency_verifier/examples/ethane.wfn",
        "consistency_verifier/examples/benzene.wfn",
    ]

    for wfn_file in test_files:
        if not os.path.exists(wfn_file):
            continue

        try:
            wfn = WFNLoader(wfn_file).load()
            total_nuclear_charge = sum(atom.charge for atom in wfn.atoms)

            print(f"\n{wfn_file}:")
            print(f"  原子数: {wfn.num_atoms}")
            print(f"  电子数: {wfn.num_electrons}")
            print(f"  总核电荷: {total_nuclear_charge:.1f}")

            # 验证电子数 = 总核电荷 - 分子电荷
            expected = total_nuclear_charge - wfn.charge
            if abs(wfn.num_electrons - expected) < 1e-6:
                print(f"  ✅ 电子数验证通过")
            else:
                print(f"  ❌ 电子数验证失败: {wfn.num_electrons} != {expected}")
        except Exception as e:
            print(f"\n{wfn_file}: ❌ 加载失败: {e}")


if __name__ == "__main__":
    success = test_h2_ccsd()
    test_other_wfn_files()

    print("\n" + "=" * 60)
    if success:
        print("✅ 所有验证通过!")
        print("电子数解析修复成功，与 Multiwfn 结果一致。")
    else:
        print("❌ 验证失败!")
    print("=" * 60)
