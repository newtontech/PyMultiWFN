#!/usr/bin/env python3
"""
测试 WFN 解析器的电子数计算修复。

这个测试验证：
1. 电子数正确从原子核电荷计算
2. 对于中性分子，电子数 = 总核电荷
3. 对于带电分子，电子数 = 总核电荷 - 分子电荷
"""

from pymultiwfn.io.parsers.wfn import WFNLoader
import os


def test_electron_count_h2():
    """测试 H2 分子（2个电子）"""
    print("测试 H2_CCSD.wfn...")
    wfn = WFNLoader("consistency_verifier/examples/H2_CCSD.wfn").load()

    # H2 分子：2个H原子，每个核电荷1.0，总核电荷2.0
    # 中性分子，所以电子数 = 2.0
    assert wfn.num_atoms == 2
    assert wfn.num_electrons == 2.0
    print("✅ H2 电子数正确: 2.0")


def test_electron_count_cobh3():
    """测试 COBH3 分子（22个电子）"""
    print("\n测试 COBH3_CCSD.wfn...")
    wfn = WFNLoader("consistency_verifier/examples/COBH3_CCSD.wfn").load()

    # COBH3 分子：C(6) + O(8) + B(5) + 3*H(1*3) = 22个电子
    total_nuclear_charge = sum(atom.charge for atom in wfn.atoms)
    expected_electrons = total_nuclear_charge - wfn.charge

    assert wfn.num_electrons == expected_electrons
    assert wfn.num_electrons == 22.0
    print(f"✅ COBH3 电子数正确: {wfn.num_electrons}")


def test_electron_count_ethane():
    """测试乙烷分子（18个电子）"""
    print("\n测试 ethane.wfn...")
    wfn = WFNLoader("consistency_verifier/examples/ethane.wfn").load()

    # C2H6: 2*6 + 6*1 = 18个电子
    total_nuclear_charge = sum(atom.charge for atom in wfn.atoms)
    expected_electrons = total_nuclear_charge - wfn.charge

    assert wfn.num_electrons == expected_electrons
    assert wfn.num_electrons == 18.0
    print(f"✅ Ethane 电子数正确: {wfn.num_electrons}")


def test_electron_count_benzene():
    """测试苯分子（42个电子）"""
    print("\n测试 benzene.wfn...")
    wfn = WFNLoader("consistency_verifier/examples/benzene.wfn").load()

    # C6H6: 6*6 + 6*1 = 42个电子
    total_nuclear_charge = sum(atom.charge for atom in wfn.atoms)
    expected_electrons = total_nuclear_charge - wfn.charge

    assert wfn.num_electrons == expected_electrons
    assert wfn.num_electrons == 42.0
    print(f"✅ Benzene 电子数正确: {wfn.num_electrons}")


def test_wavefunction_completeness():
    """测试 Wavefunction 对象的完整性"""
    print("\n测试 Wavefunction 对象完整性...")
    wfn = WFNLoader("consistency_verifier/examples/H2_CCSD.wfn").load()

    # 验证所有必要属性都被正确设置
    assert wfn.num_atoms > 0
    assert wfn.num_electrons > 0
    assert wfn.num_mos > 0
    assert wfn.num_basis > 0
    assert wfn.coefficients is not None
    assert wfn.energies is not None
    assert wfn.occupations is not None

    # 验证别名属性
    assert wfn.mo_coefficients is wfn.coefficients
    assert wfn.mo_energies is wfn.energies
    assert wfn.mo_occupations is wfn.occupations

    print("✅ Wavefunction 对象完整")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 WFN 解析器电子数修复")
    print("=" * 60)

    try:
        test_electron_count_h2()
        test_electron_count_cobh3()
        test_electron_count_ethane()
        test_electron_count_benzene()
        test_wavefunction_completeness()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise
