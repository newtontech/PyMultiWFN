#!/usr/bin/env python3
import os
import subprocess

# 设置环境变量
os.environ['MULTIWFN_BIN'] = '/home/yhm/software/PyMultiWFN/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn'

# 测试 Multiwfn
result = subprocess.run(
    [os.environ['MULTIWFN_BIN'], 'consistency_verifier/examples/H2_CCSD.wfn'],
    input='18\n1\nq\n',
    capture_output=True,
    text=True,
    timeout=10
)

print("Multiwfn 输出:")
print(result.stdout[:500])

# 测试 PyMultiWFN
from pymultiwfn.io.file_manager import FileManager
fm = FileManager()
wfn = fm.load_wavefunction('consistency_verifier/examples/H2_CCSD.wfn')
print(f"\nPyMultiWFN 加载成功:")
print(f"  标题: {wfn.title}")
print(f"  电子数: {wfn.num_electrons}")
print(f"  原子数: {len(wfn.atoms)}")
