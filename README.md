# PyMultiWFN

**PyMultiWFN** is a modernization project aiming to refactor the renowned wavefunction analysis program **Multiwfn** from Fortran to Python. This project seeks to maintain the powerful analytical capabilities of the original software while leveraging Python's ecosystem for better modularity, readability, and extensibility.

## Project Goal

To create a high-performance, modular, and user-friendly Python package for wavefunction analysis, serving as a modern alternative or companion to the original Multiwfn.

## Architecture

The project follows a modular architecture designed to separate data, logic, and visualization.

### Directory Structure

```text
pymultiwfn/
├── core/           # Core data structures (Wavefunction, Atom, BasisSet)
├── io/             # File I/O (Parsers for .fch, .wfn, .molden, etc.)
├── math/           # Mathematical engines (Grid generation, Integration, GTF evaluation)
├── analysis/       # Analysis modules (Topology, Population, Bonding, etc.)
├── vis/            # Visualization (Matplotlib/PyQt plotting)
├── utils/          # Utilities (Logging, Config)
└── config.py       # Configuration management
```

## Technology Stack

*   **Language**: Python 3.10+
*   **Core Computation**: NumPy (Vectorization), Numba (JIT Compilation for performance-critical loops)
*   **Visualization**: Matplotlib (Static), PyQt/Side (Interactive GUI - Future)
*   **Quantum Chemistry Interop**: PySCF (Optional, for complex integrals)

## Development Status

### Phase 1: Reconnaissance & Architecture (Completed)
- Analyzed original Fortran source code structure.
- Designed Python package structure.
- Established strategies for handling global variables and performance bottlenecks.

### Phase 2: Foundation (In Progress)
- Setting up project skeleton.
- Implementing Core Data Structures (`Wavefunction`).
- Implementing basic File I/O (`.fch` parser).

## License

[License Information to be added]
Original citing:
LICENSE INFORMATION: To download and use Multiwfn, you are required to read and agree the following terms:
(a) Currently Multiwfn is free of charge and open-source for both academic and commerical usages, anyone is allowed to freely distribute the original or their modified Multiwfn codes to others.
(b) Multiwfn can be distributed as a free component of commercial code. Selling modified version of Multiwfn may also be granted, however, obtaining prior consent from the original author of Multiwfn (Tian Lu) is needed.
(c) If Multiwfn is utilized in your work, or your own code incorporated any part of Multiwfn code, at least the following original papers of Multiwfn MUST BE cited in main text of your paper or code:
Tian Lu, Feiwu Chen, J. Comput. Chem., 33, 580-592 (2012)
Tian Lu, J. Chem. Phys., 161, 082503 (2024)
(d) There is no warranty of correctness of the results produced by Multiwfn, the author of Multiwfn does not hold responsibility in any way for any consequences arising from the use of the Multiwfn.

Whenever possible, please mention and cite Multiwfn in main text rather than in supplemental information, otherwise not only Multiwfn will be difficult for readers to notice, but also the paper will not be included in citation statistics.

郑重提示：在研究文章里恰当引用研究中使用的所有学术程序是最基本的学术规范和道德。最合理的引用Multiwfn的方法见Multiwfn可执行文件包里的How to cite Multiwfn.pdf文档。如果使用了Multiwfn（包括其中任何功能）的文章中甚至连上面红字提到的这篇Multiwfn原文都没在文章正文里引用的话，一经发现，作者会被列入黑名单，并禁止在未来使用Multiwfn。给实验组做代算时也必须明确告知对方需要在文章中对Multiwfn进行正确引用。

PyMultiWFN citing: 
PyMultiWFN is a modernization project aiming to refactor the renowned wavefunction analysis program Multiwfn from Fortran to Python. This project seeks to maintain the powerful analytical capabilities of the original software while leveraging Python's ecosystem for better modularity, readability, and extensibility.