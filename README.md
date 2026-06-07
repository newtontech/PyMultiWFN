# PyMultiWFN

[![Tests](https://github.com/pymultiwfn/pymultiwfn/workflows/Tests/badge.svg)](https://github.com/pymultiwfn/pymultiwfn/actions/workflows/tests.yml)
[![Code Quality](https://github.com/pymultiwfn/pymultiwfn/workflows/Code%20Quality/badge.svg)](https://github.com/pymultiwfn/pymultiwfn/actions/workflows/code-quality.yml)
[![Documentation](https://github.com/pymultiwfn/pymultiwfn/workflows/Documentation/badge.svg)](https://github.com/pymultiwfn/pymultiwfn/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/pymultiwfn/pymultiwfn/branch/main/graph/badge.svg)](https://codecov.io/gh/pymultiwfn/pymultiwfn)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-first refactor of the Multiwfn wavefunction analysis program.

## Features

- **Electron Density Calculations**: Fast and accurate density evaluation
- **Bond Order Analysis**: Mayer, Wiberg, Mulliken bond orders
- **Population Analysis**: Mulliken, Hirshfeld, Becke populations
- **Orbital Analysis**: MO visualization and analysis
- **High Performance**: Optimized with NumPy, caching, and parallel processing
- **Well Tested**: 290+ tests with comprehensive coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/newtontech/PyMultiWFN.git
cd PyMultiWFN

# Install in development mode
pip install -e .[dev]
```

The current package is a Python-first refactor that keeps performance-sensitive numerical kernels explicit. When Fortran kernels are present under `pymultiwfn/math/fortran/`, treat them as implementation details behind Python APIs and keep equivalent Python tests for behavior.

## Quick Start

```python
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density
from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order

# Load wavefunction
wfn = load_wavefunction("molecule.wfn")

# Calculate density
import numpy as np
coords = np.array([[0.0, 0.0, 0.0]])
density = calc_density(wfn, coords)

# Calculate bond orders
bond_orders = calculate_mayer_bond_order(wfn)
print(bond_orders['total'])
```

## Typical Workflows

### 1. Inspect a wavefunction file

```bash
pymultiwfn --help
```

Use the CLI for smoke testing installation, then move to Python scripts for reproducible analysis.

### 2. Density or orbital analysis

Start from `examples/density_analysis.py`, replace the input wavefunction path, and save generated arrays or plots beside your experiment notes.

### 3. Bond order comparison

Use `examples/bond_analysis.py` to compare Mayer, Wiberg, or Mulliken-style outputs across molecules. Include the input file, method, and expected tolerance when adding tests.

## Documentation

- **[User Guide](docs/user_guide.md)**: Complete guide to using PyMultiWFN
- **[Testing Guide](docs/TESTING.md)**: How to run and write tests
- **[Examples](examples/)**: Example scripts for common tasks

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Basic operations
- `density_analysis.py` - Density grid analysis
- `bond_analysis.py` - Bond order analysis

## Testing

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run with coverage
pytest --cov=pymultiwfn --cov-report=html
```

See [Testing Guide](docs/TESTING.md) for more details.

## Performance

Performance benchmarks (tested on typical laptop):

- **Density calculation**: 278K-1.66M points/s
- **Bond order calculation**: 2.4K-19K atoms/s
- **Cache speedup**: 1.1-2.2x for repeated calculations
- **Memory efficiency**: Linear scaling, < 1 MB for 200-atom systems

Run benchmarks:
```bash
python benchmark_performance.py
```

## Project Status

**Version**: 0.1.2 (Alpha)

**Test Status**:
- Tests: 291 passing, 10 skipped
- Coverage: Comprehensive
- Quality: 0 violations

**Development Progress**:
- ✅ Code Quality: 100% complete
- ✅ Test Framework: 70% complete
- 🔄 Performance: 40% complete
- 🔄 Documentation: 60% complete
- 🔄 Consistency: 50% complete

## Contributing

Contributions are welcome! Please see our development guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use PyMultiWFN in your research, please cite:

1. The original Multiwfn paper
2. This repository

```bibtex
@software{pymultiwfn_newtontech,
  title = {PyMultiWFN: Python-first wavefunction analysis tools},
  author = {PyMultiWFN contributors},
  year = {2026},
  url = {https://github.com/newtontech/PyMultiWFN}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/pymultiwfn/pymultiwfn/issues)
- **Documentation**: [User Guide](docs/user_guide.md)
- **Examples**: [examples/](examples/)

## Acknowledgments

Based on the original [Multiwfn](http://sobereva.com/multiwfn/) program by Tian Lu.

**Additional References:**
- [szczypinski-group/pyMultiwfn](https://github.com/szczypinski-group/pyMultiwfn) - Reference implementation for wavefunction analysis algorithms

---

**Maintained by**: PyMultiWFN Team  
**Last Updated**: 2026-02-27
