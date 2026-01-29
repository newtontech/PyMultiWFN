# PyMultiWFN Test Suite

This directory contains the test suite for PyMultiWFN.

## Directory Structure

```
tests/
├── __init__.py                 # Test package marker
├── conftest.py                 # Shared pytest fixtures
├── README.md                   # This file
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_core_data.py      # Tests for data structures
│   ├── test_io_loader.py      # Tests for file loading
│   └── ...
├── integration/                # Integration tests
│   └── test_file_loading.py   # End-to-end file loading tests
└── test_data/                  # Test data files
    ├── wfn/                    # WFN format files
    ├── fchk/                   # Gaussian fchk files
    └── molden/                 # Molden format files
```

## Running Tests

### Run all tests:
```bash
uv run pytest
```

### Run only unit tests:
```bash
uv run pytest -m unit
```

### Run only integration tests:
```bash
uv run pytest -m integration
```

### Run tests with coverage:
```bash
uv run pytest --cov=pymultiwfn --cov-report=html
```

### Run a specific test file:
```bash
uv run pytest tests/unit/test_core_data.py
```

### Run a specific test:
```bash
uv run pytest tests/unit/test_core_data.py::TestAtom::test_atom_creation
```

### Skip slow tests:
```bash
uv run pytest -m "not slow"
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.requires_data`: Tests that require test data files

## Writing Tests

### Example Unit Test:

```python
import pytest
from pymultiwfn.core.data import Atom

@pytest.mark.unit
class TestAtom:
    def test_atom_creation(self):
        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        assert atom.element == "H"
```

### Example Test Using Fixtures:

```python
def test_with_fixture(sample_atom):
    """Test using the sample_atom fixture from conftest.py"""
    assert sample_atom.index == 1
```

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_atom`: A single hydrogen atom
- `sample_atoms`: List of atoms (H2O molecule)
- `sample_wavefunction`: Minimal wavefunction object
- `test_data_dir`: Path to test data directory
- `temp_output_dir`: Temporary directory for test outputs
- `numpy_rng`: Seeded random number generator

## Test Data

Test data files should be placed in the `test_data/` subdirectories:

- Use small, minimal examples
- Document the source of test data
- Compress large files
- Don't commit proprietary data

## CI/CD

Tests are run automatically on:
- Pull requests
- Main branch commits
- Release tags
