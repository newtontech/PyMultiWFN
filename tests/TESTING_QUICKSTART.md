# PyMultiWFN Testing Quickstart

## Setup (One-time)

```bash
uv sync --group dev
```

## Run Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# With coverage
uv run pytest --cov=pymultiwfn --cov-report=html
open htmlcov/index.html
```

## Write Tests

```python
import pytest
from pymultiwfn.core.data import Atom

@pytest.mark.unit
class TestAtom:
    def test_creation(self):
        atom = Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)
        assert atom.element == "H"

    def test_with_fixture(self, sample_atom):
        # Uses fixture from conftest.py
        assert sample_atom.index == 1
```

## Available Fixtures

- `sample_atom` - Single hydrogen atom
- `sample_atoms` - Water molecule atoms (H2O)
- `sample_wavefunction` - Minimal wavefunction object
- `test_data_dir` - Path to test data directory
- `temp_output_dir` - Temporary directory for outputs
- `numpy_rng` - Seeded random number generator

## Markers

```python
@pytest.mark.unit          # Fast, isolated tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Slow tests (skip with -m "not slow")
@pytest.mark.requires_data # Needs test data files
```

## Test Data Structure

```
tests/test_data/
├── wfn/     # WFN format files
├── fchk/    # Gaussian FCHK files
└── molden/  # Molden format files
```

## Common Commands

```bash
# List all tests
uv run pytest --collect-only

# Run specific test
uv run pytest tests/unit/test_core_data.py::TestAtom::test_creation

# Verbose with local variables
uv run pytest -v --showlocals

# Stop on first failure
uv run pytest -x

# Run with debugger on failure
uv run pytest --pdb
```

## See Also

- Full documentation: `tests/README.md`
- Setup summary: `tests/SETUP_SUMMARY.md`
