# PyMultiWFN Testing Guide

Comprehensive guide to testing PyMultiWFN.

---

## Quick Start

```bash
# Install development tooling
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests in parallel (auto-detect CPU cores)
pytest -n auto

# Run quick tests only (skip slow tests)
pytest -m "not slow"

# Run with coverage
pytest --cov=pymultiwfn --cov-report=html
```

---

## Git Quality Gates

PyMultiWFN keeps local git hooks and GitHub CI aligned through
`.pre-commit-config.yaml`.

```bash
# Install both pre-commit and pre-push hooks
pre-commit install --install-hooks

# Run the fast commit gate manually
pre-commit run --all-files --hook-stage pre-commit

# Run the full push gate manually
pre-commit run --all-files --hook-stage pre-push
```

The pre-commit gate runs Python syntax checks, critical flake8 errors, and the
focused fuzzy bond-order tests. The pre-push gate runs the full pytest suite and
builds the package.

GitHub Actions runs the same two hook stages in `Git Hooks` CI, then verifies
that retained Multiwfn reference assets are not included in the built wheel.

---

## Test Organization

### Test Categories

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (require data)
├── test_*.py         # Various test modules
├── conftest.py       # Shared fixtures
└── test_data/        # Test data files
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_data` - Needs test data files
- `@pytest.mark.benchmark` - Performance benchmarks

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_density.py

# Run specific test class
pytest tests/unit/test_density.py::TestDensityCalc

# Run specific test
pytest tests/unit/test_density.py::TestDensityCalc::test_basic

# Run tests matching pattern
pytest -k "density"
```

### Parallel Testing

```bash
# Auto-detect CPU cores
pytest -n auto

# Use 4 workers
pytest -n 4

# Parallel with coverage
pytest -n auto --cov=pymultiwfn
```

### Test Selection

```bash
# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run integration tests
pytest --runintegration

# Run benchmarks
pytest -m benchmark --benchmark
```

### Verbosity & Debugging

```bash
# Verbose output
pytest -v

# Extra verbose
pytest -vv

# Show local variables in tracebacks
pytest -l

# Full tracebacks
pytest --tb=long

# Enter debugger on failure
pytest --pdb
```

---

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=pymultiwfn

# Show missing lines
pytest --cov=pymultiwfn --cov-report=term-missing

# HTML report
pytest --cov=pymultiwfn --cov-report=html
open htmlcov/index.html
```

### Coverage Targets

- **Unit tests**: > 90% coverage
- **Integration tests**: > 80% coverage
- **Overall**: > 85% coverage

---

## Writing Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from pymultiwfn.core.data import Atom, Shell, Wavefunction


class TestMyFeature:
    """Test suite for my feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_test_data()
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality works."""
        result = my_function(sample_data)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Using Fixtures

```python
# Use predefined fixtures
def test_with_h2(h2_molecule):
    """Test using H2 molecule fixture."""
    assert h2_molecule.num_atoms == 2

def test_with_coords(random_coords):
    """Test using random coordinates."""
    coords = random_coords(n_points=100)
    assert coords.shape == (100, 3)
```

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_fast_calculation():
    """Fast unit test."""
    pass

@pytest.mark.slow
def test_long_calculation():
    """Slow test (> 1 second)."""
    pass

@pytest.mark.integration
@pytest.mark.requires_data
def test_with_file(sample_wfn_file):
    """Integration test requiring data file."""
    pass

@pytest.mark.benchmark
def test_performance(benchmark_timer):
    """Performance benchmark test."""
    benchmark_timer.start()
    # ... code to benchmark ...
    elapsed = benchmark_timer.stop()
    assert elapsed < 1.0  # Should complete in < 1 second
```

---

## Test Best Practices

### 1. Keep Tests Isolated

```python
# Good: Each test is independent
def test_feature_a():
    data = create_fresh_data()
    result = process(data)
    assert result == expected

# Bad: Tests depend on each other
shared_data = None

def test_setup():
    global shared_data
    shared_data = create_data()

def test_feature():
    result = process(shared_data)  # Depends on previous test
```

### 2. Use Descriptive Names

```python
# Good
def test_density_positive_at_all_points():
    pass

# Bad
def test_density():
    pass
```

### 3. Test Edge Cases

```python
def test_empty_input():
    """Test with empty input."""
    with pytest.raises(ValueError):
        process([])

def test_large_input():
    """Test with large input."""
    large_data = np.random.randn(100000, 3)
    result = process(large_data)
    assert result.shape == (100000,)
```

### 4. Use Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("n_electrons,expected", [
    (2, "closed_shell"),
    (1, "open_shell"),
    (3, "open_shell"),
])
def test_spin_state(n_electrons, expected):
    result = determine_spin_state(n_electrons)
    assert result == expected
```

### 5. Clean Up Resources

```python
# Automatic cleanup with fixture
@pytest.fixture
def temporary_file():
    filepath = create_temp_file()
    yield filepath
    # Cleanup happens automatically
    Path(filepath).unlink()
```

---

## Test Utilities

### Custom Assertions

```python
def assert_allclose_sorted(a, b, rtol=1e-5):
    """Assert arrays are close after sorting."""
    np.testing.assert_allclose(
        np.sort(a),
        np.sort(b),
        rtol=rtol
    )
```

### Temporary Wavefunctions

```python
def test_with_temp_wavefunction(temporary_wavefunction):
    """Test with temporary wavefunction."""
    wfn = temporary_wavefunction(n_atoms=5)
    assert wfn.num_atoms == 5
```

---

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_density_performance(benchmark_timer, h2_molecule):
    """Benchmark density calculation."""
    coords = np.random.randn(10000, 3)
    
    benchmark_timer.start()
    density = calc_density(h2_molecule, coords)
    elapsed = benchmark_timer.stop()
    
    assert elapsed < 1.0  # Must complete in < 1 second
    assert len(density) == 10000
```

### Memory Profiling

```python
import tracemalloc

def test_memory_usage():
    """Test memory usage doesn't grow unbounded."""
    tracemalloc.start()
    
    # Run test
    wfn = create_large_wavefunction()
    process(wfn)
    
    # Check memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak should be < 100 MB
    assert peak < 100 * 1024 * 1024
```

---

## Troubleshooting

### Common Issues

**Tests hang or timeout:**
```bash
# Increase timeout
pytest --timeout=1200  # 20 minutes
```

**Out of memory errors:**
```bash
# Run with fewer workers
pytest -n 2
```

**Import errors:**
```bash
# Ensure package is installed
pip install -e .
```

**Coverage not working:**
```bash
# Install coverage plugin
pip install pytest-cov
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to `main`
- Every pull request
- Daily scheduled runs

### CI Configuration

```yaml
# Run all tests
pytest -n auto --cov=pymultiwfn

# Upload coverage
codecov
```

---

## Quick Reference

```bash
# Most common commands
pytest                                    # Run all tests
pytest -n auto                           # Parallel tests
pytest -m "not slow"                     # Quick tests
pytest --cov=pymultiwfn                  # With coverage
pytest -x                                # Stop on first failure
pytest -k "density"                      # Run matching tests
pytest tests/unit/                       # Run unit tests only
pytest --runintegration                  # Include integration tests
```

---

## Getting Help

- **Test failures**: Check test output and traceback
- **Coverage issues**: Run with `-v` to see which lines are missed
- **Performance issues**: Use `--durations=10` to see slowest tests
- **CI failures**: Check GitHub Actions logs

---

**Last Updated**: 2026-02-21
**Maintainer**: PyMultiWFN Team
