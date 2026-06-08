# Test Framework Quick Reference

## Essential Commands

### Run Tests
```bash
# All tests with coverage
pytest --cov=pymultiwfn

# Parallel tests (auto-detect CPU cores)
pytest -n auto --cov=pymultiwfn

# Unit tests only
pytest -m "unit" -n auto

# Skip slow tests
pytest -m "not slow" -n auto

# Specific test file
pytest tests/unit/test_core_data.py -v

# Verbose with local variables
pytest -v -l --tb=short
```

### Code Quality
```bash
# Code style (Flake8)
flake8 pymultiwfn/

# Type checking (MyPy)
mypy pymultiwfn/

# Coverage report
pytest --cov=pymultiwfn --cov-report=html
open htmlcov/index.html
```

## Available Fixtures

```python
def test_something(
    test_data_dir,              # Path to test data directory
    sample_atom,                 # Sample Atom object
    sample_atoms,                # Sample molecule (H2O)
    sample_shell,                # Sample Shell object
    sample_wavefunction,         # Sample Wavefunction object
    temp_output_dir,             # Temporary output directory
    numpy_rng,                   # Seeded random number generator
    parallel_safe,               # Parallel testing utilities
    isolated_environment,        # Test isolation
    performance_timer,           # Performance timing
    assert_allclose_tolerance,   # Tolerance-aware assertion
    mock_wavefunction_file,      # Mock WFN file path
):
    pass
```

## Test Markers

```python
@pytest.mark.unit                # Fast, isolated tests
@pytest.mark.integration         # Requires external resources
@pytest.mark.slow               # Long-running tests
@pytest.mark.requires_data      # Requires test data files
@pytest.mark.benchmark          # Performance tests
@pytest.mark.expensive          # Heavy computational tests
```

## Key Configurations

### pytest.ini → pyproject.toml
- ✅ Config unified in pyproject.toml
- ✅ Coverage: pytest-cov
- ✅ Parallel: pytest-xdist
- ✅ Timeout: pytest-timeout
- ✅ Reruns: pytest-rerunfailures

### setup.cfg
- ✅ Flake8: Code style (max-line-length=88)
- ✅ MyPy: Type checking (python_version=3.10)
- ✅ Coverage: Exclude rules, HTML reports

## Test Results

```
tests/unit tests/math -n 2 --no-cov
======================== 86 passed, 1 skipped in 2.65s =========================

tests/unit/test_core_data.py --cov=pymultiwfn
TOTAL                                                       9195   9099   3158      2   0.79%
```

## Files

- `pyproject.toml` - Main configuration
- `setup.cfg` - Code quality tools
- `tests/conftest.py` - Test fixtures
- `ISSUE1_TEST_FRAMEWORK_OPTIMIZATION_REPORT.md` - Full report
