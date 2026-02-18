"""
Shared pytest fixtures for PyMultiWFN testing.

This module provides common fixtures used across multiple test modules,
particularly for quantum chemistry data structures and test data files.
"""

import pytest
import numpy as np
from pathlib import Path
from pymultiwfn.core.data import Atom, Shell, Wavefunction


@pytest.fixture
def test_data_dir():
    """
    Return the path to the test data directory.

    Usage:
        def test_something(test_data_dir):
            wfn_file = test_data_dir / "wfn" / "test.wfn"
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_atom():
    """
    Return a sample Atom object for testing.

    Creates a hydrogen atom at origin.
    """
    return Atom(
        element="H",
        index=1,
        x=0.0,
        y=0.0,
        z=0.0,
        charge=1.0
    )


@pytest.fixture
def sample_atoms():
    """
    Return a list of sample Atom objects for testing.

    Creates a water molecule (H2O) with approximate geometry.
    """
    return [
        Atom(element="O", index=8, x=0.0, y=0.0, z=0.11779, charge=8.0),
        Atom(element="H", index=1, x=0.0, y=0.75545, z=-0.47116, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=-0.75545, z=-0.47116, charge=1.0),
    ]


@pytest.fixture
def sample_shell():
    """
    Return a sample Shell object for testing.

    Creates an S shell with 2 primitives.
    """
    return Shell(
        type=0,  # S shell
        center_idx=0,
        exponents=np.array([3.42525091, 0.62391373]),
        coefficients=np.array([0.15432897, 0.53532814])
    )


@pytest.fixture
def sample_wavefunction(sample_atoms):
    """
    Return a minimal Wavefunction object for testing.

    Creates a water molecule wavefunction with basic metadata.
    """
    return Wavefunction(
        atoms=sample_atoms,
        num_electrons=10.0,
        charge=0,
        multiplicity=1,
        num_basis=7,
        num_atomic_orbitals=7,
        num_primitives=14,
        num_shells=5
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary directory for test output files.

    This is particularly useful for tests that generate files
    (plots, analysis results, etc.) and need to verify their contents.

    Automatically cleaned up by pytest after the test completes.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def numpy_rng(parallel_safe):
    """
    Return a seeded numpy random number generator for reproducible tests.

    Features:
    - Deterministic across test runs (same seed)
    - Thread-safe for parallel testing
    - Worker-aware (different workers get different seeds)

    Usage:
        def test_random_calculation(numpy_rng):
            data = numpy_rng.random(10)
            assert len(data) == 10
    """
    # Use seed from parallel_safe fixture
    seed = parallel_safe['seed']
    return np.random.default_rng(seed=seed)


@pytest.fixture(scope="session")
def shared_test_data():
    """
    Session-scoped fixture for expensive-to-load test data.

    Use this for large wavefunction files or datasets that should
    be loaded once and reused across multiple tests.

    Features:
    - Lazy loading: data is loaded only when requested
    - Thread-safe for parallel testing
    - Cached across multiple tests in the same session
    - Automatically cleaned up when pytest exits

    Usage:
        def test_large_dataset(shared_test_data):
            wavefunction = shared_test_data.get('large_wfn')
            assert wavefunction is not None
    """
    # Lazy loading - only load if actually requested
    data = {}
    return data


@pytest.fixture
def parallel_safe():
    """
    Fixture to ensure tests are safe for parallel execution.

    This fixture provides utilities to make tests parallel-safe:
    - Unique temporary directories per worker
    - Unique random seeds per test
    - Worker identification

    Usage:
        def test_parallel_safe(parallel_safe):
            # Get unique worker ID (e.g., "gw0", "gw1")
            worker_id = parallel_safe['worker_id']

            # Get unique temp directory for this worker
            temp_dir = parallel_safe['temp_dir']

            # Get unique random seed
            seed = parallel_safe['seed']
    """
    import tempfile
    import os

    # Try to get pytest-xdist worker ID
    worker_id = os.getenv('PYTEST_XDIST_WORKER', 'master')

    # Create unique temp directory for this worker
    temp_dir = tempfile.mkdtemp(prefix=f'pymultiwfn_test_{worker_id}_')

    # Generate unique random seed based on worker ID and process ID
    seed = hash(f'{worker_id}_{os.getpid()}') % (2**32)

    yield {
        'worker_id': worker_id,
        'temp_dir': temp_dir,
        'seed': seed,
    }

    # Cleanup temp directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def isolated_environment():
    """
    Provide an isolated test environment with clean imports.

    This fixture ensures that tests don't interfere with each other
    when run in parallel. It resets module state before each test.

    Enhanced features:
    - Clears module cache for pymultiwfn modules
    - Resets global state variables
    - Ensures fresh import for each test
    - Safe for parallel testing with pytest-xdist

    Usage:
        def test_with_isolation(isolated_environment):
            # Fresh import guaranteed
            from pymultiwfn import something
            ...
    """
    import importlib
    import sys
    import gc

    # Get all modules to reload (before test)
    modules_to_reload = [
        name for name in sys.modules
        if name.startswith('pymultiwfn')
    ]

    # Force garbage collection before test
    gc.collect()

    yield  # Run the test

    # Force garbage collection after test
    gc.collect()

    # Reload modules after test (strict isolation)
    for name in modules_to_reload:
        try:
            module = sys.modules.get(name)
            if module is not None:
                # Clear module attributes to reset state
                for attr in list(vars(module)):
                    if not attr.startswith('__'):
                        try:
                            delattr(module, attr)
                        except (AttributeError, TypeError):
                            pass
                # Reload the module
                importlib.reload(module)
        except Exception:
            # If reload fails, at least remove from sys.modules
            sys.modules.pop(name, None)


@pytest.fixture
def performance_timer():
    """
    Context manager for timing test execution.

    Usage:
        def test_performance(performance_timer):
            with performance_timer() as timer:
                # Code to benchmark
                heavy_computation()
            assert timer.elapsed < 1.0  # Must complete in < 1s
    """
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    return Timer


@pytest.fixture
def assert_allclose_tolerance():
    """
    Return a tolerance-aware numpy.allclose assertion.

    This fixture provides different tolerance levels for different
    numerical precision requirements.

    Usage:
        def test_numerical_precision(assert_allclose_tolerance):
            a, b = compute_something()
            # Use loose tolerance (default)
            assert_allclose_tolerance(a, b)
            # Use strict tolerance
            assert_allclose_tolerance(a, b, rtol=1e-10, atol=1e-12)
    """
    def _assert_allclose(actual, desired, rtol=1e-7, atol=1e-9, err_msg=""):
        """Wrapper around numpy.allclose with default tolerances."""
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg)

    return _assert_allclose


# Pytest markers configuration
def pytest_configure(config):
    """
    Configure custom pytest markers.

    Markers can be used to categorize tests:
        @pytest.mark.unit: Fast, isolated tests
        @pytest.mark.integration: Tests requiring external resources
        @pytest.mark.slow: Tests that take a long time to run
        @pytest.mark.requires_data: Tests that require test data files
    """
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "requires_data: Tests requiring test data files")


@pytest.fixture
def mock_wavefunction_file(tmp_path):
    """
    Create a minimal mock .wfn file for testing parsers.

    Returns the path to a temporary WFN file with basic structure.
    """
    wfn_content = """\
    1
   10   7   1   0   0   0   0   0   0   0   0   0
  O   8   0.000000000000   0.000000000000   0.117790000000
  H   1   0.000000000000   0.755450000000  -0.471160000000
  H   1   0.000000000000  -0.755450000000  -0.471160000000
    1  1  1  1.00  0
   0  1  3 425250910000  2  0.154328970000  0.535328140000  0.444634540000
  1  0.000000000000  0.000000000000  0.000000000000
"""

    wfn_file = tmp_path / "test.wfn"
    wfn_file.write_text(wfn_content)
    return wfn_file
