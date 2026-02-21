"""
PyTest configuration and fixtures for PyMultiWFN.

This file provides:
- Common fixtures for all tests
- Test configuration
- Parallel testing support
- Coverage configuration
"""

import pytest
import numpy as np
from pathlib import Path
from pymultiwfn.core.data import Atom, Shell, Wavefunction


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require external data files"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip integration tests that require data unless data is available
    if not config.getoption("--runintegration", default=False):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords and "requires_data" in item.keywords:
                item.add_marker(skip_integration)


# ============================================================================
# Command Line Options
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="Run integration tests that require external data"
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run performance benchmark tests"
    )


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def simple_h_atom():
    """Create a simple hydrogen atom for testing."""
    atoms = [Atom(element="H", index=1, x=0.0, y=0.0, z=0.0, charge=1.0)]
    
    shells = [
        Shell(
            type=0,  # S orbital
            center_idx=0,
            exponents=np.array([1.0]),
            coefficients=np.array([1.0]),
        )
    ]
    
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=1.0,
        charge=0,
        multiplicity=2,
        num_basis=1,
        num_atomic_orbitals=1,
        num_primitives=1,
        num_shells=1,
        shells=shells,
        occupations=np.array([1.0]),
        coefficients=np.array([[1.0]]),
    )
    
    return wfn


@pytest.fixture
def h2_molecule():
    """Create H2 molecule at equilibrium geometry."""
    # H-H bond: 0.74 Å = 1.40 bohr
    atoms = [
        Atom(element="H", index=1, x=0.0, y=0.0, z=-0.70, charge=1.0),
        Atom(element="H", index=1, x=0.0, y=0.0, z=0.70, charge=1.0),
    ]
    
    shells = [
        Shell(type=0, center_idx=0, exponents=np.array([1.0]), coefficients=np.array([1.0])),
        Shell(type=0, center_idx=1, exponents=np.array([1.0]), coefficients=np.array([1.0])),
    ]
    
    coeff = 1.0 / np.sqrt(2)
    wfn = Wavefunction(
        atoms=atoms,
        num_electrons=2.0,
        charge=0,
        multiplicity=1,
        num_basis=2,
        num_atomic_orbitals=2,
        num_primitives=2,
        num_shells=2,
        shells=shells,
        occupations=np.array([1.0, 1.0]),
        coefficients=np.array([[coeff, coeff], [coeff, -coeff]]),
        overlap_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
        Ptot=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    
    return wfn


@pytest.fixture
def random_coords():
    """Generate random coordinates for testing."""
    def _generate(n_points=100, scale=3.0):
        """Generate n_points random coordinates within ±scale bohr."""
        np.random.seed(42)  # For reproducibility
        return np.random.randn(n_points, 3) * scale
    return _generate


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "tests" / "test_data"


@pytest.fixture
def sample_wfn_file(test_data_dir):
    """Path to sample wavefunction file."""
    wfn_file = test_data_dir / "H2_CCSD.wfn"
    if not wfn_file.exists():
        pytest.skip("Test data file not available")
    return wfn_file


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_allclose_sorted():
    """Assert two arrays are close after sorting."""
    def _compare(a, b, rtol=1e-5, atol=1e-8):
        """Compare sorted arrays."""
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        np.testing.assert_allclose(a_sorted, b_sorted, rtol=rtol, atol=atol)
    return _compare


@pytest.fixture
def temporary_wavefunction():
    """Create a temporary wavefunction for testing."""
    def _create(n_atoms=2, n_basis=None):
        """Create a temporary wavefunction with n_atoms."""
        if n_basis is None:
            n_basis = n_atoms
        
        # Create atoms
        atoms = []
        for i in range(n_atoms):
            atoms.append(
                Atom(
                    element="H",
                    index=1,
                    x=float(i * 2.0),
                    y=0.0,
                    z=0.0,
                    charge=1.0,
                )
            )
        
        # Create shells
        shells = []
        for i in range(n_atoms):
            shells.append(
                Shell(
                    type=0,
                    center_idx=i,
                    exponents=np.array([1.0]),
                    coefficients=np.array([1.0]),
                )
            )
        
        # Create random coefficients
        coeffs = np.random.randn(n_basis, n_basis) * 0.1
        coeffs, _ = np.linalg.qr(coeffs)  # Orthonormalize
        
        # Create overlap matrix
        S = np.eye(n_basis)
        for i in range(min(n_atoms - 1, n_basis - 1)):
            S[i, i + 1] = S[i + 1, i] = 0.3
        
        # Create density matrix
        P = coeffs @ np.diag(np.ones(min(n_basis, n_atoms))) @ coeffs.T
        
        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=float(n_atoms),
            charge=0,
            multiplicity=1,
            num_basis=n_basis,
            num_atomic_orbitals=n_basis,
            num_primitives=n_basis,
            num_shells=len(shells),
            shells=shells,
            occupations=np.ones(n_basis),
            coefficients=coeffs,
            overlap_matrix=S,
            Ptot=P,
        )
        
        return wfn
    
    return _create


# ============================================================================
# Performance Testing
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def start(self):
            self._start = time.time()
        
        def stop(self):
            elapsed = time.time() - self._start
            self.times.append(elapsed)
            return elapsed
        
        def average(self):
            return np.mean(self.times) if self.times else 0.0
        
        def min(self):
            return np.min(self.times) if self.times else 0.0
        
        def max(self):
            return np.max(self.times) if self.times else 0.0
    
    return Timer()


# ============================================================================
# Test Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_density_cache():
    """Automatically clean up density cache after each test."""
    yield
    # Clean up after test
    try:
        from pymultiwfn.math.density import clear_density_cache
        clear_density_cache()
    except ImportError:
        pass


# ============================================================================
# Helper Functions
# ============================================================================

def skip_if_no_module(module_name):
    """Decorator to skip test if module is not available."""
    def decorator(func):
        try:
            __import__(module_name)
            return func
        except ImportError:
            return pytest.skip(f"Module {module_name} not available")(func)
    return decorator


# ============================================================================
# Reporting
# ============================================================================

@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information."""
    yield
    
    # Print custom summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    # Count tests by marker
    markers = {}
    for item in terminalreporter.stats.get("passed", []):
        for marker in item.keywords:
            if marker not in ["test", "passed", "failed"]:
                markers[marker] = markers.get(marker, 0) + 1
    
    if markers:
        print("\nTests by marker:")
        for marker, count in sorted(markers.items()):
            print(f"  {marker}: {count}")
    
    print("\n" + "=" * 60)
