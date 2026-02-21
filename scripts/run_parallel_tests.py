#!/usr/bin/env python3
"""
Test runner script for PyMultiWFN.

Provides convenient commands for running tests with various configurations.

Usage:
    python run_parallel_tests.py                    # Run all tests in parallel
    python run_parallel_tests.py --quick            # Quick test (skip slow tests)
    python run_parallel_tests.py --coverage         # Run with coverage report
    python run_parallel_tests.py --benchmark        # Run performance benchmarks
    python run_parallel_tests.py --integration      # Run integration tests
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(args):
    """Run tests with specified configuration."""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Parallel execution
    if args.parallel:
        if args.workers == "auto":
            cmd.extend(["-n", "auto"])
        else:
            cmd.extend(["-n", str(args.workers)])
    
    # Coverage
    if args.coverage:
        cmd.extend([
            "--cov=pymultiwfn",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    
    # Test selection
    if args.quick:
        cmd.extend(["-m", "not slow"])
    
    if args.unit_only:
        cmd.extend(["-m", "unit"])
    
    if args.integration:
        cmd.extend(["--runintegration"])
    
    if args.benchmark:
        cmd.extend(["-m", "benchmark", "--benchmark"])
    
    # Verbosity
    if args.verbose:
        cmd.append("-v")
    
    if args.debug:
        cmd.extend(["-vv", "-l", "--tb=long"])
    
    # Specific test file or directory
    if args.test_path:
        cmd.append(args.test_path)
    
    # Additional options
    if args.failfast:
        cmd.append("-x")
    
    if args.pdb:
        cmd.append("--pdb")
    
    # Print command
    print("=" * 70)
    print("Running tests:")
    print(" ".join(cmd))
    print("=" * 70)
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run PyMultiWFN tests with various configurations"
    )
    
    # Test execution options
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run tests in parallel (default: True)"
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution"
    )
    
    # Coverage options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    
    # Test selection
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow tests)"
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark tests"
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extra verbose output with full tracebacks"
    )
    
    # Other options
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Start debugger on failure"
    )
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test file or directory to run"
    )
    
    args = parser.parse_args()
    
    # Handle --no-parallel
    if args.no_parallel:
        args.parallel = False
    
    # Run tests
    exit_code = run_tests(args)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
