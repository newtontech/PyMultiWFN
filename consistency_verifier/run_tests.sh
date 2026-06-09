#!/usr/bin/env bash

# PyMultiWFN consistency verifier wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_CMD="${PYTHON:-python3}"
SUITE="smoke"
MULTIWFN_EXE="${MULTIWFN_BIN:-$PROJECT_ROOT/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn}"
RESULTS_DIR="$PROJECT_ROOT/consistency_verifier/results"
EXTRA_ARGS=()

usage() {
    cat <<USAGE
Usage: $0 [smoke|quick|pr|full] [options]

Options:
  --suite NAME       Run smoke, pr, or full
  --multiwfn PATH    Path to the Multiwfn executable oracle
  --results-dir DIR  Directory for generated verifier reports
  --no-report        Do not write report files
  --help, -h         Show this help message

Environment:
  PYTHON             Python executable to use
  MULTIWFN_BIN       Default Multiwfn executable path
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        quick|smoke)
            SUITE="smoke"
            shift
            ;;
        pr)
            SUITE="pr"
            shift
            ;;
        full)
            SUITE="full"
            shift
            ;;
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --multiwfn)
            MULTIWFN_EXE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --no-report)
            EXTRA_ARGS+=("--no-report")
            shift
            ;;
        --skip-oracle-if-unavailable)
            EXTRA_ARGS+=("--skip-oracle-if-unavailable")
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SUITE" in
    smoke|pr|full)
        ;;
    *)
        echo "Unknown suite: $SUITE" >&2
        usage >&2
        exit 2
        ;;
esac

cd "$PROJECT_ROOT"

echo "PyMultiWFN consistency verifier"
echo "Python: $($PYTHON_CMD --version 2>&1)"
echo "Suite: $SUITE"
echo "Multiwfn oracle: $MULTIWFN_EXE"
echo "Results: $RESULTS_DIR"
echo

"$PYTHON_CMD" -m consistency_verifier run \
    --suite "$SUITE" \
    --multiwfn-bin "$MULTIWFN_EXE" \
    --results-dir "$RESULTS_DIR" \
    "${EXTRA_ARGS[@]}"
