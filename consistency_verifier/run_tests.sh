#!/bin/bash

# PyMultiWFN Consistency Test Runner for Linux/macOS
# This script runs the consistency testing suite

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}PyMultiWFN Consistency Test Runner${NC}"
echo -e "${BLUE}==================================${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found in PATH${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Python: $PYTHON_VERSION${NC}"

# Check if PyMultiWFN is available
if ! $PYTHON_CMD -c "import pymultiwfn" 2>/dev/null; then
    echo -e "${RED}Error: PyMultiWFN not found or not installed${NC}"
    echo -e "${YELLOW}Please install PyMultiWFN first: pip install -e .${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check Multiwfn executable
MULTIWFN_EXE=""
# Try different possible locations
MULTIWFN_PATHS=(
    "$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Win64/Multiwfn.exe"
    "$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Linux/Multiwfn"
    "$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Mac/Multiwfn"
    "$PROJECT_ROOT/Multiwfn_3.8_dev_src_Linux_*/Multiwfn"
    "/usr/local/bin/Multiwfn"
    "/usr/bin/Multiwfn"
)

for path in "${MULTIWFN_PATHS[@]}"; do
    if [[ -f "$path" ]] || [[ -L "$path" ]]; then
        # Check if it's executable
        if [[ -x "$path" ]] || [[ "$path" == *.exe ]]; then
            MULTIWFN_EXE="$path"
            break
        fi
    fi
done

# Handle wildcard path
if [[ -z "$MULTIWFN_EXE" ]]; then
    for path in $PROJECT_ROOT/Multiwfn_3.8_dev_src_Linux_*/Multiwfn; do
        if [[ -f "$path" ]] && [[ -x "$path" ]]; then
            MULTIWFN_EXE="$path"
            break
        fi
    done
fi

if [[ -z "$MULTIWFN_EXE" ]]; then
    echo -e "${RED}Error: Multiwfn executable not found${NC}"
    echo -e "${YELLOW}Searched locations:${NC}"
    for path in "${MULTIWFN_PATHS[@]}"; do
        echo "  - $path"
    done
    echo -e "${YELLOW}Please ensure Multiwfn is properly installed or specify the path with --multiwfn${NC}"
    exit 1
fi

echo -e "${GREEN}Found Multiwfn: $MULTIWFN_EXE${NC}"

# Parse command line arguments
TEST_TYPE="all"
PARALLEL="1"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        "quick")
            TEST_TYPE="quick"
            shift
            ;;
        "parallel")
            PARALLEL="1"
            shift
            ;;
        "sequential")
            PARALLEL="0"
            shift
            ;;
        "--multiwfn")
            MULTIWFN_EXE="$2"
            shift 2
            ;;
        "--examples")
            EXAMPLES_DIR="$2"
            shift 2
            ;;
        "--help"|"-h")
            echo "Usage: $0 [quick|parallel|sequential] [--multiwfn PATH] [--examples PATH]"
            echo ""
            echo "Options:"
            echo "  quick      Run quick tests only"
            echo "  parallel   Run tests in parallel (default)"
            echo "  sequential Run tests sequentially"
            echo "  --multiwfn PATH   Specify Multiwfn executable path"
            echo "  --examples PATH   Specify examples directory"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Test Type: ${YELLOW}$TEST_TYPE${NC}"
echo -e "  Parallel Mode: ${YELLOW}$PARALLEL${NC}"
echo ""

# Set default examples directory if not specified
if [[ -z "$EXAMPLES_DIR" ]]; then
    if [[ -d "$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Win64/examples" ]]; then
        EXAMPLES_DIR="$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Win64/examples"
    elif [[ -d "$PROJECT_ROOT/examples" ]]; then
        EXAMPLES_DIR="$PROJECT_ROOT/examples"
    else
        echo -e "${YELLOW}Warning: Examples directory not found, using default${NC}"
        EXAMPLES_DIR="$PROJECT_ROOT/Multiwfn_3.8_dev_bin_Win64/examples"
    fi
fi

if [[ ! -d "$EXAMPLES_DIR" ]]; then
    echo -e "${YELLOW}Warning: Examples directory not found at $EXAMPLES_DIR${NC}"
    echo "Some tests may fail if example files are required"
fi

echo -e "${BLUE}Examples Directory: $EXAMPLES_DIR${NC}"
echo ""

# Create output directory
REPORT_DIR="$PROJECT_ROOT/consistency_verifier/test_reports"
mkdir -p "$REPORT_DIR"
echo -e "${GREEN}Reports directory: $REPORT_DIR${NC}"

# Change to project root to ensure relative imports work
cd "$PROJECT_ROOT"

# Run the tests
if [[ "$TEST_TYPE" == "quick" ]]; then
    echo -e "${BLUE}Running quick tests...${NC}"
    if $PYTHON_CMD "$SCRIPT_DIR/quick_test.py" --multiwfn "$MULTIWFN_EXE" $EXTRA_ARGS; then
        echo -e "${GREEN}✅ Quick tests completed successfully${NC}"
    else
        echo -e "${RED}❌ Quick tests failed${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}Running comprehensive test suite...${NC}"
    if $PYTHON_CMD "$SCRIPT_DIR/test_runner.py" \
        --multiwfn "$MULTIWFN_EXE" \
        --examples "$EXAMPLES_DIR" \
        --parallel "$PARALLEL" $EXTRA_ARGS; then
        echo -e "${GREEN}✅ Comprehensive tests completed successfully${NC}"
    else
        echo -e "${RED}❌ Comprehensive tests failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Test completed.${NC}"
echo -e "${BLUE}Check the reports directory for detailed results:${NC}"
echo -e "${YELLOW}$REPORT_DIR${NC}"

# Show latest reports if they exist
LATEST_REPORT=$(ls -t "$REPORT_DIR"/test_report_*.txt 2>/dev/null | head -1)
if [[ -n "$LATEST_REPORT" ]]; then
    echo ""
    echo -e "${BLUE}Latest report summary:${NC}"
    if [[ -f "$LATEST_REPORT" ]]; then
        # Show first 20 lines of the report
        head -20 "$LATEST_REPORT" | while IFS= read -r line; do
            echo "  $line"
        done
        echo "  ..."
        echo -e "${BLUE}Full report available at: $(basename "$LATEST_REPORT")${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"