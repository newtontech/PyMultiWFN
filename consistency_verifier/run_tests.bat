@echo off
REM Windows batch script for running PyMultiWFN consistency tests

echo PyMultiWFN Consistency Test Runner
echo ==================================

REM Set Python environment
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Check if PyMultiWFN is available
python -c "import pymultiwfn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: PyMultiWFN not found or not installed
    echo Please install PyMultiWFN first: pip install -e .
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Check Multiwfn executable
set MULTIWFN_EXE=%PROJECT_ROOT%\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe
if not exist "%MULTIWFN_EXE%" (
    echo Error: Multiwfn executable not found at:
    echo %MULTIWFN_EXE%
    echo Please ensure Multiwfn is properly extracted
    exit /b 1
)

echo Found Multiwfn: %MULTIWFN_EXE%

REM Parse command line arguments
set TEST_TYPE=all
set PARALLEL=1

if "%1"=="quick" set TEST_TYPE=quick
if "%1"=="parallel" set PARALLEL=1
if "%1"=="sequential" set PARALLEL=0

echo Test Type: %TEST_TYPE%
echo Parallel Mode: %PARALLEL%
echo.

REM Create output directory
set REPORT_DIR=%PROJECT_ROOT%\consistency_verifier\test_reports
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

if "%TEST_TYPE%"=="quick" (
    echo Running quick tests...
    python "%SCRIPT_DIR%quick_test.py" --multiwfn "%MULTIWFN_EXE%"
) else (
    echo Running comprehensive test suite...
    python "%SCRIPT_DIR%test_runner.py" --multiwfn "%MULTIWFN_EXE%" --examples "%PROJECT_ROOT%\Multiwfn_3.8_dev_bin_Win64\examples"
)

echo.
echo Test completed. Check the reports directory for detailed results.
echo Reports location: %REPORT_DIR%

pause