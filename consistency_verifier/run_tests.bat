@echo off
REM PyMultiWFN consistency verifier wrapper for Windows shells.

setlocal

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set SUITE=smoke
set RESULTS_DIR=%PROJECT_ROOT%\consistency_verifier\results

if "%MULTIWFN_BIN%"=="" (
    set MULTIWFN_EXE=%PROJECT_ROOT%\Multiwfn_3.8_bin_Linux_noGUI\Multiwfn
) else (
    set MULTIWFN_EXE=%MULTIWFN_BIN%
)

if "%1"=="quick" set SUITE=smoke
if "%1"=="smoke" set SUITE=smoke
if "%1"=="pr" set SUITE=pr
if "%1"=="full" set SUITE=full

echo PyMultiWFN consistency verifier
python --version
echo Suite: %SUITE%
echo Multiwfn oracle: %MULTIWFN_EXE%
echo Results: %RESULTS_DIR%
echo.

python -m consistency_verifier run ^
    --suite "%SUITE%" ^
    --multiwfn-bin "%MULTIWFN_EXE%" ^
    --results-dir "%RESULTS_DIR%"

endlocal
