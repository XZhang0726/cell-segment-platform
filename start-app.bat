@echo off
setlocal
cd /d "%~dp0"
echo Cell Segmentation Platform
echo.

if exist ".venv\Scripts\python.exe" (
    set "APP_PYTHON=.venv\Scripts\python.exe"
) else (
    set "APP_PYTHON=python"
)

"%APP_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit was not found in the selected Python environment.
    echo Follow installation.md, activate your environment, and try again.
    pause
    exit /b 1
)

echo Starting the application. Press Ctrl+C to stop.
"%APP_PYTHON%" -m streamlit run app_enhanced.py --server.headless=false
endlocal
