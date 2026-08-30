@echo off
chcp 65001 >nul
title LEGALMET AI — Backend Server
cd /d "%~dp0"

echo ================================================================
echo   LEGALMET AI — AI-Assisted Legal Metrology Inspection Platform
echo   SIH Problem Statement ID: SIH26034
echo   Ministry of Consumer Affairs, Food & Public Distribution
echo ================================================================
echo.

:: Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [!] No virtual environment found. Using system Python.
)

:: Check if Python is accessible
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo.
echo [*] Starting FastAPI Backend on http://0.0.0.0:8000 ...
echo [*] Inspector Web Portal: http://localhost:8000/
echo [*] API Documentation:   http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo ================================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
