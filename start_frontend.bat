@echo off
chcp 65001 >nul
title LEGALMET AI — Frontend Portal
cd /d "%~dp0"

echo ================================================================
echo   LEGALMET AI — Inspector Web Portal
echo   SIH Problem Statement ID: SIH26034
echo ================================================================
echo.
echo [*] The frontend is automatically served by the FastAPI Backend at:
echo     http://localhost:8000/
echo.
echo [*] If you wish to run a dedicated frontend server on port 3000:
echo.

where npx >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Starting dedicated Node.js static server on http://localhost:3000 ...
    npx serve -s frontend -l 3000
) else (
    echo [*] Starting Python HTTP server for frontend on http://localhost:3000 ...
    python -m http.server 3000 --directory frontend
)

pause
