@echo off
title Rotten and Fresh Food Checker - Unified Launcher
cd /d "%~dp0"

echo ======================================================================
echo       ROTTEN AND FRESH FOOD CHECKER - UNIFIED WEB LAUNCHER
echo ======================================================================
echo.

set "PYTHON_EXE=%~dp0rotten and fresh fruit dectection\.venv\Scripts\python.exe"

if not exist "!PYTHON_EXE!" (
    echo [-] Python virtual environment was not found at:
    echo     !PYTHON_EXE!
    pause
    exit /b 1
)

echo [*] Starting Unified AI Vision and Web Server on http://localhost:8080 ...
cd /d "%~dp0rotten and fresh fruit dectection"
start "Fruit Freshness Backend" /b "%PYTHON_EXE%" web_app.py

:: Give server 1.5 seconds to bind socket
timeout /t 2 /nobreak >nul

echo [*] Opening application in your default browser...
start "" "http://localhost:8080"

echo.
echo ======================================================================
echo  [+] Rotten and Fresh Food Checker is ACTIVE!
echo      - Web Platform:    http://localhost:8080
echo      - AI Vision API:   http://localhost:8080/api/predict
echo      - Classic Mode:    http://localhost:8080/classic
echo ======================================================================
echo.
echo Press any key to exit this launcher window (server remains running in background)...
pause >nul
