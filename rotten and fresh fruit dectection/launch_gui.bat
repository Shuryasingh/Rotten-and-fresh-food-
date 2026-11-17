@echo off
title Fruit Freshness Detector - Web GUI
cd /d "%~dp0"

echo ======================================================================
echo       FRUIT FRESHNESS DETECTOR - WEB / DESKTOP GUI LAUNCHER
echo ======================================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
) else (
    echo [-] Virtual environment .venv was not found!
    pause
    exit /b 1
)

echo [*] Starting Web GUI server and opening your browser...
"%PYTHON_EXE%" web_app.py

if errorlevel 1 (
    echo.
    echo [-] Application stopped with an exit code: %errorlevel%
    pause
)
