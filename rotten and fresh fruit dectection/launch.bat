@echo off
setlocal enabledelayedexpansion
title Fruit Freshness Detector - Launcher
cd /d "%~dp0"

echo ======================================================================
echo           FRUIT FRESHNESS DETECTOR - WINDOWS LAUNCHER
echo ======================================================================
echo.

:: Detect Python executable in virtual environment
if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
) else (
    echo [-] Virtual environment .venv was not found at %~dp0.venv!
    echo     Please make sure Python 3.12+ is installed and run:
    echo     python -m venv .venv
    echo     .\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check if trained ONNX model exists
if not exist "artifacts\fruit.onnx" (
    echo [!] Trained model artifacts\fruit.onnx was not found.
    echo.
    echo Choose an option:
    echo   1. Download dataset and train model now
    echo   2. Run dataset audit
    echo   3. Launch scanner anyway
    echo   4. Exit
    echo.
    set /p CHOICE="Enter choice (1-4): "
    if "!CHOICE!"=="1" (
        echo.
        echo [*] Downloading dataset...
        "%PYTHON_EXE%" download_dataset.py --variety apple
        echo.
        echo [*] Auditing dataset...
        "%PYTHON_EXE%" audit_dataset.py
        echo.
        echo [*] Training model (15 epochs)...
        "%PYTHON_EXE%" train.py --epochs 15 --batch-size 16
    ) else if "!CHOICE!"=="2" (
        "%PYTHON_EXE%" audit_dataset.py
        pause
        exit /b 0
    ) else if "!CHOICE!"=="4" (
        exit /b 0
    )
)

echo.
echo ======================================================================
echo Launching Live Fruit Freshness Scanner...
echo Controls:
echo   - Place ONE fruit inside the inspection square
echo   - Automatic classification runs continuously (0.5s interval)
echo   - Press 'S' to deliberately LOG and COUNT an item to batch & CSV
echo   - Press 'B' to start a new batch
echo   - Press 'Q' or ESC to quit
echo ======================================================================
echo.

"%PYTHON_EXE%" app.py scan --threshold 0.80 --auto-interval 0.5

if errorlevel 1 (
    echo.
    echo [-] Application stopped with an exit code: %errorlevel%
    pause
)
