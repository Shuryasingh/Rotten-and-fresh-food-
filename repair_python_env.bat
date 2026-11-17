@echo off
setlocal
set "PROJECT=D:\rotten and fresh fruit dectection\rotten and fresh fruit dectection"
set "PYTHON313=C:\Users\Tejas mishra\AppData\Local\Programs\Python\Python313\python.exe"
set "LOG=%PROJECT%\repair_log.txt"

echo Repairing Python environment for the fruit app > "%LOG%"
echo Project: %PROJECT% >> "%LOG%"
if exist "%PROJECT%\.venv" (
  echo Removing stale virtual environment... >> "%LOG%"
  rmdir /s /q "%PROJECT%\.venv" >> "%LOG%" 2>&1
)

echo Creating Python 3.13 virtual environment... >> "%LOG%"
"%PYTHON313%" -m venv "%PROJECT%\.venv" >> "%LOG%" 2>&1

echo Upgrading pip... >> "%LOG%"
"%PROJECT%\.venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1

echo Installing project dependencies... >> "%LOG%"
"%PROJECT%\.venv\Scripts\python.exe" -m pip install -r "%PROJECT%\requirements.txt" >> "%LOG%" 2>&1

echo Verifying imports... >> "%LOG%"
"%PROJECT%\.venv\Scripts\python.exe" -c "import sys, numpy, cv2, torch, PIL, onnx, matplotlib, pytest; print(sys.version); print('PKG_OK')" >> "%LOG%" 2>&1

echo Completed. >> "%LOG%"
