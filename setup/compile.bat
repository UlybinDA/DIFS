@echo off

REM Activate virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    echo Please check if the path is correct: ..\venv\Scripts\activate.bat
    pause
    exit /b 1
)

echo ========================================
echo    CYTHON MODULES COMPILATION
echo ========================================
echo.

REM Pass all arguments directly to Python
python setup.py %*

if errorlevel 1 (
    echo.
    echo Compilation failed!
) else (
    echo.
    echo Compilation completed!
)

pause