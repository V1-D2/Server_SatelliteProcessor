@echo off
echo ========================================
echo Building SatelliteProcessor Executable
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install/upgrade PyInstaller if needed
echo Installing/upgrading PyInstaller...
pip install --upgrade pyinstaller

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del /q "*.spec"

REM Create icon if it doesn't exist (placeholder)
if not exist "assets\icon.ico" (
    echo Creating placeholder icon...
    mkdir assets 2>nul
    echo. > assets\icon.ico
)

REM Build executable
echo.
echo Building executable...
pyinstaller --onefile ^
    --windowed ^
    --name="SatelliteProcessor" ^
    --icon="assets\icon.ico" ^
    --add-data="config;config" ^
    --add-data="assets;assets" ^
    --hidden-import="tkinter" ^
    --hidden-import="h5py" ^
    --hidden-import="numpy" ^
    --hidden-import="matplotlib" ^
    --hidden-import="pyproj" ^
    --hidden-import="PIL" ^
    --hidden-import="scipy" ^
    --hidden-import="xarray" ^
    --hidden-import="tqdm" ^
    --collect-all="pyproj" ^
    --collect-all="matplotlib" ^
    main.py

REM Check if build was successful
if exist "dist\SatelliteProcessor.exe" (
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo Executable location: dist\SatelliteProcessor.exe
    echo ========================================

    REM Create distribution folder structure
    echo.
    echo Creating distribution folder...
    mkdir "dist\config" 2>nul
    mkdir "dist\temp" 2>nul

    REM Copy any necessary files
    REM xcopy /y "README.md" "dist\" 2>nul

) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo Please check the error messages above.
    echo ========================================
)

echo.
pause