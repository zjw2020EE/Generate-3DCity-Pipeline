@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ================= CONFIGURATION =================
:: Set QGIS Path
set "QGIS_ROOT=D:\Programs\QGIS 3.40.11"
set "PYQGIS_BAT=%QGIS_ROOT%\bin\python-qgis-ltr.bat"

:: Set Blender Path
set "BLENDER_EXE=D:\Blender Foundation\Blender 3.6\blender.exe"

:: Set Conda Environment Name
set "CONDA_ENV=G3DC"
:: =============================================

:: 1. Ask for the number of rounds
set /p rounds="Enter the number of rounds to generate: "

:: 2. Activate Conda Environment
echo [INFO] Activating Conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Conda activate failed. Please run this script from Anaconda Prompt or ensure conda is in your PATH.
    pause
    exit /b
)
echo [INFO] Conda environment activated.

:: 3. Start Loop
for /L %%i in (1,1,%rounds%) do (
    echo.
    echo ==========================================
    echo        Processing Round %%i of %rounds%
    echo ==========================================

    REM  --- Step A: Calculate ID and Create Directory ---
    if not exist "dataset" mkdir "dataset"

    set "count=0"
    for /f %%A in ('dir "dataset" /b /ad 2^>nul ^| find /c /v ""') do set "count=%%A"

    set "scene_id=!count!"
    set "target_dir=dataset\scene_!scene_id!"
    
    echo [INFO] Current Scene ID: !scene_id!
    echo [INFO] Creating directory: !target_dir!
    mkdir "!target_dir!"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create directory: !target_dir!
        pause
        exit /b 1
    )

    REM  --- Step B: Execute Scripts Sequentially ---
    
    echo [EXEC] python roi.py 0
    python roi.py 0
    if !errorlevel! neq 0 (
        echo [ERROR] Execution failed at: roi.py
        echo [INFO] Stopping script to prevent further errors.
        pause
        exit /b 1
    )
    
    echo [EXEC] QGIS Script
    cmd /c "call "%PYQGIS_BAT%" q_gis.py"
    @REM if !errorlevel! neq 0 (
    @REM     echo [ERROR] Execution failed at: QGIS Script (q_gis.py)
    @REM     pause
    @REM     exit /b 1
    @REM )

    echo [EXEC] python vox.py
    python vox.py
    if !errorlevel! neq 0 (
        echo [ERROR] Execution failed at: vox.py
        pause
        exit /b 1
    )
    
    echo [EXEC] python gen_scene.py
    python gen_scene.py
    if !errorlevel! neq 0 (
        echo [ERROR] Execution failed at: gen_scene.py
        pause
        exit /b 1
    )
    
    echo [EXEC] Blender Script
    "%BLENDER_EXE%" --background --python blender_to_mitsuba.py
    if !errorlevel! neq 0 (
        echo [ERROR] Execution failed at: Blender Script
        pause
        exit /b 1
    )

    REM  --- Step C: Move Files and Clean Up ---
    
    if exist "tmp" (
        echo [MOVE] Moving files from tmp to !target_dir!
        xcopy "tmp\*" "!target_dir!\" /E /I /Y /H >nul
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to move files via XCOPY.
            pause
            exit /b 1
        )
        
        echo [CLEAN] Deleting tmp folder
        rmdir /s /q "tmp"
    ) else (
        echo [WARNING] 'tmp' folder not found. Skipping move.
    )

    echo [DONE] Round %%i finished.
)

echo.
echo ==========================================
echo All %rounds% rounds completed successfully.
echo ==========================================
pause