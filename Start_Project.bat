@echo off
setlocal
cd /d "%~dp0"

rem ====================== SETTINGS: START ======================
title Bangla Fake News Detection

set "SITE_URL=http://127.0.0.1:8502"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "SERVER_FILE=%~dp0web\server.py"
set "MODEL_FILE=%~dp0models\banfakenews_tfidf_svm.joblib"
set "DATASET_FILE=%~dp0data\All60Kdataset.csv"
set "OFFLINE_PACKAGES=%~dp0offline_packages"
set "SERVER_LOG=%TEMP%\BanglaFakeNews_server.log"
rem ======================= SETTINGS: END =======================

rem ================ REQUIRED FILE CHECK: START ================
if not exist "%SERVER_FILE%" (
    echo ERROR: The website server file is missing.
    pause
    exit /b 1
)

if not exist "%MODEL_FILE%" (
    echo ERROR: The trained model file is missing.
    pause
    exit /b 1
)
rem ================= REQUIRED FILE CHECK: END ==================

rem ============== PYTHON ENVIRONMENT SETUP: START ==============
if not exist "%PYTHON_EXE%" (
    echo First-time setup: creating the project environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.14 -m venv ".venv" >nul 2>nul
        if errorlevel 1 py -3.12 -m venv ".venv" >nul 2>nul
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo.
            echo ERROR: Python 3.12 or 3.14 is not installed.
            echo Install a supported 64-bit Python version, then run this file again.
            pause
            exit /b 1
        )
        python -m venv ".venv"
    )

    if not exist "%PYTHON_EXE%" (
        echo ERROR: The Python environment could not be created.
        pause
        exit /b 1
    )
)
rem =============== PYTHON ENVIRONMENT SETUP: END ===============

rem ================= PACKAGE INSTALL: START ====================
"%PYTHON_EXE%" -c "import joblib, narwhals, numpy, scipy, sklearn" >nul 2>nul
if errorlevel 1 (
    echo Installing the required Python packages from the local folder...
    if exist "%OFFLINE_PACKAGES%\*.whl" (
        "%PYTHON_EXE%" -m pip install --disable-pip-version-check --no-index --find-links "%OFFLINE_PACKAGES%" -r "requirements.txt"
        if errorlevel 1 (
            echo.
            echo Offline installation was incomplete. Trying the online package source...
            "%PYTHON_EXE%" -m pip install --disable-pip-version-check -r "requirements.txt"
        )
    ) else (
        echo Local package files are missing. Trying the online package source...
        "%PYTHON_EXE%" -m pip install --disable-pip-version-check -r "requirements.txt"
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Required packages could not be installed.
        echo Check the offline_packages folder and run this file again.
        pause
        exit /b 1
    )

    "%PYTHON_EXE%" -c "import joblib, narwhals, numpy, scipy, sklearn" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ERROR: Package installation finished, but the environment is incomplete.
        echo Check Python compatibility and run this file again.
        pause
        exit /b 1
    )
)
rem ================== PACKAGE INSTALL: END =====================

rem ================= DATASET NOTICE: START =====================
if not exist "%DATASET_FILE%" (
    echo NOTE: The local dataset is missing, but the trained website can still run.
)
rem ================== DATASET NOTICE: END ======================

rem ================= SERVER STARTUP: START =====================
echo Checking the local website...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri '%SITE_URL%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 goto open_site

echo Starting the website...
if exist "%SERVER_LOG%" del /q "%SERVER_LOG%" >nul 2>nul
start "BanglaFakeNewsServer" /min cmd /d /c ""%PYTHON_EXE%" -B -u "%SERVER_FILE%" 1>"%SERVER_LOG%" 2>&1"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$url = '%SITE_URL%'; for ($i = 0; $i -lt 40; $i++) { try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1" >nul 2>nul
if not errorlevel 1 goto open_site

echo.
echo ERROR: The website could not start.
if exist "%SERVER_LOG%" (
    echo.
    type "%SERVER_LOG%"
)
pause
exit /b 1
rem ================== SERVER STARTUP: END ======================

rem ================== BROWSER OPEN: START ======================
:open_site
echo Website ready. Opening the browser...
start "" "%SITE_URL%"
if errorlevel 1 (
    echo Open this address manually: %SITE_URL%
    pause
    exit /b 1
)

exit /b 0
rem =================== BROWSER OPEN: END =======================
