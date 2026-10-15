@echo off
setlocal
cd /d "%~dp0"

rem ====================== SETTINGS: START ======================
title Bangla Fake News Project Check
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
rem ======================= SETTINGS: END =======================

rem ================= ENVIRONMENT CHECK: START ==================
if not exist "%PYTHON_EXE%" (
    echo ERROR: The local environment is not ready.
    echo Run Start_Project.bat once, then run this check again.
    pause
    exit /b 1
)
rem ================== ENVIRONMENT CHECK: END ===================

rem ================= PROJECT CHECKS: START =====================
echo [1/5] Checking required files...
if not exist "data\All60Kdataset.csv" goto missing_file
if not exist "models\banfakenews_tfidf_svm.joblib" goto missing_file
if not exist "web\server.py" goto missing_file
if not exist "web\index.html" goto missing_file

echo [2/5] Checking Python files...
"%PYTHON_EXE%" -m compileall -q src training evaluation web
if errorlevel 1 goto failed

echo [3/5] Checking evidence decisions...
"%PYTHON_EXE%" evaluation\evaluate_evidence_logic.py
if errorlevel 1 goto failed

echo [4/5] Checking RQ4 and RQ5...
"%PYTHON_EXE%" evaluation\evaluate_disagreement.py
if errorlevel 1 goto failed
"%PYTHON_EXE%" evaluation\evaluate_generalization.py
if errorlevel 1 goto failed

echo [5/5] Checking website GET and POST prediction...
"%PYTHON_EXE%" -m unittest evaluation\test_functional.py
if errorlevel 1 goto failed

echo.
echo SUCCESS: All independent project checks passed.
pause
exit /b 0
rem ================== PROJECT CHECKS: END ======================

rem ================= ERROR HANDLING: START =====================
:missing_file
echo ERROR: A required project file is missing.
pause
exit /b 1

:failed
echo.
echo ERROR: One or more checks failed. Review the output above.
pause
exit /b 1
rem ================== ERROR HANDLING: END ======================
