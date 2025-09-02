@echo off
REM Script to start both frontend and backend servers on Windows

echo ==========================================
echo Brain Tumor Detection Application
echo ==========================================

REM Check if backend directory exists
if not exist "..\brain_mri-master" (
    echo Error: Backend directory not found!
    echo Please ensure brain_mri-master is in the parent directory.
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "..\brain_mri-master\models\best_model.pth" (
    echo Error: Model file not found!
    echo Please ensure best_model.pth is in brain_mri-master\models\
    pause
    exit /b 1
)

echo Starting Brain Tumor Detection Application...

REM Start the FastAPI backend server in a new command window
echo [1/2] Starting FastAPI backend server on http://localhost:12000...
start "Backend Server - Brain Tumor Detection" /D "..\brain_mri-master" cmd /c "python manage.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start the React frontend server
echo [2/2] Starting React frontend server on http://localhost:5173...
npm run dev

echo.
echo Application started successfully!
echo Frontend: http://localhost:5173
echo Backend: http://localhost:12000
echo.
echo Press Ctrl+C to stop the servers...
pause >nul