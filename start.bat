@echo off
echo ========================================
echo SHL Assessment Recommender
echo ========================================
echo.

echo Starting API server...
echo.
echo API will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:8080
echo API docs available at: http://localhost:8000/docs
echo.

start "SHL API" cmd /k "uvicorn api:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

start "SHL Frontend" cmd /k "python -m http.server 8080 --directory frontend"

timeout /t 2 /nobreak > nul

echo.
echo ========================================
echo Services started!
echo ========================================
echo.
echo Opening browser...
start http://localhost:8080

echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WindowTitle eq SHL API*" /T /F
taskkill /FI "WindowTitle eq SHL Frontend*" /T /F

echo.
echo Services stopped.
