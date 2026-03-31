@echo off
setlocal

echo ============================================
echo   AI Thermostat Agent - Startup
echo ============================================
echo.

set AGENT_DIR=%~dp0

:: Check llama-server exists (needed on-demand)
if not exist "C:\VashwarTests\EmailOrganizer\llama-server\llama-server.exe" (
    echo ERROR: llama-server not found
    pause
    exit /b 1
)

cd /d "%AGENT_DIR%"

:: Install requirements if needed
pip install -q -r requirements.txt 2>nul

echo LLM server will start on-demand during evaluations.
echo Bot and monitoring are always active.
echo.

:: Launch the agent (LLM server managed automatically)
python agent.py

:: If agent exits, keep window open
echo.
echo Agent has stopped.
pause
