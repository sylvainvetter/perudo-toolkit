@echo off
chcp 65001 >nul
title Perudo Toolkit

echo === Perudo Toolkit ===
echo.

REM --- Stop any process already using port 8000 ---
echo Arret du serveur existant (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM --- Install / update dependencies ---
echo Installation des dependances...
pip install -e . -q
if %errorlevel% neq 0 (
    echo ERREUR : pip install a echoue.
    pause
    exit /b 1
)

REM --- Launch server ---
echo.
echo Demarrage sur http://localhost:8000
echo Ctrl+C pour arreter.
echo.
start "" "http://localhost:8000"
python -m uvicorn perudo.web.app:app --reload --port 8000
