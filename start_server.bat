@echo off
title Deep Research Agent - Server
cd /d "%~dp0"
set PYTHONPATH=src

echo ============================================
echo   Deep Research Agent starting...
echo.
echo   Open browser: http://127.0.0.1:8000
echo   Press Ctrl+C to stop the server.
echo ============================================
echo.

"D:\python\python.exe" -m uvicorn dra.server.app:app --host 127.0.0.1 --port 8000
pause