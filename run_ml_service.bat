@echo off
title Petrophysical AI GPU Reasoning Service
cls
echo =======================================================================
echo   Starting Petrophysical AI GPU Reasoning & Inference Service...
echo =======================================================================
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe run_ml_service.py
) else (
    python run_ml_service.py
)
pause
