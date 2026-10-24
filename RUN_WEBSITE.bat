@echo off
title Synthetic Media Detection
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please set up the project environment before running this launcher.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
streamlit run app.py

if errorlevel 1 pause
