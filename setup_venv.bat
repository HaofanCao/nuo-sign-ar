@echo off
cd /d %~dp0
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
