@echo off
cd /d %~dp0
if not exist ".venv\Scripts\python.exe" (
  powershell -ExecutionPolicy Bypass -File setup_venv.ps1
)
.venv\Scripts\python.exe app.py --camera 0 --smoothing 5 --threshold 0.55 --margin 0.06 --backend tasks
