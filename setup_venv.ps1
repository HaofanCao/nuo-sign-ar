param(
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python -m venv $VenvDir

$py = Join-Path $root "$VenvDir\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Write-Host "[OK] venv ready -> $VenvDir"
Write-Host "Run demo with:"
Write-Host "`"$py`" app.py --camera 0 --smoothing 5 --threshold 0.55 --margin 0.06 --backend tasks"
