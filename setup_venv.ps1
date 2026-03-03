param(
    [string]$VenvDir = ".venv",
    [string]$PyVersion = "3.12",
    [string]$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Invoke-OrThrow {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$ErrorMessage
    )
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Get-TrustedHost {
    param([string]$Url)
    try {
        return ([uri]$Url).Host
    } catch {
        return $null
    }
}

function Resolve-PythonSpec {
    param([string]$Preferred)

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pyCmd) {
            throw "Python is not installed. Please install Python 3.11 or 3.12 first."
        }

        $ver = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to query Python version."
        }
        if ($ver -notin @("3.10", "3.11", "3.12")) {
            throw "Detected python=$ver. This project requires Python 3.10-3.12 (recommended 3.12)."
        }
        return @{ Kind = "python"; Spec = $ver }
    }

    $candidates = @($Preferred, "3.11", "3.12", "3.10") | Select-Object -Unique
    foreach ($spec in $candidates) {
        & py -$spec -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return @{ Kind = "py"; Spec = $spec }
        }
    }

    throw "No suitable Python runtime found. Install Python 3.11 or 3.12, then rerun setup_venv.ps1."
}

$resolved = Resolve-PythonSpec -Preferred $PyVersion

if ($resolved.Kind -eq "py") {
    Write-Host "[info] creating venv with py -$($resolved.Spec)"
    Invoke-OrThrow -Exe "py" -Args @("-$($resolved.Spec)", "-m", "venv", $VenvDir) -ErrorMessage "Failed to create virtual environment."
} else {
    Write-Host "[info] creating venv with python $($resolved.Spec)"
    Invoke-OrThrow -Exe "python" -Args @("-m", "venv", $VenvDir) -ErrorMessage "Failed to create virtual environment."
}

$py = Join-Path $root "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Virtual environment python not found: $py"
}

$trustedHost = Get-TrustedHost -Url $IndexUrl

$pipUpgradeArgs = @("-m", "pip", "install", "--upgrade", "pip")
$pipInstallArgs = @("-m", "pip", "install", "--only-binary=:all:", "-r", "requirements.txt")
if ($IndexUrl) {
    $pipUpgradeArgs += @("-i", $IndexUrl)
    $pipInstallArgs += @("-i", $IndexUrl)
    if ($trustedHost) {
        $pipUpgradeArgs += @("--trusted-host", $trustedHost)
        $pipInstallArgs += @("--trusted-host", $trustedHost)
    }
}

Write-Host "[info] upgrading pip ..."
Invoke-OrThrow -Exe $py -Args $pipUpgradeArgs -ErrorMessage "pip upgrade failed."

Write-Host "[info] installing requirements ..."
try {
    Invoke-OrThrow -Exe $py -Args $pipInstallArgs -ErrorMessage "requirements installation failed."
} catch {
    if ($IndexUrl) {
        Write-Host "[warn] install via mirror failed, retrying with default index ..."
        Invoke-OrThrow -Exe $py -Args @("-m", "pip", "install", "--only-binary=:all:", "-r", "requirements.txt") -ErrorMessage "requirements installation failed on both mirror and default index."
    } else {
        throw
    }
}

Write-Host "[OK] venv ready -> $VenvDir"
Write-Host "Run demo with:"
Write-Host "`"$py`" app.py --camera 0 --smoothing 5 --threshold 0.55 --margin 0.06 --backend tasks"
