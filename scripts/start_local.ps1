param(
    [switch]$Install,
    [switch]$PullModel,
    [switch]$StartDatabase,
    [switch]$InitializeDatabase,
    [switch]$CheckOnly,
    [string]$EnvFile = '',
    [int]$Port = 7860
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not $EnvFile) { $EnvFile = Join-Path $projectRoot '.env' }
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        if (-not $Install) { throw 'Missing .venv. Rerun with -Install (Python 3.10+ required).' }
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Check Python installation.' }
    }
    if ($Install) {
        & $pythonPath -m pip install -r requirements/local.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
        & $pythonPath -m pip install -e .
        if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
    }
    $runtimeArgs = @('scripts/setup_local.py', '--env-file', $EnvFile, '--port', "$Port")
    if ($PullModel) { $runtimeArgs += '--pull-model' }
    if ($StartDatabase) { $runtimeArgs += '--start-database' }
    if ($InitializeDatabase) { $runtimeArgs += '--initialize-database' }
    if ($CheckOnly) { $runtimeArgs += '--check-only' }
    & $pythonPath @runtimeArgs
    if ($LASTEXITCODE -ne 0) { throw 'Startup failed. Resolve the error above and rerun.' }
} finally {
    Pop-Location
}
