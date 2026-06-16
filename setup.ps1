<#
.SYNOPSIS
    Set up VoiceTransfer on Windows using uv: create venv, install requirements,
    pre-download model weights. Skips any step that is already done.

.NOTES
    Requires: uv  https://github.com/astral-sh/uv
    Install uv (pick one):
        pip install uv
        winget install astral-sh.uv
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    If PowerShell blocks this script, run once in an elevated prompt:
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    Or bypass per-invocation:
        powershell -ExecutionPolicy Bypass -File setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}
function Write-Skip([string]$msg) {
    Write-Host "    [skip] $msg" -ForegroundColor DarkGray
}
function Write-Done([string]$msg) {
    Write-Host "    [done] $msg" -ForegroundColor Green
}
function Write-Fail([string]$msg) {
    Write-Host "    [fail] $msg" -ForegroundColor Red
    exit 1
}

# ── 1. Check uv is installed ───────────────────────────────────────────────────

Write-Step "Checking for uv..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "  'uv' is not installed. Install it with one of:" -ForegroundColor Yellow
    Write-Host "    pip install uv"
    Write-Host "    winget install astral-sh.uv"
    Write-Host "    powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    Write-Fail "uv not found — aborting."
}
$uvVer = (uv --version 2>&1)
Write-Done "Found: $uvVer"

# ── 2. Create virtual environment ─────────────────────────────────────────────

Write-Step "Virtual environment..."
if (Test-Path ".venv") {
    Write-Skip ".venv already exists — skipping creation."
} else {
    Write-Host "    Creating .venv with uv..." -ForegroundColor White
    uv venv .venv
    Write-Done ".venv created."
}

# ── 3. Activate virtual environment ───────────────────────────────────────────

Write-Step "Activating virtual environment..."
$activateScript = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Fail "Activate script not found at: $activateScript"
}
. $activateScript
Write-Done "Active: $env:VIRTUAL_ENV"

# ── 4. Install requirements ────────────────────────────────────────────────────

Write-Step "Installing requirements..."

# Check whether PyTorch is already importable in this venv.
$torchVer = & python -c "import torch; print(torch.__version__)" 2>$null
if ($LASTEXITCODE -eq 0 -and $torchVer) {
    Write-Skip "PyTorch $torchVer already installed — skipping."
    Write-Host "    (To force reinstall: uv pip install -r requirements.txt)" -ForegroundColor DarkGray
} else {
    Write-Host "    Running: uv pip install -r requirements.txt" -ForegroundColor White
    Write-Host "    (First run fetches CPU-only torch wheels — may take a few minutes)" -ForegroundColor DarkGray
    uv pip install -r requirements.txt
    Write-Done "Requirements installed."
}

# ── 5. Pre-download model weights ─────────────────────────────────────────────

Write-Step "Model weights..."
$sentinel = ".\models\hub\checkpoints\wavlm_large_finetune.pt"
if (Test-Path $sentinel) {
    Write-Skip "Weights already present at $sentinel — skipping download."
} else {
    Write-Host "    Running download_models.py (~650 MB on first run)..." -ForegroundColor White
    python download_models.py
    Write-Done "Weights downloaded and cached."
}

# ── Summary ────────────────────────────────────────────────────────────────────

$line = "-" * 54
Write-Host "`n$line" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Streamlit UI  :  streamlit run app.py"
Write-Host "  CLI           :  python run.py --config config.yaml"
Write-Host "  Tests         :  pytest tests/ -v"
Write-Host "$line" -ForegroundColor Cyan
