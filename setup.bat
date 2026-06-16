@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  VoiceTransfer — Windows Command Prompt setup script
::  Uses uv for fast dependency installation.
::  Skips any step that is already done. Safe to re-run.
::
::  Requires: uv  https://github.com/astral-sh/uv
::  Install uv (pick one):
::    pip install uv
::    winget install astral-sh.uv
:: ============================================================

echo.
echo ==================================================
echo   VoiceTransfer Setup
echo ==================================================

:: ── 1. Check uv ───────────────────────────────────────────
echo.
echo [1/4] Checking for uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: 'uv' is not installed. Install it with one of:
    echo     pip install uv
    echo     winget install astral-sh.uv
    echo.
    exit /b 1
)
for /f "tokens=*" %%v in ('uv --version 2^>^&1') do set UV_VER=%%v
echo   Found: %UV_VER%

:: ── 2. Create virtual environment ─────────────────────────
echo.
echo [2/4] Virtual environment...
if exist ".venv\" (
    echo   [skip] .venv already exists.
) else (
    echo   Creating .venv with uv...
    uv venv .venv
    if %errorlevel% neq 0 exit /b 1
    echo   [done] .venv created.
)

:: ── 3. Activate and install requirements ──────────────────
echo.
echo [3/4] Installing requirements...
call .venv\Scripts\activate.bat

python -c "import torch" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('python -c "import torch; print(torch.__version__)"') do set TORCH_VER=%%v
    echo   [skip] PyTorch !TORCH_VER! already installed.
    echo   (To force reinstall: uv pip install -r requirements.txt)
) else (
    echo   Running: uv pip install -r requirements.txt
    echo   (First run fetches CPU-only torch wheels -- may take a few minutes)
    uv pip install -r requirements.txt
    if %errorlevel% neq 0 exit /b 1
    echo   [done] Requirements installed.
)

:: ── 4. Pre-download model weights ─────────────────────────
echo.
echo [4/4] Model weights...
if exist "models\hub\checkpoints\wavlm_large_finetune.pt" (
    echo   [skip] Weights already present -- skipping download.
) else (
    echo   Running download_models.py (~650 MB on first run^)...
    python download_models.py
    if %errorlevel% neq 0 exit /b 1
    echo   [done] Weights downloaded and cached.
)

:: ── Summary ───────────────────────────────────────────────
echo.
echo ==================================================
echo   Setup complete!
echo.
echo   Streamlit UI  :  streamlit run app.py
echo   CLI           :  python run.py --config config.yaml
echo   Tests         :  pytest tests/ -v
echo ==================================================
echo.

endlocal
