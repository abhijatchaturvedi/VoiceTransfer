#!/usr/bin/env bash
# setup.sh — Set up VoiceTransfer on macOS / Linux using uv.
# Skips any step that has already been completed.
#
# Requires: uv  https://github.com/astral-sh/uv
#   Install:
#     pip install uv
#     OR: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   bash setup.sh          (safe to re-run — idempotent)
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────

CYAN='\033[0;36m'; GREEN='\033[0;32m'; GRAY='\033[0;90m'
YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'

step()  { echo -e "\n${CYAN}==> $*${NC}"; }
skip()  { echo -e "    ${GRAY}[skip] $*${NC}"; }
done_() { echo -e "    ${GREEN}[done] $*${NC}"; }
fail()  { echo -e "${RED}[fail] $*${NC}" >&2; exit 1; }

# Resolve the directory this script lives in so relative paths work regardless
# of where it is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Check uv is installed ───────────────────────────────────────────────────

step "Checking for uv..."
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}"
    echo "  'uv' is not installed. Install it with one of:"
    echo "    pip install uv"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo -e "${NC}"
    fail "uv not found — aborting."
fi
done_ "Found: $(uv --version)"

# ── 2. Create virtual environment ─────────────────────────────────────────────

step "Virtual environment..."
if [ -d ".venv" ]; then
    skip ".venv already exists — skipping creation."
else
    echo "    Creating .venv with uv..."
    uv venv .venv
    done_ ".venv created."
fi

# ── 3. Activate virtual environment ───────────────────────────────────────────

step "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate
done_ "Active: $VIRTUAL_ENV"

# ── 4. Install requirements ────────────────────────────────────────────────────

step "Installing requirements..."
if python -c "import torch" &>/dev/null 2>&1; then
    torch_ver=$(python -c "import torch; print(torch.__version__)")
    skip "PyTorch $torch_ver already installed — skipping."
    echo -e "    ${GRAY}(To force reinstall: uv pip install -r requirements.txt)${NC}"
else
    echo "    Running: uv pip install -r requirements.txt"
    echo -e "    ${GRAY}(First run fetches CPU-only torch wheels — may take a few minutes)${NC}"
    uv pip install -r requirements.txt
    done_ "Requirements installed."
fi

# ── 5. Pre-download model weights ─────────────────────────────────────────────

step "Model weights..."
SENTINEL="./models/hub/checkpoints/wavlm_large_finetune.pt"
if [ -f "$SENTINEL" ]; then
    skip "Weights already present at $SENTINEL — skipping download."
else
    echo "    Running download_models.py (~650 MB on first run)..."
    python download_models.py
    done_ "Weights downloaded and cached."
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo ""
echo "  Streamlit UI  :  streamlit run app.py"
echo "  CLI           :  python run.py --config config.yaml"
echo "  Tests         :  pytest tests/ -v"
echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"
