#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Kubix — Full Workspace Setup
#  Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}"
  echo "  ██╗  ██╗██╗   ██╗██████╗ ██╗██╗  ██╗"
  echo "  ██║ ██╔╝██║   ██║██╔══██╗██║╚██╗██╔╝"
  echo "  █████╔╝ ██║   ██║██████╔╝██║ ╚███╔╝ "
  echo "  ██╔═██╗ ██║   ██║██╔══██╗██║ ██╔██╗ "
  echo "  ██║  ██╗╚██████╔╝██████╔╝██║██╔╝ ██╗"
  echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═╝"
  echo -e "${RESET}"
  echo -e "  ${BOLD}AI GPU Performance Engineer${RESET} · kubix.ai"
  echo ""
}

step()  { echo -e "\n${CYAN}${BOLD}▶ $1${RESET}"; }
ok()    { echo -e "  ${GREEN}✓ $1${RESET}"; }
warn()  { echo -e "  ${YELLOW}⚠ $1${RESET}"; }
fail()  { echo -e "  ${RED}✗ $1${RESET}"; exit 1; }

# ─────────────────────────────────────────────
# 1. Check prerequisites
# ─────────────────────────────────────────────
banner

step "Checking prerequisites"

# Python
if command -v python3 &>/dev/null; then
  PY=$(python3 --version 2>&1)
  ok "Python: $PY"
  PYTHON=python3
elif command -v python &>/dev/null; then
  PY=$(python --version 2>&1)
  ok "Python: $PY"
  PYTHON=python
else
  fail "Python 3.10+ is required. Install from https://python.org"
fi

# pip
if ! $PYTHON -m pip --version &>/dev/null; then
  fail "pip not found. Run: $PYTHON -m ensurepip"
fi
ok "pip: $($PYTHON -m pip --version | awk '{print $2}')"

# Git
if command -v git &>/dev/null; then
  ok "git: $(git --version | awk '{print $3}')"
else
  fail "git is required. Install from https://git-scm.com"
fi

# Node / npm (optional — for kubix-mcp)
if command -v node &>/dev/null && command -v npm &>/dev/null; then
  ok "node: $(node --version)  npm: $(npm --version)"
  HAS_NODE=true
else
  warn "Node/npm not found — skipping kubix-mcp build (optional)"
  HAS_NODE=false
fi

# GPU (informational)
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  GPU=$($PYTHON -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, f'{p.total_memory/1e9:.0f}GB')")
  ok "GPU detected: $GPU"
  IS_GPU=true
else
  warn "No CUDA GPU detected — kernel profiling requires a GPU (Colab/RunPod)"
  IS_GPU=false
fi

# ─────────────────────────────────────────────
# 2. kubix-cli
# ─────────────────────────────────────────────
step "Installing kubix-cli"

if [ -d "$SCRIPT_DIR/kubix-cli" ]; then
  $PYTHON -m pip install -e "$SCRIPT_DIR/kubix-cli" -q
  ok "kubix-cli installed"
else
  warn "kubix-cli directory not found — skipping"
fi

# ─────────────────────────────────────────────
# 3. kubix-agent
# ─────────────────────────────────────────────
step "Installing kubix-agent"

if [ -d "$SCRIPT_DIR/kubix-agent" ]; then
  if [ "$IS_GPU" = true ]; then
    $PYTHON -m pip install -e "$SCRIPT_DIR/kubix-agent[cuda,models,kernelbench]" -q
    ok "kubix-agent installed (cuda + models + kernelbench)"
  else
    $PYTHON -m pip install -e "$SCRIPT_DIR/kubix-agent[models,kernelbench]" -q
    ok "kubix-agent installed (models + kernelbench, no cuda)"
  fi
else
  warn "kubix-agent directory not found — skipping"
fi

# triton (Linux + CUDA only)
if [ "$IS_GPU" = true ] && [[ "$(uname)" == "Linux" ]]; then
  step "Installing Triton"
  $PYTHON -m pip install triton -q
  ok "triton installed"
fi

# ─────────────────────────────────────────────
# 4. kubix-ci
# ─────────────────────────────────────────────
step "Installing kubix-ci"

if [ -d "$SCRIPT_DIR/kubix-ci" ]; then
  $PYTHON -m pip install -e "$SCRIPT_DIR/kubix-ci" -q
  ok "kubix-ci installed"
else
  warn "kubix-ci directory not found — skipping"
fi

# ─────────────────────────────────────────────
# 5. kubix-mcp (TypeScript — optional)
# ─────────────────────────────────────────────
step "Building kubix-mcp (MCP server for VS Code / Cursor)"

if [ "$HAS_NODE" = true ] && [ -d "$SCRIPT_DIR/kubix-mcp" ]; then
  cd "$SCRIPT_DIR/kubix-mcp"
  npm install -q
  npm run build
  cd "$SCRIPT_DIR"
  ok "kubix-mcp built → kubix-mcp/dist/"
else
  warn "Skipping kubix-mcp (requires Node.js)"
fi

# ─────────────────────────────────────────────
# 6. API key setup
# ─────────────────────────────────────────────
step "Setting up API key"

CONFIG_DIR="$HOME/.kubix-cli"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ] && $PYTHON -c "import json; d=json.load(open('$CONFIG_FILE')); assert d.get('openrouter_api_key','')" 2>/dev/null; then
  ok "API key already configured ($CONFIG_FILE)"
else
  echo ""
  echo -e "  ${BOLD}Enter your OpenAI API key${RESET} (sk-proj-...)"
  echo -e "  ${YELLOW}Leave blank to skip and configure later${RESET}"
  echo -n "  Key: "
  read -rs API_KEY
  echo ""

  if [ -n "$API_KEY" ]; then
    $PYTHON -c "
import json, pathlib
p = pathlib.Path('$CONFIG_FILE')
p.write_text(json.dumps({'openrouter_api_key': '$API_KEY'}))
print('  Key saved.')
"
    ok "API key saved to $CONFIG_FILE"
  else
    warn "Skipped — run later: echo '{\"openrouter_api_key\":\"sk-...\"}' > $CONFIG_FILE"
  fi
fi

# ─────────────────────────────────────────────
# 7. Verify installs
# ─────────────────────────────────────────────
step "Verifying installs"

check_cmd() {
  if command -v "$1" &>/dev/null; then
    ok "$1 → $(command -v $1)"
  else
    warn "$1 not found in PATH (may need to restart shell)"
  fi
}

check_cmd kubix
check_cmd kubix-cli
check_cmd kubix-ci

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  ✓ Kubix setup complete!${RESET}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  ${BOLD}Quick start:${RESET}"
echo -e "  ${CYAN}kubix${RESET}                          → interactive AI optimizer"
echo -e "  ${CYAN}kubix-ci --help${RESET}                → GPU CI runner"

if [ "$IS_GPU" = true ]; then
  echo -e "  ${CYAN}cd kubix-agent && python profile.py --pretrained facebook/opt-125m${RESET}"
else
  echo -e "  ${YELLOW}GPU profiling:${RESET} open kubix_agent_colab.ipynb in Google Colab"
fi

echo ""
echo -e "  ${BOLD}Docs:${RESET} TESTING.md · Kubix.ai.md"
echo ""
