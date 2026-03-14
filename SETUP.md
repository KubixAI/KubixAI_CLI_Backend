# Kubix Setup Guide

## Requirements

- Python 3.10+
- Git
- Node.js + npm (optional — for MCP server)
- CUDA GPU (optional — for kernel profiling; use Colab otherwise)

---

## Install

Clone the repo and run the setup script:

```bash
git clone https://github.com/Kubix-AI/kubix.git
cd kubix
bash setup.sh
```

That's it. The script installs all components and asks for your API key once.

---

## API Key

The script will prompt for your OpenAI API key (`sk-proj-...`) and save it to:

```
~/.kubix-cli/config.json
```

To set it manually:

```bash
echo '{"openrouter_api_key": "sk-proj-..."}' > ~/.kubix-cli/config.json
```

---

## What Gets Installed

| Package | Command | Purpose |
|---|---|---|
| kubix-cli | `kubix` | Interactive AI GPU optimizer REPL |
| kubix-agent | — | Kernel profiler + optimizer scripts |
| kubix-ci | `kubix-ci` | Multi-GPU CI runner |
| kubix-mcp | — | VS Code / Cursor MCP server (Node) |

---

## Quick Start

```bash
# Launch interactive optimizer
kubix

# Profile a model (requires GPU)
cd kubix-agent
python profile.py --pretrained facebook/opt-125m

# Run on Colab (no local GPU needed)
# Open: kubix_agent_colab.ipynb → Runtime > T4 GPU > Run All
```

---

## GPU Server (Linux + CUDA)

The setup script auto-detects CUDA and installs Triton. Nothing extra needed.

To install manually:

```bash
pip install -e kubix-agent[cuda,models,kernelbench]
pip install triton
```

---

## MCP Server (VS Code / Cursor)

Requires Node.js 18+.

```bash
cd kubix-mcp
npm install
npm run build
```

Then add to your Cursor / VS Code `mcp.json`:

```json
{
  "mcpServers": {
    "kubix": {
      "command": "node",
      "args": ["/path/to/kubix-mcp/dist/index.js"]
    }
  }
}
```

---

## Troubleshooting

**`kubix` command not found after install**
```bash
# Restart your shell or run:
source ~/.zshrc
```

**Triton install fails on Mac**
```bash
# Triton is Linux + CUDA only. Skip it on Mac:
pip install -e kubix-agent[models,kernelbench]
```

**`No CUDA GPU` warning**
Run profiling on Google Colab — open `kubix_agent_colab.ipynb` and select a T4 GPU runtime.

**API key errors (402 / 429)**
Make sure you're using an OpenAI key (`sk-proj-...`), not an OpenRouter key.
