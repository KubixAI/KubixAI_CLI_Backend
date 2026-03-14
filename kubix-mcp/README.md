<p align="center">
  <img src="https://raw.githubusercontent.com/Kubix-AI/kubix-mcp/main/kubix-logo.jpg" alt="Kubix" width="120" />
</p>

<h1 align="center">Kubix MCP Server</h1>

<p align="center">
  <strong>Swarm agents that turn slow PyTorch into fast CUDA/Triton kernels, from any AI coding agent.</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#tools">Tools</a> ·
  <a href="#resources">Resources</a> ·
  <a href="#prompts">Prompts</a> ·
  <a href="#security">Security</a> ·
  <a href="#development">Development</a>
</p>

---

## Overview

Kubix transforms PyTorch models into production-grade CUDA/Triton kernels through automated multi-agent optimization. Using **32 parallel AI agents** with inference-time scaling, it achieves up to **14x faster inference** than `torch.compile(mode='max-autotune-no-cudagraphs')` while maintaining 100% numerical correctness.

This MCP server connects any MCP-compatible AI coding agent to Kubix. Your agent submits PyTorch code, Kubix optimizes it with swarm agents on real datacenter GPUs, and returns the fastest kernel as a drop-in replacement.

### What it does

- **Optimize existing kernels** - Submit PyTorch code, get back an optimized Triton/CUDA kernel benchmarked against `torch.compile(max-autotune)`
- **Generate new kernels** - Describe an operation (e.g. "fused LayerNorm + GELU + Dropout"), get a production-ready optimized kernel
- **32 parallel swarm agents** - Coder+Judge agent pairs compete to discover optimal kernels, exploring tensor core utilization, memory coalescing, shared memory tiling, and kernel fusion simultaneously
- **Real datacenter GPU benchmarking** - Every kernel is compiled, tested for correctness, and profiled on actual datacenter hardware
- **250k tokens/sec inference** - Results in minutes, not hours
- **Smart detection** - The agent automatically recognizes when your code would benefit from GPU optimization
- **One-click auth** - Browser-based OAuth sign-in. No API keys to manage.

### Supported GPUs

All optimization and benchmarking runs on datacenter-grade hardware:

| GPU | Architecture |
|-----|-------------|
| **B200** | Blackwell |
| **H200** | Hopper |
| **H100** | Hopper |
| **L40S** | Ada Lovelace |
| **A100** | Ampere |
| **L4** | Ada Lovelace |
| **A10** | Ampere |
| **T4** | Turing |

### Supported clients

| Client | Status |
|--------|--------|
| Claude Code | Fully supported |
| Claude Desktop | Fully supported |
| OpenCode | Fully supported |
| Cursor | Fully supported |
| Windsurf | Fully supported |
| VS Code + Copilot | Fully supported |
| Any MCP client | Fully supported via stdio |

---

## Installation

### Claude Code

**macOS / Linux:**

```bash
claude mcp add kubix-mcp -- npx -y @kubix/kubix-mcp
```

**Windows:**

```bash
claude mcp add kubix-mcp -- cmd /c npx -y @kubix/kubix-mcp
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

<details>
<summary>macOS: <code>~/Library/Application Support/Claude/claude_desktop_config.json</code></summary>

```json
{
  "mcpServers": {
    "kubix-mcp": {
      "command": "npx",
      "args": ["-y", "@kubix/kubix-mcp"]
    }
  }
}
```
</details>

<details>
<summary>Windows: <code>%APPDATA%\Claude\claude_desktop_config.json</code></summary>

```json
{
  "mcpServers": {
    "kubix-mcp": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@kubix/kubix-mcp"]
    }
  }
}
```
</details>

### VS Code / Copilot

Add to your `.vscode/mcp.json` (workspace) or user settings:

```json
{
  "servers": {
    "kubix-mcp": {
      "command": "npx",
      "args": ["-y", "@kubix/kubix-mcp"]
    }
  }
}
```

> **Windows:** Use `"command": "cmd"` with `"args": ["/c", "npx", "-y", "@kubix/kubix-mcp"]`

### Cursor

Add to your Cursor MCP settings (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "kubix-mcp": {
      "command": "npx",
      "args": ["-y", "@kubix/kubix-mcp"]
    }
  }
}
```

> **Windows:** Use `"command": "cmd"` with `"args": ["/c", "npx", "-y", "@kubix/kubix-mcp"]`

### Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "kubix-mcp": {
      "command": "npx",
      "args": ["-y", "@kubix/kubix-mcp"]
    }
  }
}
```

> **Windows:** Use `"command": "cmd"` with `"args": ["/c", "npx", "-y", "@kubix/kubix-mcp"]`

### OpenCode

Add to your `opencode.json`:

```json
{
  "mcp": {
    "kubix-mcp": {
      "command": "npx",
      "args": ["-y", "@kubix/kubix-mcp"]
    }
  }
}
```

---

## Tools

### `kubix_auth`

Authenticate with the Kubix service. Opens your browser to sign in via the Kubix dashboard. Required before using any other tool.

- Inputs:
  - `force` (boolean, optional): Force re-authentication even if valid tokens exist
- Returns: Authentication status, email, plan type, and credit balance

### `kubix_optimize`

Submit PyTorch code for GPU kernel optimization. 32 swarm agents generate optimized Triton or CUDA kernels, evaluate them on real datacenter GPUs, and return the best result with speedup metrics.

The agent will automatically use this tool when it detects:
- PyTorch custom operations (`torch.autograd.Function`, custom `forward`/`backward`)
- Manual CUDA kernels that could be faster
- Performance-critical tensor operations (attention, convolution, normalization, softmax)
- Code with comments like `"slow"`, `"bottleneck"`, `"optimize"`
- `torch.compile()` targets or `triton.jit` kernels
- Any `nn.Module` with significant compute in `forward()`
- Matrix multiplication, reduction, or scan operations
- Custom loss functions with reduction operations
- Fused operation opportunities (e.g., LayerNorm + activation)

- Inputs:
  - `pytorch_code` (string, required): Complete PyTorch code to optimize. Max 500 KB.
  - `kernel_name` (string, required): Short name for the kernel (e.g., `"flash_attention"`)
  - `output_format` (enum, optional): `"triton"` (default) or `"native_cuda"`
  - `target_speedup` (number, optional): Target speedup multiplier. Default `2.0`
  - `max_iterations` (number, optional): Max optimization iterations (1-100). Default `10`
  - `gpu` (enum, optional): Target GPU. Default `"H100"`. Options: `B200`, `H200`, `H100`, `L40S`, `A100`, `L4`, `A10`, `T4`
  - `user_prompt` (string, optional): Guidance for the optimizer (e.g., `"focus on memory bandwidth"`)
- Returns: Optimized kernel code, speedup metrics, latency comparison, iteration history

### `kubix_generate`

Generate an optimized GPU kernel from scratch based on a natural-language specification. Kubix creates a PyTorch baseline, then optimizes it into Triton or CUDA.

- Inputs:
  - `operation` (string, required): Operation name (e.g., `"fused_attention"`, `"softmax"`)
  - `description` (string, required): Detailed description of what the kernel should do
  - `input_shapes` (number[][], required): Input tensor shapes (e.g., `[[8, 512, 768]]`)
  - `output_shape` (number[], optional): Expected output shape
  - `dtype` (string, optional): Data type. Default `"float16"`
  - `output_format` (enum, optional): `"triton"` (default) or `"native_cuda"`
  - `target_speedup` (number, optional): Target speedup. Default `2.0`
  - `max_iterations` (number, optional): Max iterations (1-100). Default `10`
  - `gpu` (enum, optional): Target GPU. Default `"H100"`
  - `user_prompt` (string, optional): Additional guidance
- Returns: Generated kernel code, speedup metrics, iteration history

### `kubix_credits`

Check your current Kubix credit balance.

- Inputs: None
- Returns: Credit balance, total purchased, total used, plan type

### `kubix_status`

Check the status of a running or completed optimization job.

- Inputs:
  - `session_id` (string, required): Session ID from `kubix_optimize` or `kubix_generate`
- Returns: Job status, current iteration, best speedup

### `kubix_cancel`

Cancel a running optimization job.

- Inputs:
  - `session_id` (string, required): Session ID of the job to cancel
- Returns: Cancellation confirmation

### `kubix_sessions`

List past optimization sessions with results.

- Inputs:
  - `limit` (number, optional): Number of sessions to return (1-100). Default `10`
  - `status` (enum, optional): Filter by status: `"all"`, `"completed"`, `"failed"`, `"running"`. Default `"all"`
- Returns: Table of sessions with task name, GPU, speedup, status, and date

### Tool Annotations

| Tool | Read-only | Idempotent | Destructive |
|------|-----------|------------|-------------|
| `kubix_auth` | No | Yes | No |
| `kubix_optimize` | No | No | No |
| `kubix_generate` | No | No | No |
| `kubix_credits` | Yes | Yes | No |
| `kubix_status` | Yes | Yes | No |
| `kubix_cancel` | No | No | Yes |
| `kubix_sessions` | Yes | Yes | No |

---

## Resources

| URI | Description |
|-----|-------------|
| `kubix://auth/status` | Current authentication state (authenticated, token expiry, has refresh token) |
| `kubix://credits` | Credit balance, usage, and plan information |

---

## Prompts

### `kubix-optimize`

Guided workflow for optimizing a GPU kernel. Instructs the agent to:
1. Check credit balance
2. Analyze the code for optimization targets
3. Call `kubix_optimize` with appropriate parameters
4. Explain the results and suggest integration

### `kubix-analyze`

Teaches the agent to scan a codebase for GPU optimization opportunities, ranked by expected impact:

| Priority | Pattern |
|----------|---------|
| **HIGH** | Custom autograd functions, attention mechanisms, fused operations |
| **MEDIUM** | Standard `nn.Module` compositions, normalization + activation fusion |
| **LOW** | Element-wise operations, simple reductions |

---

## How It Works

```
┌──────────────┐     stdio      ┌──────────────────┐     HTTPS      ┌──────────────────┐
│  AI Agent    │ ──────────────>│  Kubix MCP       │ ──────────────>│  Kubix API       │
│  (Claude,    │                │  Server          │                │  (Kubix AI)      │
│   Cursor,    │<──────────────│                  │<──────────────│                  │
│   etc.)      │   MCP result   │  - OAuth + PKCE  │   SSE stream   │  - 32 swarm      │
└──────────────┘                │  - SSE streaming │                │    agents        │
                                │  - Token mgmt    │                │  - Real GPU      │
                                └──────────────────┘                │    benchmarking  │
                                                                    └──────────────────┘
```

1. **Authenticate**: The agent calls `kubix_auth`, which opens your browser. Sign in once, tokens are stored locally at `~/.kubix/tokens.json` and auto-refresh.
2. **Optimize**: The agent sends your PyTorch code via `kubix_optimize`. The MCP server POSTs to the Kubix API and streams SSE events in real time.
3. **Benchmark**: 32 parallel Coder+Judge agents generate kernels, compile them, test correctness against the PyTorch reference, and profile performance on real datacenter GPUs.
4. **Return**: The MCP server collects all results and returns the optimized code, speedup metrics, and iteration history. The output is a drop-in replacement for your original code.

Each optimization costs **1 credit**. Credits are only charged for successful runs (speedup >= 1.1x). Failed runs and cancelled jobs are not charged.

---

## Configuration

### Authentication

No API keys needed. The server uses **OAuth 2.0 with PKCE** for secure browser-based authentication:

1. Agent calls `kubix_auth`
2. Your default browser opens to the Kubix dashboard
3. Sign in or create an account
4. Authorization completes automatically
5. Tokens are stored locally at `~/.kubix/tokens.json` (mode `0600`)
6. Access tokens auto-refresh, you only sign in once

### Credits

Kubix uses a **pay-as-you-go** credit system. Each optimization or generation run costs **1 credit**.

| Credits | Price | Per Credit |
|---------|-------|------------|
| 1-9 | $15.00 each | $15.00 |
| 10+ | 25% off | $11.25 |
| 50 | $562.50 | $11.25 |
| Enterprise | Custom volume pricing | [Contact us](#) |

**Free trial**: optimize 1 kernel, no credit card required.

**100% refund guarantee**: if Kubix doesn't beat `torch.compile`, you get your credit back.

Purchase credits at the Kubix dashboard.

---

## Benchmarks

End-to-end latency on NVIDIA B200. Kubix vs `torch.compile(mode='max-autotune-no-cudagraphs')`:

| Model | torch.compile | Kubix | Speedup |
|-------|--------------|-------|---------|
| Llama-3.1-8B | 42.3ms | 8.2ms | **5.16x** |
| Qwen2.5-7B | 38.5ms | 9.1ms | **4.23x** |
| Mistral-7B | 35.2ms | 10.4ms | **3.38x** |
| Phi-3-mini | 18.7ms | 6.8ms | **2.75x** |
| SDXL UNet | 89.4ms | 31.2ms | **2.87x** |
| Whisper-large | 52.1ms | 19.8ms | **2.63x** |
| BERT-large | 12.4ms | 5.1ms | **2.43x** |

---

## Security

### Token Protection

- **No tokens in errors**: All error messages are sanitized through regex filters that strip JWTs, Bearer tokens, hex tokens, and credential parameters before reaching the agent
- **Local storage only**: Tokens are stored at `~/.kubix/tokens.json` with file mode `0600` (owner read/write only)
- **Auto-refresh**: Access tokens expire in 1 hour and auto-refresh using the stored refresh token
- **PKCE flow**: OAuth uses Proof Key for Code Exchange (SHA-256), preventing authorization code interception
- **No secrets in config**: The MCP server requires zero environment variables or API keys

### Input Validation

- PyTorch code input is capped at **500 KB** to prevent memory exhaustion
- User prompts are capped at **10 KB**
- All string inputs have maximum length validation via Zod schemas
- Numeric inputs have min/max bounds (e.g., `max_iterations: 1-100`)

### Network Security

- All API communication uses **HTTPS**
- Non-SSE requests have a **30-second timeout** to prevent hanging
- SSE streams have a **10-minute timeout** with automatic cleanup
- Token refresh uses a **mutex** to prevent race conditions from concurrent requests

### What the server can access

- **Network**: Only the Kubix dashboard and Kubix API endpoints
- **Filesystem**: Only reads/writes `~/.kubix/tokens.json`
- **No codebase access**: The MCP server never reads your files. The agent passes code to it explicitly through tool parameters.

---

## Development

### Build from source

```bash
git clone [url]
cd kubix-mcp
npm install
npm run build
```

### Run locally

```bash
npm run dev
```

### Type check

```bash
npm run typecheck
```

### Debug with MCP Inspector

```bash
npx @modelcontextprotocol/inspector node dist/index.js
```

This opens a web UI where you can invoke each tool, inspect inputs/outputs, and debug the server interactively.

### Project structure

```
kubix-mcp/
├── src/
│   ├── index.ts              # Entry point (McpServer + StdioServerTransport)
│   ├── server.ts             # Registers all tools, resources, prompts
│   ├── constants.ts          # URLs, client IDs, timeouts, limits
│   ├── types.ts              # TypeScript interfaces + type guards + sanitization
│   ├── auth/
│   │   ├── oauth-client.ts   # PKCE flow, token refresh, access token management
│   │   └── token-store.ts    # ~/.kubix/tokens.json read/write/clear
│   ├── api/
│   │   ├── kubix-client.ts   # HTTP client for all Kubix API endpoints
│   │   └── sse-consumer.ts   # SSE stream parser via native fetch + ReadableStream
│   ├── tools/                # 7 MCP tools
│   ├── resources/            # 2 MCP resources
│   └── prompts/              # 2 MCP prompts
├── .github/workflows/
│   ├── ci.yml                # Typecheck + build on push/PR
│   └── release.yml           # npm publish on version tags
├── package.json
├── tsconfig.json
└── tsup.config.ts
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run `npm run typecheck` and `npm run build`
5. Commit and push
6. Open a pull request

---

## License

[MIT](LICENSE)

Part of the Kubix AI ecosystem. Member of the NVIDIA Inception Program.
