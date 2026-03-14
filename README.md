# Kubix.ai
### The AI GPU Performance Engineer — Smarter, Faster, Broader

> **One-liner:** Kubix.ai is an AI-powered GPU kernel optimization platform that profiles, diagnoses, patches, and verifies GPU workloads  making every team's models run 2–10x faster without needing $2M/year CUDA engineers.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Solution — What Kubix.ai Does](#2-the-solution)
3. [Core Product Components](#3-core-product-components)
4. [The AI Agent Pipeline](#4-the-ai-agent-pipeline)
5. [Technical Architecture](#5-technical-architecture)
6. [Key Features & Differentiators](#6-key-features--differentiators)



---

## 1. The Problem

### GPUs Are Wasting 80% of Their Potential

Companies running AI inference and training workloads use roughly **20% of the GPU capacity they pay for** — not because of poor scheduling or small batch sizes, but because the **kernels running on that hardware are not optimized for the specific silicon they run on**.

Every GPU model — NVIDIA A100, H100, B200, AMD MI300X — has a unique memory hierarchy, SM count, warp size, and instruction set. A kernel written for H100 runs poorly on B200. A kernel benchmarked on NVIDIA runs even worse on AMD. The gap between "it runs" and "it runs optimally" is 2–10x in compute efficiency.

### The Human Expertise Gap

Fixing this requires **GPU performance engineers** — people who understand:
- CUDA/HIP programming model
- Memory hierarchies (L1/L2/HBM bandwidth, shared memory bank conflicts)
- Warp execution, occupancy, and instruction-level parallelism
- PTX, SASS, and compiled output analysis
- Profiler data interpretation (NSight Compute, ROCprofiler)

These engineers are among the rarest in tech. **Top GPU engineers at NVIDIA and OpenAI earn $2M+ in total comp.** Most companies — even well-funded ones — simply don't have access to them. The optimization work that would save $5,000/day in compute costs doesn't happen because it would take a $300,000/year engineer months to implement.

### AI Changes the Cost Function

AI agents can now run the same profile → diagnose → patch → verify loop that human GPU engineers run — at a fraction of the cost and time. A $600 compute run can find and implement optimizations that pay back in hours. **Kubix.ai is the infrastructure that makes this possible.**

---

## 2. The Solution

### Kubix.ai — The AI GPU Performance Engineer

Kubix.ai is a developer platform that gives every ML/infra team access to GPU performance engineering at scale. It combines:

1. **A VS Code / Cursor IDE extension** — Profiling, PTX/SASS analysis, trace viewing, and GPU docs, all inside your editor
2. **A CLI (`kubix`)** — For scripting, automation, and coding agent integration
3. **An AI agent** — That runs the full optimization loop autonomously, with real profiler data
4. **On-demand GPU sandboxes** — Per-second-billed hardware for benchmarking and verification
5. **Production monitoring** — Continuous watching of live inference workloads

The result: ML teams can ship models that run **2–10x faster** without hiring a single dedicated CUDA engineer.

---

## 3. Core Product Components

### 3.1 Kubix IDE Extension (VS Code / Cursor)

A developer tool that brings GPU engineering capabilities directly into the editor:

| Feature | What It Does |
|---|---|
| **Profiler Integration** | Run NSight Compute (NCU) and ROCprofiler from inside VS Code — get structured metric reports without leaving the editor |
| **PTX / SASS Explorer** | Cloud-compile CUDA files, view PTX and SASS for 12+ GPU architectures (sm_80 through sm_120a). No local CUDA or GPU required |
| **Perfetto Trace Viewer** | Open Chrome trace JSON files natively in the IDE — timeline, flamegraph, and SQL query interface |
| **ROCprofiler Compute** | AMD GPU profiling in VS Code — hardware metrics and roofline analysis |
| **GPU Docs Search** | Instant, citation-grounded search across CUDA programming guides, API references, and optimization best practices |

**Key insight:** These are the same tools a senior GPU engineer uses daily — Kubix makes them accessible to any developer without a multi-day setup.

---

### 3.2 Kubix CLI (`kubix`)

A command-line interface designed for scripting and coding agent integration.

```bash
# Install
pip install kubix-cli

# Authenticate
kubix auth login

# Evaluate a custom kernel against a reference on real hardware
kubix evaluate pytorch --impl my_attention.cu --reference torch.nn.MultiheadAttention --benchmark

# Provision on-demand GPU compute
kubix workspaces create --gpu b200 --name optimization-run-1

# Invoke the AI agent
kubix agent --template attention_optimization --args '{"arch": "blackwell", "model": "llama3"}'

# SQL query against a profiler trace
kubix tool perfetto query trace.json "SELECT name, dur FROM slice ORDER BY dur DESC LIMIT 10"
```

**Design principle:** Every tool the agent uses is accessible to humans too. No magic black boxes.

---

### 3.3 AI Agent with Skills

The agent is invoked via `kubix agent` or through a **skill** — a structured prompt file that teaches coding agents (Cursor, Claude Code, etc.) exactly how to use Kubix's tools.

```bash
# Install the Kubix skill into Cursor
kubix skill install --target cursor

# Or Claude Code
kubix skill install --target claude-code
```

The skill file (~150 lines of markdown) provides scaffolding for:
- When and how to call `kubix tool` profiling commands
- How to interpret profiler output and identify bottlenecks
- How to validate fixes with `kubix evaluate`
- How to use GPU docs for architecture-specific optimization

This means **Cursor, Claude Code, or any coding agent becomes a GPU performance engineer** when the Kubix skill is installed.

---

### 3.4 GPU Workspaces

On-demand GPU compute for coding agents and developers:

- **Hardware:** NVIDIA B200 (180GB HBM3e), AMD MI300X (192GB HBM3)
- **Billing:** Per-second, only when commands are running — no idle charges
- **Setup:** No Docker, no SSH configuration, no environment setup
- **Features:** File sync, remote exec, SSH access

```bash
kubix workspaces create --gpu b200 --name my-workspace
kubix workspaces sync ./my_kernel/ my-workspace:/workspace/
kubix workspaces exec my-workspace "python benchmark.py"
```

---

### 3.5 Production Monitoring

After initial optimization, Kubix watches your production inference workloads:
- Detects regressions when models are updated or hardware changes
- Surfaces new optimization opportunities as model usage patterns evolve
- Alerts when a kernel's performance degrades vs. its verified baseline
- Acts as a GPU performance engineer **on call, 24/7**

---

## 4. The AI Agent Pipeline

### Profile → Diagnose → Patch → Verify

This is the core loop that Kubix automates. Every step uses the same tools a human GPU engineer would use.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PROFILE   │ -> │  DIAGNOSE   │ -> │    PATCH    │ -> │   VERIFY    │
│             │    │             │    │             │    │             │
│ NCU / RCP   │    │ Bottleneck  │    │ AI generates│    │ Real hw     │
│ Trace data  │    │ analysis    │    │ code diffs  │    │ benchmark   │
│ Structured  │    │ vs arch     │    │ User approves    │ Correctness │
│ metrics     │    │ docs        │    │ or auto     │    │ validation  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Step 1: Profile
- Execute NCU or ROCprofiler on the target workload
- Collect: memory bandwidth utilization, occupancy, compute throughput, warp efficiency, instruction mix
- Output: structured JSON with per-kernel metrics, not human-readable flamegraphs

### Step 2: Diagnose
- Cross-reference profiler data with GPU architecture docs (Hopper/Blackwell/CDNA memory hierarchy, SM counts, warp sizes)
- Identify the highest-leverage bottleneck: memory-bound, compute-bound, low occupancy, bank conflicts, warp divergence
- Explain findings in plain language — *"This kernel launches only 64 blocks on a 145-SM GPU (6.25% occupancy) — the bottleneck is register pressure limiting block count"*

### Step 3: Patch
- Generate optimized code grounded in profiler evidence
- Inspect PTX, SASS, and IR to validate changes at the compiler level before running
- Present diff to user for approval, or run fully autonomously

### Step 4: Verify
- Run patched kernel on **real hardware** in a Kubix Workspace
- Validate correctness against reference implementation with bitwise-exact checking
- Measure speedup vs. baseline
- Report: before/after metrics, speedup factor, correctness status

### Real-World Example Results

| Kernel | Hardware | Speedup | Technique |
|---|---|---|---|
| Kimi Delta Attention | NVIDIA (sm_90) | **11.65x** | Register vectorization (128 floats → 32 float4s) + loop unrolling |
| AMD topk_sigmoid (MoE routing) | AMD MI300X | **9x** | DPP instructions + ISA-level `row_bcast`, eliminated shared memory shuffles |
| Nordlys Labs routing kernel | NVIDIA | **8x** | float4 loads + kernel fusion + shared memory padding (bank conflict elimination) |
| vLLM Fused MOE | NVIDIA B200 | Significant | Autonomous trace analysis → identified missing B200 tuned config |

---

## 5. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubix Platform                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  IDE Layer   │  │  CLI Layer   │  │   Agent Layer        │  │
│  │              │  │              │  │                      │  │
│  │ VS Code Ext  │  │ kubix-cli    │  │ Skill files for:     │  │
│  │ Cursor Ext   │  │ (Python 3.8+)│  │ - Cursor             │  │
│  │              │  │              │  │ - Claude Code        │  │
│  └──────┬───────┘  └──────┬───────┘  │ - Any coding agent   │  │
│         └─────────────────┘          └──────────┬───────────┘  │
│                      │                          │               │
│         ┌────────────▼──────────────────────────▼────────────┐ │
│         │              Kubix Tool Layer                       │ │
│         │                                                     │ │
│         │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │ │
│         │  │  NVIDIA  │ │   AMD    │ │ Compiler │            │ │
│         │  │   NCU    │ │ ROCProf  │ │ Explorer │            │ │
│         │  │ Perfetto │ │ ISA Tool │ │ PTX/SASS │            │ │
│         │  └──────────┘ └──────────┘ └──────────┘            │ │
│         │                                                     │ │
│         │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │ │
│         │  │ GPU Docs │ │  Eval    │ │ Reward   │            │ │
│         │  │ Search   │ │ Harness  │ │ Hack Det │            │ │
│         │  └──────────┘ └──────────┘ └──────────┘            │ │
│         └─────────────────────────────────────────────────────┘ │
│                                │                                │
│         ┌──────────────────────▼──────────────────────────────┐ │
│         │              GPU Workspace Layer                     │ │
│         │   NVIDIA B200 (180GB HBM3e) | AMD MI300X (192GB)    │ │
│         │   Per-second billing | Baremetal + VM options        │ │
│         └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Profiler Integration Layer

- **NVIDIA:** NSight Compute (ncu), NSight Systems (nsys), PyTorch Profiler, Perfetto
- **AMD:** ROCprofiler SDK, ROCprofiler Systems, ISA analysis (DPP, GFX9 instructions)
- **Output format:** Structured JSON/SQL-queryable data, not raw HTML reports

### Compiler Explorer 

- Supports sm_80 through sm_120a (Ampere → Blackwell)
- CUTLASS 4.x and PyTorch headers bundled
- No local CUDA installation needed
- Returns PTX, SASS, and intermediate IR

### Evaluation Harness

- Bitwise-exact correctness validation against reference implementations
- Reproducible benchmarking with hardware-locked seeds
- Reward hack detection (10-pattern static analysis catalog)
- Multi-input validation (prevents identity/no-op kernel exploits)

---

## 6. Key Features & Differentiators

### vs. Doing It Manually
| | Manual CUDA Engineer | Kubix.ai |
|---|---|---|
| Cost | $300k–$2M+/year | Pay-per-use credits |
| Time to first speedup | Weeks to months | Hours |
| Hardware coverage | Usually 1 vendor | NVIDIA + AMD |
| Availability | Business hours | 24/7 |
| Reproducibility | Varies | Verified, benchmarked |




