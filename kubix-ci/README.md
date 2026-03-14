<p align="center">
  <h1 align="center">kubix_ci</h1>
  <p align="center">
    <strong>Test CUDA kernels across multiple GPUs via SSH</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#providers">Providers</a> •
    <a href="#github-actions">GitHub Actions</a> •
    <a href="docs/providers.md">Documentation</a>
  </p>
</p>

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   $ kubix_ci test matmul.cu                                                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    kubix_ci results: matmul.cu                         │   │
│   ├─────────────┬───────────────────┬────────┬──────────┬───────────────┤   │
│   │ Target      │ GPU               │ Status │   Median │ Compile       │   │
│   ├─────────────┼───────────────────┼────────┼──────────┼───────────────┤   │
│   │ h100-cloud  │ NVIDIA H100       │  PASS  │  0.42ms  │   2.1s        │   │
│   │ a100-server │ NVIDIA A100       │  PASS  │  0.61ms  │   2.3s        │   │
│   │ rtx-5070ti  │ RTX 5070 Ti       │  PASS  │  1.03ms  │   3.0s        │   │
│   └─────────────┴───────────────────┴────────┴──────────┴───────────────┘   │
│                                                                             │
│   All 3 targets passed                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Multi-GPU       │  │  Accurate        │  │  6 Cloud         │
│  Testing         │  │  Timing          │  │  Providers       │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ Run kernels on   │  │ CUDA events for  │  │ SSH, RunPod,     │
│ multiple GPUs    │  │ microsecond      │  │ Lambda, Vast.ai  │
│ simultaneously   │  │ precision        │  │ FluidStack, Brev │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Easy Setup      │  │  GitHub Actions  │  │  CI/CD Ready     │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ Simple YAML      │  │ First-class CI   │  │ Exit codes for   │
│ configuration    │  │ integration with │  │ automation and   │
│ + init wizard    │  │ PR comments      │  │ pipelines        │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │     │   Compile   │     │   Execute   │     │   Report    │
│   Kernel    │────▶│   on GPU    │────▶│   Timed     │────▶│   Results   │
│             │     │   Machine   │     │   Runs      │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
 ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
 │ SFTP    │        │ nvcc    │        │ CUDA    │        │ Rich    │
 │ Upload  │        │ -arch   │        │ Events  │        │ Table   │
 └─────────┘        │ auto    │        │ Timing  │        └─────────┘
                    └─────────┘        └─────────┘

                    Architecture
                    auto-detected               Warmup + Benchmark
                    via nvidia-smi              runs with stats
```

### Execution Flow

```
Your Machine                          GPU Target(s)
┌──────────────────┐                 ┌──────────────────┐
│                  │    SSH/SFTP     │                  │
│  kubix_ci test      │ ──────────────▶ │  /tmp/kubix_ci/     │
│  kernel.cu       │                 │  kernel.cu       │
│                  │                 │                  │
│  ┌────────────┐  │                 │  ┌────────────┐  │
│  │ Wrap with  │  │                 │  │ nvcc       │  │
│  │ timing     │  │                 │  │ compile    │  │
│  │ macros     │  │                 │  └────────────┘  │
│  └────────────┘  │                 │        │         │
│                  │                 │        ▼         │
│                  │                 │  ┌────────────┐  │
│                  │                 │  │ Execute    │  │
│                  │  ◀───────────── │  │ 3 warmup   │  │
│  Parse results   │    stdout       │  │ 10 timed   │  │
│  KUBIXCI_*         │                 │  └────────────┘  │
│                  │                 │                  │
└──────────────────┘                 └──────────────────┘
        │
        ▼
┌──────────────────┐
│  Display Table   │
│  with timing     │
│  statistics      │
└──────────────────┘
```

---

## Installation

```bash
# Basic installation
pip install kubix_ci

# With RunPod support
pip install kubix_ci[runpod]

# With Vast.ai support
pip install kubix_ci[vastai]

# All cloud providers
pip install kubix_ci[cloud]
```

### From Source

```bash
git clone [url]
cd kubix_ci
pip install -e .
```

---

## Quick Start

### 1. Initialize Configuration

```bash
kubix_ci init
```

This creates `kubix_ci.yml` with your GPU targets.

### 2. Configure Your Targets

```yaml
# kubix_ci.yml
targets:
  - name: my-gpu-server
    provider: ssh
    host: gpu.mycompany.com
    user: ubuntu
    gpu: RTX 4090

warmup_runs: 3
benchmark_runs: 10
timeout: 120
```

### 3. Write a CUDA Kernel

```cuda
// kernel.cu
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### 4. Run Tests

```bash
kubix_ci test kernel.cu
```

---

## Providers

kubix_ci supports **6 GPU providers**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GPU PROVIDERS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    SSH      │  │   RunPod    │  │   Lambda    │  │   Vast.ai   │        │
│  │  ───────    │  │  ────────   │  │  ────────   │  │  ─────────  │        │
│  │ Your own    │  │ On-demand   │  │ High-perf   │  │ GPU market  │        │
│  │ machines    │  │ cloud GPUs  │  │ cloud GPUs  │  │ place       │        │
│  │             │  │             │  │             │  │             │        │
│  │ FREE        │  │ $0.20-4/hr  │  │ $1.10-3/hr  │  │ $0.10-2/hr  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐                                          │
│  │ FluidStack  │  │    Brev     │                                          │
│  │ ──────────  │  │  ────────   │                                          │
│  │ Enterprise  │  │ NVIDIA      │                                          │
│  │ GPUs        │  │ managed     │                                          │
│  │             │  │             │                                          │
│  │ Custom      │  │ $0.50-4/hr  │                                          │
│  └─────────────┘  └─────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Provider Comparison

| Provider | Type | Setup | Best For |
|:---------|:-----|:------|:---------|
| **SSH** | Your machines | SSH key | Dedicated hardware, no cost |
| **RunPod** | Cloud | API key | Quick testing, simple pricing |
| **Lambda Labs** | Cloud | API key | Production workloads |
| **Vast.ai** | Marketplace | API key | Cost optimization |
| **FluidStack** | Enterprise | API key | Large scale deployments |
| **Brev** | Cloud | CLI login | Development & testing |

### Configuration Examples

<details>
<summary><b>SSH (Your Own Machine)</b></summary>

```yaml
- name: my-server
  provider: ssh
  host: gpu.example.com
  user: ubuntu
  port: 22
  key: ~/.ssh/id_rsa
  gpu: RTX 4090
```

**Requirements:**
- NVIDIA GPU with drivers
- CUDA toolkit (nvcc)
- SSH access

</details>

<details>
<summary><b>RunPod</b></summary>

```yaml
- name: runpod-a100
  provider: runpod
  gpu: "NVIDIA A100 80GB PCIe"
  gpu_count: 1
  image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
```

```bash
export RUNPOD_API_KEY=your_key
pip install kubix_ci[runpod]
```

</details>

<details>
<summary><b>Lambda Labs</b></summary>

```yaml
- name: lambda-h100
  provider: lambdalabs
  gpu: gpu_1x_h100_pcie
  region: us-west-1
```

```bash
export LAMBDA_API_KEY=your_key
```

</details>

<details>
<summary><b>Vast.ai</b></summary>

```yaml
- name: vastai-4090
  provider: vastai
  gpu: RTX_4090
  max_price: 0.50
  min_gpu_ram: 24
```

```bash
export VASTAI_API_KEY=your_key
pip install kubix_ci[vastai]
```

</details>

<details>
<summary><b>FluidStack</b></summary>

```yaml
- name: fluidstack-h100
  provider: fluidstack
  gpu: H100_SXM_80GB
```

```bash
export FLUIDSTACK_API_KEY=your_key
```

</details>

<details>
<summary><b>Brev (NVIDIA)</b></summary>

```yaml
- name: brev-h100
  provider: brev
  gpu: H100
```

```bash
curl -fsSL [url] | bash
brev login
```

</details>

> **Full documentation:** [docs/providers.md](docs/providers.md)

---

## Kernel Formats

kubix_ci supports two kernel formats:

### Format 1: Full Kernel (with main)

Use KUBIXCI macros for timing control:

```cuda
#include <cuda_runtime.h>

__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

int main() {
    // Setup...

    // Warmup (not timed)
    KUBIXCI_WARMUP_START()
    myKernel<<<blocks, threads>>>(d_data, N);
    KUBIXCI_WARMUP_END()

    // Benchmark (timed with CUDA events)
    KUBIXCI_BENCHMARK_START()
    myKernel<<<blocks, threads>>>(d_data, N);
    KUBIXCI_BENCHMARK_END()

    kubix_ci_print_results();
    return 0;
}
```

### Format 2: Kernel Only (auto-harness)

Just provide the kernel - kubix_ci generates the test harness:

```cuda
__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}
```

---

## CLI Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI COMMANDS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  kubix_ci init                     Create kubix_ci.yml configuration              │
│  kubix_ci test <kernel.cu>         Run kernel tests                            │
│  kubix_ci targets                  List configured targets                     │
│  kubix_ci check                    Verify target connectivity                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              TEST OPTIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  --target <name>                Test specific target only                   │
│  --config <path>                Use specific config file                    │
│  --runs <n>                     Number of benchmark runs (default: 10)      │
│  --warmup <n>                   Number of warmup runs (default: 3)          │
│  --verbose                      Show detailed output                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples

```bash
# Test all targets
kubix_ci test kernel.cu

# Test specific target
kubix_ci test kernel.cu --target h100-cloud

# Custom benchmark settings
kubix_ci test kernel.cu --runs 20 --warmup 5

# Verbose output
kubix_ci test kernel.cu --verbose

# Use specific config
kubix_ci test kernel.cu --config /path/to/kubix_ci.yml
```

---

## GitHub Actions

### Quick Setup

```yaml
# .github/workflows/gpu-test.yml
name: GPU Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kubix-ai/kubix_ci@v1
        with:
          kernel: 'kernels/*.cu'
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

### Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GitHub Actions Workflow                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  on: [push, pull_request]                                                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Checkout   │───▶│  Setup      │───▶│  Run kubix_ci  │───▶│  Comment    │  │
│  │  Code       │    │  Python     │    │  test       │    │  on PR      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                              │                              │
│                                              ▼                              │
│                           ┌─────────────────────────────────┐              │
│                           │  Cloud GPUs (RunPod/Lambda/etc) │              │
│                           │  ┌─────┐ ┌─────┐ ┌─────┐        │              │
│                           │  │ H100│ │ A100│ │ 4090│        │              │
│                           │  └─────┘ └─────┘ └─────┘        │              │
│                           └─────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Full guide:** [docs/github-actions.md](docs/github-actions.md)

---

## Configuration Reference

```yaml
# kubix_ci.yml - Full Configuration Reference

# ┌─────────────────────────────────────────────────────────────────────────┐
# │                              TARGETS                                    │
# └─────────────────────────────────────────────────────────────────────────┘
targets:
  # SSH Target
  - name: my-server           # Unique identifier
    provider: ssh             # Provider type
    host: gpu.example.com     # Hostname or IP
    user: ubuntu              # SSH username
    port: 22                  # SSH port (optional)
    key: ~/.ssh/id_rsa        # SSH key path (optional)
    gpu: RTX 4090             # GPU name (for display)

  # Cloud Target (RunPod example)
  - name: runpod-a100
    provider: runpod
    gpu: "NVIDIA A100 80GB PCIe"
    gpu_count: 1
    image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ┌─────────────────────────────────────────────────────────────────────────┐
# │                           TIMING SETTINGS                               │
# └─────────────────────────────────────────────────────────────────────────┘
warmup_runs: 3                # Warmup iterations (not timed)
benchmark_runs: 10            # Timed iterations
timeout: 120                  # Seconds per target

# ┌─────────────────────────────────────────────────────────────────────────┐
# │                         COMPILATION FLAGS                               │
# └─────────────────────────────────────────────────────────────────────────┘
nvcc_flags:
  - "-O3"                     # Optimization level
  - "-lineinfo"               # Debug info (optional)
```

---

## Project Structure

```
kubix_ci/
├── kubix_ci/                    # Main package
│   ├── __init__.py           # Version
│   ├── __main__.py           # python -m kubix_ci
│   ├── cli.py                # Click CLI commands
│   ├── config.py             # YAML configuration
│   ├── runner.py             # Parallel execution
│   ├── reporter.py           # Rich table output
│   ├── timing.py             # CUDA timing wrapper
│   ├── exceptions.py         # Error hierarchy
│   └── providers/            # GPU providers
│       ├── base.py           # Abstract base class
│       ├── ssh.py            # Direct SSH
│       ├── runpod.py         # RunPod SDK
│       ├── lambdalabs.py     # Lambda REST API
│       ├── vastai.py         # Vast.ai SDK
│       ├── fluidstack.py     # FluidStack REST API
│       └── brev.py           # Brev CLI
├── examples/                 # Example kernels
│   ├── vector_add.cu
│   ├── matmul.cu
│   └── simple_kernel.cu
├── docs/                     # Documentation
│   ├── providers.md
│   └── github-actions.md
├── .github/workflows/        # CI/CD workflows
├── action.yml                # GitHub Action
├── pyproject.toml            # Package config
├── LICENSE                   # MIT License
├── CHANGELOG.md              # Version history
└── README.md                 # This file
```

---

## Timing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUDA Event Timing                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    CPU Timeline
    ────────────────────────────────────────────────────────────────────────▶

    cudaEventRecord(start)     cudaEventRecord(stop)
           │                          │
           ▼                          ▼
    ┌──────┴──────────────────────────┴──────┐
    │              GPU Execution              │
    │    ┌────────────────────────────┐      │
    │    │      Kernel Execution      │      │
    │    │      (what we measure)     │      │
    │    └────────────────────────────┘      │
    └────────────────────────────────────────┘
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
              cudaEventElapsedTime()
              ─────────────────────
              Microsecond precision
              GPU-side measurement
              Independent of CPU load
```

### Benchmark Process

```
Run 1   Run 2   Run 3   Run 4   Run 5   Run 6   Run 7   Run 8   Run 9   Run 10
  │       │       │       │       │       │       │       │       │       │
  ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│W W│   │W W│   │W W│   │ T │   │ T │   │ T │   │ T │   │ T │   │ T │   │ T │
│A A│   │A A│   │A A│   │ I │   │ I │   │ I │   │ I │   │ I │   │ I │   │ I │
│R R│   │R R│   │R R│   │ M │   │ M │   │ M │   │ M │   │ M │   │ M │   │ M │
│M M│   │M M│   │M M│   │ E │   │ E │   │ E │   │ E │   │ E │   │ E │   │ E │
│U U│   │U U│   │U U│   │ D │   │ D │   │ D │   │ D │   │ D │   │ D │   │ D │
│P P│   │P P│   │P P│   │   │   │   │   │   │   │   │   │   │   │   │   │   │
└───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘
  │       │       │       │       │       │       │       │       │       │
  └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Statistics     │
                          │  ─────────────  │
                          │  Median: 0.42ms │
                          │  Mean:   0.43ms │
                          │  Min:    0.40ms │
                          │  Max:    0.45ms │
                          └─────────────────┘
```

---

## Links

| Resource | Link |
|:---------|:-----|
| GitHub Repository | [github.com/kubix-ai/kubix_ci](#) |
| Issue Tracker | [github.com/kubix-ai/kubix_ci/issues](#) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Provider Docs | [docs/providers.md](docs/providers.md) |
| GitHub Actions | [docs/github-actions.md](docs/github-actions.md) |

### Provider Documentation

| Provider | Website | API Docs |
|:---------|:--------|:---------|
| RunPod | [runpod.io](#) | [docs.runpod.io](#) |
| Lambda Labs | [lambdalabs.com](#) | [docs.lambda.ai](#) |
| Vast.ai | [vast.ai](#) | [docs.vast.ai](#) |
| FluidStack | [fluidstack.io](#) | [docs.fluidstack.io](#) |
| NVIDIA Brev | [brev.dev](#) | [docs.nvidia.com/brev](#) |

---

## License

**PolyForm Noncommercial License 1.0.0** - Free for non-commercial use.

| Use Case | Allowed |
|----------|---------|
| Personal projects | ✅ |
| Academic/Research | ✅ |
| Education/Learning | ✅ |
| Non-profits | ✅ |
| Commercial use | ❌ (contact for license) |

See [LICENSE](LICENSE) for full terms.

---

<p align="center">
  Part of the Kubix AI ecosystem.<br>
  For the complete GPU development environment, see Kubix Code Editor.
</p>
