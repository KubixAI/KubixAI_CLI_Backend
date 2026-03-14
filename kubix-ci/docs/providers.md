# GPU Providers

kubix_ci supports 6 GPU providers for testing CUDA kernels. This guide covers setup, configuration, and best practices for each provider.

## Provider Overview

| Provider | Type | Best For | Pricing | Instance Lifecycle |
|----------|------|----------|---------|-------------------|
| **SSH** | Your machines | Dedicated hardware | Free | Always on |
| **RunPod** | Cloud | Quick testing | $0.20-4.00/hr | On-demand pods |
| **Lambda Labs** | Cloud | Production workloads | $1.10-3.00/hr | On-demand instances |
| **Vast.ai** | Marketplace | Cost optimization | $0.10-2.00/hr | Spot-like rentals |
| **FluidStack** | Enterprise | Large scale | Custom | Managed instances |
| **Brev** | Cloud | Development | $0.50-4.00/hr | CLI-managed |

## Quick Start

```yaml
# kubix_ci.yml - Example with multiple providers
targets:
  # Your own machine
  - name: local-4090
    provider: ssh
    host: gpu.local
    user: ubuntu
    gpu: RTX 4090

  # Cloud providers (pick one or more)
  - name: runpod-a100
    provider: runpod
    gpu: "NVIDIA A100 80GB PCIe"

  - name: lambda-h100
    provider: lambdalabs
    gpu: gpu_1x_h100_pcie

  - name: vastai-4090
    provider: vastai
    gpu: RTX_4090
    max_price: 0.50
```

---

## SSH Provider

Direct SSH connection to your own GPU machines. No cloud costs, full control.

### Configuration

```yaml
- name: my-server
  provider: ssh
  host: gpu.example.com     # Required: hostname or IP
  user: ubuntu              # Required: SSH username
  port: 22                  # Optional: SSH port (default: 22)
  key: ~/.ssh/id_rsa        # Optional: path to SSH private key
  gpu: RTX 4090             # Required: GPU name (for display)
```

### Setup

1. Ensure SSH access to your GPU machine:
   ```bash
   ssh ubuntu@gpu.example.com
   ```

2. Verify CUDA is installed on remote:
   ```bash
   ssh ubuntu@gpu.example.com "nvcc --version && nvidia-smi"
   ```

3. (Optional) Add SSH key for passwordless access:
   ```bash
   ssh-copy-id ubuntu@gpu.example.com
   ```

### Requirements on Remote Machine

- NVIDIA GPU with drivers installed
- CUDA toolkit with `nvcc` compiler
- SSH access (password or key-based)
- Sufficient disk space for kernel compilation

### Troubleshooting

**Connection refused**: Check firewall and SSH service status
```bash
sudo systemctl status ssh
sudo ufw allow 22/tcp
```

**CUDA not found**: Ensure CUDA is in PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

---

## RunPod Provider

RunPod offers on-demand GPU pods with simple pricing and fast spin-up times.

### Installation

```bash
pip install kubix_ci[runpod]
```

### Configuration

```yaml
- name: runpod-a100
  provider: runpod
  gpu: "NVIDIA A100 80GB PCIe"    # Required: GPU type ID
  gpu_count: 1                     # Optional: number of GPUs (default: 1)
  image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04  # Optional
  volume_size: 20                  # Optional: volume size in GB
```

### Setup

1. Get your API key from RunPod Settings

2. Set environment variable:
   ```bash
   export RUNPOD_API_KEY=your_api_key_here
   ```

3. For CI/CD, add as a secret:
   ```yaml
   # GitHub Actions
   env:
     RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
   ```

### Available GPU Types

| GPU Type ID | GPU | VRAM | Approx. Price |
|-------------|-----|------|---------------|
| `NVIDIA RTX 4090` | RTX 4090 | 24GB | $0.44/hr |
| `NVIDIA A100 80GB PCIe` | A100 | 80GB | $1.89/hr |
| `NVIDIA A100-SXM4-80GB` | A100 SXM | 80GB | $1.99/hr |
| `NVIDIA H100 PCIe` | H100 | 80GB | $3.89/hr |
| `NVIDIA L40S` | L40S | 48GB | $1.14/hr |

Get current availability:
```python
import runpod
runpod.api_key = "your_key"
print(runpod.get_gpus())
```

### How It Works

1. kubix_ci creates a new pod with your specified GPU
2. Waits for pod to reach RUNNING state
3. Connects via SSH (port 22 exposed automatically)
4. Uploads kernel, compiles, runs benchmarks
5. Terminates pod on completion (you only pay for runtime)

---

## Lambda Labs Provider

Lambda Labs provides high-performance cloud GPUs with pre-installed CUDA and ML frameworks.

### Configuration

```yaml
- name: lambda-h100
  provider: lambdalabs
  gpu: gpu_1x_h100_pcie        # Required: instance type name
  region: us-west-1            # Optional: preferred region
  ssh_key_name: my-key         # Optional: SSH key name (uses first key if not specified)
```

### Setup

1. Get your API key from Lambda Cloud API Keys

2. Set environment variable:
   ```bash
   export LAMBDA_API_KEY=your_api_key_here
   ```

3. Add an SSH key to your account at Lambda SSH Keys

### Available Instance Types

| Instance Type | GPU | VRAM | Price |
|---------------|-----|------|-------|
| `gpu_1x_a10` | A10 | 24GB | $0.60/hr |
| `gpu_1x_a100` | A100 40GB | 40GB | $1.10/hr |
| `gpu_1x_a100_sxm4` | A100 80GB | 80GB | $1.29/hr |
| `gpu_1x_h100_pcie` | H100 | 80GB | $2.49/hr |
| `gpu_8x_a100` | 8x A100 | 320GB | $8.80/hr |
| `gpu_8x_h100_sxm5` | 8x H100 | 640GB | $23.92/hr |

Check availability:
```bash
curl -u $LAMBDA_API_KEY: [url]
```

### Regions

- `us-west-1` (California)
- `us-south-1` (Texas)
- `us-east-1` (New York)
- `me-central-1` (Middle East)
- `asia-south-1` (India)

---

## Vast.ai Provider

Vast.ai is a GPU marketplace where you can rent from independent providers at competitive prices.

### Installation

```bash
pip install kubix_ci[vastai]
```

### Configuration

```yaml
- name: vastai-4090
  provider: vastai
  gpu: RTX_4090                # Required: GPU name
  max_price: 0.50              # Optional: max $/hour (filters expensive offers)
  min_gpu_ram: 24              # Optional: minimum GPU RAM in GB
  image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel  # Optional: Docker image
```

### Setup

1. Create account at Vast.ai

2. Get your API key from Account Settings

3. Set environment variable:
   ```bash
   export VASTAI_API_KEY=your_api_key_here
   ```

4. Add credits to your account (prepaid model)

### GPU Names

| GPU Name | Typical VRAM | Price Range |
|----------|--------------|-------------|
| `RTX_3090` | 24GB | $0.15-0.30/hr |
| `RTX_4090` | 24GB | $0.30-0.50/hr |
| `A100_PCIE` | 40/80GB | $0.80-1.50/hr |
| `A100_SXM4` | 80GB | $1.00-2.00/hr |
| `H100_PCIE` | 80GB | $2.00-3.00/hr |
| `H100_SXM5` | 80GB | $2.50-4.00/hr |

### Search Filters

kubix_ci automatically searches for:
- Unrented, rentable machines
- Verified hosts (more reliable)
- Matches your GPU name, RAM, and price requirements

The cheapest matching offer is selected automatically.

### Notes

- Vast.ai uses a marketplace model - prices and availability fluctuate
- Offers are sorted by price; kubix_ci picks the cheapest match
- Instances are destroyed after testing (you only pay for runtime)
- Consider setting `max_price` to avoid expensive offers during high demand

---

## FluidStack Provider

FluidStack provides enterprise-grade GPU cloud infrastructure.

### Configuration

```yaml
- name: fluidstack-h100
  provider: fluidstack
  gpu: H100_SXM_80GB           # Required: GPU type
  ssh_key_name: my-key         # Optional: SSH key name
```

### Setup

1. Create account at FluidStack Console

2. Get your API key from API Keys

3. Set environment variable:
   ```bash
   export FLUIDSTACK_API_KEY=your_api_key_here
   ```

4. Add an SSH key at SSH Keys

### Available GPU Types

| GPU Type | GPU | VRAM |
|----------|-----|------|
| `RTX_A6000_48GB` | RTX A6000 | 48GB |
| `A100_PCIE_40GB` | A100 | 40GB |
| `A100_SXM_80GB` | A100 SXM | 80GB |
| `H100_PCIE_80GB` | H100 PCIe | 80GB |
| `H100_SXM_80GB` | H100 SXM | 80GB |

### Notes

- Enterprise pricing - contact FluidStack for rates
- Best for organizations needing guaranteed capacity
- SLA-backed availability

---

## Brev Provider

NVIDIA Brev (formerly Brev.dev) provides managed cloud GPUs with CLI-based management.

### Configuration

```yaml
- name: brev-h100
  provider: brev
  gpu: H100                    # Required: GPU type for display
```

### Setup

1. Install Brev CLI:
   ```bash
   curl -fsSL [url] | bash
   ```

2. Login:
   ```bash
   brev login
   ```

3. Verify:
   ```bash
   brev ls
   ```

### How It Works

Unlike other providers, Brev uses CLI commands rather than API:

1. kubix_ci runs `brev create` to create an instance
2. Polls `brev ls` until instance is ready
3. Extracts SSH info and connects via paramiko
4. Runs kernel tests
5. Runs `brev delete` to clean up

### Notes

- Requires Brev CLI installed locally
- Uses your Brev account credentials
- Good for development and testing
- GUI available at [console.brev.dev](#)

---

## Environment Variables Reference

| Provider | Variable | Description |
|----------|----------|-------------|
| RunPod | `RUNPOD_API_KEY` | API key from RunPod settings |
| Lambda Labs | `LAMBDA_API_KEY` | API key from Lambda Cloud |
| Vast.ai | `VASTAI_API_KEY` | API key from Vast.ai account |
| FluidStack | `FLUIDSTACK_API_KEY` | API key from FluidStack console |

**Note**: SSH and Brev providers don't require environment variables (SSH uses keys, Brev uses CLI login).

---

## GitHub Actions Integration

Set up secrets in your repository settings:

```yaml
# .github/workflows/gpu-test.yml
name: GPU Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install kubix_ci
        run: pip install kubix_ci[cloud]

      - name: Run GPU tests
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          LAMBDA_API_KEY: ${{ secrets.LAMBDA_API_KEY }}
          VASTAI_API_KEY: ${{ secrets.VASTAI_API_KEY }}
        run: kubix_ci test kernel.cu
```

---

## Cost Optimization Tips

1. **Use SSH for frequent testing**: If you have dedicated hardware, SSH provider is free.

2. **Set max_price on Vast.ai**: Prevents accidentally renting expensive machines during high demand.

3. **Use appropriate GPU**: Don't rent an H100 for simple kernels that run fine on a 4090.

4. **Clean up instances**: kubix_ci automatically terminates instances, but verify in provider dashboards.

5. **Use spot/preemptible when available**: Some providers offer discounted interruptible instances.

6. **Test locally first**: Debug on local GPU before running on expensive cloud instances.

---

## Troubleshooting

### "API key not set"

Ensure the environment variable is set:
```bash
echo $RUNPOD_API_KEY  # Should show your key
```

### "No instances available"

- Try a different GPU type
- Try a different region
- Check provider status page
- For Vast.ai, increase `max_price`

### "SSH connection failed"

- Instance may still be initializing
- Check that SSH key is added to provider account
- Verify firewall allows SSH (port 22)

### "CUDA not found on remote"

- Use a CUDA-enabled Docker image
- Verify CUDA toolkit is installed
- Check PATH includes `/usr/local/cuda/bin`

### Instance not terminating

If kubix_ci crashes before cleanup:
- RunPod: Check RunPod console
- Lambda Labs: Check Lambda instances
- Vast.ai: Check Vast.ai instances
- FluidStack: Check FluidStack console
