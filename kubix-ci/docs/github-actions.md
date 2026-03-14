# GitHub Actions Integration

kubix_ci provides a GitHub Action for easy CI/CD integration. Test your CUDA kernels on every push or PR.

## Quick Start

### 1. Add kubix_ci.yml to your repo

```yaml
# kubix_ci.yml
targets:
  - name: gpu-server
    provider: ssh
    host: gpu.yourcompany.com
    user: ubuntu
    gpu: A100
```

### 2. Add SSH key as secret

1. Go to your repo → Settings → Secrets → Actions
2. Add `GPU_SSH_KEY` with your private SSH key

### 3. Create workflow

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
          ssh-key: ${{ secrets.GPU_SSH_KEY }}
```

That's it! GPU tests will run on every push and PR.

---

## Action Reference

### Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `kernel` | Yes | - | Path to kernel file(s). Supports globs: `src/*.cu` |
| `config` | No | `kubix_ci.yml` | Path to config file |
| `target` | No | - | Filter to specific target(s) |
| `runs` | No | `10` | Number of benchmark iterations |
| `warmup` | No | `3` | Number of warmup iterations |
| `ssh-key` | No | - | SSH private key for GPU machines |
| `brev-token` | No | - | Brev.dev API token |
| `post-comment` | No | `true` | Post results as PR comment |
| `fail-on-error` | No | `true` | Fail workflow if tests fail |

### Outputs

| Output | Description |
|--------|-------------|
| `results` | Full text output from kubix_ci |
| `passed` | `true` if all tests passed, `false` otherwise |

### Example with all options

```yaml
- uses: kubix-ai/kubix_ci@v1
  with:
    kernel: 'src/kernels/*.cu'
    config: 'config/kubix_ci.yml'
    target: 'h100'
    runs: '20'
    warmup: '5'
    ssh-key: ${{ secrets.GPU_SSH_KEY }}
    post-comment: 'true'
    fail-on-error: 'true'
```

---

## Setting Up Secrets

### SSH Key (recommended)

1. Generate a key pair (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -f kubix_ci_key -N ""
   ```

2. Add public key to your GPU server:
   ```bash
   ssh-copy-id -i kubix_ci_key.pub user@gpu-server.com
   ```

3. Add private key to GitHub:
   - Go to: Repo → Settings → Secrets → Actions
   - Click "New repository secret"
   - Name: `GPU_SSH_KEY`
   - Value: Contents of `kubix_ci_key` (private key)

### Brev Token (for cloud GPUs)

1. Get your token from [brev.dev](https://brev.dev) dashboard
2. Add to GitHub Secrets as `BREV_TOKEN`

---

## Workflow Examples

### Basic: Test on push

```yaml
name: GPU Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kubix-ai/kubix_ci@v1
        with:
          kernel: '**/*.cu'
          ssh-key: ${{ secrets.GPU_SSH_KEY }}
```

### Matrix: Test on multiple GPUs

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        gpu: [h100, a100, rtx4090]
    steps:
      - uses: actions/checkout@v4
      - uses: kubix-ai/kubix_ci@v1
        with:
          kernel: 'kernels/*.cu'
          target: ${{ matrix.gpu }}
          ssh-key: ${{ secrets.GPU_SSH_KEY }}
```

### Only on kernel changes

```yaml
on:
  push:
    paths:
      - '**/*.cu'
      - '**/*.cuh'
      - 'kubix_ci.yml'
```

### Manual trigger

```yaml
on:
  workflow_dispatch:
    inputs:
      kernel:
        description: 'Kernel to test'
        required: true
        default: 'kernels/matmul.cu'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kubix-ai/kubix_ci@v1
        with:
          kernel: ${{ inputs.kernel }}
          ssh-key: ${{ secrets.GPU_SSH_KEY }}
```

---

## PR Comments

When `post-comment: true` (default), kubix_ci posts results as a PR comment:

```
## GPU Kernel Test Results

✅ All GPU tests passed

<details>
<summary>Full Results</summary>

┌─────────────┬───────────────────┬────────┬──────────┐
│ Target      │ GPU               │ Status │   Median │
├─────────────┼───────────────────┼────────┼──────────┤
│ h100-server │ NVIDIA H100       │  PASS  │  0.42ms  │
│ a100-server │ NVIDIA A100       │  PASS  │  0.61ms  │
└─────────────┴───────────────────┴────────┴──────────┘

</details>
```

The comment is updated on each push, not duplicated.

---

## Self-Hosted Runners

If you have GPU machines, you can run the GitHub Actions runner directly on them:

1. Install runner on GPU machine:
   ```bash
   # Follow GitHub's instructions for self-hosted runners
   # Add labels: self-hosted, gpu, cuda
   ```

2. Use in workflow:
   ```yaml
   jobs:
     test:
       runs-on: [self-hosted, gpu, cuda]
       steps:
         - uses: actions/checkout@v4
         - run: |
             nvcc -O3 -o test kernel.cu
             ./test
   ```

---

## Troubleshooting

### "Connection refused"

- Check that the SSH key is correct
- Verify the host is reachable from GitHub Actions
- Try adding the host to known_hosts

### "nvcc not found"

- Ensure CUDA toolkit is installed on the target machine
- Check that nvcc is in PATH

### "Permission denied"

- Check SSH key permissions
- Verify the user has access to run CUDA programs

### Timeouts

- Increase the `timeout` in kubix_ci.yml
- Check network connectivity to GPU machines

---

## Security Notes

- SSH keys are stored as GitHub Secrets (encrypted)
- Keys are written to disk only during workflow execution
- Keys are cleaned up after the job completes
- Use deploy keys with minimal permissions when possible
