# Changelog

All notable changes to kubix_ci will be documented in this file.

The format is based on [Keep a Changelog](#),
and this project adheres to [Semantic Versioning](#).

## [0.1.0] - 2025-01-25

### License
- **PolyForm Noncommercial 1.0.0** - Free for non-commercial use
- Commercial use requires separate license from Kubix AI

### Added

#### Core Features
- **Multi-GPU Testing**: Run CUDA kernels across multiple GPU targets simultaneously
- **CUDA Event Timing**: Microsecond-precision kernel timing using `cudaEventRecord`/`cudaEventElapsedTime`
- **Parallel Execution**: ThreadPoolExecutor-based parallel dispatch to all targets
- **Rich Output**: Beautiful terminal output with Rich tables

#### GPU Providers (6 total)
- **SSH**: Direct connection to your own GPU machines
- **RunPod**: On-demand cloud GPUs via RunPod SDK
- **Lambda Labs**: High-performance cloud GPUs via REST API
- **Vast.ai**: GPU marketplace via vastai-sdk
- **FluidStack**: Enterprise GPUs via REST API
- **Brev**: NVIDIA Brev cloud GPUs via CLI

#### CLI Commands
- `kubix_ci init`: Interactive configuration wizard
- `kubix_ci test <kernel.cu>`: Run kernel tests across targets
- `kubix_ci targets`: List configured GPU targets
- `kubix_ci check`: Verify connectivity to all targets

#### Configuration
- YAML-based configuration (`kubix_ci.yml`)
- Automatic config file discovery (searches parent directories)
- Customizable warmup runs, benchmark runs, and timeout
- Per-target nvcc flags support

#### Kernel Support
- **Full kernels**: Kernels with `main()` using KUBIXCI macros
- **Kernel-only**: Auto-generated test harness for `__global__` functions
- Automatic GPU architecture detection via `nvidia-smi`

#### GitHub Actions Integration
- Reusable GitHub Action (`action.yml`)
- PR comment support with benchmark results
- Multiple workflow examples included

#### Documentation
- Comprehensive README with examples
- Detailed provider setup guide (`docs/providers.md`)
- GitHub Actions integration guide (`docs/github-actions.md`)

### Verified
- Tested on NVIDIA GeForce RTX 5070 Ti (Vast.ai)
- All example kernels pass: `vector_add.cu`, `matmul.cu`, `simple_kernel.cu`

---

## Future Releases

### Planned for v0.2.0
- [ ] JSON output format for programmatic use
- [ ] Retry logic for transient cloud provider failures
- [ ] Cost estimation before running tests
- [ ] Historical benchmark tracking

### Planned for v0.3.0
- [ ] Web dashboard for results visualization
- [ ] Notifications
- [ ] Custom metrics beyond timing
- [ ] Multi-kernel test files

---

[0.1.0]: #
