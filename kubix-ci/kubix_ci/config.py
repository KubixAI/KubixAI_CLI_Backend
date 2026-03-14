"""Configuration loading and validation for kubix_ci."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
import yaml

from kubix_ci.exceptions import ConfigError


# =============================================================================
# Target Dataclasses - One per provider
# =============================================================================


@dataclass
class SSHTarget:
    """Direct SSH connection target.

    Connect to your own GPU machines via SSH.

    Required:
        name: Unique identifier for this target
        host: SSH hostname or IP address
        user: SSH username
        gpu: GPU name (for display purposes)

    Optional:
        key: Path to SSH private key (default: uses ssh-agent or ~/.ssh/id_rsa)
        port: SSH port (default: 22)
    """

    name: str
    host: str
    user: str
    gpu: str
    provider: str = "ssh"
    key: str | None = None
    port: int = 22


@dataclass
class BrevTarget:
    """NVIDIA Brev cloud GPU target.

    Uses brev CLI to manage instances.
    Requires: `brev login` to authenticate.

    Docs: https://docs.nvidia.com/brev/

    Required:
        name: Unique identifier for this target
        gpu: GPU type (e.g., H100, A100, T4, L4)

    Optional:
        instance_name: Existing Brev instance name (default: creates new)
    """

    name: str
    gpu: str
    provider: str = "brev"
    instance_name: str | None = None


@dataclass
class RunPodTarget:
    """RunPod cloud GPU target.

    Uses RunPod Python SDK.
    Requires: RUNPOD_API_KEY environment variable.

    Docs: https://docs.runpod.io/

    Required:
        name: Unique identifier for this target
        gpu: GPU type (e.g., "NVIDIA A100 80GB PCIe", "NVIDIA RTX 4090")

    Optional:
        gpu_count: Number of GPUs (default: 1)
        image: Docker image to use (default: runpod/pytorch:latest)
        volume_size: Persistent volume size in GB (default: 20)
    """

    name: str
    gpu: str
    provider: str = "runpod"
    gpu_count: int = 1
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    volume_size: int = 20


@dataclass
class LambdaLabsTarget:
    """Lambda Labs cloud GPU target.

    Uses Lambda Cloud REST API.
    Requires: LAMBDA_API_KEY environment variable.

    Docs: https://docs.lambda.ai/

    Required:
        name: Unique identifier for this target
        gpu: Instance type (e.g., gpu_1x_h100_pcie, gpu_1x_a100)

    Optional:
        region: Preferred region (default: any available)
        ssh_key_name: Name of SSH key in Lambda account
    """

    name: str
    gpu: str  # Actually instance_type_name like "gpu_1x_h100_pcie"
    provider: str = "lambdalabs"
    region: str | None = None
    ssh_key_name: str | None = None


@dataclass
class VastAITarget:
    """Vast.ai cloud GPU target.

    Uses Vast.ai Python SDK.
    Requires: VASTAI_API_KEY environment variable.

    Docs: https://docs.vast.ai/

    Required:
        name: Unique identifier for this target
        gpu: GPU name filter (e.g., RTX_4090, A100, H100)

    Optional:
        min_gpu_ram: Minimum GPU RAM in GB (default: 0)
        max_price: Maximum price per hour in $ (default: no limit)
        image: Docker image (default: pytorch/pytorch:latest)
    """

    name: str
    gpu: str
    provider: str = "vastai"
    min_gpu_ram: int = 0
    max_price: float | None = None
    image: str = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"


@dataclass
class FluidStackTarget:
    """FluidStack cloud GPU target.

    Uses FluidStack REST API.
    Requires: FLUIDSTACK_API_KEY environment variable.

    Docs: https://docs.fluidstack.io/

    Required:
        name: Unique identifier for this target
        gpu: GPU type (e.g., RTX_A6000_48GB, H100_SXM_80GB)

    Optional:
        ssh_key_name: Name of SSH key in FluidStack account
    """

    name: str
    gpu: str
    provider: str = "fluidstack"
    ssh_key_name: str | None = None


# Union of all target types
Target = Union[
    SSHTarget,
    BrevTarget,
    RunPodTarget,
    LambdaLabsTarget,
    VastAITarget,
    FluidStackTarget,
]


# =============================================================================
# Main Config
# =============================================================================


@dataclass
class KUBIXCIConfig:
    """Main configuration for kubix_ci."""

    targets: list[Target]
    nvcc_flags: list[str] = field(default_factory=lambda: ["-O3"])
    warmup_runs: int = 3
    benchmark_runs: int = 10
    timeout: int = 120  # seconds per target


# =============================================================================
# Config Loading
# =============================================================================


def find_config_file(start_dir: Path | None = None) -> Path | None:
    """Walk up from start_dir looking for kubix_ci.yml or kubix_ci.yaml."""
    if start_dir is None:
        start_dir = Path.cwd()

    current = start_dir.resolve()

    while True:
        for name in ["kubix_ci.yml", "kubix_ci.yaml"]:
            config_path = current / name
            if config_path.exists():
                return config_path

        parent = current.parent
        if parent == current:
            return None
        current = parent


def parse_target(target_dict: dict) -> Target:
    """Parse a single target entry from YAML into typed dataclass."""
    if "name" not in target_dict:
        raise ConfigError("Target missing required field 'name'")

    provider = target_dict.get("provider", "ssh")

    if provider == "ssh":
        required = ["host", "user", "gpu"]
        for field_name in required:
            if field_name not in target_dict:
                raise ConfigError(
                    f"SSH target '{target_dict['name']}' missing required field '{field_name}'"
                )

        return SSHTarget(
            name=target_dict["name"],
            host=target_dict["host"],
            user=target_dict["user"],
            gpu=target_dict["gpu"],
            key=target_dict.get("key"),
            port=target_dict.get("port", 22),
        )

    elif provider == "brev":
        if "gpu" not in target_dict:
            raise ConfigError(f"Brev target '{target_dict['name']}' missing required field 'gpu'")

        return BrevTarget(
            name=target_dict["name"],
            gpu=target_dict["gpu"],
            instance_name=target_dict.get("instance_name"),
        )

    elif provider == "runpod":
        if "gpu" not in target_dict:
            raise ConfigError(f"RunPod target '{target_dict['name']}' missing required field 'gpu'")

        return RunPodTarget(
            name=target_dict["name"],
            gpu=target_dict["gpu"],
            gpu_count=target_dict.get("gpu_count", 1),
            image=target_dict.get("image", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"),
            volume_size=target_dict.get("volume_size", 20),
        )

    elif provider == "lambdalabs":
        if "gpu" not in target_dict:
            raise ConfigError(
                f"Lambda Labs target '{target_dict['name']}' missing required field 'gpu'"
            )

        return LambdaLabsTarget(
            name=target_dict["name"],
            gpu=target_dict["gpu"],
            region=target_dict.get("region"),
            ssh_key_name=target_dict.get("ssh_key_name"),
        )

    elif provider == "vastai":
        if "gpu" not in target_dict:
            raise ConfigError(f"Vast.ai target '{target_dict['name']}' missing required field 'gpu'")

        return VastAITarget(
            name=target_dict["name"],
            gpu=target_dict["gpu"],
            min_gpu_ram=target_dict.get("min_gpu_ram", 0),
            max_price=target_dict.get("max_price"),
            image=target_dict.get("image", "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"),
        )

    elif provider == "fluidstack":
        if "gpu" not in target_dict:
            raise ConfigError(
                f"FluidStack target '{target_dict['name']}' missing required field 'gpu'"
            )

        return FluidStackTarget(
            name=target_dict["name"],
            gpu=target_dict["gpu"],
            ssh_key_name=target_dict.get("ssh_key_name"),
        )

    else:
        raise ConfigError(f"Unknown provider '{provider}' for target '{target_dict['name']}'")


def load_config(path: Path | None = None) -> KUBIXCIConfig:
    """Load config from kubix_ci.yml, searching up directory tree if path not specified."""
    if path is None:
        path = find_config_file()
        if path is None:
            raise ConfigError(
                "No kubix_ci.yml found. Run 'kubix_ci init' to create one, "
                "or specify --config path."
            )

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    if data is None:
        raise ConfigError(f"Config file is empty: {path}")

    if "targets" not in data:
        raise ConfigError("Config missing required 'targets' section")

    if not isinstance(data["targets"], list):
        raise ConfigError("'targets' must be a list")

    if len(data["targets"]) == 0:
        raise ConfigError("'targets' list is empty. Add at least one target.")

    targets = [parse_target(t) for t in data["targets"]]

    return KUBIXCIConfig(
        targets=targets,
        nvcc_flags=data.get("nvcc_flags", ["-O3"]),
        warmup_runs=data.get("warmup_runs", 3),
        benchmark_runs=data.get("benchmark_runs", 10),
        timeout=data.get("timeout", 120),
    )


def create_default_config() -> str:
    """Generate default config template for kubix_ci init."""
    return '''# kubix_ci configuration
# Documentation: https://github.com/kubix-ai/kubix_ci

targets:
  # ==========================================================================
  # SSH - Your own GPU machines
  # ==========================================================================
  - name: my-gpu-server
    provider: ssh
    host: gpu.example.com
    user: ubuntu
    key: ~/.ssh/id_rsa  # optional
    port: 22            # optional
    gpu: RTX 4090       # for display

  # ==========================================================================
  # RunPod - Cloud GPUs with Python SDK
  # Requires: pip install runpod && export RUNPOD_API_KEY=xxx
  # Docs: https://docs.runpod.io/
  # ==========================================================================
  # - name: runpod-a100
  #   provider: runpod
  #   gpu: "NVIDIA A100 80GB PCIe"
  #   gpu_count: 1
  #   image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

  # ==========================================================================
  # Lambda Labs - Cloud GPUs with REST API
  # Requires: export LAMBDA_API_KEY=xxx
  # Docs: https://docs.lambda.ai/
  # ==========================================================================
  # - name: lambda-h100
  #   provider: lambdalabs
  #   gpu: gpu_1x_h100_pcie  # instance type name
  #   region: us-west-1      # optional

  # ==========================================================================
  # Vast.ai - GPU marketplace
  # Requires: pip install vastai-sdk && export VASTAI_API_KEY=xxx
  # Docs: https://docs.vast.ai/
  # ==========================================================================
  # - name: vastai-4090
  #   provider: vastai
  #   gpu: RTX_4090
  #   max_price: 0.50  # max $/hour
  #   image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

  # ==========================================================================
  # FluidStack - Enterprise GPU cloud
  # Requires: export FLUIDSTACK_API_KEY=xxx
  # Docs: https://docs.fluidstack.io/
  # ==========================================================================
  # - name: fluidstack-h100
  #   provider: fluidstack
  #   gpu: H100_SXM_80GB

  # ==========================================================================
  # NVIDIA Brev - Cloud GPUs via CLI
  # Requires: brev login
  # Docs: https://docs.nvidia.com/brev/
  # ==========================================================================
  # - name: brev-h100
  #   provider: brev
  #   gpu: H100

# Compilation flags
nvcc_flags:
  - "-O3"

# Timing configuration
warmup_runs: 3      # warmup iterations
benchmark_runs: 10  # timed iterations
timeout: 120        # seconds per target
'''


def save_config(config_str: str, path: Path | None = None) -> Path:
    """Save config string to file."""
    if path is None:
        path = Path.cwd() / "kubix_ci.yml"

    with open(path, "w") as f:
        f.write(config_str)

    return path
