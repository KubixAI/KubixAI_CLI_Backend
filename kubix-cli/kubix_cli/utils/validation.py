"""
Kubix CLI - Configuration Validation

Uses Pydantic for robust input validation and configuration management.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizationLevel(str, Enum):
    """Optimization level for compilation."""
    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"


class GPUArchitecture(str, Enum):
    """Supported GPU architectures."""
    # NVIDIA
    SM_60 = "sm_60"  # Pascal
    SM_70 = "sm_70"  # Volta
    SM_75 = "sm_75"  # Turing
    SM_80 = "sm_80"  # Ampere
    SM_86 = "sm_86"  # Ampere (RTX 30xx)
    SM_89 = "sm_89"  # Ada Lovelace
    SM_90 = "sm_90"  # Hopper

    # AMD (GCN/RDNA)
    GFX900 = "gfx900"  # Vega
    GFX906 = "gfx906"  # Vega 20
    GFX908 = "gfx908"  # MI100
    GFX90A = "gfx90a"  # MI200
    GFX1030 = "gfx1030"  # RDNA2


class KernelConstraints(BaseModel):
    """Validated kernel compilation constraints."""

    max_registers: int = Field(
        default=255,
        ge=16,
        le=255,
        description="Maximum registers per thread"
    )

    shared_memory_kb: int = Field(
        default=48,
        ge=1,
        le=164,
        description="Maximum shared memory per block in KB"
    )

    target_gpu: GPUArchitecture = Field(
        default=GPUArchitecture.SM_70,
        description="Target GPU architecture"
    )

    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.O3,
        description="Compiler optimization level"
    )

    use_fast_math: bool = Field(
        default=True,
        description="Enable fast math optimizations"
    )

    unroll_loops: bool = Field(
        default=True,
        description="Enable automatic loop unrolling"
    )

    max_threads_per_block: int = Field(
        default=1024,
        ge=32,
        le=1024,
        description="Maximum threads per block"
    )

    @field_validator('max_threads_per_block')
    @classmethod
    def validate_thread_count(cls, v):
        """Ensure thread count is power of 2."""
        if v & (v - 1) != 0:
            raise ValueError(f"max_threads_per_block must be power of 2, got {v}")
        return v

    model_config = ConfigDict(use_enum_values=True)


class OptimizationConfig(BaseModel):
    """Configuration for kernel optimization."""

    variants: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of optimization variants to generate"
    )

    benchmark_iterations: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of benchmark iterations"
    )

    warmup_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of warmup iterations before benchmarking"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Maximum time for optimization in seconds"
    )

    force_regenerate: bool = Field(
        default=False,
        description="Force regeneration even if cached"
    )

    enable_profiling: bool = Field(
        default=True,
        description="Enable detailed profiling"
    )

    parallel_compilation: bool = Field(
        default=True,
        description="Compile variants in parallel"
    )

    max_parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum parallel compilation workers"
    )

    constraints: KernelConstraints = Field(
        default_factory=KernelConstraints,
        description="Kernel compilation constraints"
    )

    @field_validator('max_parallel_workers')
    @classmethod
    def validate_workers(cls, v, info):
        """Ensure workers don't exceed variants."""
        if info.data and 'variants' in info.data and v > info.data['variants']:
            return info.data['variants']
        return v


class APIConfig(BaseModel):
    """Configuration for OpenRouter API."""

    api_key: str = Field(
        ...,
        min_length=20,
        description="OpenRouter API key"
    )

    model: str = Field(
        default="openai/gpt-4",
        description="AI model to use for optimization"
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )

    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=32000,
        description="Maximum tokens in response"
    )

    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    rate_limit_per_minute: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum API calls per minute"
    )

    @field_validator('api_key')
    @classmethod
    def validate_api_key_format(cls, v):
        """Basic validation of API key format."""
        if not v or v.isspace():
            raise ValueError("API key cannot be empty")
        return v.strip()

    model_config = ConfigDict(env_prefix='KUBIX_')


class CacheConfig(BaseModel):
    """Configuration for caching system."""

    enabled: bool = Field(
        default=True,
        description="Enable caching"
    )

    max_size_mb: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum cache size in MB"
    )

    max_age_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Maximum cache entry age in days"
    )

    compression_enabled: bool = Field(
        default=True,
        description="Enable cache compression"
    )

    auto_cleanup: bool = Field(
        default=True,
        description="Automatically clean old cache entries"
    )


class BackendConfig(BaseModel):
    """Configuration for GPU backend selection."""

    preferred_backend: Optional[str] = Field(
        default=None,
        description="Preferred GPU backend (cuda, rocm, sycl, vulkan)"
    )

    device_id: int = Field(
        default=0,
        ge=0,
        description="GPU device ID to use"
    )

    multi_gpu_enabled: bool = Field(
        default=False,
        description="Enable multi-GPU optimization"
    )

    fallback_to_cpu: bool = Field(
        default=False,
        description="Fallback to CPU if no GPU available"
    )

    @field_validator('preferred_backend')
    @classmethod
    def validate_backend_name(cls, v):
        """Validate backend name."""
        if v is not None:
            valid_backends = ['cuda', 'rocm', 'sycl', 'vulkan', 'metal']
            v_lower = v.lower()
            if v_lower not in valid_backends:
                raise ValueError(f"Invalid backend '{v}'. Must be one of {valid_backends}")
            return v_lower
        return v


class GlobalConfig(BaseModel):
    """Global Kubix CLI configuration."""

    api: APIConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of {valid_levels}")
        return v_upper

    @model_validator(mode='after')
    def validate_config_consistency(self):
        """Validate configuration consistency across fields."""
        # Ensure multi-GPU is only enabled if backend supports it
        if self.backend and self.backend.multi_gpu_enabled:
            if self.backend.preferred_backend not in ['cuda', 'rocm']:
                raise ValueError("Multi-GPU only supported on CUDA and ROCm backends")

        return self

    model_config = ConfigDict(validate_assignment=True, extra='forbid')


# ============================================================================
# Helper Functions
# ============================================================================


def load_config_from_dict(data: Dict[str, Any]) -> GlobalConfig:
    """Load configuration from dictionary with validation."""
    return GlobalConfig(**data)


def load_config_from_file(file_path: str) -> GlobalConfig:
    """Load configuration from JSON/YAML file with validation."""
    import json
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    return load_config_from_dict(data)


def create_default_config(api_key: str) -> GlobalConfig:
    """Create default configuration with provided API key."""
    return GlobalConfig(
        api=APIConfig(api_key=api_key),
        optimization=OptimizationConfig(),
        cache=CacheConfig(),
        backend=BackendConfig()
    )
