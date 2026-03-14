"""Orchestrates parallel kernel execution across GPU targets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kubix_ci.config import (
    KUBIXCIConfig,
    Target,
    SSHTarget,
    BrevTarget,
    RunPodTarget,
    LambdaLabsTarget,
    VastAITarget,
    FluidStackTarget,
)
from kubix_ci.providers.base import BaseProvider, ExecutionResult
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.timing import wrap_kernel_with_timing
from kubix_ci.exceptions import ConfigError


def create_provider(target: Target) -> BaseProvider:
    """Factory: instantiate the correct provider for a target."""
    if isinstance(target, SSHTarget):
        return SSHProvider(target)
    elif isinstance(target, BrevTarget):
        # Import here to avoid circular imports and optional dependencies
        from kubix_ci.providers.brev import BrevProvider
        return BrevProvider(target)
    elif isinstance(target, RunPodTarget):
        from kubix_ci.providers.runpod import RunPodProvider
        return RunPodProvider(target)
    elif isinstance(target, LambdaLabsTarget):
        from kubix_ci.providers.lambdalabs import LambdaLabsProvider
        return LambdaLabsProvider(target)
    elif isinstance(target, VastAITarget):
        from kubix_ci.providers.vastai import VastAIProvider
        return VastAIProvider(target)
    elif isinstance(target, FluidStackTarget):
        from kubix_ci.providers.fluidstack import FluidStackProvider
        return FluidStackProvider(target)
    else:
        raise ConfigError(f"Unknown target type: {type(target)}")


def run_on_target(
    target: Target,
    wrapped_source: str,
    nvcc_flags: list[str],
    timeout: int,
    verbose: bool = False,
) -> ExecutionResult:
    """Execute the full test workflow on a single target.

    This function is called in a separate thread for each target.

    Args:
        target: Target configuration
        wrapped_source: CUDA source with timing wrapper
        nvcc_flags: Compilation flags
        timeout: Execution timeout
        verbose: Print verbose output

    Returns:
        ExecutionResult with timing and status
    """
    provider = create_provider(target)

    try:
        with provider:
            result = provider.compile_and_run(
                wrapped_source=wrapped_source,
                nvcc_flags=nvcc_flags,
                timeout=timeout,
            )
        return result

    except Exception as e:
        # Catch any unhandled exceptions and return as error result
        return ExecutionResult(
            target_name=target.name,
            gpu_name=target.gpu,
            status="error",
            error=str(e),
        )


def filter_targets(
    targets: list[Target],
    target_filter: str | None,
) -> list[Target]:
    """Filter targets by name or GPU type."""
    if not target_filter:
        return targets

    filter_lower = target_filter.lower()
    return [
        t for t in targets
        if filter_lower in t.name.lower() or filter_lower in t.gpu.lower()
    ]


def run_all(
    kernel_path: str | Path,
    config: KUBIXCIConfig,
    target_filter: str | None = None,
    verbose: bool = False,
    progress_callback=None,
) -> list[ExecutionResult]:
    """Run kernel on all (or filtered) targets in parallel.

    Args:
        kernel_path: Path to the CUDA kernel source file
        config: kubix_ci configuration
        target_filter: Optional filter for target names/GPU types
        verbose: Enable verbose output
        progress_callback: Optional callback(target_name, stage) for progress updates

    Returns:
        List of ExecutionResult, one per target
    """
    kernel_path = Path(kernel_path)

    if not kernel_path.exists():
        raise ConfigError(f"Kernel file not found: {kernel_path}")

    kernel_source = kernel_path.read_text()

    # Wrap kernel with timing infrastructure
    wrapped_source = wrap_kernel_with_timing(
        kernel_source,
        warmup=config.warmup_runs,
        runs=config.benchmark_runs,
    )

    # Filter targets if specified
    targets = filter_targets(config.targets, target_filter)

    if not targets:
        if target_filter:
            raise ConfigError(
                f"No targets match filter '{target_filter}'. "
                f"Available targets: {', '.join(t.name for t in config.targets)}"
            )
        raise ConfigError("No targets configured. Run 'kubix_ci init' to add targets.")

    results: list[ExecutionResult] = []

    # Run all targets in parallel
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        future_to_target = {
            executor.submit(
                run_on_target,
                target,
                wrapped_source,
                config.nvcc_flags,
                config.timeout,
                verbose,
            ): target
            for target in targets
        }

        for future in as_completed(future_to_target):
            target = future_to_target[future]

            if progress_callback:
                progress_callback(target.name, "completed")

            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # This shouldn't happen since run_on_target catches exceptions
                results.append(ExecutionResult(
                    target_name=target.name,
                    gpu_name=target.gpu,
                    status="error",
                    error=f"Unexpected error: {e}",
                ))

    return results
