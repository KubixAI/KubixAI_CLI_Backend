"""
Kubix CLI - Async Kernel Optimizer

Parallel kernel optimization using async/await for significant performance improvements.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..backends.base import GPUBackend, CompileOptions
from ..utils.validation import OptimizationConfig, KernelConstraints
from ..exceptions import (
    OptimizationError, OptimizationTimeoutError,
    NoValidVariantsError, CompilationError
)


@dataclass
class OptimizationResult:
    """Result of kernel optimization."""
    kernel_name: str
    original_code: str
    optimized_code: str
    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    compilation_time_ms: float
    total_time_ms: float
    variants_generated: int
    variants_compiled: int
    best_variant_index: int
    metrics: Dict[str, Any]
    backend_type: str


class AsyncOptimizer:
    """
    Async kernel optimizer for parallel variant generation and compilation.

    Features:
    - Parallel AI variant generation
    - Concurrent kernel compilation
    - Async benchmarking
    - Timeout handling
    - Progress tracking
    """

    def __init__(
        self,
        backend: GPUBackend,
        api_client: Any,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize async optimizer.

        Args:
            backend: GPU backend for compilation and benchmarking
            api_client: API client for AI generation
            config: Optimization configuration
        """
        self.backend = backend
        self.api_client = api_client
        self.config = config or OptimizationConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_workers
        )

    async def optimize_kernel_async(
        self,
        code: str,
        kernel_name: str,
        analysis: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize kernel asynchronously with parallel variant generation.

        Args:
            code: Original kernel code
            kernel_name: Name of kernel function
            analysis: Kernel analysis results

        Returns:
            OptimizationResult with best optimized variant

        Raises:
            OptimizationTimeoutError: If optimization times out
            NoValidVariantsError: If no valid variants generated
        """
        start_time = time.time()

        try:
            # Run optimization with timeout
            result = await asyncio.wait_for(
                self._optimize_internal(code, kernel_name, analysis),
                timeout=self.config.timeout_seconds
            )

            result.total_time_ms = (time.time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            raise OptimizationTimeoutError(self.config.timeout_seconds)

    async def _optimize_internal(
        self,
        code: str,
        kernel_name: str,
        analysis: Dict[str, Any]
    ) -> OptimizationResult:
        """Internal optimization logic."""

        # Step 1: Generate variants in parallel
        print(f"[GENERATE] Generating {self.config.variants} variants in parallel...")
        generation_tasks = [
            self._generate_variant_async(code, analysis, i)
            for i in range(self.config.variants)
        ]

        variant_results = await asyncio.gather(
            *generation_tasks,
            return_exceptions=True
        )

        # Filter out failed generations
        variants = [
            v for v in variant_results
            if not isinstance(v, Exception) and v is not None
        ]

        if not variants:
            raise NoValidVariantsError(self.config.variants)

        print(f"[GENERATE] Successfully generated {len(variants)} variants")

        # Step 2: Compile variants in parallel
        print(f"[COMPILE] Compiling {len(variants)} variants in parallel...")
        compilation_start = time.time()

        compilation_tasks = [
            self._compile_variant_async(variant, kernel_name)
            for variant in variants
        ]

        compiled_results = await asyncio.gather(
            *compilation_tasks,
            return_exceptions=True
        )

        # Filter out failed compilations
        compiled_variants = [
            c for c in compiled_results
            if not isinstance(c, Exception) and c is not None
        ]

        compilation_time_ms = (time.time() - compilation_start) * 1000

        if not compiled_variants:
            raise NoValidVariantsError(len(variants))

        print(f"[COMPILE] Successfully compiled {len(compiled_variants)} variants")

        # Step 3: Benchmark variants in parallel (if configured)
        if self.config.parallel_compilation:
            print(f"[BENCHMARK] Benchmarking {len(compiled_variants)} variants in parallel...")
            benchmark_tasks = [
                self._benchmark_variant_async(cv, analysis)
                for cv in compiled_variants
            ]

            benchmark_results = await asyncio.gather(
                *benchmark_tasks,
                return_exceptions=True
            )
        else:
            # Sequential benchmarking
            benchmark_results = []
            for cv in compiled_variants:
                result = await self._benchmark_variant_async(cv, analysis)
                benchmark_results.append(result)

        # Filter valid benchmarks
        valid_benchmarks = [
            b for b in benchmark_results
            if not isinstance(b, Exception) and b is not None
        ]

        if not valid_benchmarks:
            raise OptimizationError("All variants failed benchmarking")

        # Step 4: Select best variant
        best = min(valid_benchmarks, key=lambda x: x['time_ms'])
        best_index = valid_benchmarks.index(best)

        # Benchmark original for comparison
        original_time = await self._benchmark_code_async(code, kernel_name, analysis)

        # Create result
        return OptimizationResult(
            kernel_name=kernel_name,
            original_code=code,
            optimized_code=best['code'],
            original_time_ms=original_time,
            optimized_time_ms=best['time_ms'],
            speedup=original_time / best['time_ms'] if best['time_ms'] > 0 else 0,
            compilation_time_ms=compilation_time_ms,
            total_time_ms=0,  # Set by caller
            variants_generated=len(variants),
            variants_compiled=len(compiled_variants),
            best_variant_index=best_index,
            metrics=best.get('metrics', {}),
            backend_type=self.backend.get_backend_type().value
        )

    async def _generate_variant_async(
        self,
        code: str,
        analysis: Dict[str, Any],
        variant_index: int
    ) -> Optional[str]:
        """Generate a single variant asynchronously."""
        try:
            # Run AI generation in thread pool (blocking I/O)
            loop = asyncio.get_event_loop()
            variant_code = await loop.run_in_executor(
                self._executor,
                self._generate_variant_sync,
                code,
                analysis,
                variant_index
            )
            return variant_code
        except Exception as e:
            print(f"[WARNING] Variant {variant_index} generation failed: {e}")
            return None

    def _generate_variant_sync(
        self,
        code: str,
        analysis: Dict[str, Any],
        variant_index: int
    ) -> str:
        """Synchronous variant generation (called in thread pool)."""
        # Use the API client to generate variant
        # This is a placeholder - actual implementation depends on API client
        constraints = {
            'max_registers': self.config.constraints.max_registers,
            'shared_memory_kb': self.config.constraints.shared_memory_kb,
            'target_gpu': self.config.constraints.target_gpu
        }

        variants = self.api_client.generate_kernel_optimizations(
            original_code=code,
            analysis=analysis,
            constraints=constraints,
            num_variants=1
        )

        if variants:
            return variants[0].code
        return None

    async def _compile_variant_async(
        self,
        code: str,
        kernel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Compile a variant asynchronously."""
        try:
            # Run compilation in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            compiled = await loop.run_in_executor(
                self._executor,
                self._compile_variant_sync,
                code,
                kernel_name
            )
            return compiled
        except CompilationError as e:
            print(f"[WARNING] Compilation failed: {e.message}")
            return None
        except Exception as e:
            print(f"[WARNING] Compilation error: {e}")
            return None

    def _compile_variant_sync(
        self,
        code: str,
        kernel_name: str
    ) -> Dict[str, Any]:
        """Synchronous compilation (called in thread pool)."""
        options = CompileOptions(
            optimization_level=self.config.constraints.optimization_level.value[-1],
            use_fast_math=self.config.constraints.use_fast_math,
            max_registers=self.config.constraints.max_registers,
            target_arch=self.config.constraints.target_gpu
        )

        compiled_kernel = self.backend.compile_kernel(code, kernel_name, options)

        return {
            'code': code,
            'kernel': compiled_kernel,
            'kernel_name': kernel_name
        }

    async def _benchmark_variant_async(
        self,
        compiled_variant: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Benchmark a compiled variant asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._benchmark_variant_sync,
                compiled_variant,
                analysis
            )
            return result
        except Exception as e:
            print(f"[WARNING] Benchmark failed: {e}")
            return None

    def _benchmark_variant_sync(
        self,
        compiled_variant: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous benchmarking (called in thread pool)."""
        kernel = compiled_variant['kernel']

        # Create test configuration based on analysis
        grid_size, block_size, test_args = self._create_test_config(analysis)

        # Benchmark
        time_ms = self.backend.benchmark_kernel(
            kernel,
            grid_size=grid_size,
            block_size=block_size,
            args=test_args,
            iterations=self.config.benchmark_iterations,
            warmup=self.config.warmup_iterations
        )

        result = {
            'code': compiled_variant['code'],
            'time_ms': time_ms,
            'kernel': kernel
        }

        # Add profiling if enabled
        if self.config.enable_profiling:
            try:
                metrics = self.backend.profile_kernel(
                    kernel,
                    grid_size=grid_size,
                    block_size=block_size,
                    args=test_args
                )
                result['metrics'] = metrics
            except:
                pass

        return result

    async def _benchmark_code_async(
        self,
        code: str,
        kernel_name: str,
        analysis: Dict[str, Any]
    ) -> float:
        """Benchmark original code asynchronously."""
        try:
            # Compile
            compiled = await self._compile_variant_async(code, kernel_name)
            if not compiled:
                return float('inf')

            # Benchmark
            result = await self._benchmark_variant_async(compiled, analysis)
            if not result:
                return float('inf')

            return result['time_ms']
        except:
            return float('inf')

    def _create_test_config(
        self,
        analysis: Dict[str, Any]
    ) -> tuple:
        """Create test configuration based on kernel analysis."""
        # Default configuration
        size = 1024 * 1024  # 1M elements
        block_size = (256, 1, 1)
        grid_size = ((size + 255) // 256, 1, 1)

        # Adjust based on patterns
        patterns = analysis.get('patterns', [])

        if 'matrix multiplication' in patterns:
            m, n, k = 1024, 1024, 1024
            block_size = (16, 16, 1)
            grid_size = ((n + 15) // 16, (m + 15) // 16, 1)
            # Create test matrices (simplified)
            import numpy as np
            test_args = [
                np.random.randn(m, k).astype(np.float32),
                np.random.randn(k, n).astype(np.float32),
                np.zeros((m, n), dtype=np.float32)
            ]
        elif 'reduction' in patterns:
            block_size = (256, 1, 1)
            grid_size = ((size + 255) // 256, 1, 1)
            import numpy as np
            test_args = [
                np.random.randn(size).astype(np.float32),
                np.zeros(1, dtype=np.float32)
            ]
        else:
            # Generic
            import numpy as np
            test_args = [
                np.random.randn(size).astype(np.float32),
                np.zeros(size, dtype=np.float32)
            ]

        return grid_size, block_size, test_args

    async def optimize_multiple_kernels(
        self,
        kernels: List[Dict[str, Any]]
    ) -> List[OptimizationResult]:
        """
        Optimize multiple kernels concurrently.

        Args:
            kernels: List of kernel dicts with 'code', 'name', 'analysis'

        Returns:
            List of OptimizationResult
        """
        tasks = [
            self.optimize_kernel_async(
                k['code'],
                k['name'],
                k['analysis']
            )
            for k in kernels
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r for r in results
            if not isinstance(r, Exception)
        ]

    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
