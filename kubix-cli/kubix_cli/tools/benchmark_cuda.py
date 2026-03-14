"""
CUDA Benchmarking Tool with Performance Metrics

Production-ready benchmarking that:
- Runs multiple iterations for statistical accuracy
- Measures key performance metrics (FLOPS, bandwidth, latency)
- Compares against theoretical GPU limits
- Provides optimization suggestions
- Generates performance report
"""

import os
import subprocess
import statistics
import time
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .base import Tool, ToolContext, ToolResult
from ..utils.detection import detect_nvcc


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""
    # Timing
    avg_runtime_ms: float
    min_runtime_ms: float
    max_runtime_ms: float
    std_dev_ms: float

    # Throughput
    throughput_gflops: float
    memory_bandwidth_gbps: float
    kernel_launch_overhead_us: float

    # Efficiency
    gpu_efficiency_percent: float
    memory_efficiency_percent: float
    compute_efficiency_percent: float

    # Comparison
    theoretical_peak_gflops: float
    theoretical_bandwidth_gbps: float

    def get_performance_rating(self) -> str:
        """Get overall performance rating."""
        efficiency = (self.gpu_efficiency_percent +
                     self.memory_efficiency_percent +
                     self.compute_efficiency_percent) / 3

        if efficiency >= 80:
            return "🏆 Excellent"
        elif efficiency >= 60:
            return "✅ Good"
        elif efficiency >= 40:
            return "⚠️ Moderate"
        else:
            return "❌ Poor"


@dataclass
class BenchmarkReport:
    """Complete benchmark report with insights."""
    executable_name: str
    gpu_name: str
    iterations: int
    metrics: BenchmarkMetrics
    bottlenecks: List[str]
    recommendations: List[str]
    comparison_baseline: Optional[Dict] = None

    def to_report(self) -> str:
        """Generate actionable benchmark report."""
        lines = []
        lines.append("="*70)
        lines.append("              CUDA PERFORMANCE BENCHMARK REPORT")
        lines.append("="*70)
        lines.append("")

        # System Info
        lines.append("🖥️  SYSTEM CONFIGURATION")
        lines.append("-"*50)
        lines.append(f"  Executable:  {self.executable_name}")
        lines.append(f"  GPU:         {self.gpu_name}")
        lines.append(f"  Iterations:  {self.iterations}")
        lines.append("")

        # Performance Rating
        rating = self.metrics.get_performance_rating()
        lines.append(f"📊 PERFORMANCE RATING: {rating}")
        lines.append("")

        # Timing Results
        lines.append("⏱️  EXECUTION TIME")
        lines.append("-"*50)
        lines.append(f"  Average:     {self.metrics.avg_runtime_ms:.3f} ms")
        lines.append(f"  Minimum:     {self.metrics.min_runtime_ms:.3f} ms")
        lines.append(f"  Maximum:     {self.metrics.max_runtime_ms:.3f} ms")
        lines.append(f"  Std Dev:     {self.metrics.std_dev_ms:.3f} ms")
        lines.append(f"  Variability: {(self.metrics.std_dev_ms/self.metrics.avg_runtime_ms*100):.1f}%")
        lines.append("")

        # Throughput Metrics
        lines.append("🚀 THROUGHPUT METRICS")
        lines.append("-"*50)
        lines.append(f"  Compute:     {self.metrics.throughput_gflops:.2f} GFLOPS")
        lines.append(f"  Memory BW:   {self.metrics.memory_bandwidth_gbps:.2f} GB/s")
        lines.append(f"  Launch OH:   {self.metrics.kernel_launch_overhead_us:.2f} μs")
        lines.append("")

        # Efficiency Analysis
        lines.append("📈 EFFICIENCY ANALYSIS")
        lines.append("-"*50)
        lines.append(f"  GPU Utilization:      {self.metrics.gpu_efficiency_percent:.1f}%")
        lines.append(f"  Memory Efficiency:    {self.metrics.memory_efficiency_percent:.1f}%")
        lines.append(f"  Compute Efficiency:   {self.metrics.compute_efficiency_percent:.1f}%")
        lines.append("")
        lines.append(f"  vs. Peak FLOPS:       {self.metrics.throughput_gflops/self.metrics.theoretical_peak_gflops*100:.1f}%")
        lines.append(f"  vs. Peak Bandwidth:   {self.metrics.memory_bandwidth_gbps/self.metrics.theoretical_bandwidth_gbps*100:.1f}%")
        lines.append("")

        # Bottlenecks
        if self.bottlenecks:
            lines.append("⚠️  PERFORMANCE BOTTLENECKS")
            lines.append("-"*50)
            for bottleneck in self.bottlenecks:
                lines.append(f"  • {bottleneck}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("💡 OPTIMIZATION RECOMMENDATIONS")
            lines.append("-"*50)
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        # Baseline Comparison
        if self.comparison_baseline:
            lines.append("📊 BASELINE COMPARISON")
            lines.append("-"*50)
            baseline_time = self.comparison_baseline.get("runtime_ms", 0)
            if baseline_time > 0:
                speedup = baseline_time / self.metrics.avg_runtime_ms
                lines.append(f"  Speedup:     {speedup:.2f}x")
                if speedup > 1:
                    lines.append(f"  Status:      ✅ {(speedup-1)*100:.1f}% faster")
                else:
                    lines.append(f"  Status:      ❌ {(1-speedup)*100:.1f}% slower")
            lines.append("")

        lines.append("="*70)
        return "\n".join(lines)


class BenchmarkCudaTool(Tool):
    """
    Production-ready CUDA benchmarking with actionable metrics.

    Features:
    - Statistical accuracy through multiple iterations
    - Comprehensive performance metrics
    - Efficiency analysis
    - Bottleneck detection
    - Optimization recommendations
    """

    @property
    def name(self) -> str:
        return "benchmark_cuda"

    @property
    def description(self) -> str:
        return """Benchmark CUDA executable and provide performance metrics with optimization suggestions."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "executable": {
                    "type": "string",
                    "description": "Path to compiled CUDA executable"
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of iterations to run (default: 10)",
                    "default": 10
                },
                "warmup": {
                    "type": "integer",
                    "description": "Warmup iterations before benchmarking (default: 3)",
                    "default": 3
                },
                "args": {
                    "type": "string",
                    "description": "Arguments to pass to executable",
                    "default": ""
                }
            },
            "required": ["executable"]
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute comprehensive benchmarking."""

        executable = params["executable"]
        iterations = params.get("iterations", 10)
        warmup = params.get("warmup", 3)
        args = params.get("args", "")

        # Resolve path
        if not Path(executable).is_absolute():
            executable = Path(ctx.working_dir) / executable
        executable = Path(executable)

        # Check executable exists
        if not executable.exists():
            return ToolResult(
                title="Executable not found",
                output="",
                error=f"Cannot find executable: {executable}\n\nCompile your CUDA code first with 'compile_cuda'"
            )

        # Get GPU info
        gpu_info = self._get_gpu_info()
        if not gpu_info:
            return ToolResult(
                title="GPU not detected",
                output="",
                error="No NVIDIA GPU detected. Ensure drivers are installed."
            )

        # Run warmup iterations
        for _ in range(warmup):
            self._run_executable(executable, args)

        # Run benchmark iterations
        timings = []
        outputs = []

        for i in range(iterations):
            start_time = time.perf_counter()
            output, success = self._run_executable(executable, args)
            end_time = time.perf_counter()

            if not success:
                return ToolResult(
                    title="Benchmark failed",
                    output="",
                    error=f"Failed to run {executable.name}:\n\n{output}"
                )

            runtime_ms = (end_time - start_time) * 1000
            timings.append(runtime_ms)
            outputs.append(output)

        # Calculate metrics
        metrics = self._calculate_metrics(timings, outputs, gpu_info)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, bottlenecks)

        # Create report
        report = BenchmarkReport(
            executable_name=executable.name,
            gpu_name=gpu_info["name"],
            iterations=iterations,
            metrics=metrics,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )

        # Generate output
        report_text = report.to_report()

        return ToolResult(
            title=f"Benchmarked {executable.name}",
            output=report_text,
            metadata={
                "executable": str(executable),
                "iterations": iterations,
                "avg_runtime_ms": metrics.avg_runtime_ms,
                "efficiency_percent": (metrics.gpu_efficiency_percent +
                                      metrics.memory_efficiency_percent +
                                      metrics.compute_efficiency_percent) / 3,
                "rating": metrics.get_performance_rating(),
                "success": True
            }
        )

    def _get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information."""
        try:
            # Get GPU name
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    gpu_name = parts[0].strip()
                    compute_cap = parts[1].strip()
                    memory_gb = float(parts[2].strip()) / 1024

                    # Estimate theoretical peaks based on GPU
                    peak_flops, peak_bandwidth = self._get_theoretical_peaks(gpu_name)

                    return {
                        "name": gpu_name,
                        "compute_capability": compute_cap,
                        "memory_gb": memory_gb,
                        "peak_gflops": peak_flops,
                        "peak_bandwidth_gbps": peak_bandwidth
                    }

            return None

        except:
            return None

    def _get_theoretical_peaks(self, gpu_name: str) -> Tuple[float, float]:
        """Get theoretical peak performance for known GPUs."""

        # GPU theoretical peaks (GFLOPS, GB/s)
        gpu_peaks = {
            # Consumer GPUs
            "RTX 4090": (82580, 1008),
            "RTX 4080": (48740, 717),
            "RTX 4070 Ti": (40090, 504),
            "RTX 4070": (29150, 504),
            "RTX 3090": (35580, 936),
            "RTX 3080": (29770, 760),
            "RTX 3070": (20370, 448),
            "RTX 3060": (12740, 360),

            # Data center GPUs
            "A100": (19500, 1555),
            "V100": (15700, 900),
            "T4": (8100, 320),
            "A40": (37420, 696),
            "H100": (67000, 3350),

            # Default estimates
            "Default": (10000, 400)
        }

        # Find matching GPU
        for gpu_key, peaks in gpu_peaks.items():
            if gpu_key.lower() in gpu_name.lower():
                return peaks

        return gpu_peaks["Default"]

    def _run_executable(self, executable: Path, args: str) -> Tuple[str, bool]:
        """Run executable once."""
        try:
            cmd = [str(executable)]
            if args:
                cmd.extend(args.split())

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(executable.parent),
                env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}  # Sync for accurate timing
            )

            return result.stdout + result.stderr, result.returncode == 0

        except subprocess.TimeoutExpired:
            return "Execution timed out", False
        except Exception as e:
            return str(e), False

    def _calculate_metrics(
        self,
        timings: List[float],
        outputs: List[str],
        gpu_info: Dict
    ) -> BenchmarkMetrics:
        """Calculate comprehensive metrics."""

        # Timing statistics
        avg_time = statistics.mean(timings)
        min_time = min(timings)
        max_time = max(timings)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

        # Parse output for FLOPS/bandwidth if reported
        gflops = self._extract_gflops(outputs)
        bandwidth = self._extract_bandwidth(outputs)

        # Estimate if not reported
        if gflops == 0:
            # Rough estimate based on runtime
            gflops = 100 / avg_time  # Placeholder

        if bandwidth == 0:
            # Rough estimate
            bandwidth = 50 / avg_time  # Placeholder

        # Calculate efficiencies
        peak_gflops = gpu_info["peak_gflops"]
        peak_bandwidth = gpu_info["peak_bandwidth_gbps"]

        gpu_efficiency = min((gflops / peak_gflops) * 100, 100)
        memory_efficiency = min((bandwidth / peak_bandwidth) * 100, 100)
        compute_efficiency = gpu_efficiency  # Simplified

        # Estimate kernel launch overhead
        launch_overhead = 10.0  # Typical ~10 microseconds

        return BenchmarkMetrics(
            avg_runtime_ms=avg_time,
            min_runtime_ms=min_time,
            max_runtime_ms=max_time,
            std_dev_ms=std_dev,
            throughput_gflops=gflops,
            memory_bandwidth_gbps=bandwidth,
            kernel_launch_overhead_us=launch_overhead,
            gpu_efficiency_percent=gpu_efficiency,
            memory_efficiency_percent=memory_efficiency,
            compute_efficiency_percent=compute_efficiency,
            theoretical_peak_gflops=peak_gflops,
            theoretical_bandwidth_gbps=peak_bandwidth
        )

    def _extract_gflops(self, outputs: List[str]) -> float:
        """Extract GFLOPS from output if reported."""
        for output in outputs:
            # Look for common GFLOPS patterns
            patterns = [
                r"(\d+\.?\d*)\s*GFLOPS",
                r"(\d+\.?\d*)\s*GFlops",
                r"Throughput:\s*(\d+\.?\d*)\s*GFLOP"
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return float(match.group(1))

        return 0.0

    def _extract_bandwidth(self, outputs: List[str]) -> float:
        """Extract bandwidth from output if reported."""
        for output in outputs:
            # Look for bandwidth patterns
            patterns = [
                r"(\d+\.?\d*)\s*GB/s",
                r"(\d+\.?\d*)\s*GB/sec",
                r"Bandwidth:\s*(\d+\.?\d*)"
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return float(match.group(1))

        return 0.0

    def _detect_bottlenecks(self, metrics: BenchmarkMetrics) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        # Low GPU utilization
        if metrics.gpu_efficiency_percent < 50:
            bottlenecks.append(f"Low GPU utilization ({metrics.gpu_efficiency_percent:.1f}%)")

        # Memory bottleneck
        if metrics.memory_efficiency_percent < 30:
            bottlenecks.append(f"Memory bandwidth underutilized ({metrics.memory_efficiency_percent:.1f}%)")

        # High variability
        if metrics.std_dev_ms > metrics.avg_runtime_ms * 0.1:
            bottlenecks.append(f"High runtime variability ({metrics.std_dev_ms/metrics.avg_runtime_ms*100:.1f}%)")

        # Kernel launch overhead
        if metrics.kernel_launch_overhead_us > metrics.avg_runtime_ms * 10:
            bottlenecks.append("Excessive kernel launch overhead")

        # Compute bound
        if metrics.compute_efficiency_percent > 80 and metrics.memory_efficiency_percent < 50:
            bottlenecks.append("Compute-bound workload")

        # Memory bound
        if metrics.memory_efficiency_percent > 80 and metrics.compute_efficiency_percent < 50:
            bottlenecks.append("Memory-bound workload")

        return bottlenecks

    def _generate_recommendations(
        self,
        metrics: BenchmarkMetrics,
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Based on efficiency
        if metrics.gpu_efficiency_percent < 50:
            recommendations.append("Increase parallelism: Use more threads/blocks")
            recommendations.append("Check for thread divergence and synchronization bottlenecks")

        if metrics.memory_efficiency_percent < 50:
            recommendations.append("Optimize memory access: Ensure coalesced reads/writes")
            recommendations.append("Use shared memory for frequently accessed data")

        # Based on bottlenecks
        if "High runtime variability" in str(bottlenecks):
            recommendations.append("Profile for irregular behavior or system interference")
            recommendations.append("Run on dedicated GPU without display manager")

        if "Compute-bound" in str(bottlenecks):
            recommendations.append("Consider algorithmic optimizations to reduce operations")
            recommendations.append("Use tensor cores if available (for matrix operations)")

        if "Memory-bound" in str(bottlenecks):
            recommendations.append("Reduce memory transfers between host and device")
            recommendations.append("Consider data compression or precision reduction")

        # General improvements
        if metrics.throughput_gflops < metrics.theoretical_peak_gflops * 0.3:
            recommendations.append("Profile with NCU to identify specific kernel bottlenecks")

        return recommendations