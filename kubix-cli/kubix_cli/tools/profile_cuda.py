"""
CUDA Profiling Tool with Actionable Insights

Production-ready profiler that:
- Detects available profilers (NCU, NSYS, NVPROF)
- Executes profiling with proper parameters
- Parses output for key metrics
- Provides actionable optimization suggestions
- Shows performance bottlenecks clearly
"""

import os
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .base import Tool, ToolContext, ToolResult
from ..utils.detection import detect_ncu, detect_nsys, detect_nvprof


@dataclass
class KernelMetrics:
    """Parsed kernel performance metrics."""
    name: str
    duration_ms: float
    occupancy: float
    memory_throughput_gbps: float
    compute_throughput_percentage: float
    shared_memory_usage_kb: float
    register_usage: int
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]

    def get_bottleneck(self) -> str:
        """Identify primary bottleneck."""
        if self.occupancy < 50:
            return "Low Occupancy"
        elif self.compute_throughput_percentage < 60:
            return "Compute Bound"
        elif self.memory_throughput_gbps < 100:
            return "Memory Bound"
        else:
            return "Well Balanced"


@dataclass
class ProfilingInsights:
    """Actionable profiling insights."""
    total_runtime_ms: float
    kernel_count: int
    top_kernels: List[KernelMetrics]
    bottlenecks: List[str]
    recommendations: List[str]
    warnings: List[str]

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("="*70)
        lines.append("                    CUDA PROFILING REPORT")
        lines.append("="*70)
        lines.append("")

        # Overview
        lines.append("📊 PERFORMANCE OVERVIEW")
        lines.append("-"*50)
        lines.append(f"  Total Runtime:     {self.total_runtime_ms:.2f} ms")
        lines.append(f"  Kernels Executed:  {self.kernel_count}")
        if self.top_kernels:
            avg_occupancy = sum(k.occupancy for k in self.top_kernels) / len(self.top_kernels)
            lines.append(f"  Avg Occupancy:     {avg_occupancy:.1f}%")
        lines.append("")

        # Top kernels
        if self.top_kernels:
            lines.append("🔥 TOP KERNELS BY TIME")
            lines.append("-"*50)
            for i, kernel in enumerate(self.top_kernels[:5], 1):
                lines.append(f"  {i}. {kernel.name[:40]}")
                lines.append(f"     Time: {kernel.duration_ms:.3f} ms | Occupancy: {kernel.occupancy:.1f}%")
                lines.append(f"     Bottleneck: {kernel.get_bottleneck()}")
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
            for rec in self.recommendations:
                lines.append(f"  → {rec}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("⚠️  WARNINGS")
            lines.append("-"*50)
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        lines.append("="*70)
        return "\n".join(lines)


class ProfileCudaTool(Tool):
    """
    Production-ready CUDA profiling with actionable insights.

    Features:
    - Automatic profiler detection
    - Smart metric extraction
    - Performance bottleneck analysis
    - Actionable recommendations
    - Cross-platform support
    """

    @property
    def name(self) -> str:
        return "profile_cuda"

    @property
    def description(self) -> str:
        return """Profile CUDA kernel performance and provide actionable insights.
Automatically selects best available profiler and analyzes results."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "executable": {
                    "type": "string",
                    "description": "Path to compiled CUDA executable"
                },
                "args": {
                    "type": "string",
                    "description": "Arguments to pass to executable (optional)",
                    "default": ""
                },
                "profile_mode": {
                    "type": "string",
                    "description": "Profile mode: 'quick', 'detailed', 'memory' (optional)",
                    "default": "quick"
                }
            },
            "required": ["executable"]
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute profiling with insights."""

        executable = params["executable"]
        args = params.get("args", "")
        mode = params.get("profile_mode", "quick")

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

        # Detect available profiler
        profiler = self._select_profiler()
        if not profiler:
            return ToolResult(
                title="No profiler available",
                output="",
                error=self._get_profiler_install_instructions()
            )

        # Execute profiling
        raw_output, success = self._run_profiler(profiler, executable, args, mode)

        if not success:
            return ToolResult(
                title="Profiling failed",
                output="",
                error=f"Failed to profile {executable.name}:\n\n{raw_output}"
            )

        # Parse output for insights
        insights = self._parse_profiler_output(profiler["type"], raw_output, mode)

        # Generate report
        report = insights.to_report()

        return ToolResult(
            title=f"Profiled {executable.name}",
            output=report,
            metadata={
                "profiler": profiler["type"],
                "executable": str(executable),
                "mode": mode,
                "total_runtime_ms": insights.total_runtime_ms,
                "kernel_count": insights.kernel_count,
                "bottlenecks": insights.bottlenecks,
                "success": True
            }
        )

    def _select_profiler(self) -> Optional[Dict[str, str]]:
        """Select best available profiler."""

        # Check NCU (Nsight Compute - most detailed)
        ncu = detect_ncu()
        if ncu.available:
            return {
                "type": "ncu",
                "path": ncu.path or "ncu",
                "version": ncu.version
            }

        # Check NSYS (Nsight Systems - good timeline)
        nsys = detect_nsys()
        if nsys.available:
            return {
                "type": "nsys",
                "path": nsys.path or "nsys",
                "version": nsys.version
            }

        # Check NVPROF (deprecated but works)
        nvprof = detect_nvprof()
        if nvprof.available:
            return {
                "type": "nvprof",
                "path": nvprof.path or "nvprof",
                "version": nvprof.version
            }

        return None

    def _get_profiler_install_instructions(self) -> str:
        """Get installation instructions for profilers."""
        return """No CUDA profiler found!

To install a profiler, choose one:

1. **NVIDIA Nsight Compute (Recommended)**
   → Best for kernel-level analysis
   → Download: https://developer.nvidia.com/nsight-compute

2. **NVIDIA Nsight Systems**
   → Best for application timeline
   → Download: https://developer.nvidia.com/nsight-systems

3. **Install CUDA Toolkit (includes profilers)**
   → https://developer.nvidia.com/cuda-downloads

After installation, restart your terminal."""

    def _run_profiler(
        self,
        profiler: Dict[str, str],
        executable: Path,
        args: str,
        mode: str
    ) -> Tuple[str, bool]:
        """Run profiler and capture output."""

        profiler_path = profiler["path"]
        profiler_type = profiler["type"]

        # Build command based on profiler and mode
        if profiler_type == "ncu":
            # Nsight Compute
            if mode == "quick":
                cmd = [profiler_path, "--print-summary", "per-kernel", str(executable)]
            elif mode == "detailed":
                cmd = [profiler_path, "--set", "full", str(executable)]
            else:  # memory
                cmd = [profiler_path, "--metrics", "dram__throughput,l2_cache_throughput", str(executable)]

        elif profiler_type == "nsys":
            # Nsight Systems
            cmd = [profiler_path, "profile", "--stats=true", "--cuda-memory-usage=true", str(executable)]

        else:  # nvprof
            # Legacy nvprof
            if mode == "quick":
                cmd = [profiler_path, "--print-gpu-summary", str(executable)]
            elif mode == "detailed":
                cmd = [profiler_path, "--print-gpu-trace", str(executable)]
            else:  # memory
                cmd = [profiler_path, "--print-gpu-trace", "--print-api-trace", str(executable)]

        # Add user args
        if args:
            cmd.extend(args.split())

        try:
            # Run profiler
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(executable.parent)
            )

            # Combine stdout and stderr (profilers output to both)
            output = result.stdout + "\n" + result.stderr

            # Check for common errors
            if "no kernels were profiled" in output.lower():
                return "No CUDA kernels were executed. Check if your program uses GPU.", False

            if "cuda driver error" in output.lower():
                return "CUDA driver error. Ensure NVIDIA drivers are installed and GPU is available.", False

            return output, result.returncode == 0

        except subprocess.TimeoutExpired:
            return "Profiling timed out after 2 minutes.", False
        except Exception as e:
            return f"Failed to run profiler: {str(e)}", False

    def _parse_profiler_output(
        self,
        profiler_type: str,
        raw_output: str,
        mode: str
    ) -> ProfilingInsights:
        """Parse profiler output for insights."""

        kernels = []
        total_time = 0.0
        bottlenecks = []
        recommendations = []
        warnings = []

        if profiler_type == "ncu":
            kernels, total_time = self._parse_ncu_output(raw_output)
        elif profiler_type == "nsys":
            kernels, total_time = self._parse_nsys_output(raw_output)
        else:  # nvprof
            kernels, total_time = self._parse_nvprof_output(raw_output)

        # Analyze for bottlenecks and recommendations
        if kernels:
            # Sort by time
            kernels.sort(key=lambda k: k.duration_ms, reverse=True)

            # Check occupancy
            low_occupancy = [k for k in kernels if k.occupancy < 50]
            if low_occupancy:
                bottlenecks.append(f"{len(low_occupancy)} kernels with <50% occupancy")
                recommendations.append("Increase occupancy: Reduce register usage or shared memory per block")

            # Check for serialized kernels
            if len(kernels) > 10:
                recommendations.append("Many kernel launches detected. Consider kernel fusion to reduce overhead.")

            # Check memory throughput
            low_memory = [k for k in kernels if k.memory_throughput_gbps < 50]
            if low_memory:
                bottlenecks.append(f"{len(low_memory)} kernels with low memory throughput")
                recommendations.append("Optimize memory access: Use coalesced access patterns")

            # Top kernel analysis
            top_kernel = kernels[0] if kernels else None
            if top_kernel:
                if top_kernel.duration_ms > total_time * 0.5:
                    bottlenecks.append(f"'{top_kernel.name}' takes {top_kernel.duration_ms/total_time*100:.1f}% of runtime")
                    recommendations.append(f"Focus optimization on '{top_kernel.name[:30]}' kernel")

        else:
            warnings.append("No kernel metrics could be extracted from profiler output")

        return ProfilingInsights(
            total_runtime_ms=total_time,
            kernel_count=len(kernels),
            top_kernels=kernels[:10],
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            warnings=warnings
        )

    def _parse_ncu_output(self, output: str) -> Tuple[List[KernelMetrics], float]:
        """Parse NCU output for metrics."""
        kernels = []
        total_time = 0.0

        # NCU output patterns
        kernel_pattern = r"(\w+)\s+\((\d+)\)"
        duration_pattern = r"Duration\s+([\d.]+)\s+(us|ms|s)"
        occupancy_pattern = r"Occupancy\s+([\d.]+)\s*%"

        lines = output.split('\n')
        current_kernel = None

        for line in lines:
            # Extract kernel name
            kernel_match = re.search(kernel_pattern, line)
            if kernel_match and "kernel" in line.lower():
                if current_kernel:
                    kernels.append(current_kernel)

                current_kernel = KernelMetrics(
                    name=kernel_match.group(1),
                    duration_ms=0.0,
                    occupancy=0.0,
                    memory_throughput_gbps=0.0,
                    compute_throughput_percentage=0.0,
                    shared_memory_usage_kb=0.0,
                    register_usage=0,
                    grid_size=(0, 0, 0),
                    block_size=(0, 0, 0)
                )

            # Extract metrics
            if current_kernel:
                duration_match = re.search(duration_pattern, line)
                if duration_match:
                    value = float(duration_match.group(1))
                    unit = duration_match.group(2)
                    if unit == "us":
                        current_kernel.duration_ms = value / 1000
                    elif unit == "ms":
                        current_kernel.duration_ms = value
                    elif unit == "s":
                        current_kernel.duration_ms = value * 1000
                    total_time += current_kernel.duration_ms

                occupancy_match = re.search(occupancy_pattern, line)
                if occupancy_match:
                    current_kernel.occupancy = float(occupancy_match.group(1))

        if current_kernel:
            kernels.append(current_kernel)

        return kernels, total_time

    def _parse_nsys_output(self, output: str) -> Tuple[List[KernelMetrics], float]:
        """Parse NSYS output for metrics."""
        kernels = []
        total_time = 0.0

        # NSYS patterns
        cuda_kernel_section = False

        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "CUDA Kernel Statistics" in line:
                cuda_kernel_section = True
                continue

            if cuda_kernel_section and "Time(%)" in line:
                # Parse table data
                for j in range(i+1, min(i+20, len(lines))):
                    data_line = lines[j].strip()
                    if not data_line or "---" in data_line:
                        continue

                    parts = data_line.split()
                    if len(parts) >= 6:
                        try:
                            time_percent = float(parts[0].replace('%', ''))
                            time_ms = float(parts[1]) / 1000 if parts[2] == "us" else float(parts[1])
                            kernel_name = " ".join(parts[6:])

                            kernel = KernelMetrics(
                                name=kernel_name,
                                duration_ms=time_ms,
                                occupancy=50.0,  # NSYS doesn't provide occupancy
                                memory_throughput_gbps=0.0,
                                compute_throughput_percentage=time_percent,
                                shared_memory_usage_kb=0.0,
                                register_usage=0,
                                grid_size=(0, 0, 0),
                                block_size=(0, 0, 0)
                            )
                            kernels.append(kernel)
                            total_time += time_ms
                        except:
                            pass

        return kernels, total_time

    def _parse_nvprof_output(self, output: str) -> Tuple[List[KernelMetrics], float]:
        """Parse nvprof output for metrics."""
        kernels = []
        total_time = 0.0

        # NVPROF patterns
        gpu_activities_section = False

        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "GPU activities:" in line:
                gpu_activities_section = True
                continue

            if gpu_activities_section:
                # Parse percentage lines
                match = re.match(r'\s+([\d.]+)%\s+([\d.]+)(us|ms|s)\s+.*\s+(\w+)', line)
                if match:
                    percent = float(match.group(1))
                    time_val = float(match.group(2))
                    time_unit = match.group(3)
                    kernel_name = match.group(4)

                    # Convert to ms
                    if time_unit == "us":
                        time_ms = time_val / 1000
                    elif time_unit == "ms":
                        time_ms = time_val
                    else:  # s
                        time_ms = time_val * 1000

                    kernel = KernelMetrics(
                        name=kernel_name,
                        duration_ms=time_ms,
                        occupancy=50.0,  # NVPROF doesn't always provide
                        memory_throughput_gbps=0.0,
                        compute_throughput_percentage=percent,
                        shared_memory_usage_kb=0.0,
                        register_usage=0,
                        grid_size=(0, 0, 0),
                        block_size=(0, 0, 0)
                    )
                    kernels.append(kernel)
                    total_time += time_ms

        return kernels, total_time