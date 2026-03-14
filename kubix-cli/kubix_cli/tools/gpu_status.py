"""
GPU Status and Performance Analyzer

Rock-solid GPU detection and monitoring that:
- Detects all NVIDIA GPUs
- Shows real-time utilization
- Reports memory usage
- Checks CUDA availability
- Provides performance capabilities
"""

import subprocess
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .base import Tool, ToolContext, ToolResult


@dataclass
class GPUInfo:
    """Complete GPU information."""
    index: int
    name: str
    uuid: str
    driver_version: str
    cuda_version: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: int
    utilization_memory: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    compute_capability: str
    clock_graphics_mhz: int
    clock_memory_mhz: int
    pcie_generation: int
    pcie_width: int

    def get_health_status(self) -> str:
        """Get GPU health status."""
        if self.temperature_c > 85:
            return "🔥 Overheating"
        elif self.utilization_gpu > 90:
            return "⚡ High Load"
        elif self.memory_used_mb / self.memory_total_mb > 0.9:
            return "💾 Memory Full"
        else:
            return "✅ Healthy"

    def get_performance_estimate(self) -> Dict[str, float]:
        """Estimate performance capabilities."""
        # Rough estimates based on compute capability
        cc_map = {
            "8.9": {"tflops": 82.6, "bandwidth": 1008},  # Ada Lovelace (RTX 40xx)
            "8.6": {"tflops": 40.0, "bandwidth": 760},   # Ampere (RTX 30xx)
            "8.0": {"tflops": 19.5, "bandwidth": 1555},  # A100
            "7.5": {"tflops": 14.0, "bandwidth": 448},   # Turing (RTX 20xx)
            "7.0": {"tflops": 15.7, "bandwidth": 900},   # V100
            "6.1": {"tflops": 11.3, "bandwidth": 480},   # Pascal (GTX 10xx)
        }

        cc = self.compute_capability
        if cc in cc_map:
            return cc_map[cc]
        else:
            # Default estimate
            return {"tflops": 10.0, "bandwidth": 400}


class GPUStatusTool(Tool):
    """
    Rock-solid GPU status and performance analyzer.

    Features:
    - Complete GPU detection
    - Real-time monitoring
    - Health status
    - Performance capabilities
    - CUDA readiness check
    """

    @property
    def name(self) -> str:
        return "gpu_status"

    @property
    def description(self) -> str:
        return """Get comprehensive GPU status, utilization, and performance capabilities."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "detailed": {
                    "type": "boolean",
                    "description": "Show detailed GPU information (default: True)",
                    "default": True
                }
            }
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Get GPU status with comprehensive error handling."""

        detailed = params.get("detailed", True)

        # Check if nvidia-smi is available
        if not self._check_nvidia_smi():
            return ToolResult(
                title="GPU detection failed",
                output="",
                error=self._get_gpu_error_help()
            )

        # Get all GPUs
        gpus = self._get_gpu_info()

        if not gpus:
            return ToolResult(
                title="No GPUs detected",
                output="",
                error="No NVIDIA GPUs found. Check if:\n1. NVIDIA GPU is installed\n2. Drivers are installed\n3. GPU is enabled in BIOS"
            )

        # Generate report
        report = self._generate_report(gpus, detailed)

        return ToolResult(
            title="GPU Status Report",
            output=report,
            metadata={
                "gpu_count": len(gpus),
                "primary_gpu": gpus[0].name if gpus else None,
                "cuda_available": self._check_cuda_available(),
                "total_memory_gb": sum(g.memory_total_mb for g in gpus) / 1024,
                "success": True
            }
        )

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _get_gpu_error_help(self) -> str:
        """Get helpful error message for GPU issues."""
        import platform

        system = platform.system()

        if system == "Windows":
            return """NVIDIA GPU or drivers not detected!

To fix:
1. Check if you have an NVIDIA GPU:
   → Device Manager > Display adapters

2. Install/Update NVIDIA drivers:
   → https://www.nvidia.com/drivers

3. If using laptop with hybrid graphics:
   → Ensure NVIDIA GPU is active (not Intel/AMD)
   → Check NVIDIA Control Panel settings

4. Restart after driver installation"""

        elif system == "Linux":
            return """NVIDIA GPU or drivers not detected!

To fix:
1. Check if you have an NVIDIA GPU:
   $ lspci | grep -i nvidia

2. Install NVIDIA drivers:
   Ubuntu: $ sudo apt install nvidia-driver-535
   Fedora: $ sudo dnf install akmod-nvidia

3. Verify installation:
   $ nvidia-smi

4. If using laptop, check prime-select:
   $ sudo prime-select nvidia"""

        else:
            return "NVIDIA GPU not detected. Ensure drivers are installed."

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _get_gpu_info(self) -> List[GPUInfo]:
        """Get information for all GPUs."""
        gpus = []

        try:
            # Query comprehensive GPU information
            query = (
                "index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
                "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,"
                "compute_cap,clocks.gr,clocks.mem,pci.link.gen.current,pci.link.width.current"
            )

            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return []

            # Get CUDA version separately
            cuda_version = self._get_cuda_version()

            # Parse each GPU
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 17:
                    continue

                try:
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        uuid=parts[2],
                        driver_version=parts[3],
                        cuda_version=cuda_version,
                        memory_total_mb=int(float(parts[4])),
                        memory_used_mb=int(float(parts[5])),
                        memory_free_mb=int(float(parts[6])),
                        utilization_gpu=int(parts[7]) if parts[7] else 0,
                        utilization_memory=int(parts[8]) if parts[8] else 0,
                        temperature_c=int(parts[9]) if parts[9] else 0,
                        power_draw_w=float(parts[10]) if parts[10] and parts[10] != '[N/A]' else 0,
                        power_limit_w=float(parts[11]) if parts[11] and parts[11] != '[N/A]' else 0,
                        compute_capability=parts[12] if parts[12] else "Unknown",
                        clock_graphics_mhz=int(parts[13]) if parts[13] else 0,
                        clock_memory_mhz=int(parts[14]) if parts[14] else 0,
                        pcie_generation=int(parts[15]) if parts[15] else 0,
                        pcie_width=int(parts[16]) if parts[16] else 0
                    )
                    gpus.append(gpu)
                except:
                    continue

        except:
            pass

        return gpus

    def _get_cuda_version(self) -> str:
        """Get CUDA runtime version."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Extract CUDA version from nvidia-smi output
                match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
                if match:
                    return match.group(1)

            # Try nvcc as fallback
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                match = re.search(r"release\s*([\d.]+)", result.stdout)
                if match:
                    return match.group(1)

        except:
            pass

        return "Not installed"

    def _generate_report(self, gpus: List[GPUInfo], detailed: bool) -> str:
        """Generate comprehensive GPU report."""
        lines = []
        lines.append("="*70)
        lines.append("                    GPU STATUS REPORT")
        lines.append("="*70)
        lines.append("")

        # Summary
        total_memory_gb = sum(g.memory_total_mb for g in gpus) / 1024
        total_used_gb = sum(g.memory_used_mb for g in gpus) / 1024
        avg_utilization = sum(g.utilization_gpu for g in gpus) / len(gpus) if gpus else 0

        lines.append("📊 SYSTEM OVERVIEW")
        lines.append("-"*50)
        lines.append(f"  GPU Count:        {len(gpus)}")
        lines.append(f"  Driver Version:   {gpus[0].driver_version}")
        lines.append(f"  CUDA Version:     {gpus[0].cuda_version}")
        lines.append(f"  Total Memory:     {total_memory_gb:.1f} GB")
        lines.append(f"  Memory Used:      {total_used_gb:.1f} GB ({total_used_gb/total_memory_gb*100:.1f}%)")
        lines.append(f"  Avg Utilization:  {avg_utilization:.0f}%")
        lines.append("")

        # Per-GPU details
        for gpu in gpus:
            lines.append(f"🎮 GPU {gpu.index}: {gpu.name}")
            lines.append("-"*50)

            # Health status
            health = gpu.get_health_status()
            lines.append(f"  Status:           {health}")
            lines.append("")

            # Utilization
            lines.append("  Performance:")
            lines.append(f"    GPU Usage:      {gpu.utilization_gpu}%")
            lines.append(f"    Memory Usage:   {gpu.utilization_memory}%")
            lines.append(f"    Temperature:    {gpu.temperature_c}°C")
            if gpu.power_draw_w > 0:
                lines.append(f"    Power Draw:     {gpu.power_draw_w:.1f}W / {gpu.power_limit_w:.1f}W")
            lines.append("")

            # Memory
            lines.append("  Memory:")
            lines.append(f"    Total:          {gpu.memory_total_mb} MB")
            lines.append(f"    Used:           {gpu.memory_used_mb} MB")
            lines.append(f"    Free:           {gpu.memory_free_mb} MB")
            lines.append("")

            if detailed:
                # Capabilities
                lines.append("  Capabilities:")
                lines.append(f"    Compute Cap:    {gpu.compute_capability}")
                lines.append(f"    Graphics Clock: {gpu.clock_graphics_mhz} MHz")
                lines.append(f"    Memory Clock:   {gpu.clock_memory_mhz} MHz")
                if gpu.pcie_generation > 0:
                    lines.append(f"    PCIe:           Gen{gpu.pcie_generation} x{gpu.pcie_width}")

                # Performance estimate
                perf = gpu.get_performance_estimate()
                lines.append("")
                lines.append("  Performance Estimate:")
                lines.append(f"    Peak TFLOPS:    {perf['tflops']:.1f}")
                lines.append(f"    Peak BW:        {perf['bandwidth']:.0f} GB/s")
                lines.append("")

        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-"*50)

        recommendations = []

        # Check for issues
        for gpu in gpus:
            if gpu.temperature_c > 80:
                recommendations.append(f"GPU {gpu.index} running hot ({gpu.temperature_c}°C) - improve cooling")
            if gpu.memory_used_mb / gpu.memory_total_mb > 0.9:
                recommendations.append(f"GPU {gpu.index} memory nearly full - reduce batch size")
            if gpu.utilization_gpu < 10 and gpu.memory_used_mb > 100:
                recommendations.append(f"GPU {gpu.index} has allocated memory but low usage - check for idle processes")

        if gpus[0].cuda_version == "Not installed":
            recommendations.append("Install CUDA Toolkit for development: https://developer.nvidia.com/cuda-downloads")

        if not recommendations:
            recommendations.append("All GPUs operating normally")

        for rec in recommendations:
            lines.append(f"  • {rec}")

        lines.append("")
        lines.append("="*70)
        return "\n".join(lines)