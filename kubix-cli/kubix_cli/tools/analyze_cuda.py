"""
Analyze CUDA Tool - Deep analysis of CUDA kernels.
"""

from pathlib import Path
from typing import Dict, Any
from .base import Tool, ToolContext, ToolResult
from ..ui.progress import show_spinner


class AnalyzeCudaTool(Tool):
    """Analyze a CUDA kernel for optimization opportunities."""

    @property
    def name(self) -> str:
        return "analyze_cuda"

    @property
    def description(self) -> str:
        return """Analyze a CUDA kernel file to identify optimization opportunities.
Detects patterns, memory access, shared memory usage, and suggests improvements.
Use this before optimizing a kernel to understand its characteristics."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the CUDA file to analyze (.cu). Also accepts 'file_path' for compatibility."
                },
                "file_path": {
                    "type": "string",
                    "description": "Alternative parameter name for filepath (for compatibility)"
                }
            },
            "required": []
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Analyze CUDA kernel."""
        # Support both 'filepath' and 'file_path' for compatibility
        filepath = params.get("filepath") or params.get("file_path")
        if not filepath:
            raise ValueError("Missing required parameter: filepath or file_path")

        # Resolve path
        if not Path(filepath).is_absolute():
            filepath = Path(ctx.working_dir) / filepath

        filepath = Path(filepath)

        # Check if file exists
        if not filepath.exists():
            return ToolResult(
                title=f"File not found: {filepath.name}",
                output="",
                error=f"CUDA file does not exist: {filepath}"
            )

        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # Import analyzer (reuse existing one)
            from kubix_cli.kernel_analyzer import KernelAnalyzer

            # Analyze with progress indicator
            with show_spinner(f"Analyzing {filepath.name}"):
                analyzer = KernelAnalyzer()
                analysis = analyzer.analyze_kernel(code)

            # Format analysis as readable output
            output_lines = []
            output_lines.append("=== KERNEL ANALYSIS ===\n")
            output_lines.append(f"Kernel Name: {analysis.get('kernel_name', 'unknown')}")
            output_lines.append(f"Parameters: {len(analysis.get('parameters', []))}")
            output_lines.append(f"Complexity: {analysis.get('complexity', 'unknown')}")
            output_lines.append(f"Arithmetic Intensity: {analysis.get('arithmetic_intensity', 0):.2f}")

            if analysis.get('patterns'):
                output_lines.append(f"\nDetected Patterns:")
                for pattern in analysis['patterns']:
                    output_lines.append(f"  • {pattern}")

            if analysis.get('optimization_opportunities'):
                output_lines.append(f"\nOptimization Opportunities:")
                for i, opp in enumerate(analysis['optimization_opportunities'], 1):
                    output_lines.append(f"  {i}. {opp}")

            if analysis.get('performance_hints'):
                output_lines.append(f"\nPerformance Hints:")
                for hint in analysis['performance_hints']:
                    output_lines.append(f"  💡 {hint}")

            output = "\n".join(output_lines)

            return ToolResult(
                title=f"Analysis of {filepath.name}",
                output=output,
                metadata={
                    "filepath": str(filepath),
                    "analysis": analysis
                }
            )

        except Exception as e:
            return ToolResult(
                title=f"Error analyzing {filepath.name}",
                output="",
                error=str(e)
            )
