"""
List Files Tool - List CUDA files in directory.
"""

from pathlib import Path
from typing import Dict, Any
from .base import Tool, ToolContext, ToolResult


class ListFilesTool(Tool):
    """List CUDA files in the current directory."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return """List all CUDA-related files in the working directory.
Shows .cu, .cuh, .h, and .cpp files with their sizes.
Use this to discover what files are available."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.cu', '*.cuh'). Optional, defaults to all CUDA files.",
                    "default": "*"
                }
            },
            "required": []
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """List files matching pattern."""
        pattern = params.get("pattern", "*")

        working_dir = Path(ctx.working_dir)

        # Define CUDA-related extensions
        cuda_extensions = [".cu", ".cuh", ".h", ".hpp", ".cpp", ".c"]

        # Find files
        files = []

        if pattern == "*":
            # List all CUDA files
            for ext in cuda_extensions:
                files.extend(working_dir.glob(f"*{ext}"))
        else:
            # Use custom pattern
            files.extend(working_dir.glob(pattern))

        # Sort by name
        files = sorted(files, key=lambda f: f.name)

        if not files:
            return ToolResult(
                title="No files found",
                output=f"No files matching '{pattern}' in {working_dir}",
                metadata={"count": 0}
            )

        # Format output
        output_lines = []
        output_lines.append(f"=== FILES IN {working_dir.name}/ ===\n")

        for f in files:
            size = f.stat().st_size
            size_str = self._format_size(size)
            output_lines.append(f"  {f.name:<40} {size_str:>10}")

        output_lines.append(f"\nTotal: {len(files)} files")

        return ToolResult(
            title=f"Found {len(files)} files",
            output="\n".join(output_lines),
            metadata={
                "count": len(files),
                "files": [str(f) for f in files],
                "directory": str(working_dir)
            }
        )

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
