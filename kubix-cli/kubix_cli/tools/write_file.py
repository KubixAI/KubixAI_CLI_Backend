"""
Write File Tool - Create or overwrite CUDA files.
"""

from pathlib import Path
from typing import Dict, Any
from .base import Tool, ToolContext, ToolResult
from ..ui.syntax import print_cuda_code
from ..ui.theme import console


class WriteFileTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return """Write content to a file. Creates the file if it doesn't exist, overwrites if it does.
Use this to create new CUDA kernels or modify existing ones.
Always provide complete file contents, not just snippets."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to write (absolute or relative to working directory). Also accepts 'file_path' for compatibility."
                },
                "file_path": {
                    "type": "string",
                    "description": "Alternative parameter name for filepath (for compatibility)"
                },
                "content": {
                    "type": "string",
                    "description": "Complete content to write to the file"
                }
            },
            "required": ["content"]
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Write content to file."""
        # Support both 'filepath' and 'file_path' for compatibility
        filepath = params.get("filepath") or params.get("file_path")
        if not filepath:
            raise ValueError("Missing required parameter: filepath or file_path")

        content = params["content"]

        # Resolve path
        if not Path(filepath).is_absolute():
            filepath = Path(ctx.working_dir) / filepath

        filepath = Path(filepath)

        # Check if file exists
        file_existed = filepath.exists()

        # Check permissions if permission manager is available
        if ctx.permission_manager:
            operation = "update" if file_existed else "create"
            if not ctx.permission_manager.check_edit(str(filepath), operation):
                return ToolResult(
                    title=f"Permission denied: {filepath.name}",
                    output="",
                    error=f"User denied permission to {operation} file: {filepath}"
                )

        try:
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            # Count lines
            lines = content.split('\n')
            line_count = len(lines)

            action = "Updated" if file_existed else "Created"
            is_cuda = filepath.suffix.lower() in ['.cu', '.cuh', '.h', '.cpp', '.c']

            # Don't print preview - let tool_display handle it smoothly
            # (Removed bulky preview box for better UX)

            return ToolResult(
                title=f"{action} {filepath.name}",
                output=f"Successfully {action.lower()} {filepath.name} ({line_count} lines)",
                metadata={
                    "filepath": str(filepath),
                    "lines": line_count,
                    "bytes": len(content),
                    "existed": file_existed,
                    "is_cuda": is_cuda
                }
            )

        except PermissionError:
            return ToolResult(
                title=f"Permission denied: {filepath.name}",
                output="",
                error=f"No permission to write to: {filepath}"
            )
        except Exception as e:
            return ToolResult(
                title=f"Error writing {filepath.name}",
                output="",
                error=str(e)
            )
