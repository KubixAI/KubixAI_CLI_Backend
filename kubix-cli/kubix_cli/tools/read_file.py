"""
Read File Tool - Read CUDA source files.
"""

from pathlib import Path
from typing import Dict, Any
from .base import Tool, ToolContext, ToolResult
from ..ui.syntax import highlight_cuda_code
from ..ui.theme import console


class ReadFileTool(Tool):
    """Read a CUDA source file (.cu, .cuh, .h, .cpp)."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return """Read the contents of a file.
Supports reading CUDA files (.cu, .cuh), C/C++ headers, and other text files.
Returns the file with line numbers for easy reference.
Use offset and limit parameters for large files."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to working directory). Also accepts 'file_path' for compatibility."
                },
                "file_path": {
                    "type": "string",
                    "description": "Alternative parameter name for filepath (for compatibility)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Optional.",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional, defaults to 2000.",
                    "default": 2000
                }
            },
            "required": []
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Read file contents with line numbers."""
        # Support both 'filepath' and 'file_path' for compatibility
        filepath = params.get("filepath") or params.get("file_path")
        if not filepath:
            raise ValueError("Missing required parameter: filepath or file_path")
        offset = params.get("offset", 0)
        limit = params.get("limit", 2000)

        # Resolve path
        if not Path(filepath).is_absolute():
            filepath = Path(ctx.working_dir) / filepath

        filepath = Path(filepath)

        # Check if file exists
        if not filepath.exists():
            return ToolResult(
                title=f"File not found: {filepath.name}",
                output="",
                error=f"File does not exist: {filepath}"
            )

        # Check if it's a file
        if not filepath.is_file():
            return ToolResult(
                title=f"Not a file: {filepath.name}",
                output="",
                error=f"Path is not a file: {filepath}"
            )

        # Read file
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply offset and limit
            start = offset
            end = min(offset + limit, total_lines)
            selected_lines = lines[start:end]

            # Check if this is a CUDA file for syntax highlighting
            is_cuda = filepath.suffix.lower() in ['.cu', '.cuh', '.h', '.cpp', '.c']

            if is_cuda:
                # Show only first 15 lines in terminal with syntax highlighting
                preview_lines = selected_lines[:15]
                code_preview = "".join(preview_lines)

                # Display with syntax highlighting
                console.print(f"\n[nvidia]📄 {filepath.name} (lines {start + 1}-{end})[/nvidia]")
                syntax = highlight_cuda_code(code_preview, line_numbers=True, start_line=start + 1)
                console.print(syntax)

                # Show truncation message if there are more lines
                if len(selected_lines) > 15:
                    remaining = len(selected_lines) - 15
                    console.print(f"[dim]... ({remaining} more lines not shown)[/dim]\n")

                # For the AI, provide plain text with line numbers (ALL lines)
                formatted_lines = []
                for i, line in enumerate(selected_lines, start=start + 1):
                    formatted_lines.append(f"{i:5d}→{line.rstrip()}")
                output = "\n".join(formatted_lines)
            else:
                # Show only first 15 lines for non-CUDA files
                preview_lines = selected_lines[:15]

                # Format preview with line numbers
                formatted_lines = []
                for i, line in enumerate(preview_lines, start=start + 1):
                    formatted_lines.append(f"{i:5d}→{line.rstrip()}")

                # Display preview
                console.print(f"\n[nvidia]📄 {filepath.name} (lines {start + 1}-{end})[/nvidia]")
                for line in formatted_lines:
                    console.print(line)

                # Show truncation message if there are more lines
                if len(selected_lines) > 15:
                    remaining = len(selected_lines) - 15
                    console.print(f"[dim]... ({remaining} more lines not shown)[/dim]\n")

                # For the AI, provide ALL lines
                formatted_lines = []
                for i, line in enumerate(selected_lines, start=start + 1):
                    formatted_lines.append(f"{i:5d}→{line.rstrip()}")
                output = "\n".join(formatted_lines)

            # Add note if file has more lines
            if end < total_lines:
                output += f"\n\n(File has {total_lines - end} more lines. Use offset={end} to continue reading)"

            return ToolResult(
                title=f"Read {filepath.name} (lines {start + 1}-{end})",
                output=output,
                metadata={
                    "filepath": str(filepath),
                    "total_lines": total_lines,
                    "lines_read": len(selected_lines),
                    "offset": start,
                    "has_more": end < total_lines,
                    "is_cuda": is_cuda
                }
            )

        except UnicodeDecodeError:
            return ToolResult(
                title=f"Cannot read {filepath.name}",
                output="",
                error="File appears to be binary or uses an unsupported encoding"
            )
        except Exception as e:
            return ToolResult(
                title=f"Error reading {filepath.name}",
                output="",
                error=str(e)
            )
