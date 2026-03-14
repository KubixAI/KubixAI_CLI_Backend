"""
Bash Execution Tool - Run shell commands.
"""

from typing import Dict, Any
import subprocess
from .base import Tool, ToolContext, ToolResult


class BashTool(Tool):
    """Execute shell commands."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return """Execute a shell command in the working directory.
Use for running CUDA benchmarks, git operations, file operations, etc.
Returns stdout, stderr, and exit code."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Optional, defaults to 120.",
                    "default": 120
                }
            },
            "required": ["command"]
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute bash command."""
        command = params["command"]
        timeout = params.get("timeout", 120)

        # Check permissions if permission manager is available
        if ctx.permission_manager:
            if not ctx.permission_manager.check_bash(command):
                return ToolResult(
                    title=f"Permission denied: {command[:50]}",
                    output="",
                    error=f"User denied permission to execute command: {command}"
                )

        try:
            # Run command
            result = subprocess.run(
                command,
                shell=True,
                cwd=ctx.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Combine stdout and stderr
            output_lines = []
            if result.stdout:
                output_lines.append("=== STDOUT ===")
                output_lines.append(result.stdout.rstrip())
            if result.stderr:
                output_lines.append("=== STDERR ===")
                output_lines.append(result.stderr.rstrip())

            output = "\n".join(output_lines) if output_lines else "(no output)"

            # Truncate if too long
            MAX_OUTPUT = 5000
            if len(output) > MAX_OUTPUT:
                output = output[:MAX_OUTPUT] + f"\n\n... (truncated {len(output) - MAX_OUTPUT} characters)"

            success = result.returncode == 0

            if success:
                title = f"Command succeeded: {command[:50]}"
            else:
                title = f"Command failed (exit {result.returncode}): {command[:50]}"

            return ToolResult(
                title=title,
                output=output,
                metadata={
                    "command": command,
                    "exit_code": result.returncode,
                    "success": success
                },
                error=None if success else f"Exit code: {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                title=f"Command timeout: {command[:50]}",
                output="",
                error=f"Command exceeded {timeout} second timeout"
            )
        except Exception as e:
            return ToolResult(
                title=f"Command error: {command[:50]}",
                output="",
                error=str(e)
            )
