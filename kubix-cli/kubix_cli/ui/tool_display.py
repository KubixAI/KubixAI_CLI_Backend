"""
Beautiful animated tool execution display with ASCII animations.

Shows tool execution with:
- ASCII spinner animations
- Context-aware messages (analyzing, generating, compiling, etc.)
- Minimal, colorful output
- Code truncation with syntax highlighting
- NO HARDCODED VALUES - fully configurable
- Post-edit action prompts for agentic workflows
"""

import sys
import time
import threading
from typing import Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box
from .theme import console  # Use themed console with NVIDIATheme
from .post_edit_prompts import show_post_edit_prompt


@dataclass
class DisplayConfig:
    """Configuration for display formatting - NO HARDCODED VALUES."""
    max_code_lines: int = 10
    max_output_chars: int = 200
    max_bash_lines: int = 5
    max_file_list_items: int = 10
    max_insight_items: int = 5
    spinner_fps: float = 0.08  # seconds between frames (faster = smoother)
    enable_post_edit_prompts: bool = True  # Show action prompts after file edits
    auto_mode: bool = False  # Skip prompts in auto mode


# Default config - can be overridden
DEFAULT_CONFIG = DisplayConfig()


# ASCII spinner frames
SPINNERS = {
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "line": ["-", "\\", "|", "/"],
    "simple": [".", "..", "...", ""],
    "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
}

# Tool-specific styles
TOOL_STYLES = {
    "read_file": {
        "emoji": "📖",
        "color": "cyan",
        "verb": "Reading",
        "spinner": "dots"
    },
    "write_file": {
        "emoji": "✍️",
        "color": "green",
        "verb": "Writing",
        "spinner": "line"
    },
    "compile_cuda": {
        "emoji": "⚙️",
        "color": "yellow",
        "verb": "Compiling",
        "spinner": "line"
    },
    "analyze_cuda": {
        "emoji": "🔍",
        "color": "magenta",
        "verb": "Analyzing",
        "spinner": "dots"
    },
    "bash": {
        "emoji": "💻",
        "color": "blue",
        "verb": "Executing",
        "spinner": "dots"
    },
    "list_files": {
        "emoji": "📂",
        "color": "cyan",
        "verb": "Scanning",
        "spinner": "dots"
    }
}


class SpinnerThread(threading.Thread):
    """Background thread that shows ASCII spinner."""

    def __init__(self, message: str, spinner_frames: list, color: str, fps: float = 0.1):
        super().__init__(daemon=True)
        self.message = message
        self.frames = spinner_frames
        self.color = color
        self.fps = fps
        self.stop_flag = threading.Event()

    def run(self):
        """Run spinner animation - using raw stdout for real-time updates."""
        i = 0
        while not self.stop_flag.is_set():
            frame = self.frames[i % len(self.frames)]
            # Use raw stdout for unbuffered, real-time output
            # ANSI color codes for proper coloring
            color_code = self._get_ansi_color(self.color)
            sys.stdout.write(f"\r{color_code}{frame} {self.message}\033[0m")
            sys.stdout.flush()
            i += 1
            time.sleep(self.fps)

    def stop(self):
        """Stop the spinner."""
        self.stop_flag.set()
        # Clear the line using raw stdout
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    def _get_ansi_color(self, color: str) -> str:
        """Get ANSI color code for the given color name."""
        colors = {
            "green": "\033[32m",
            "cyan": "\033[36m",
            "yellow": "\033[33m",
            "magenta": "\033[35m",
            "blue": "\033[34m",
            "red": "\033[31m",
            "white": "\033[37m",
        }
        return colors.get(color, "\033[37m")


class ToolExecutionDisplay:
    """Animated display for tool execution with ASCII art."""

    def __init__(self, config: DisplayConfig = None):
        self.console = console
        self.config = config or DEFAULT_CONFIG

    def execute_with_animation(
        self,
        tool_name: str,
        params: Dict[str, Any],
        executor: callable
    ) -> Any:
        """
        Execute a tool with animated display.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            executor: Function that returns ToolResult

        Returns:
            Tool result (ToolResult object or string)
        """
        # Get tool style
        style = TOOL_STYLES.get(tool_name, {
            "emoji": "🛠️",
            "color": "white",
            "verb": "Running",
            "spinner": "dots"
        })

        # Create display message
        display_msg = self._create_display_message(tool_name, params, style)

        # Get spinner frames
        spinner_frames = SPINNERS.get(style["spinner"], SPINNERS["dots"])

        # Start spinner
        spinner = SpinnerThread(display_msg, spinner_frames, style["color"], self.config.spinner_fps)
        spinner.start()

        # Execute tool with progress tracking
        result = None
        error = None
        try:
            # Show what we're doing based on tool type
            if tool_name == "write_file":
                # Show writing stages
                result = executor(tool_name, params)
            elif tool_name == "compile_cuda":
                # Show compile stages
                result = executor(tool_name, params)
            elif tool_name == "analyze_cuda":
                # Show analysis stages
                result = executor(tool_name, params)
            else:
                result = executor(tool_name, params)
        except Exception as e:
            error = e
        finally:
            spinner.stop()
            spinner.join(timeout=0.5)

        # Show result
        if error:
            self._show_error(tool_name, error, style)
            return f"Error: {error}"
        else:
            # Check if result is ToolResult object
            if hasattr(result, 'title') and hasattr(result, 'output'):
                # It's a ToolResult - use structured data
                self._show_tool_result(tool_name, result, params, style)
                result_str = str(result)
            else:
                # It's already a string - show as-is
                self._show_success_string(tool_name, str(result), params, style)
                result_str = str(result)

            # Post-edit prompts for file writes
            if (tool_name == "write_file" and
                self.config.enable_post_edit_prompts and
                not error):
                self._handle_post_edit_prompt(params, executor)

            return result_str

    def _create_display_message(self, tool_name: str, params: Dict[str, Any], style: Dict) -> str:
        """Create minimal display message."""
        verb = style["verb"]

        if tool_name == "read_file" and "file_path" in params:
            import os
            filename = os.path.basename(params["file_path"])
            return f"{verb} {filename}"

        elif tool_name == "write_file" and "filepath" in params:
            import os
            filename = os.path.basename(params["filepath"])
            return f"{verb} {filename}"

        elif tool_name == "compile_cuda":
            return f"{verb} CUDA kernel"

        elif tool_name == "analyze_cuda":
            return f"{verb} code"

        elif tool_name == "bash" and "command" in params:
            cmd = params["command"]
            if len(cmd) > 40:
                cmd = cmd[:37] + "..."
            return f"{verb}: {cmd}"

        elif tool_name == "list_files":
            return f"{verb} directory"

        else:
            return f"{verb} {tool_name}"

    def _show_tool_result(self, tool_name: str, result, params: Dict, style: Dict):
        """Show result using ToolResult structure (production-ready)."""
        emoji = style["emoji"]
        color = style["color"]

        # Get structured data
        output = result.output
        metadata = result.metadata

        # Format based on tool type
        if tool_name == "read_file":
            self._show_file_content(output, params.get("file_path", ""), color)

        elif tool_name == "write_file":
            import os
            # Get filepath from metadata first (most reliable), then params
            filepath = metadata.get("filepath") or params.get("filepath") or params.get("file_path", "")
            filename = os.path.basename(filepath) if filepath else "file"

            # Get metadata if available
            lines = metadata.get("lines", len(params.get("content", "").split("\n")))
            bytes_written = metadata.get("bytes", 0)
            existed = metadata.get("existed", False)

            action = "Updated" if existed else "Created"
            size_str = self._format_size(bytes_written) if bytes_written else ""

            if size_str:
                self.console.print(f"[{color}]{emoji} {action} [bold]{filename}[/bold] [dim]({lines} lines, {size_str})[/dim][/{color}]")
            else:
                self.console.print(f"[{color}]{emoji} {action} [bold]{filename}[/bold] [dim]({lines} lines)[/dim][/{color}]")

        elif tool_name == "compile_cuda":
            if "success" in output.lower() or "compiled" in output.lower():
                self.console.print(f"[{color}]{emoji} Compiled successfully ✓[/{color}]")
            else:
                self._show_compile_errors(output, color)

        elif tool_name == "analyze_cuda":
            self._show_analysis(output, color)

        elif tool_name == "bash":
            self._show_bash_output(output, params.get("command", ""), color)

        elif tool_name == "list_files":
            # Use metadata if available
            file_count = metadata.get("count", 0)
            files_data = metadata.get("files", [])
            if files_data:
                self._show_file_list_structured(files_data, file_count, color)
            else:
                # Fallback to parsing output
                self._show_file_list_from_text(output, color)

        else:
            self._show_truncated(output, color, emoji)

    def _show_success_string(self, tool_name: str, result: str, params: Dict, style: Dict):
        """Fallback for string results (less ideal but works)."""
        emoji = style["emoji"]
        color = style["color"]

        if tool_name == "read_file":
            self._show_file_content(result, params.get("file_path", ""), color)

        elif tool_name == "write_file":
            import os
            # Get filepath from params (fallback method)
            filepath = params.get("filepath") or params.get("file_path", "")
            filename = os.path.basename(filepath) if filepath else "file"
            lines = len(params.get("content", "").split("\n"))
            content = params.get("content", "")
            size_str = self._format_size(len(content.encode('utf-8')))

            self.console.print(f"[{color}]{emoji} Saved [bold]{filename}[/bold] [dim]({lines} lines, {size_str})[/dim][/{color}]")

        elif tool_name == "compile_cuda":
            if "success" in result.lower() or "compiled" in result.lower():
                self.console.print(f"[{color}]{emoji} Compiled successfully ✓[/{color}]")
            else:
                self._show_compile_errors(result, color)

        elif tool_name == "analyze_cuda":
            self._show_analysis(result, color)

        elif tool_name == "bash":
            self._show_bash_output(result, params.get("command", ""), color)

        elif tool_name == "list_files":
            self._show_file_list_from_text(result, color)

        else:
            self._show_truncated(result, color, emoji)

    def _show_error(self, tool_name: str, error: Exception, style: Dict):
        """Show error in a nice way."""
        error_msg = str(error)
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        self.console.print(f"[red]❌ Error: {error_msg}[/red]")

    def _show_file_content(self, content: str, filepath: str, color: str):
        """Show minimal file read confirmation."""
        import os

        filename = os.path.basename(filepath) if filepath else "file"
        lines = content.split("\n")
        total_lines = len(lines)

        # Get file size
        try:
            if filepath and os.path.isfile(filepath):
                size_bytes = os.path.getsize(filepath)
                size_str = f", {self._format_size(size_bytes)}"
            else:
                size_str = ""
        except:
            size_str = ""

        # Minimal one-line output
        self.console.print(f"[{color}]📖 Read [bold]{filename}[/bold] [dim]({total_lines} lines{size_str})[/dim] ✓[/{color}]")

    def _show_compile_errors(self, result: str, color: str):
        """Show compilation errors concisely."""
        lines = result.split("\n")
        error_lines = [l for l in lines if "error" in l.lower() or "warning" in l.lower()]

        if error_lines:
            shown = error_lines[:3]
            remaining = len(error_lines) - len(shown)

            self.console.print(f"[yellow]⚠️  Compilation issues:[/yellow]")
            for err in shown:
                self.console.print(f"[dim]  • {err.strip()}[/dim]")

            if remaining > 0:
                self.console.print(f"[dim]  ... and {remaining} more[/dim]")
        else:
            self.console.print(f"[{color}]⚙️  Compilation output received[/{color}]")

    def _show_analysis(self, result: str, color: str):
        """Show minimal analysis summary."""
        lines = result.split("\n")

        # Count insights
        keywords = [
            "bottleneck", "optimization", "performance", "register",
            "memory", "shared", "global", "coalesced", "occupancy",
            "warning", "improvement", "issue", "detected"
        ]

        insight_count = sum(1 for line in lines if any(kw in line.lower() for kw in keywords))

        if insight_count > 0:
            self.console.print(f"[{color}]🔍 Analysis complete [dim]({insight_count} insights)[/dim] ✓[/{color}]")
        else:
            self.console.print(f"[{color}]🔍 Analysis complete ✓[/{color}]")

    def _show_bash_output(self, result: str, command: str, color: str):
        """Show minimal bash output."""
        lines = [l for l in result.strip().split("\n") if l.strip()]

        if len(command) > 40:
            cmd_display = command[:37] + "..."
        else:
            cmd_display = command

        # Just show command executed with line count
        self.console.print(f"[{color}]💻 [bold]{cmd_display}[/bold] [dim]({len(lines)} lines)[/dim] ✓[/{color}]")

    def _show_file_list_structured(self, files: list, count: int, color: str):
        """Show minimal file list summary."""
        import os

        if not files:
            self.console.print(f"[{color}]📂 No files found[/{color}]")
            return

        # Get file types
        extensions = set()
        for filepath in files[:10]:  # Sample first 10
            ext = os.path.splitext(filepath)[1]
            if ext:
                extensions.add(ext)

        ext_str = ", ".join(sorted(extensions)[:3]) if extensions else "various"

        # Minimal one-line output
        self.console.print(f"[{color}]📂 Found [bold]{count}[/bold] files [dim]({ext_str})[/dim] ✓[/{color}]")

    def _show_file_list_from_text(self, output: str, color: str):
        """Fallback: minimal file list summary from text."""
        import os
        lines = output.split("\n")

        # Extract lines that look like files
        file_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or len(stripped) < 3 or stripped.startswith("="):
                continue
            # Likely a file line if it has common extensions
            if any(ext in stripped for ext in [".cu", ".c", ".h", ".py", ".cpp", ".cuh"]):
                file_lines.append(stripped)

        if not file_lines:
            self.console.print(f"[{color}]📂 No files found[/{color}]")
            return

        # Get file types from parsed lines
        extensions = set()
        for line in file_lines[:10]:
            parts = line.split()
            if parts:
                ext = os.path.splitext(parts[0])[1]
                if ext:
                    extensions.add(ext)

        ext_str = ", ".join(sorted(extensions)[:3]) if extensions else "various"
        count = len(file_lines)

        # Minimal one-line output
        self.console.print(f"[{color}]📂 Found [bold]{count}[/bold] files [dim]({ext_str})[/dim] ✓[/{color}]")

    def _get_file_color(self, ext: str) -> str:
        """Get color for file based on extension."""
        ext = ext.lower()
        color_map = {
            '.cu': 'green',      # CUDA files - green (NVIDIA theme)
            '.cuh': 'green',     # CUDA headers
            '.cpp': 'cyan',      # C++ files
            '.c': 'cyan',        # C files
            '.h': 'blue',        # Headers
            '.py': 'yellow',     # Python
            '.json': 'magenta',  # JSON
            '.md': 'white',      # Markdown
            '.txt': 'white',     # Text
        }
        return color_map.get(ext, 'white')  # Default to white

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _show_truncated(self, result: str, color: str, emoji: str):
        """Show truncated generic result."""
        max_chars = self.config.max_output_chars

        if len(result) <= max_chars:
            self.console.print(f"[{color}]{emoji} {result}[/{color}]")
        else:
            truncated = result[:max_chars] + "..."
            self.console.print(f"[{color}]{emoji} {truncated}[/{color}]")
            self.console.print(f"[dim]  (output truncated)[/dim]")

    def _handle_post_edit_prompt(self, params: Dict[str, Any], executor: Callable):
        """
        Handle post-edit action prompt.

        Shows prompt after file edit and executes selected action.

        Args:
            params: write_file parameters (contains filepath)
            executor: Tool executor function
        """
        # Get filepath from params
        filepath = params.get('filepath') or params.get('file_path')
        if not filepath:
            return

        # Show prompt and get action
        action = show_post_edit_prompt(filepath, self.config.auto_mode)

        if not action or not action.command:
            return

        # Execute selected action based on command type
        self.console.print(f"[dim]→ {action.label}...[/dim]")
        self.console.print()

        try:
            from pathlib import Path
            import platform

            if action.command == "compile_cuda":
                executor("compile_cuda", {"file_path": filepath})

            elif action.command == "analyze_cuda":
                executor("analyze_cuda", {"file_path": filepath})

            elif action.command == "profile_cuda":
                # First compile, then profile
                stem = Path(filepath).stem
                self.console.print(f"[yellow]Compiling {Path(filepath).name} first...[/yellow]")
                self.console.print()
                executor("compile_cuda", {"file_path": filepath})

                # Determine executable name
                if platform.system() == "Windows":
                    executable = f"{stem}.exe"
                else:
                    executable = stem

                self.console.print(f"[dim]→ Profiling {stem}...[/dim]")
                self.console.print()
                executor("profile_cuda", {"executable": executable})

            elif action.command == "benchmark_cuda":
                # First compile, then benchmark
                stem = Path(filepath).stem
                self.console.print(f"[yellow]Compiling {Path(filepath).name} first...[/yellow]")
                self.console.print()
                executor("compile_cuda", {"file_path": filepath})

                # Determine executable name
                if platform.system() == "Windows":
                    executable = f"{stem}.exe"
                else:
                    executable = stem

                self.console.print(f"[dim]→ Benchmarking {stem}...[/dim]")
                self.console.print()
                executor("benchmark_cuda", {"executable": executable, "iterations": 10})

            elif action.command == "gpu_status":
                executor("gpu_status", {"detailed": True})

            elif action.command:
                # Execute as bash command (fallback)
                executor("bash", {"command": action.command})

        except Exception as e:
            self.console.print(f"[yellow]⚠ Action failed: {e}[/yellow]")
            self.console.print()


# Singleton instance
_display = ToolExecutionDisplay()


def execute_tool_with_display(
    tool_name: str,
    params: Dict[str, Any],
    executor: callable,
    config: DisplayConfig = None
) -> Any:
    """
    Execute a tool with beautiful animated display.

    Args:
        tool_name: Name of the tool
        params: Tool parameters
        executor: Function to execute
        config: Optional display configuration

    Returns:
        Tool result
    """
    if config:
        display = ToolExecutionDisplay(config)
        return display.execute_with_animation(tool_name, params, executor)
    else:
        return _display.execute_with_animation(tool_name, params, executor)


def set_display_config(config: DisplayConfig):
    """Set global display configuration."""
    global _display
    _display = ToolExecutionDisplay(config)
