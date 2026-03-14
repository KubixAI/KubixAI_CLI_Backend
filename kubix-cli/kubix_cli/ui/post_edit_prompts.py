"""
Post-edit action prompts for agentic AI workflows.

After editing files, intelligently prompt user for next actions:
- CUDA files: compile, profile, benchmark, analyze
- Python files: run, test, lint
- Other files: relevant actions
"""

import os
import platform
from pathlib import Path
from typing import Optional, List, Dict, Callable
from .theme import console
from ..utils.detection import detect_nsys, detect_nvprof, detect_ncu


class PostEditAction:
    """Represents a post-edit action the user can take."""

    def __init__(self, key: str, label: str, description: str, command: str):
        """
        Initialize action.

        Args:
            key: Single letter key to trigger action
            label: Short label (e.g., "Compile")
            description: What this action does
            command: Command or function to execute
        """
        self.key = key
        self.label = label
        self.description = description
        self.command = command


def get_cuda_actions(filepath: str) -> List[PostEditAction]:
    """Get actions for CUDA files."""
    filename = os.path.basename(filepath)
    stem = Path(filepath).stem

    return [
        PostEditAction(
            key="a",
            label="Analyze",
            description="Analyze code for optimizations",
            command="analyze_cuda"  # Use analyze_cuda tool
        ),
        PostEditAction(
            key="p",
            label="Profile",
            description="Profile with actionable insights",
            command="profile_cuda"  # Use new profiling tool with insights
        ),
        PostEditAction(
            key="b",
            label="Benchmark",
            description="Benchmark with performance metrics",
            command="benchmark_cuda"  # Use new benchmarking tool
        ),
        PostEditAction(
            key="g",
            label="GPU Status",
            description="Check GPU status and health",
            command="gpu_status"  # GPU analyzer
        ),
        PostEditAction(
            key="s",
            label="Skip",
            description="Continue without running",
            command=None
        ),
    ]


def get_python_actions(filepath: str) -> List[PostEditAction]:
    """Get actions for Python files."""
    filename = os.path.basename(filepath)

    return [
        PostEditAction(
            key="r",
            label="Run",
            description=f"Run {filename}",
            command=f"python {filepath}"
        ),
        PostEditAction(
            key="t",
            label="Test",
            description="Run tests",
            command="pytest"
        ),
        PostEditAction(
            key="l",
            label="Lint",
            description="Check code style",
            command=f"pylint {filepath}"
        ),
        PostEditAction(
            key="s",
            label="Skip",
            description="Continue without running",
            command=None
        ),
    ]


def get_general_actions(filepath: str) -> List[PostEditAction]:
    """Get actions for general files."""
    return [
        PostEditAction(
            key="s",
            label="Skip",
            description="Continue",
            command=None
        ),
    ]


def get_actions_for_file(filepath: str) -> List[PostEditAction]:
    """
    Get appropriate actions based on file type.

    Args:
        filepath: Path to the edited file

    Returns:
        List of available actions
    """
    ext = Path(filepath).suffix.lower()

    if ext in ['.cu', '.cuh']:
        return get_cuda_actions(filepath)
    elif ext == '.py':
        return get_python_actions(filepath)
    else:
        return get_general_actions(filepath)


def show_post_edit_prompt(filepath: str, auto_mode: bool = False) -> Optional[PostEditAction]:
    """
    Show clean post-edit action prompt.

    Args:
        filepath: Path to the edited file
        auto_mode: If True, skip prompt and return None

    Returns:
        Selected action or None
    """
    # Skip if not a code file we care about
    ext = Path(filepath).suffix.lower()
    if ext not in ['.cu', '.cuh', '.py', '.cpp', '.c', '.h']:
        return None

    # Skip in auto mode
    if auto_mode:
        return None

    # Get available actions
    actions = get_actions_for_file(filepath)

    # If only "skip" action, don't prompt
    if len(actions) == 1 and actions[0].key == 's':
        return None

    # Show clean prompt
    console.print()
    console.print(f"[dim]File saved:[/dim] [green]{os.path.basename(filepath)}[/green]")
    console.print()
    console.print("[dim]Next action?[/dim]")
    console.print()

    # Show options in clean format
    for action in actions:
        console.print(f"  [green]{action.key}[/green]  {action.label:<12} [dim]{action.description}[/dim]")

    console.print()
    console.print("[dim]Choose:[/dim] ", end="")

    import sys
    sys.stdout.flush()

    # Get single keypress - USE INPUT() for reliability
    try:
        # Clear any buffered input first
        if sys.platform == 'win32':
            import msvcrt
            # Flush input buffer
            while msvcrt.kbhit():
                msvcrt.getch()

        # Use simple input() for maximum reliability
        choice = input().strip().lower()

        # Take only first character if multiple entered
        if len(choice) > 0:
            choice = choice[0]
        else:
            choice = 's'  # Default to skip if empty

        console.print()

        # Find matching action
        for action in actions:
            if action.key == choice:
                return action

        # Invalid choice - default to skip
        if choice != 's':
            console.print(f"[yellow]'{choice}' not recognized, skipping[/yellow]")
            console.print()
        return None

    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print("[dim]Skipped[/dim]")
        console.print()
        return None


def prompt_and_execute(
    filepath: str,
    tool_executor: Callable,
    auto_mode: bool = False
) -> bool:
    """
    Show prompt and execute selected action.

    Args:
        filepath: Path to edited file
        tool_executor: Function to execute tools (takes tool_name, params)
        auto_mode: Skip prompting if True

    Returns:
        True if action was executed, False otherwise
    """
    action = show_post_edit_prompt(filepath, auto_mode)

    if not action or not action.command:
        return False

    # Execute action based on command type
    if action.command == "analyze_cuda":
        # Use analyze_cuda tool
        console.print(f"[dim]→ Analyzing {os.path.basename(filepath)}...[/dim]")
        console.print()
        try:
            tool_executor("analyze_cuda", {"filepath": filepath})
        except Exception as e:
            console.print(f"[red]Error during analysis: {str(e)[:200]}[/red]")
            console.print()

    elif action.command == "profile_cuda":
        # Profile existing executable
        stem = Path(filepath).stem
        console.print(f"[dim]→ Profiling {stem}...[/dim]")
        console.print()

        try:
            # The executable should be in the same directory with .exe extension on Windows
            if platform.system() == "Windows":
                executable = f"{stem}.exe"
            else:
                executable = stem

            # Execute profiling
            tool_executor("profile_cuda", {"executable": executable})
        except Exception as e:
            console.print(f"[red]Error during profiling: {str(e)[:200]}[/red]")
            console.print()

    elif action.command == "benchmark_cuda":
        # Benchmark existing executable
        stem = Path(filepath).stem
        console.print(f"[dim]→ Benchmarking {stem}...[/dim]")
        console.print()

        try:
            # The executable should be in the same directory with .exe extension on Windows
            if platform.system() == "Windows":
                executable = f"{stem}.exe"
            else:
                executable = stem

            # Execute benchmarking
            tool_executor("benchmark_cuda", {"executable": executable, "iterations": 10})
        except Exception as e:
            console.print(f"[red]Error during benchmarking: {str(e)[:200]}[/red]")
            console.print()

    elif action.command == "gpu_status":
        # Use gpu_status tool
        console.print(f"[dim]→ Checking GPU status and health...[/dim]")
        console.print()
        tool_executor("gpu_status", {"detailed": True})


    else:
        # Execute as bash command (fallback)
        console.print(f"[dim]→ Running: {action.label}[/dim]")
        console.print()
        tool_executor("bash", {"command": action.command})

    return True


# Simple yes/no prompt helper
def prompt_yes_no(question: str, default: bool = True) -> bool:
    """
    Simple yes/no prompt.

    Args:
        question: Question to ask
        default: Default answer

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    console.print(f"[dim]{question}[/dim] [{default_str}]: ", end="")

    try:
        import sys
        response = input().strip().lower()

        if not response:
            return default

        return response in ['y', 'yes']

    except (KeyboardInterrupt, EOFError):
        console.print()
        return False
