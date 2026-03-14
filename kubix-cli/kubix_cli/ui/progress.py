"""
Progress indicators with NVIDIA styling
"""

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.spinner import Spinner
from .theme import console, NVIDIA_GREEN
from contextlib import contextmanager


def show_progress(description: str, total: int):
    """
    Create a progress bar with NVIDIA styling for long operations.

    Args:
        description: Progress description
        total: Total number of steps

    Returns:
        Progress context manager

    Example:
        with show_progress("Compiling kernels", 10) as progress:
            task = progress.add_task(description, total=10)
            for i in range(10):
                # Do work
                progress.update(task, advance=1)
    """
    return Progress(
        SpinnerColumn(spinner_name="dots", style="nvidia"),
        TextColumn("[nvidia]{task.description}[/nvidia]"),
        BarColumn(complete_style="nvidia", finished_style="success"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )


@contextmanager
def show_spinner(description: str):
    """
    Show a spinner for indeterminate operations.

    Args:
        description: Operation description

    Example:
        with show_spinner("Analyzing kernel"):
            # Long operation
            analyze_code()
    """
    from rich.live import Live

    spinner = Spinner("dots", text=f"[nvidia]{description}...[/nvidia]", style="nvidia")

    with Live(spinner, console=console, refresh_per_second=10):
        yield


def show_compilation_progress(files: list[str]):
    """
    Show progress for compiling multiple CUDA files.

    Args:
        files: List of file paths to compile

    Returns:
        Progress context manager with task ID
    """
    progress = show_progress("Compiling CUDA files", len(files))
    task = progress.add_task("[nvidia]Compiling...[/nvidia]", total=len(files))
    return progress, task


def show_analysis_progress():
    """
    Show indeterminate progress for code analysis.

    Returns:
        Spinner context manager
    """
    return show_spinner("Analyzing CUDA code")
