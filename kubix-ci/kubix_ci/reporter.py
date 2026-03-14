"""Rich table output formatting for kubix_ci results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from kubix_ci.providers.base import ExecutionResult


console = Console()


def format_time(ms: float | None) -> str:
    """Format time in milliseconds with appropriate precision."""
    if ms is None:
        return "-"
    if ms < 0.001:
        return f"{ms * 1000:.3f}us"
    if ms < 1:
        return f"{ms:.4f}ms"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


def format_compile_time(s: float | None) -> str:
    """Format compile time in seconds."""
    if s is None:
        return "-"
    return f"{s:.1f}s"


def print_results(
    results: list[ExecutionResult],
    kernel_name: str,
    verbose: bool = False,
) -> None:
    """Print formatted results table.

    Args:
        results: List of execution results from all targets
        kernel_name: Name of the kernel file (for display)
        verbose: Show additional details
    """
    console.print()
    console.print(Panel(
        f"[bold cyan]kubix_ci[/bold cyan] results: [bold]{kernel_name}[/bold]",
        style="cyan",
        expand=False,
    ))
    console.print()

    # Create table
    table = Table(box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Target", style="bold cyan", no_wrap=True)
    table.add_column("GPU", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Median", justify="right", style="bold green")
    table.add_column("Min", justify="right", style="dim")
    table.add_column("Max", justify="right", style="dim")
    table.add_column("Compile", justify="right", style="dim")

    if verbose:
        table.add_column("CUDA", style="dim")
        table.add_column("Arch", style="dim")

    # Sort by median time (fastest first), errors at end
    def sort_key(r: ExecutionResult):
        if r.status != "success":
            return (1, float('inf'))
        return (0, r.kernel_time_ms or float('inf'))

    sorted_results = sorted(results, key=sort_key)

    # Track summary stats
    success_count = 0
    error_count = 0

    for r in sorted_results:
        # Status formatting
        if r.status == "success":
            status_str = "[green]PASS[/green]"
            success_count += 1
        elif r.status == "compile_error":
            status_str = "[red]COMPILE[/red]"
            error_count += 1
        elif r.status == "runtime_error":
            status_str = "[red]RUNTIME[/red]"
            error_count += 1
        elif r.status == "timeout":
            status_str = "[yellow]TIMEOUT[/yellow]"
            error_count += 1
        else:
            status_str = f"[red]{r.status.upper()}[/red]"
            error_count += 1

        # GPU name - prefer detected name over config
        gpu_name = r.device_name or r.gpu_name
        if len(gpu_name) > 20:
            gpu_name = gpu_name[:17] + "..."

        row = [
            r.target_name,
            gpu_name,
            status_str,
            format_time(r.kernel_time_ms),
            format_time(r.min_time_ms),
            format_time(r.max_time_ms),
            format_compile_time(r.compile_time_s),
        ]

        if verbose:
            cuda_ver = r.cuda_version or "-"
            arch = r.compute_capability or "-"
            row.extend([cuda_ver, arch])

        table.add_row(*row)

    console.print(table)

    # Summary
    console.print()
    total = len(results)

    if error_count == 0:
        console.print(f"[bold green]All {total} targets passed[/bold green]")
    elif success_count == 0:
        console.print(f"[bold red]All {total} targets failed[/bold red]")
    else:
        console.print(
            f"[bold]{success_count}/{total} passed[/bold], "
            f"[red]{error_count} failed[/red]"
        )

    # Print errors if any
    errors = [r for r in results if r.error and r.status != "success"]
    if errors:
        console.print()
        console.print("[bold red]Errors:[/bold red]")
        for r in errors:
            error_msg = r.error[:200] if r.error else "Unknown error"
            console.print(f"  [cyan]{r.target_name}[/cyan]: {error_msg}")
            if verbose and r.stderr:
                for line in r.stderr.split('\n')[:5]:
                    if line.strip():
                        console.print(f"    [dim]{line}[/dim]")


def print_progress(target_name: str, stage: str) -> None:
    """Print real-time progress update."""
    console.print(f"  [dim]{target_name}[/dim]: {stage}")


def print_connecting(targets: list[str]) -> None:
    """Print connection status."""
    console.print(f"[dim]Connecting to {len(targets)} target(s)...[/dim]")
    for name in targets:
        console.print(f"  [dim]- {name}[/dim]")
    console.print()


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[cyan]{message}[/cyan]")
