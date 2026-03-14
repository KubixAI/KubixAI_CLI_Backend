"""Click CLI for kubix_ci."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm

from kubix_ci import __version__
from kubix_ci.config import load_config, create_default_config, save_config
from kubix_ci.runner import run_all
from kubix_ci.reporter import print_results, print_error, print_info
from kubix_ci.exceptions import KUBIXCIError, ConfigError


console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="kubix_ci")
def cli():
    """kubix_ci - Test CUDA kernels across multiple GPUs via SSH.

    \b
    Quick start:
      1. kubix_ci init          # Create config file
      2. kubix_ci test kernel.cu # Run tests

    \b
    Documentation: https://github.com/kubix-ai/kubix_ci
    """
    pass


@cli.command()
@click.option(
    "--path", "-p",
    type=click.Path(),
    default=None,
    help="Path for config file (default: ./kubix_ci.yml)",
)
def init(path):
    """Interactive setup - creates kubix_ci.yml configuration file.

    This will guide you through setting up GPU targets for testing.
    """
    console.print("[bold cyan]kubix_ci init[/bold cyan] - Configure GPU targets\n")

    # Check if config already exists
    if path:
        config_path = Path(path)
    else:
        config_path = Path.cwd() / "kubix_ci.yml"

    if config_path.exists():
        if not Confirm.ask(f"[yellow]{config_path} already exists. Overwrite?[/yellow]"):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Interactive configuration
    targets = []

    console.print("Let's add your first GPU target.\n")

    while True:
        console.print("[bold]Add a target:[/bold]")

        # Provider selection
        provider = Prompt.ask(
            "Provider",
            choices=["ssh", "brev", "runpod"],
            default="ssh",
        )

        if provider == "ssh":
            target = _configure_ssh_target()
        elif provider == "brev":
            target = _configure_brev_target()
        else:
            console.print("[yellow]RunPod support coming soon![/yellow]")
            continue

        if target:
            targets.append(target)
            console.print(f"[green]Added target: {target['name']}[/green]\n")

        if not Confirm.ask("Add another target?", default=False):
            break

    if not targets:
        console.print("[yellow]No targets configured. Creating template file.[/yellow]")
        config_content = create_default_config()
    else:
        # Generate YAML
        config_content = _generate_config_yaml(targets)

    # Save config
    save_config(config_content, config_path)
    console.print(f"\n[green]Created {config_path}[/green]")
    console.print("\n[dim]Edit the file to adjust settings, then run:[/dim]")
    console.print("  [bold]kubix_ci test your_kernel.cu[/bold]")


def _configure_ssh_target() -> dict | None:
    """Interactive SSH target configuration."""
    name = Prompt.ask("  Target name", default="my-gpu-server")
    host = Prompt.ask("  SSH host")
    if not host:
        console.print("[red]Host is required[/red]")
        return None

    user = Prompt.ask("  SSH user", default="ubuntu")
    port = Prompt.ask("  SSH port", default="22")
    key = Prompt.ask("  SSH key path (optional)", default="")
    gpu = Prompt.ask("  GPU name (for display)", default="GPU")

    target = {
        "name": name,
        "provider": "ssh",
        "host": host,
        "user": user,
        "port": int(port),
        "gpu": gpu,
    }

    if key:
        target["key"] = key

    return target


def _configure_brev_target() -> dict | None:
    """Interactive Brev target configuration."""
    name = Prompt.ask("  Target name", default="brev-gpu")
    gpu = Prompt.ask("  GPU type (e.g., H100, A100, T4)", default="T4")

    return {
        "name": name,
        "provider": "brev",
        "gpu": gpu,
    }


def _generate_config_yaml(targets: list[dict]) -> str:
    """Generate YAML config from target list."""
    lines = [
        "# kubix_ci configuration",
        "# Documentation: https://github.com/kubix-ai/kubix_ci",
        "",
        "targets:",
    ]

    for target in targets:
        lines.append(f"  - name: {target['name']}")
        lines.append(f"    provider: {target['provider']}")

        if target["provider"] == "ssh":
            lines.append(f"    host: {target['host']}")
            lines.append(f"    user: {target['user']}")
            lines.append(f"    port: {target.get('port', 22)}")
            if target.get("key"):
                lines.append(f"    key: {target['key']}")
            lines.append(f"    gpu: {target['gpu']}")

        elif target["provider"] == "brev":
            lines.append(f"    gpu: {target['gpu']}")

        lines.append("")

    lines.extend([
        "# Compilation flags",
        "nvcc_flags:",
        '  - "-O3"',
        "",
        "# Timing configuration",
        "warmup_runs: 3",
        "benchmark_runs: 10",
        "timeout: 120",
    ])

    return "\n".join(lines)


@cli.command()
@click.argument("kernel_file", type=click.Path(exists=True))
@click.option(
    "--target", "-t",
    help="Run on specific target (name or GPU type filter)",
)
@click.option(
    "--runs", "-n",
    type=int,
    default=None,
    help="Number of benchmark iterations (default: from config)",
)
@click.option(
    "--warmup", "-w",
    type=int,
    default=None,
    help="Number of warmup iterations (default: from config)",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Timeout per target in seconds (default: from config)",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file (default: searches for kubix_ci.yml)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show verbose output including CUDA versions",
)
def test(kernel_file, target, runs, warmup, timeout, config, verbose):
    """Test a CUDA kernel across configured GPU targets.

    \b
    Examples:
      kubix_ci test kernel.cu                 # Test on all targets
      kubix_ci test kernel.cu --target h100   # Test on H100 targets only
      kubix_ci test kernel.cu -n 20 -v        # 20 runs, verbose output
    """
    try:
        # Load config
        config_path = Path(config) if config else None
        cfg = load_config(config_path)

        # Override config with CLI flags
        if runs is not None:
            cfg.benchmark_runs = runs
        if warmup is not None:
            cfg.warmup_runs = warmup
        if timeout is not None:
            cfg.timeout = timeout

        kernel_path = Path(kernel_file)

        # Show what we're doing
        print_info(f"Testing: {kernel_path.name}")
        target_count = len(cfg.targets)
        if target:
            print_info(f"Filter: {target}")
        print_info(f"Targets: {target_count} configured")
        print_info(f"Runs: {cfg.warmup_runs} warmup + {cfg.benchmark_runs} benchmark")
        console.print()

        # Run tests
        results = run_all(
            kernel_path=kernel_path,
            config=cfg,
            target_filter=target,
            verbose=verbose,
        )

        # Print results
        print_results(results, kernel_path.name, verbose=verbose)

        # Exit with error code if any failures
        if any(r.status != "success" for r in results):
            sys.exit(1)

    except ConfigError as e:
        print_error(str(e))
        if "No kubix_ci.yml found" in str(e):
            console.print("\n[dim]Run 'kubix_ci init' to create a configuration file.[/dim]")
        sys.exit(1)

    except KUBIXCIError as e:
        print_error(str(e))
        sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
def targets():
    """List configured targets."""
    try:
        cfg = load_config()

        console.print("[bold cyan]Configured targets:[/bold cyan]\n")

        for t in cfg.targets:
            console.print(f"  [bold]{t.name}[/bold]")
            console.print(f"    Provider: {t.provider}")
            console.print(f"    GPU: {t.gpu}")

            if hasattr(t, "host"):
                console.print(f"    Host: {t.host}:{getattr(t, 'port', 22)}")
                console.print(f"    User: {t.user}")

            console.print()

    except ConfigError as e:
        print_error(str(e))
        sys.exit(1)


@cli.command()
def check():
    """Check connectivity to all configured targets."""
    try:
        cfg = load_config()

        console.print("[bold cyan]Checking targets...[/bold cyan]\n")

        from kubix_ci.runner import create_provider

        for t in cfg.targets:
            console.print(f"  {t.name}... ", end="")
            try:
                provider = create_provider(t)
                provider.connect()

                # Try a simple command
                stdout, stderr, code = provider.execute("nvidia-smi --query-gpu=name --format=csv,noheader")
                provider.disconnect()

                if code == 0:
                    gpu_name = stdout.strip().split('\n')[0]
                    console.print(f"[green]OK[/green] ({gpu_name})")
                else:
                    console.print("[yellow]Connected but nvidia-smi failed[/yellow]")

            except Exception as e:
                console.print(f"[red]FAILED[/red] ({e})")

    except ConfigError as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    cli()
