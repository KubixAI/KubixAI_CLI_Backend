"""
Interactive Setup Wizard for Compiler Paths

Guides users through configuring compiler paths with auto-detection.
"""

from pathlib import Path
from typing import Optional, Dict
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from ..ui.theme import console, show_success, show_warning, show_error
from ..utils.detection import ToolchainDetector, ToolInfo
from ..config.manager import ConfigManager
from ..config.schema import CompilerPaths


def run_setup_wizard(config_manager: ConfigManager) -> bool:
    """
    Run interactive setup wizard for compiler paths.

    Args:
        config_manager: Configuration manager instance

    Returns:
        True if setup completed, False if cancelled
    """
    console.clear()
    console.print("\n[nvidia]🔧 COMPILER SETUP WIZARD[/nvidia]\n")
    console.print("[dim]Configure compiler paths for CUDA development[/dim]\n")

    # Auto-detect all compilers
    console.print("[nvidia]▸[/nvidia] Auto-detecting compilers...\n")
    detector = ToolchainDetector()
    tools = detector.get_all_tools()

    # Show detection results
    _show_detection_results(tools)

    # Ask if user wants to configure paths
    console.print()
    choice = Prompt.ask(
        "[nvidia]Configure compiler paths?[/nvidia]",
        choices=["yes", "no", "cancel"],
        default="yes"
    )

    if choice == "cancel":
        show_warning("Setup cancelled")
        return False

    if choice == "no":
        show_success("Using auto-detected paths")
        return True

    # Interactive path configuration
    console.print("\n[nvidia]CONFIGURE PATHS[/nvidia]")
    console.print("[dim]Press Enter to use auto-detected path, or enter custom path[/dim]\n")

    paths = {}

    # NVCC
    paths['nvcc'] = _configure_path(
        "NVCC (CUDA Compiler)",
        tools['nvcc'],
        "nvcc.exe" if detector.platform == "Windows" else "nvcc"
    )

    # C++ Compiler
    paths['cl'] = _configure_path(
        "C++ Compiler (cl.exe/g++)",
        tools['cpp_compiler'],
        "cl.exe" if detector.platform == "Windows" else "g++"
    )

    # NCU (optional)
    paths['ncu'] = _configure_path(
        "NCU (Nsight Compute)",
        tools['ncu'],
        "ncu.exe" if detector.platform == "Windows" else "ncu",
        optional=True
    )

    # NSYS (optional)
    paths['nsys'] = _configure_path(
        "NSYS (Nsight Systems)",
        tools['nsys'],
        "nsys.exe" if detector.platform == "Windows" else "nsys",
        optional=True
    )

    # Confirm and save
    console.print("\n[nvidia]SUMMARY[/nvidia]\n")
    _show_path_summary(paths)

    console.print()
    save_choice = Prompt.ask(
        "[nvidia]Save these paths to global config?[/nvidia]",
        choices=["yes", "no"],
        default="yes"
    )

    if save_choice == "yes":
        _save_compiler_paths(config_manager, paths)
        show_success("Compiler paths saved successfully!")
        console.print("[dim]Paths saved to:[/dim]", config_manager.global_config_file)
        return True
    else:
        show_warning("Paths not saved")
        return False


def _show_detection_results(tools: Dict[str, ToolInfo]):
    """Show auto-detection results in a table."""
    table = Table(title="Auto-Detection Results", box=box.ROUNDED, border_style="nvidia")
    table.add_column("Tool", style="bold", width=20)
    table.add_column("Status", width=12)
    table.add_column("Path / Error", style="dim")

    for name, info in tools.items():
        if name == 'nvprof':
            continue  # Skip deprecated tool

        display_name = {
            'nvcc': 'NVCC',
            'cpp_compiler': 'C++ Compiler',
            'ncu': 'NCU',
            'nsys': 'NSYS'
        }.get(name, name.upper())

        if info.available:
            status = "[green]✓ Found[/green]"
            path_info = f"{info.path}"
            if info.version:
                path_info += f"\n[dim]Version: {info.version}[/dim]"
        else:
            status = "[red]✗ Not Found[/red]"
            path_info = f"[yellow]{info.error}[/yellow]" if info.error else "Not found"

        table.add_row(display_name, status, path_info)

    console.print(table)


def _configure_path(
    name: str,
    tool_info: ToolInfo,
    executable_name: str,
    optional: bool = False
) -> Optional[str]:
    """
    Configure a single compiler path interactively.

    Args:
        name: Display name of the tool
        tool_info: Auto-detected tool information
        executable_name: Name of the executable to look for
        optional: Whether this tool is optional

    Returns:
        Configured path or None
    """
    # Show current status
    if tool_info.available:
        console.print(f"[green]✓[/green] [bold]{name}[/bold]")
        console.print(f"  [dim]Auto-detected: {tool_info.path}[/dim]")
        default_path = tool_info.path
    else:
        console.print(f"[yellow]![/yellow] [bold]{name}[/bold]")
        console.print(f"  [dim]Not auto-detected[/dim]")
        if tool_info.error:
            console.print(f"  [dim]Tip: {tool_info.error}[/dim]")
        default_path = ""

    # Get user input
    if optional:
        prompt_text = f"  Path (optional, press Enter to skip)"
    else:
        prompt_text = f"  Path"

    user_input = Prompt.ask(
        prompt_text,
        default=default_path if default_path else ""
    ).strip()

    # Validate path
    if user_input:
        path = Path(user_input)
        if path.exists():
            console.print(f"  [green]✓ Valid path[/green]\n")
            return str(path)
        else:
            console.print(f"  [red]✗ Path does not exist[/red]")
            # Ask if they want to retry
            retry = Prompt.ask(
                "  Try again?",
                choices=["yes", "no"],
                default="no"
            )
            if retry == "yes":
                return _configure_path(name, tool_info, executable_name, optional)
            console.print()
            return None
    elif tool_info.available:
        # Use auto-detected path
        console.print(f"  [dim]Using auto-detected path[/dim]\n")
        return tool_info.path
    else:
        # No path provided and not auto-detected
        if not optional:
            console.print(f"  [yellow]⚠ No path configured for required tool[/yellow]\n")
        else:
            console.print(f"  [dim]Skipped[/dim]\n")
        return None


def _show_path_summary(paths: Dict[str, Optional[str]]):
    """Show summary of configured paths."""
    table = Table(box=box.SIMPLE, show_header=False, border_style="nvidia")
    table.add_column("Tool", style="bold nvidia", width=20)
    table.add_column("Path", style="dim")

    for key, path in paths.items():
        display_name = {
            'nvcc': 'NVCC',
            'cl': 'C++ Compiler',
            'ncu': 'NCU',
            'nsys': 'NSYS'
        }.get(key, key.upper())

        if path:
            table.add_row(display_name, path)
        else:
            table.add_row(display_name, "[dim]Not configured[/dim]")

    console.print(table)


def _save_compiler_paths(config_manager: ConfigManager, paths: Dict[str, Optional[str]]):
    """Save compiler paths to global config."""
    # Load current config
    config = config_manager.get()

    # Update compiler paths
    config.cuda.compiler_paths.nvcc = paths.get('nvcc')
    config.cuda.compiler_paths.cl = paths.get('cl')
    config.cuda.compiler_paths.ncu = paths.get('ncu')
    config.cuda.compiler_paths.nsys = paths.get('nsys')

    # Save to global config
    config_manager.save_global(config)


def show_compiler_status(config_manager: ConfigManager):
    """
    Show current compiler configuration status.

    Args:
        config_manager: Configuration manager instance
    """
    console.print("\n[nvidia]COMPILER CONFIGURATION[/nvidia]\n")

    # Load config
    config = config_manager.get()

    # Get configured paths
    compiler_paths = {}
    if config.cuda.compiler_paths:
        if 'nvcc' in config.cuda.compiler_paths and config.cuda.compiler_paths['nvcc']:
            compiler_paths['nvcc'] = config.cuda.compiler_paths.nvcc
        if 'cl' in config.cuda.compiler_paths and config.cuda.compiler_paths['cl']:
            compiler_paths['cl'] = config.cuda.compiler_paths.cl
        if 'ncu' in config.cuda.compiler_paths and config.cuda.compiler_paths['ncu']:
            compiler_paths['ncu'] = config.cuda.compiler_paths.ncu
        if 'nsys' in config.cuda.compiler_paths and config.cuda.compiler_paths['nsys']:
            compiler_paths['nsys'] = config.cuda.compiler_paths.nsys

    # Detect with configured paths
    detector = ToolchainDetector(compiler_paths)
    tools = detector.get_all_tools()

    # Show status
    table = Table(box=box.ROUNDED, border_style="nvidia")
    table.add_column("Tool", style="bold", width=18)
    table.add_column("Status", width=12)
    table.add_column("Path", style="dim")
    table.add_column("Source", width=12)

    for name, info in tools.items():
        if name == 'nvprof':
            continue

        display_name = {
            'nvcc': 'NVCC',
            'cpp_compiler': 'C++ Compiler',
            'ncu': 'NCU',
            'nsys': 'NSYS'
        }.get(name, name.upper())

        if info.available:
            status = "[green]✓ Found[/green]"
            path_display = info.path or "N/A"

            # Determine source
            config_key = {'nvcc': 'nvcc', 'cpp_compiler': 'cl', 'ncu': 'ncu', 'nsys': 'nsys'}.get(name)
            if config_key and config_key in compiler_paths and compiler_paths[config_key] == info.path:
                source = "[cyan]Config[/cyan]"
            else:
                source = "[dim]Auto[/dim]"
        else:
            status = "[red]✗ Missing[/red]"
            path_display = "Not found"
            source = "-"

        table.add_row(display_name, status, path_display, source)

    console.print(table)
    console.print()
    console.print("[dim]Run[/dim] [green]/setup[/green] [dim]to configure compiler paths[/dim]\n")
