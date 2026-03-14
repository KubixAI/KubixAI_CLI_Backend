"""
NVIDIA-Themed UI Components

Colors:
- NVIDIA Green: #76B900
- Dark Gray: #1E1E1E
- Light Gray: #CCCCCC
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich import box

# NVIDIA Color Palette
NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#1E1E1E"
NVIDIA_GRAY = "#666666"
NVIDIA_LIGHT = "#CCCCCC"
NVIDIA_WHITE = "#FFFFFF"

# Create NVIDIA theme
NVIDIATheme = Theme({
    "info": f"{NVIDIA_GREEN}",
    "success": f"bold {NVIDIA_GREEN}",
    "warning": "bold yellow",
    "error": "bold red",
    "nvidia": f"bold {NVIDIA_GREEN}",
    "nvidia_dim": f"dim {NVIDIA_GREEN}",
    "agent": f"{NVIDIA_GREEN}",
    "tool": f"cyan",
    "command": f"{NVIDIA_GREEN}",
    "session": "bold cyan",
})

# Create console with NVIDIA theme and force UTF-8 encoding
import sys
import os

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    import io
    # Only re-wrap if not already UTF-8
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except (AttributeError, OSError):
            pass  # Already wrapped or can't wrap
    if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        except (AttributeError, OSError):
            pass  # Already wrapped or can't wrap
    os.environ['PYTHONIOENCODING'] = 'utf-8'

console = Console(theme=NVIDIATheme, force_terminal=True)


def themed_panel(
    content: str,
    title: str = "",
    subtitle: str = "",
    border_style: str = "nvidia",
    **kwargs
) -> Panel:
    """
    Create a panel with NVIDIA styling.

    Args:
        content: Panel content
        title: Panel title
        subtitle: Panel subtitle
        border_style: Border style (nvidia, success, warning, error)
        **kwargs: Additional Panel arguments

    Returns:
        Styled Panel
    """
    return Panel(
        content,
        title=f"[bold]{title}[/bold]" if title else None,
        subtitle=subtitle if subtitle else None,
        border_style=border_style,
        box=box.ROUNDED,
        **kwargs
    )


def themed_table(
    title: str = "",
    show_header: bool = True,
    **kwargs
) -> Table:
    """
    Create a table with NVIDIA styling.

    Args:
        title: Table title
        show_header: Show table header
        **kwargs: Additional Table arguments

    Returns:
        Styled Table
    """
    return Table(
        title=f"[nvidia]{title}[/nvidia]" if title else None,
        show_header=show_header,
        box=box.ROUNDED,
        border_style="nvidia",
        header_style="bold nvidia",
        **kwargs
    )


# Global flag to track if logo has been shown
_LOGO_SHOWN = False

def show_logo():
    """Display NVIDIA-themed Kubix logo."""
    # Prevent duplicate display
    global _LOGO_SHOWN
    if _LOGO_SHOWN:
        return
    _LOGO_SHOWN = True

    logo = f"""
[nvidia]    ██╗  ██╗██╗   ██╗██████╗ ██╗██╗  ██╗
    ██║ ██╔╝██║   ██║██╔══██╗██║╚██╗██╔╝
    █████╔╝ ██║   ██║██████╔╝██║ ╚███╔╝
    ██╔═██╗ ██║   ██║██╔══██╗██║ ██╔██╗
    ██║  ██╗╚██████╔╝██████╔╝██║██╔╝ ██╗
    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═╝[/nvidia]
    """
    console.print(logo)
    console.print("[bold nvidia]AI GPU Performance Engineer[/bold nvidia] [nvidia_dim]• Profile → Diagnose → Patch → Verify[/nvidia_dim]")
    console.print(f"[nvidia_dim]⚡ NVIDIA & AMD GPU Kernel Optimization[/nvidia_dim]")
    console.print()
    console.print("[nvidia_dim]🌐 [link=https://kubix.ai]kubix.ai[/link] • 💬 Discord • 🐙 github.com/Kubix-AI[/nvidia_dim]")


def show_agent_switch(from_agent: str, to_agent: str):
    """Show agent switch with NVIDIA styling."""
    console.print(f"\n[nvidia_dim]→ Routing to [nvidia]{to_agent}[/nvidia][/nvidia_dim]\n")


def show_tool_call(tool_name: str, params_str: str = ""):
    """Show tool call with NVIDIA styling."""
    if params_str:
        console.print(f"[nvidia_dim]🛠️  {tool_name}[/nvidia_dim] [dim]({params_str})[/dim]")
    else:
        console.print(f"[nvidia_dim]🛠️  {tool_name}[/nvidia_dim]")


def show_success(message: str):
    """Show success message."""
    console.print(f"\n[success]✓ {message}[/success]\n")


def show_error(message: str):
    """Show error message."""
    console.print(f"\n[bold red]✗ {message}[/bold red]\n")


def show_warning(message: str):
    """Show warning message."""
    console.print(f"\n[warning]⚠ {message}[/warning]\n")


def show_info(message: str):
    """Show info message."""
    console.print(f"\n[info]ℹ {message}[/info]\n")


def show_status_line(agent: str, routing: bool, session: str = None):
    """Show status line at startup."""
    status_parts = [
        f"[nvidia]Agent:[/nvidia] [bold]{agent}[/bold]",
        f"[nvidia]Auto-routing:[/nvidia] {'on' if routing else 'off'}",
    ]
    if session:
        status_parts.append(f"[nvidia]Session:[/nvidia] {session}")

    console.print(f"\n[nvidia_dim]{' • '.join(status_parts)}[/nvidia_dim]")
    console.print(f"[nvidia_dim]Type [nvidia]/help[/nvidia] for commands • Ctrl+C to exit[/nvidia_dim]\n")


def get_input_box(width: int = None) -> tuple:
    """
    Get styled input box borders.

    Returns:
        Tuple of (top_border, bottom_border)
    """
    import shutil
    if width is None:
        width = shutil.get_terminal_size().columns

    box_width = max(40, width - 2)
    top_border = f"[nvidia_dim]┌{'─' * (box_width - 2)}┐[/nvidia_dim]"
    bottom_border = f"[nvidia_dim]└{'─' * (box_width - 2)}┘[/nvidia_dim]"

    return top_border, bottom_border


def show_session_saved(name: str):
    """Show session saved confirmation."""
    panel = themed_panel(
        f"[success]Session saved successfully[/success]\n\n"
        f"[nvidia]Name:[/nvidia] {name}\n"
        f"[nvidia_dim]You can load it later with:[/nvidia_dim] [command]/load {name}[/command]",
        title="💾 Session Saved",
        border_style="success"
    )
    console.print()
    console.print(panel)
    console.print()


def show_session_loaded(name: str, message_count: int):
    """Show session loaded confirmation."""
    panel = themed_panel(
        f"[success]Session loaded successfully[/success]\n\n"
        f"[nvidia]Name:[/nvidia] {name}\n"
        f"[nvidia]Messages:[/nvidia] {message_count} restored",
        title="📂 Session Loaded",
        border_style="success"
    )
    console.print()
    console.print(panel)
    console.print()


def show_welcome_tips():
    """Show welcome tips for new users."""
    tips = [
        "[nvidia]💡 Tip:[/nvidia] Try asking: [command]Optimize this CUDA kernel[/command]",
        "[nvidia]💡 Tip:[/nvidia] Save your work with [command]/save my-session[/command]",
        "[nvidia]💡 Tip:[/nvidia] The system auto-routes to the best expert agent",
    ]

    import random
    console.print(f"\n{random.choice(tips)}\n")
