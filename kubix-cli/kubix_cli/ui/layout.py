"""
Enhanced Layout Components for Professional CLI Interface
"""

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich import box
from .theme import console, NVIDIA_GREEN, NVIDIA_DARK, NVIDIA_GRAY


def create_header(agent_name: str, routing: bool, session: str = None) -> Panel:
    """
    Create a professional header panel.

    Args:
        agent_name: Current agent name
        routing: Auto-routing enabled
        session: Current session name

    Returns:
        Styled header panel
    """
    # Status info
    status_parts = []
    status_parts.append(f"[nvidia]●[/nvidia] [bold]{agent_name}[/bold]")
    status_parts.append(f"[nvidia]⚡[/nvidia] Routing: {'ON' if routing else 'OFF'}")
    if session:
        status_parts.append(f"[nvidia]💾[/nvidia] {session}")

    status_line = "  ".join(status_parts)

    panel = Panel(
        status_line,
        border_style="nvidia",
        box=box.ROUNDED,
        padding=(0, 2)
    )

    return panel


def create_input_section() -> str:
    """
    Create visual separator for input section.

    Returns:
        Formatted input prompt
    """
    return "\n[nvidia_dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/nvidia_dim]\n"


def create_response_header(agent_name: str, tool_used: bool = False) -> Panel:
    """
    Create header for AI response section.

    Args:
        agent_name: Agent responding
        tool_used: Whether agent used tools

    Returns:
        Response header panel
    """
    indicator = "🛠️" if tool_used else "💬"
    content = f"{indicator} [nvidia]{agent_name}[/nvidia]"

    panel = Panel(
        content,
        border_style="nvidia_dim",
        box=box.ROUNDED,
        padding=(0, 1)
    )

    return panel


def create_tool_call_panel(tool_name: str, params: str = "") -> Panel:
    """
    Create styled panel for tool calls.

    Args:
        tool_name: Name of tool being called
        params: Tool parameters summary

    Returns:
        Tool call panel
    """
    if params:
        content = f"[nvidia]🔧 {tool_name}[/nvidia]\n[nvidia_dim]{params}[/nvidia_dim]"
    else:
        content = f"[nvidia]🔧 {tool_name}[/nvidia]"

    panel = Panel(
        content,
        border_style="nvidia",
        box=box.ROUNDED,
        padding=(0, 1),
        title="[nvidia_dim]Tool[/nvidia_dim]",
        title_align="left"
    )

    return panel


def create_section_divider(text: str = None) -> str:
    """
    Create a section divider with optional text.

    Args:
        text: Optional divider text

    Returns:
        Formatted divider
    """
    if text:
        return f"\n[nvidia_dim]─── {text} {'─' * (50 - len(text))}[/nvidia_dim]\n"
    else:
        return "\n[nvidia_dim]{'─' * 60}[/nvidia_dim]\n"


def create_status_panel(agent: str, model: str, routing: bool, working_dir: str) -> Panel:
    """
    Create comprehensive status panel.

    Args:
        agent: Current agent name
        model: Current model
        routing: Auto-routing status
        working_dir: Working directory

    Returns:
        Status panel
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="nvidia", justify="right", width=15)
    table.add_column(style="white")

    table.add_row("Agent:", f"[bold]{agent}[/bold]")
    table.add_row("Model:", model)
    table.add_row("Auto-routing:", "[success]Enabled[/success]" if routing else "[nvidia_dim]Disabled[/nvidia_dim]")
    table.add_row("Directory:", f"[nvidia_dim]{working_dir}[/nvidia_dim]")

    panel = Panel(
        table,
        title="[nvidia]⚙ System Status[/nvidia]",
        border_style="nvidia",
        box=box.ROUNDED,
        padding=(1, 2)
    )

    return panel


def create_welcome_panel() -> Panel:
    """
    Create welcome panel with quick tips.

    Returns:
        Welcome panel
    """
    content = """[nvidia]Welcome to Kubix CLI![/nvidia]

[white]Quick Tips:[/white]
  [nvidia]▪[/nvidia] Type naturally - the system routes to the best expert
  [nvidia]▪[/nvidia] Use [command]/help[/command] to see all commands
  [nvidia]▪[/nvidia] Use [command]/save[/command] to preserve your session
  [nvidia]▪[/nvidia] Press [command]Ctrl+C[/command] to exit anytime

[nvidia_dim]Ready to optimize your CUDA code![/nvidia_dim]"""

    panel = Panel(
        content,
        title="[nvidia]🎮 CUDA AI Assistant[/nvidia]",
        subtitle="[nvidia_dim]Powered by Multi-Agent System[/nvidia_dim]",
        border_style="nvidia",
        box=box.DOUBLE,
        padding=(1, 2)
    )

    return panel


def create_agent_table(agents: dict, current_agent) -> Table:
    """
    Create beautiful table of available agents.

    Args:
        agents: Dictionary of agent instances
        current_agent: Current active agent

    Returns:
        Styled agents table
    """
    table = Table(
        show_header=True,
        header_style="bold nvidia",
        border_style="nvidia",
        box=box.ROUNDED,
        padding=(0, 1),
        title="[nvidia]🤖 Available Agents[/nvidia]"
    )

    table.add_column("●", style="success", width=3, justify="center")
    table.add_column("Agent", style="nvidia", width=18)
    table.add_column("Specialty", style="white", width=35)
    table.add_column("Model", style="nvidia_dim", width=25)

    for name, agent in agents.items():
        active = "●" if agent == current_agent else ""
        specialty = agent.description[:35] + "..." if len(agent.description) > 35 else agent.description

        table.add_row(
            active,
            agent.display_name,
            specialty,
            agent.model
        )

    return table


def create_command_help_table() -> Table:
    """
    Create beautiful command reference table.

    Returns:
        Command help table
    """
    table = Table(
        show_header=True,
        header_style="bold nvidia",
        border_style="nvidia",
        box=box.ROUNDED,
        padding=(0, 2),
        title="[nvidia]📚 Command Reference[/nvidia]"
    )

    table.add_column("Command", style="nvidia", width=20)
    table.add_column("Description", style="white", width=45)

    # Agent commands
    table.add_row("[bold]AGENT CONTROL[/bold]", "")
    table.add_row("/optimize", "Switch to CUDA optimizer agent")
    table.add_row("/debug", "Switch to debugging agent")
    table.add_row("/analyze", "Switch to code analysis agent")
    table.add_row("/general", "Switch to general purpose agent")
    table.add_row("/agents", "Show all available agents")

    # Session commands
    table.add_row("", "")
    table.add_row("[bold]SESSION MANAGEMENT[/bold]", "")
    table.add_row("/save [name]", "Save current conversation")
    table.add_row("/load <name>", "Load saved session")
    table.add_row("/sessions", "List all saved sessions")
    table.add_row("/fork <name>", "Branch session for experiments")
    table.add_row("/export <file>", "Export session (md/json/html/txt)")

    # System commands
    table.add_row("", "")
    table.add_row("[bold]SYSTEM[/bold]", "")
    table.add_row("/status", "Show system status")
    table.add_row("/routing", "Toggle auto-routing")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/help", "Show this help")
    table.add_row("/exit", "Exit the CLI")

    return table


def show_startup_screen(agent_name: str, routing: bool):
    """
    Show beautiful startup screen.

    Args:
        agent_name: Starting agent name
        routing: Auto-routing status
    """
    console.clear()
    console.print()

    # Welcome panel
    console.print(create_welcome_panel())
    console.print()

    # Status
    status_line = f"[nvidia_dim]Active Agent:[/nvidia_dim] [bold]{agent_name}[/bold]  [nvidia_dim]│[/nvidia_dim]  [nvidia_dim]Auto-routing:[/nvidia_dim] {'[success]ON[/success]' if routing else '[nvidia_dim]OFF[/nvidia_dim]'}"
    console.print(status_line, justify="center")
    console.print()


def show_thinking_indicator():
    """Show thinking indicator."""
    console.print("[nvidia_dim]⠿ Thinking...[/nvidia_dim]", end="\r")


def clear_thinking_indicator():
    """Clear thinking indicator."""
    console.print(" " * 30, end="\r")
