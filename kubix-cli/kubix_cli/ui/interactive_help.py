"""
Interactive help menu with keyboard navigation.
"""

import sys
from typing import List, Dict, Tuple
from rich.table import Table
from rich.panel import Panel
from rich import box
from .theme import console


class HelpMenu:
    """Interactive help menu with keyboard navigation."""

    COMMANDS = [
        # Agent Control
        ("AGENT CONTROL", [
            ("/optimize, /opt", "Switch to CUDA optimizer agent", "⚡"),
            ("/debug, /dbg", "Switch to debugging agent", "🐛"),
            ("/analyze, /anal", "Switch to code analysis agent", "🔍"),
            ("/general, /gen", "Switch to general purpose agent", "💬"),
            ("/agents", "Show all available agents", "🤖"),
            ("/routing", "Toggle auto-routing on/off", "🔄"),
        ]),

        # Session Management
        ("SESSION MANAGEMENT", [
            ("/save [name]", "Save current conversation", "💾"),
            ("/load <name>", "Load saved session", "📂"),
            ("/sessions", "List all saved sessions", "📋"),
            ("/fork <name>", "Branch session for experiments", "🌿"),
            ("/export <file>", "Export session to file", "📤"),
        ]),

        # System Commands
        ("SYSTEM", [
            ("/help", "Show this help menu", "❓"),
            ("/status", "Show system status", "ℹ️"),
            ("/clear", "Clear conversation history", "🗑️"),
            ("/exit, /quit, /q", "Exit the CLI", "👋"),
        ]),
    ]

    def show(self):
        """Display interactive help menu."""
        console.clear()
        self._show_header()
        self._show_commands()
        self._show_footer()

    def _show_header(self):
        """Show help menu header."""
        header = Panel(
            "[bold cyan]📚 Kubix CLI - Command Reference[/bold cyan]\n\n"
            "[dim]Navigate with arrow keys • Press any key to return[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(header)
        console.print()

    def _show_commands(self):
        """Show all commands organized by category."""
        for category, commands in self.COMMANDS:
            # Category header
            console.print(f"\n[bold cyan]═══ {category} ═══[/bold cyan]\n")

            # Create table for this category
            table = Table(
                show_header=False,
                border_style="dim cyan",
                box=box.SIMPLE,
                padding=(0, 2),
                expand=False
            )

            table.add_column("Icon", style="cyan", width=4)
            table.add_column("Command", style="bold green", width=25)
            table.add_column("Description", style="white", width=50)

            for cmd, desc, icon in commands:
                table.add_row(icon, cmd, desc)

            console.print(table)

    def _show_footer(self):
        """Show help menu footer with tips."""
        console.print()

        tips = Panel(
            "[bold cyan]💡 Quick Tips[/bold cyan]\n\n"
            "  • Commands can be shortened (e.g., [green]/opt[/green] for [green]/optimize[/green])\n"
            "  • Type naturally - the AI routes to the best agent automatically\n"
            "  • Save your work often with [green]/save[/green]\n"
            "  • Use [green]/status[/green] to see current agent and settings",
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(tips)
        console.print()

        # Wait for keypress
        console.print("[dim cyan]Press Enter to continue...[/dim cyan]", end="")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass

    def show_compact(self):
        """Show compact help (for inline display)."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
            title="[bold cyan]📚 Command Reference[/bold cyan]"
        )

        table.add_column("Command", style="bold green", width=20)
        table.add_column("Description", style="white", width=45)

        # Flatten commands
        for category, commands in self.COMMANDS:
            table.add_row(f"[bold]{category}[/bold]", "")
            for cmd, desc, _ in commands:
                table.add_row(cmd, desc)
            table.add_row("", "")  # Spacing

        console.print()
        console.print(table)
        console.print()


class OptionsMenu:
    """Interactive options/settings menu."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def show(self):
        """Display options menu with current settings."""
        console.clear()
        self._show_header()
        self._show_current_settings()
        self._show_options()
        self._show_footer()

    def _show_header(self):
        """Show options menu header."""
        header = Panel(
            "[bold cyan]⚙️  Kubix CLI - Settings & Options[/bold cyan]\n\n"
            "[dim]Current configuration and available options[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(header)
        console.print()

    def _show_current_settings(self):
        """Show current settings."""
        agent = self.orchestrator.get_current_agent()

        settings = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
            title="[bold cyan]Current Settings[/bold cyan]"
        )

        settings.add_column("Setting", style="cyan", width=25)
        settings.add_column("Value", style="white", width=40)

        settings.add_row("Active Agent", f"[bold]{agent.display_name}[/bold]")
        settings.add_row("Model", agent.model)
        settings.add_row("Auto-routing", "Enabled" if self.orchestrator.auto_routing else "Disabled")
        settings.add_row("Working Directory", str(self.orchestrator.working_dir))
        settings.add_row("Conversation Length", f"{len(agent.conversation)} messages")

        console.print(settings)
        console.print()

    def _show_options(self):
        """Show available options."""
        console.print("[bold cyan]═══ AVAILABLE OPTIONS ═══[/bold cyan]\n")

        options = [
            ("Switch Agent", "Use [green]/optimize[/green], [green]/debug[/green], [green]/analyze[/green], or [green]/general[/green]"),
            ("Toggle Routing", "Use [green]/routing[/green] to enable/disable auto-routing"),
            ("Save Session", "Use [green]/save [name][/green] to preserve your work"),
            ("Clear History", "Use [green]/clear[/green] to start fresh"),
            ("View Agents", "Use [green]/agents[/green] to see all available agents"),
        ]

        for title, description in options:
            console.print(f"  [bold cyan]•[/bold cyan] [bold]{title}[/bold]: {description}")

        console.print()

    def _show_footer(self):
        """Show options menu footer."""
        console.print("[dim cyan]Press Enter to continue...[/dim cyan]", end="")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass


def show_interactive_help():
    """Show interactive help menu."""
    menu = HelpMenu()
    menu.show()


def show_compact_help():
    """Show compact help (inline)."""
    menu = HelpMenu()
    menu.show_compact()


def show_options_menu(orchestrator):
    """Show interactive options menu."""
    menu = OptionsMenu(orchestrator)
    menu.show()
