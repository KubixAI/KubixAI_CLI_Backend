"""
Interactive User Prompter for Permission Confirmations

Displays permission requests to user and gets their decision.
"""

from typing import Literal, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

console = Console()


UserDecision = Literal["allow", "deny", "always_allow", "always_deny"]


class PermissionPrompter:
    """
    Prompts user for permission decisions with a clean interface.
    """

    @staticmethod
    def prompt_bash(command: str, matched_pattern: Optional[str] = None) -> UserDecision:
        """
        Prompt user to allow/deny a bash command.

        Args:
            command: The bash command to execute
            matched_pattern: Which pattern matched (if any)

        Returns:
            UserDecision
        """
        console.print()

        # Create panel with command info
        info_lines = []
        info_lines.append(f"[bold yellow]⚠️  Permission Required[/bold yellow]\n")
        info_lines.append(f"[bold]Command:[/bold] [cyan]{command}[/cyan]")

        if matched_pattern:
            info_lines.append(f"[dim]Matched pattern: {matched_pattern}[/dim]")

        panel = Panel(
            "\n".join(info_lines),
            title="[bold]Bash Command Permission[/bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print(panel)

        # Show options
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Description")

        table.add_row("[1] Allow", "Execute this command once")
        table.add_row("[2] Deny", "Block this command once")
        table.add_row("[3] Always", "Always allow this pattern")
        table.add_row("[4] Never", "Always deny this pattern")

        console.print(table)

        # Get user choice
        try:
            choice = Prompt.ask(
                "\n[bold]Your decision[/bold]",
                choices=["1", "2", "3", "4", "allow", "deny", "always", "never"],
                default="2"  # Default to deny for safety
            ).lower()

            # Map choice to decision
            if choice in ("1", "allow"):
                return "allow"
            elif choice in ("2", "deny"):
                return "deny"
            elif choice in ("3", "always"):
                console.print("[green]✓ Will always allow this pattern in this session[/green]\n")
                return "always_allow"
            elif choice in ("4", "never"):
                console.print("[red]✗ Will always deny this pattern in this session[/red]\n")
                return "always_deny"
            else:
                return "deny"

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Permission request cancelled, denying operation[/yellow]\n")
            return "deny"

    @staticmethod
    def prompt_edit(filepath: str, operation: str = "edit") -> UserDecision:
        """
        Prompt user to allow/deny file editing.

        Args:
            filepath: Path to file being edited
            operation: Type of operation (edit, write, create)

        Returns:
            UserDecision
        """
        console.print()

        # Create panel
        info_lines = []
        info_lines.append(f"[bold yellow]⚠️  Permission Required[/bold yellow]\n")
        info_lines.append(f"[bold]Operation:[/bold] {operation}")
        info_lines.append(f"[bold]File:[/bold] [cyan]{filepath}[/cyan]")

        panel = Panel(
            "\n".join(info_lines),
            title="[bold]File Edit Permission[/bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print(panel)

        # Show options
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Description")

        table.add_row("[1] Allow", "Allow this edit")
        table.add_row("[2] Deny", "Block this edit")
        table.add_row("[3] Always", "Always allow file edits")
        table.add_row("[4] Never", "Always deny file edits")

        console.print(table)

        # Get user choice
        try:
            choice = Prompt.ask(
                "\n[bold]Your decision[/bold]",
                choices=["1", "2", "3", "4", "allow", "deny", "always", "never"],
                default="1"  # Default to allow for edits (less dangerous)
            ).lower()

            if choice in ("1", "allow"):
                return "allow"
            elif choice in ("2", "deny"):
                return "deny"
            elif choice in ("3", "always"):
                console.print("[green]✓ Will always allow file edits in this session[/green]\n")
                return "always_allow"
            elif choice in ("4", "never"):
                console.print("[red]✗ Will always deny file edits in this session[/red]\n")
                return "always_deny"
            else:
                return "deny"

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Permission request cancelled, denying operation[/yellow]\n")
            return "deny"

    @staticmethod
    def prompt_webfetch(url: str) -> UserDecision:
        """
        Prompt user to allow/deny web fetching.

        Args:
            url: URL to fetch

        Returns:
            UserDecision
        """
        console.print()

        # Create panel
        info_lines = []
        info_lines.append(f"[bold yellow]⚠️  Permission Required[/bold yellow]\n")
        info_lines.append(f"[bold]URL:[/bold] [cyan]{url}[/cyan]")

        panel = Panel(
            "\n".join(info_lines),
            title="[bold]Web Fetch Permission[/bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print(panel)

        # Show options
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Description")

        table.add_row("[1] Allow", "Fetch from this URL")
        table.add_row("[2] Deny", "Block this fetch")
        table.add_row("[3] Always", "Always allow web fetches")
        table.add_row("[4] Never", "Always deny web fetches")

        console.print(table)

        # Get user choice
        try:
            choice = Prompt.ask(
                "\n[bold]Your decision[/bold]",
                choices=["1", "2", "3", "4", "allow", "deny", "always", "never"],
                default="1"  # Default to allow for web fetches
            ).lower()

            if choice in ("1", "allow"):
                return "allow"
            elif choice in ("2", "deny"):
                return "deny"
            elif choice in ("3", "always"):
                console.print("[green]✓ Will always allow web fetches in this session[/green]\n")
                return "always_allow"
            elif choice in ("4", "never"):
                console.print("[red]✗ Will always deny web fetches in this session[/red]\n")
                return "always_deny"
            else:
                return "deny"

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Permission request cancelled, denying operation[/yellow]\n")
            return "deny"

    @staticmethod
    def prompt_generic(
        operation: str,
        details: str,
        default_allow: bool = False
    ) -> UserDecision:
        """
        Generic permission prompt for other operations.

        Args:
            operation: Name of the operation
            details: Details about what's being done
            default_allow: Whether to default to allow or deny

        Returns:
            UserDecision
        """
        console.print()

        # Create panel
        info_lines = []
        info_lines.append(f"[bold yellow]⚠️  Permission Required[/bold yellow]\n")
        info_lines.append(f"[bold]Operation:[/bold] {operation}")
        info_lines.append(f"[bold]Details:[/bold] {details}")

        panel = Panel(
            "\n".join(info_lines),
            title="[bold]Permission Request[/bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print(panel)

        # Show options
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Description")

        table.add_row("[1] Allow", "Allow this operation")
        table.add_row("[2] Deny", "Block this operation")

        console.print(table)

        # Get user choice
        try:
            default = "1" if default_allow else "2"
            choice = Prompt.ask(
                "\n[bold]Your decision[/bold]",
                choices=["1", "2", "allow", "deny"],
                default=default
            ).lower()

            if choice in ("1", "allow"):
                return "allow"
            else:
                return "deny"

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Permission request cancelled, denying operation[/yellow]\n")
            return "deny"
