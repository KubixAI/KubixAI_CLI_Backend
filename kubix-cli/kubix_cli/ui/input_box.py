"""
Clean, minimal input box with expandable support
"""

import shutil
from rich.console import Console
from rich.prompt import Prompt
from .theme import console, NVIDIA_GREEN


def get_terminal_width() -> int:
    """Get current terminal width."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def create_input_box(width: int = None) -> tuple:
    """
    Create clean input box borders.

    Args:
        width: Box width (defaults to terminal width - 4)

    Returns:
        Tuple of (top_border, bottom_border, inner_width)
    """
    if width is None:
        terminal_width = get_terminal_width()
        width = max(60, terminal_width - 4)  # Leave margin

    inner_width = width - 2

    top = f"[nvidia_dim]╭{'─' * inner_width}╮[/nvidia_dim]"
    bottom = f"[nvidia_dim]╰{'─' * inner_width}╯[/nvidia_dim]"

    return top, bottom, inner_width


def get_multiline_input() -> str:
    """
    Get user input with complete box visible.

    Shows full box BEFORE getting input so user sees both borders.

    Returns:
        User input string
    """
    import sys

    top, bottom, inner_width = create_input_box()

    # Print complete box structure first
    console.print(f"\n{top}")
    console.print("[nvidia_dim]│[/nvidia_dim]")
    console.print(f"{bottom}\n")

    # Move cursor up using sys.stdout (bypasses Rich's processing)
    sys.stdout.write("\033[3A")  # Move up 3 lines
    sys.stdout.write("\033[2C")  # Move right 2 chars (past "│ ")
    sys.stdout.flush()

    try:
        # Get input with simple input() - cursor is already positioned
        user_input = input().strip()

        # Move down to clear cursor position
        sys.stdout.write("\n\n")
        sys.stdout.flush()

        return user_input

    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        raise


def show_simple_prompt() -> str:
    """
    Show minimal prompt with box.

    Returns:
        User input
    """
    try:
        return get_multiline_input()
    except (EOFError, KeyboardInterrupt):
        raise
