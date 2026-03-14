"""
Smart expandable input box - handles paste like Claude Code
"""

import sys
from rich.console import Console
from .theme import console, NVIDIA_GREEN


def get_smart_input() -> str:
    """
    Smart input that expands for pasted content.

    Features:
    - Shows clean input prompt
    - Detects and expands for pasted content
    - Shows preview of pasted text
    - Press Enter to send

    Returns:
        User input string
    """

    # Show clean prompt
    console.print()
    console.print("[nvidia_dim]╭─ Input ─────────────────────────────────────────────╮[/nvidia_dim]")
    console.print("[nvidia_dim]│[/nvidia_dim] [nvidia]▶[/nvidia] ", end="")

    try:
        # Get input - handles paste naturally
        user_input = input()

        # Check if it's a paste (contains newlines or very long)
        lines = user_input.split('\n') if '\n' in user_input else [user_input]
        is_paste = len(lines) > 1 or len(user_input) > 100

        if is_paste:
            # Show expanded box with content preview
            console.print("[nvidia_dim]│[/nvidia_dim]")
            console.print("[nvidia_dim]│ 📋 Pasted content:[/nvidia_dim]")

            # Show preview (first 5 lines)
            preview_lines = lines[:5]
            for line in preview_lines:
                display_line = line[:70] + "..." if len(line) > 70 else line
                console.print(f"[nvidia_dim]│[/nvidia_dim]   {display_line}")

            if len(lines) > 5:
                console.print(f"[nvidia_dim]│   ... and {len(lines) - 5} more lines[/nvidia_dim]")

            console.print("[nvidia_dim]│[/nvidia_dim]")
            console.print(f"[nvidia_dim]│ Total: {len(user_input)} chars, {len(lines)} lines[/nvidia_dim]")

        console.print("[nvidia_dim]╰─────────────────────────────────────────────────────╯[/nvidia_dim]")

        return user_input.strip()

    except (EOFError, KeyboardInterrupt):
        console.print("[nvidia_dim]│[/nvidia_dim]")
        console.print("[nvidia_dim]╰─────────────────────────────────────────────────────╯[/nvidia_dim]")
        raise


def get_expandable_input() -> str:
    """
    Expandable input box that shows content as you type/paste.

    Like Claude Code - expands naturally for large input.
    Properly handles multi-line paste.

    Returns:
        User input string
    """
    import sys
    import select

    console.print()

    # Start with compact prompt
    console.print("[nvidia]▶[/nvidia] ", end="")
    sys.stdout.flush()

    try:
        # Read all available input (handles paste properly)
        lines = []

        # Read first line
        first_line = sys.stdin.readline()
        if not first_line:
            return ""

        lines.append(first_line.rstrip('\n\r'))

        # Check if there's more input available (pasted content)
        # Use a small timeout to detect paste
        import time
        time.sleep(0.05)  # 50ms delay to catch pasted lines

        # Read any additional lines that were pasted
        while True:
            # Check if input is available without blocking
            if sys.platform == 'win32':
                # Windows: use msvcrt
                import msvcrt
                if msvcrt.kbhit():
                    line = sys.stdin.readline()
                    if line:
                        lines.append(line.rstrip('\n\r'))
                    else:
                        break
                else:
                    break
            else:
                # Unix: use select
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if line:
                        lines.append(line.rstrip('\n\r'))
                    else:
                        break
                else:
                    break

        # Join all lines
        user_input = '\n'.join(lines) if len(lines) > 1 else lines[0]

        # If pasted or long, show what was received
        if len(lines) > 1 or len(user_input) > 80:
            # Show box around pasted content
            width = min(max(len(line) for line in lines), 120)
            width = max(width, 40)

            console.print()
            console.print(f"[nvidia_dim]╭{'─' * (width + 2)}╮[/nvidia_dim]")

            for line in lines[:10]:  # Show first 10 lines
                display = line[:width].ljust(width)
                console.print(f"[nvidia_dim]│[/nvidia_dim] {display} [nvidia_dim]│[/nvidia_dim]")

            if len(lines) > 10:
                more = f"... {len(lines) - 10} more lines ...".ljust(width)
                console.print(f"[nvidia_dim]│[/nvidia_dim] {more} [nvidia_dim]│[/nvidia_dim]")

            console.print(f"[nvidia_dim]╰{'─' * (width + 2)}╯[/nvidia_dim]")
            console.print(f"[nvidia_dim]✓ Received {len(user_input)} characters, {len(lines)} lines[/nvidia_dim]")

        console.print()

        return user_input.strip()

    except (EOFError, KeyboardInterrupt):
        console.print()
        raise
