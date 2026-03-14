"""
Production-ready input handling with beautiful design.

Rock-solid cross-platform input with comprehensive error handling.
Clean, minimal, professional design with NVIDIA green accents.
"""

import sys
import os
import platform
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich import box
from .theme import console

# Try to import prompt_toolkit, but have fallback
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    Completer = object  # Dummy for type hints


class CleanCommandCompleter(Completer):
    """Clean, minimal autocomplete for slash commands."""

    COMMANDS = [
        # Essential Commands Only
        ("/models", "list or switch AI models"),
        ("/setkey", "set or update OpenRouter API key"),
        ("/gpu", "show GPU status"),
        ("/setup", "configure compiler paths"),
        ("/compilers", "show compiler configuration"),
        ("/clear", "clear conversation history"),
        ("/help", "show help menu"),
        ("/quit", "exit Kubix CLI"),
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if text.startswith('/'):
            command_text = text[1:].lower()

            for cmd, desc in self.COMMANDS:
                cmd_without_slash = cmd[1:]

                if cmd_without_slash.startswith(command_text):
                    display = f"{cmd:<12} {desc}"

                    yield Completion(
                        text=cmd_without_slash,
                        start_position=-len(command_text),
                        display=display,
                    )


def _ensure_cursor_visible():
    """Ensure cursor is visible (cross-platform)."""
    try:
        if platform.system() == "Windows":
            # Windows-specific cursor management
            import ctypes

            # Try to make cursor visible
            try:
                class _CursorInfo(ctypes.Structure):
                    _fields_ = [("size", ctypes.c_int),
                              ("visible", ctypes.c_byte)]

                ci = _CursorInfo()
                handle = ctypes.windll.kernel32.GetStdHandle(-11)
                ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
                ci.visible = True
                ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
            except:
                pass

            # Also try ANSI code
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
            except:
                pass
        else:
            # Unix-like systems - use ANSI escape code
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
    except:
        # Silently fail - cursor will remain in current state
        pass


def _get_terminal_width() -> int:
    """Get terminal width safely."""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except:
        return 80  # Default fallback


def _safe_flush():
    """Flush stdout and stderr safely."""
    try:
        sys.stdout.flush()
    except:
        pass
    try:
        sys.stderr.flush()
    except:
        pass


def get_modern_input(show_separator: bool = True) -> str:
    """
    Clean input with autocomplete - simple and reliable.
    Full terminal width, minimal design.

    Args:
        show_separator: Whether to show top separator (default True)

    Returns:
        User input string
    """
    # Ensure cursor is visible
    _ensure_cursor_visible()
    _safe_flush()

    # If prompt_toolkit not available, use fallback
    if not PROMPT_TOOLKIT_AVAILABLE:
        return get_minimal_input()

    try:
        terminal_width = _get_terminal_width()
        separator = "─" * terminal_width

        # Clean style with NVIDIA green
        style = Style.from_dict({
            # Autocomplete menu
            'completion-menu': 'bg:#2a2a2a #ffffff',
            'completion-menu.completion': 'bg:#2a2a2a #ffffff',
            'completion-menu.completion.current': 'bg:#76B900 #000000',

            # Input
            '': '#ffffff',
        })

        # Key bindings for better error handling
        bindings = KeyBindings()

        @bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C gracefully."""
            event.app.exit(exception=KeyboardInterrupt())

        @bindings.add('c-d')
        def _(event):
            """Handle Ctrl+D gracefully."""
            event.app.exit(exception=EOFError())

        completer = CleanCommandCompleter()

        # Top separator (only if requested)
        if show_separator:
            try:
                console.print(f"[dim]{separator}[/dim]")
            except:
                print("-" * terminal_width)

        # Ensure clean terminal state before prompt
        _safe_flush()

        try:
            # Get input with error handling
            result = prompt(
                '  ▸ ',
                completer=completer,
                complete_while_typing=True,
                style=style,
                mouse_support=False,
                key_bindings=bindings,
                vi_mode=False,  # Disable vi mode for simplicity
                enable_history_search=False,  # Avoid history issues
                complete_in_thread=False,  # Avoid threading issues
            )

            result = result.strip()

            # Only print bottom separator if input is not empty
            if result:
                try:
                    console.print(f"[dim]{separator}[/dim]")
                except:
                    print("-" * terminal_width)

            return result

        except (KeyboardInterrupt, EOFError):
            # Handle gracefully
            try:
                console.print(f"[dim]{separator}[/dim]")
            except:
                print("-" * terminal_width)
            raise

    except Exception as e:
        # Fallback to minimal input on any error
        return get_minimal_input()


def get_minimal_input() -> str:
    """
    Ultra-minimal input - just a clean prompt.
    Rock-solid fallback that always works.

    Returns:
        User input string
    """
    # Ensure cursor is visible
    _ensure_cursor_visible()
    _safe_flush()

    try:
        terminal_width = _get_terminal_width()
        separator = "─" * terminal_width

        # Print separator
        try:
            console.print()
            console.print(f"[dim]{separator}[/dim]")
        except:
            print()
            print("-" * terminal_width)

        # Show prompt
        try:
            console.print("  [green]▸[/green] ", end="")
        except:
            sys.stdout.write("  > ")

        _safe_flush()

        # Get input
        try:
            user_input = input()

            # Print bottom separator
            try:
                console.print(f"[dim]{separator}[/dim]")
                console.print()
            except:
                print("-" * terminal_width)
                print()

            return user_input.strip()

        except (EOFError, KeyboardInterrupt):
            # Print separator before raising
            try:
                console.print(f"[dim]{separator}[/dim]")
                console.print()
            except:
                print("-" * terminal_width)
                print()
            raise

    except Exception:
        # Ultimate fallback - just use plain input
        try:
            return input("  > ").strip()
        except:
            return ""


def get_boxed_input() -> str:
    """
    Beautiful boxed input with full borders.
    Uses full terminal width dynamically.

    Returns:
        User input string
    """
    # Ensure cursor is visible
    _ensure_cursor_visible()
    _safe_flush()

    try:
        terminal_width = _get_terminal_width()
        box_width = terminal_width - 4  # Account for margins

        # Print top of box
        try:
            console.print()
            console.print(f"  [green]╭{'─' * box_width}╮[/green]")
            console.print(f"  [green]│[/green] ", end="")
        except:
            print()
            print("  +" + "-" * box_width + "+")
            sys.stdout.write("  | ")

        _safe_flush()

        # Get input
        try:
            user_input = input()

            # Close the box
            try:
                console.print(f"  [green]╰{'─' * box_width}╯[/green]")
                console.print()
            except:
                print("  +" + "-" * box_width + "+")
                print()

            return user_input.strip()

        except (EOFError, KeyboardInterrupt):
            # Close box before raising
            try:
                console.print(f"  [green]╰{'─' * box_width}╯[/green]")
                console.print()
            except:
                print("  +" + "-" * box_width + "+")
                print()
            raise

    except Exception:
        # Fallback to minimal input
        return get_minimal_input()


def get_confirmation(message: str, default: bool = False) -> bool:
    """
    Get yes/no confirmation from user.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True for yes, False for no
    """
    try:
        # Build prompt
        if default:
            prompt_str = f"{message} [Y/n]: "
            valid_yes = ['y', 'yes', '']
            valid_no = ['n', 'no']
        else:
            prompt_str = f"{message} [y/N]: "
            valid_yes = ['y', 'yes']
            valid_no = ['n', 'no', '']

        while True:
            try:
                console.print(prompt_str, end="")
            except:
                sys.stdout.write(prompt_str)

            _safe_flush()

            try:
                response = input().lower().strip()
            except (KeyboardInterrupt, EOFError):
                return False

            if response in valid_yes:
                return True
            elif response in valid_no:
                return False
            else:
                try:
                    console.print("[dim]Please enter 'y' for yes or 'n' for no[/dim]")
                except:
                    print("Please enter 'y' for yes or 'n' for no")

    except Exception:
        return default


def clear_screen():
    """Clear the screen (cross-platform)."""
    try:
        if platform.system() == "Windows":
            os.system('cls')
        else:
            os.system('clear')
    except:
        # Fallback - print many newlines
        print('\n' * 50)


# Export main functions
__all__ = [
    'get_minimal_input',
    'get_modern_input',
    'get_boxed_input',
    'get_confirmation',
    'clear_screen',
]