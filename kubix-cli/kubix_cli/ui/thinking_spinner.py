"""
Rock-solid thinking/generating spinner for AI responses.

Production-ready with comprehensive error handling and cross-platform support.
FIXED: User input no longer disappears when transitioning to static status.
"""

import sys
import os
import time
import threading
import platform
from typing import Optional
from contextlib import contextmanager
from rich.console import Console

console = Console()


class ThinkingSpinner:
    """
    Production-ready ASCII spinner for AI thinking/generating.

    Features:
    - Thread-safe operations
    - Comprehensive error handling
    - Cross-platform compatibility (Windows/Linux/macOS)
    - Proper terminal state management
    - Graceful degradation on unsupported terminals
    - FIXED: Preserves user input during status transitions
    """

    # Cross-platform spinner frames
    SPINNERS = {
        # Unicode spinners (for terminals that support it)
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "dots2": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
        "arc": ["◜", "◠", "◝", "◞", "◡", "◟"],
        # ASCII fallback (always works)
        "line": ["-", "\\", "|", "/"],
        "simple": [".", "..", "...", "....", "....."],
    }

    def __init__(self, message: str = "Thinking", spinner_type: str = "auto"):
        """
        Initialize spinner with cross-platform support.

        Args:
            message: Message to show (e.g., "Thinking", "Generating")
            spinner_type: Type of spinner ("auto" detects best for platform)
        """
        self.message = message
        self.running = False
        self.thread = None
        self.content_below = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._original_cursor_state = None
        self._spinner_started = False  # Track if spinner has started displaying

        # Auto-detect best spinner for platform
        if spinner_type == "auto":
            spinner_type = self._detect_best_spinner()

        self.frames = self.SPINNERS.get(spinner_type, self.SPINNERS["line"])

        # Terminal capabilities
        self._supports_ansi = self._check_ansi_support()
        self._terminal_width = self._get_terminal_width()

    def _detect_best_spinner(self) -> str:
        """Detect best spinner type for current platform."""
        system = platform.system()

        # Windows console Unicode support is inconsistent
        if system == "Windows":
            # Check if we're in Windows Terminal or VS Code
            if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") == "vscode":
                return "dots"
            # PowerShell/CMD might not support Unicode well
            return "line"

        # Unix-like systems generally support Unicode
        return "dots"

    def _check_ansi_support(self) -> bool:
        """Check if terminal supports ANSI escape codes."""
        system = platform.system()

        # Windows 10+ supports ANSI in most terminals
        if system == "Windows":
            # Enable ANSI on Windows if possible
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                # Fallback for older Windows
                return False

        # Unix-like systems support ANSI
        return True

    def _get_terminal_width(self) -> int:
        """Get terminal width safely."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 80  # Default fallback

    def _hide_cursor(self):
        """Hide cursor with error handling."""
        if not self._supports_ansi:
            return

        try:
            if platform.system() == "Windows":
                # Windows-specific cursor hiding
                import ctypes

                class _CursorInfo(ctypes.Structure):
                    _fields_ = [("size", ctypes.c_int),
                              ("visible", ctypes.c_byte)]

                ci = _CursorInfo()
                handle = ctypes.windll.kernel32.GetStdHandle(-11)
                ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
                self._original_cursor_state = ci.visible
                ci.visible = False
                ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
            else:
                # Unix-like systems
                sys.stdout.write("\033[?25l")
                sys.stdout.flush()
        except:
            # Silently fail - cursor will remain visible
            pass

    def _show_cursor(self):
        """Show cursor with error handling."""
        if not self._supports_ansi:
            return

        try:
            if platform.system() == "Windows":
                # Restore Windows cursor
                if self._original_cursor_state is not None:
                    import ctypes

                    class _CursorInfo(ctypes.Structure):
                        _fields_ = [("size", ctypes.c_int),
                                  ("visible", ctypes.c_byte)]

                    ci = _CursorInfo()
                    handle = ctypes.windll.kernel32.GetStdHandle(-11)
                    ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
                    ci.visible = self._original_cursor_state
                    ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
            else:
                # Unix-like systems
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
        except:
            # Silently fail
            pass

    def _clear_line(self, width: Optional[int] = None):
        """Clear current line with platform-specific handling."""
        if width is None:
            width = self._terminal_width

        try:
            # Move to start of line and clear
            sys.stdout.write('\r' + ' ' * min(width, 200) + '\r')
            sys.stdout.flush()
        except:
            # Fallback - just print newline
            sys.stdout.write('\n')
            sys.stdout.flush()

    def start(self):
        """Start the spinner with comprehensive error handling."""
        with self._lock:
            if self.running:
                return

            self.running = True
            self._stop_event.clear()
            self._spinner_started = False

            # Hide cursor
            self._hide_cursor()

            # Start animation thread
            self.thread = threading.Thread(
                target=self._animate_safe,
                daemon=True,
                name="SpinnerThread"
            )
            self.thread.start()

    def update_message(self, new_message: str, content_below: bool = False):
        """
        Update spinner message thread-safely.
        FIXED: No longer clears user input when transitioning to static.

        Args:
            new_message: New message to display
            content_below: If True, content is printed below (makes spinner static)
        """
        with self._lock:
            old_message = self.message
            self.message = new_message

            try:
                if content_below and not self.content_below:
                    # Transition to static status
                    self.running = False

                    # FIXED: Don't clear line - just overwrite with static status
                    # This preserves user input that might be above
                    if self._spinner_started:
                        # Only clear if spinner was actually displaying
                        sys.stdout.write('\r')  # Return to start of line

                    if self._supports_ansi:
                        # Print static status with color
                        sys.stdout.write(f"\033[32m  ▸ \033[0m\033[2m{self.message}...\033[0m\n")
                    else:
                        # Plain text fallback
                        sys.stdout.write(f"  > {self.message}...\n")

                    sys.stdout.flush()
                    self.content_below = True

                elif self.content_below:
                    # STABILITY FIX: Don't try to update status line while streaming content
                    # This cursor manipulation can interfere with content display
                    # Just keep the message updated internally without visual changes
                    pass

                    sys.stdout.flush()
            except Exception:
                # Silently handle any display errors
                self.message = old_message

    def stop(self):
        """Stop spinner and restore terminal state."""
        with self._lock:
            if not self.running and not self.content_below:
                return

            self.running = False
            self._stop_event.set()

        # Wait for thread to finish (outside lock to avoid deadlock)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Clear display
        try:
            if self.content_below:
                # STABILITY FIX: Don't try to clear status line when content is below
                # This can interfere with content display
                pass
            elif self._spinner_started:
                # Only clear if spinner was displaying and no content below
                sys.stdout.write('\r' + ' ' * min(self._terminal_width, 200) + '\r')

            sys.stdout.flush()
        except:
            pass

        # Always restore cursor
        self._show_cursor()

        # Reset state
        with self._lock:
            self.content_below = False
            self._spinner_started = False

    def _animate_safe(self):
        """Animation loop with comprehensive error handling."""
        i = 0
        last_frame_time = time.time()
        first_frame = True

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if not self.running or self.content_below:
                        break

                    # Throttle updates for stability
                    current_time = time.time()
                    if current_time - last_frame_time < 0.08:  # ~12 FPS max
                        continue

                    frame = self.frames[i % len(self.frames)]
                    message = self.message

                    # Mark that spinner has started displaying
                    if first_frame:
                        self._spinner_started = True
                        first_frame = False

                # Write frame (outside lock for performance)
                if self._supports_ansi:
                    output = f"\r\033[32m  {frame} \033[0m\033[2m{message}...\033[0m"
                else:
                    output = f"\r  {frame} {message}..."

                # Truncate if too long
                max_width = self._terminal_width - 5
                if len(output) > max_width:
                    output = output[:max_width-3] + "..."

                sys.stdout.write(output)
                sys.stdout.flush()

                i += 1
                last_frame_time = current_time

            except Exception:
                # Silently handle any animation errors
                break

            # Sleep with interrupt checking
            self._stop_event.wait(0.1)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop()
        except:
            pass


@contextmanager
def show_thinking_spinner(message: str = "Thinking"):
    """
    Context manager for showing thinking spinner.

    Usage:
        with show_thinking_spinner("Generating"):
            # AI generates response here
            response = generate()

    Args:
        message: Message to display
    """
    spinner = ThinkingSpinner(message)
    try:
        spinner.start()
        yield spinner
    finally:
        spinner.stop()


# Global spinner instance for singleton pattern
_global_spinner: Optional[ThinkingSpinner] = None


def get_global_spinner() -> ThinkingSpinner:
    """Get or create global spinner instance (singleton pattern)."""
    global _global_spinner
    if _global_spinner is None:
        _global_spinner = ThinkingSpinner()
    return _global_spinner


def cleanup_global_spinner():
    """Cleanup global spinner on exit."""
    global _global_spinner
    if _global_spinner is not None:
        _global_spinner.stop()
        _global_spinner = None