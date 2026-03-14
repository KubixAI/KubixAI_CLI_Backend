"""
Kubix CLI - NVIDIA-Themed UI Module

Beautiful terminal UI with NVIDIA green (#76B900) theme.
"""

from .theme import NVIDIATheme, themed_panel, themed_table, show_logo
from .progress import show_progress, show_spinner
from .syntax import highlight_cuda_code
from .input_box import show_simple_prompt, create_input_box, get_multiline_input
from .tool_display import execute_tool_with_display, ToolExecutionDisplay, DisplayConfig, set_display_config
from .interactive_help import show_interactive_help, show_options_menu
from .post_edit_prompts import show_post_edit_prompt, prompt_and_execute, prompt_yes_no
from .layout import (
    create_header,
    create_input_section,
    create_response_header,
    create_tool_call_panel,
    create_section_divider,
    create_status_panel,
    create_welcome_panel,
    create_agent_table,
    create_command_help_table,
    show_startup_screen,
    show_thinking_indicator,
    clear_thinking_indicator
)

__all__ = [
    "NVIDIATheme",
    "themed_panel",
    "themed_table",
    "show_logo",
    "show_progress",
    "show_spinner",
    "highlight_cuda_code",
    "show_simple_prompt",
    "create_input_box",
    "get_multiline_input",
    "execute_tool_with_display",
    "ToolExecutionDisplay",
    "DisplayConfig",
    "set_display_config",
    "show_interactive_help",
    "show_options_menu",
    "show_post_edit_prompt",
    "prompt_and_execute",
    "prompt_yes_no",
    "create_header",
    "create_input_section",
    "create_response_header",
    "create_tool_call_panel",
    "create_section_divider",
    "create_status_panel",
    "create_welcome_panel",
    "create_agent_table",
    "create_command_help_table",
    "show_startup_screen",
    "show_thinking_indicator",
    "clear_thinking_indicator",
]
