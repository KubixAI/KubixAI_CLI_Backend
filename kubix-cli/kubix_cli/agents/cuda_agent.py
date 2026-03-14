"""
CUDA Agent - Advanced AI agent with native tool calling for CUDA development.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..tools import ToolRegistry
from ..tools.base import ToolContext
from ..tools.read_file import ReadFileTool
from ..tools.write_file import WriteFileTool
from ..tools.analyze_cuda import AnalyzeCudaTool
from ..tools.bash_exec import BashTool
from ..tools.list_files import ListFilesTool

from ..openrouter_v2 import OpenRouterClientV2, Message
from ..cache import CacheManager
from ..config import ConfigManager
from ..permissions import PermissionManager
from ..ui.tool_display import execute_tool_with_display

console = Console()

# CUDA Agent System Prompt
CUDA_AGENT_PROMPT = """You are an expert CUDA development AI assistant with native tool calling capabilities.

**#1 RULE - BE SMART ABOUT CONTEXT:**
- For GREETINGS (hi, hey, hello): Be friendly and conversational
- For TASKS (create, compile, optimize): Take IMMEDIATE ACTION without asking
- For QUESTIONS (what is, how does): Give clear answers
- When user says "yes", "yess", "ok", "sure" - IMMEDIATELY take the next action
- NO "Would you like..." questions when action is obvious
- Be HUMAN when chatting, be FAST when working

**YOUR ROLE:**
You help developers write, optimize, analyze, and debug CUDA kernels for GPU computing.

**AVAILABLE TOOLS:**
You have access to powerful tools via function calling:

1. **read_file** - Read CUDA source files (.cu, .cuh)
2. **write_file** - Create or modify CUDA files
3. **analyze_cuda** - Deep analysis for optimization opportunities
4. **bash** - Execute shell commands
5. **list_files** - List available CUDA files

**HOW TO USE TOOLS:**
- ALWAYS use tools to accomplish tasks
- Call multiple tools in parallel when possible
- Read files before editing them
- Analyze before optimizing

**BEST PRACTICES:**
1. When asked to create a kernel:
   - Use write_file to create it
   - Show the user the results

2. When asked to optimize:
   - Use read_file to see the current code
   - Use analyze_cuda to find opportunities
   - Suggest or implement optimizations

3. When debugging:
   - Use read_file to examine code
   - Suggest fixes clearly

**RESPONSE STYLE - BE FAST AND ACTION-ORIENTED:**
- IMMEDIATE action - use tools without asking permission
- Show RESULTS not intentions
- NO "I'll create..." or "Let me..." - just DO IT
- Maximum 1-2 sentences then EXECUTE TOOLS
- Be technical but FAST

**EXAMPLES:**

User: "Create a vector addition kernel"
You: [Call write_file with CUDA code] → Report success

User: "Optimize kernel.cu"
You: [Call read_file] → [Call analyze_cuda] → Suggest optimizations → [Call write_file]

User: "What CUDA files are here?"
You: [Call list_files] → Present results in a clear format

Remember: You are proactive, use tools automatically, and provide complete solutions."""


class CUDAAgent:
    """
    Advanced CUDA development agent with native tool calling.
    """

    def __init__(self, working_dir: Optional[Path] = None, config_overrides: Optional[Dict[str, Any]] = None):
        self.working_dir = working_dir or Path.cwd()
        self.conversation: List[Message] = []
        self.session_id = f"session_{int(__import__('time').time())}"

        # Load configuration
        self.config_manager = ConfigManager(self.working_dir)
        self.config = self.config_manager.load(cli_overrides=config_overrides)

        # Setup permission manager (uses config)
        self.permission_manager = PermissionManager(self.config)

        # Setup tool registry
        self.registry = ToolRegistry()
        self._register_tools()

        # Setup API client
        self.cache_manager = CacheManager()
        self.api_key = self.cache_manager.get_api_key()

        if not self.api_key:
            console.print("\n[yellow]⚠️  No API key found[/yellow]")
            self._setup_api_key()
            self.api_key = self.cache_manager.get_api_key()
            if not self.api_key:
                console.print("[red]Cannot continue without API key[/red]")
                sys.exit(1)

        # Use model from config
        model = self.config.model
        self.client = OpenRouterClientV2(self.api_key, model=model)

        # Add system message
        self.conversation.append(Message(
            role="system",
            content=CUDA_AGENT_PROMPT
        ))

    def _register_tools(self):
        """Register all available tools."""
        self.registry.register(ReadFileTool())
        self.registry.register(WriteFileTool())
        self.registry.register(AnalyzeCudaTool())
        self.registry.register(BashTool())
        self.registry.register(ListFilesTool())

    def _setup_api_key(self):
        """Prompt user for API key."""
        console.print("\n[yellow]Get your API key from: https://openrouter.ai[/yellow]\n")
        from rich.prompt import Prompt
        api_key = Prompt.ask("Enter API key", password=True)
        if api_key:
            self.cache_manager.save_api_key(api_key)
            console.print("[green]✓ API key saved[/green]")
        else:
            console.print("[red]No API key provided[/red]")

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Execute a tool with beautiful animated display.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool result as string
        """
        def _internal_executor(tool_name: str, params: Dict[str, Any]) -> str:
            """Internal executor that does the actual work."""
            # Create context with permission manager
            ctx = ToolContext(
                session_id=self.session_id,
                message_id=f"msg_{len(self.conversation)}",
                agent="cuda_agent",
                working_dir=str(self.working_dir),
                permission_manager=self.permission_manager
            )

            # Execute tool (wrapping async call)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.registry.execute(tool_name, params, ctx)
            )
            loop.close()

            return str(result)

        # Use display wrapper for beautiful output
        return execute_tool_with_display(tool_name, params, _internal_executor)

    def chat(self, user_message: str):
        """
        Send a message and get response with automatic tool execution.

        Args:
            user_message: User's message
        """
        # Add user message
        self.conversation.append(Message(
            role="user",
            content=user_message
        ))

        console.print()  # Spacing

        # Get tools in OpenRouter format
        tools = self.registry.to_openrouter_format()

        # Chat with automatic tool execution
        response = self.client.chat_with_tools(
            messages=self.conversation,
            tools=tools,
            tool_executor=self._execute_tool,
            stream=True
        )

        # Add response to conversation
        self.conversation.append(response)

        console.print()  # Spacing

    def start(self):
        """Start interactive chat loop."""
        self._show_banner()
        self._show_status()

        while True:
            try:
                # Get input
                user_input = self._get_input()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Regular chat
                    self.chat(user_input)

            except KeyboardInterrupt:
                console.print("\n\n[dim]👋 Goodbye![/dim]\n")
                break
            except EOFError:
                console.print("\n\n[dim]👋 Goodbye![/dim]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")
                import traceback
                traceback.print_exc()

    def _show_banner(self):
        """Show branded banner."""
        banner = """
    ██████╗ ██╗ ██████╗ ██╗  ██╗████████╗███╗   ██╗ ██████╗ ██╗    ██╗
    ██╔══██╗██║██╔════╝ ██║  ██║╚══██╔══╝████╗  ██║██╔═══██╗██║    ██║
    ██████╔╝██║██║  ███╗███████║   ██║   ██╔██╗ ██║██║   ██║██║ █╗ ██║
    ██╔══██╗██║██║   ██║██╔══██║   ██║   ██║╚██╗██║██║   ██║██║███╗██║
    ██║  ██║██║╚██████╔╝██║  ██║   ██║   ██║ ╚████║╚██████╔╝╚███╔███╔╝
    ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝
        """
        console.print(f"[bold cyan]{banner}[/bold cyan]")
        console.print("[bold]CUDA AI Agent V2[/bold] • Native Tool Calling")
        console.print(f"[dim]📂 {self.working_dir}[/dim]")

    def _show_status(self):
        """Show status line."""
        # Show config info if config files were loaded
        config_count = len(self.config_manager.get_loaded_paths())
        config_msg = f" • {config_count} config(s) loaded" if config_count > 0 else ""

        console.print(f"\n[dim]Using {self.client.model}{config_msg} • Type /help for commands • Ctrl+C to exit[/dim]\n")

    def _get_input(self) -> str:
        """Get user input with nice formatting."""
        import shutil
        width = shutil.get_terminal_size().columns
        box_width = max(40, width - 2)

        top_border = "┌" + "─" * (box_width - 2) + "┐"
        bottom_border = "└" + "─" * (box_width - 2) + "┘"

        console.print(f"[dim]{top_border}[/dim]")
        console.print("[dim]│[/dim] [cyan]>[/cyan] ", end="")

        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            raise

        console.print(f"[dim]{bottom_border}[/dim]")

        return user_input

    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        parts = cmd[1:].split()
        command = parts[0].lower()

        if command == "help":
            self._cmd_help()
        elif command == "clear":
            self._cmd_clear()
        elif command == "tools":
            self._cmd_tools()
        elif command == "config":
            self._cmd_config()
        elif command in ["exit", "quit"]:
            console.print("\n[dim]👋 Goodbye![/dim]\n")
            sys.exit(0)
        else:
            console.print(f"\n[yellow]Unknown command: /{command}[/yellow]")
            console.print("[dim]Type /help for available commands[/dim]\n")

    def _cmd_help(self):
        """Show help."""
        console.print("\n[bold cyan]📚 Available Commands:[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="dim")

        table.add_row("/help", "Show this help message")
        table.add_row("/tools", "List available tools")
        table.add_row("/config", "Show configuration info")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/exit", "Exit the agent")

        console.print(table)
        console.print()

    def _cmd_clear(self):
        """Clear conversation."""
        self.conversation = [Message(role="system", content=CUDA_AGENT_PROMPT)]
        os.system('cls' if os.name == 'nt' else 'clear')
        self._show_banner()
        console.print(f"\n[green]✓ Conversation cleared[/green]")
        self._show_status()

    def _cmd_tools(self):
        """List available tools."""
        console.print("\n[bold cyan]🛠️  Available Tools:[/bold cyan]\n")

        table = Table(show_header=True, box=box.ROUNDED, padding=(0, 1))
        table.add_column("Tool", style="cyan", width=20)
        table.add_column("Description", style="dim")

        for tool in self.registry.list():
            table.add_row(tool.name, tool.description.split('\n')[0])

        console.print(table)
        console.print()

    def _cmd_config(self):
        """Show configuration info."""
        self.config_manager.show_config_info()
