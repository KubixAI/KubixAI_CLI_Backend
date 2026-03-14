"""
Agent Orchestrator - Routes queries to specialized agents.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

from ..tools import ToolRegistry
from ..tools.read_file import ReadFileTool
from ..tools.write_file import WriteFileTool
from ..tools.analyze_cuda import AnalyzeCudaTool
from ..tools.bash_exec import BashTool
from ..tools.list_files import ListFilesTool
from ..tools.profile_cuda import ProfileCudaTool
from ..tools.benchmark_cuda import BenchmarkCudaTool

from ..config import ConfigManager
from ..config.schema import Config
from ..permissions import PermissionManager
from ..cache import CacheManager
from ..sessions import SessionManager

from ..ui.theme import (
    console,
    show_agent_switch,
    show_success,
    show_error,
    show_warning,
    themed_table,
    themed_panel
)

from .base_agent import BaseAgent
from .optimizer_agent import CUDAOptimizerAgent
from .debugger_agent import CUDADebuggerAgent
from .analyzer_agent import CUDAAnalyzerAgent
from .general_agent import GeneralCUDAAgent


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents.

    Responsibilities:
    - Create and manage all agents
    - Route queries to appropriate agents
    - Handle agent switching
    - Provide unified interface
    - Manage shared resources (tools, permissions, config)
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        config_overrides: Optional[Dict] = None
    ):
        """
        Initialize orchestrator and all agents.

        Args:
            working_dir: Working directory for file operations
            config_overrides: CLI config overrides
        """
        self.working_dir = working_dir or Path.cwd()

        # Load configuration
        self.config_manager = ConfigManager(self.working_dir)
        self.config = self.config_manager.load(cli_overrides=config_overrides)

        # Setup API key - Don't interrupt the experience
        self.cache_manager = CacheManager()

        # Clean up any placeholder keys that might have been saved
        self.cache_manager.clear_placeholder_api_key()

        self.api_key = self.cache_manager.get_api_key()

        # If no valid API key, use placeholder but flag that we need a real one
        if not self.api_key:
            self.api_key = "sk-temp-placeholder"
            self.needs_api_key = True
        else:
            self.needs_api_key = False

        # Setup shared resources
        self.permission_manager = PermissionManager(self.config)
        self.registry = ToolRegistry()
        self._register_tools()

        # Create all specialized agents
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

        # Current active agent
        self.current_agent: BaseAgent = self.agents["general"]

        # Auto-routing enabled by default
        self.auto_routing = True

        # Session management
        self.session_manager = SessionManager(
            working_dir=self.working_dir,
            agent_name=self.current_agent.name
        )

        # Enable auto-save (every 60 seconds)
        # DISABLED: Session saving has O(n²) performance issue
        # self.session_manager.enable_auto_save(interval=60)

    def _register_tools(self):
        """Register all available tools."""
        self.registry.register(ReadFileTool())
        self.registry.register(WriteFileTool())
        self.registry.register(AnalyzeCudaTool())
        self.registry.register(BashTool())
        self.registry.register(ListFilesTool())
        self.registry.register(ProfileCudaTool())
        self.registry.register(BenchmarkCudaTool())

    def _initialize_agents(self):
        """Create all specialized agents."""
        # Get agent-specific configs
        agent_configs = self.config.agents

        # Optimizer Agent
        optimizer_config = agent_configs.get("optimizer", {})
        optimizer_model = optimizer_config.get("model")
        self.agents["optimizer"] = CUDAOptimizerAgent(
            working_dir=self.working_dir,
            config=self.config,
            permission_manager=self.permission_manager,
            registry=self.registry,
            api_key=self.api_key,
            model=optimizer_model,
            orchestrator=self
        )

        # Debugger Agent
        debugger_config = agent_configs.get("debugger", {})
        debugger_model = debugger_config.get("model")
        self.agents["debugger"] = CUDADebuggerAgent(
            working_dir=self.working_dir,
            config=self.config,
            permission_manager=self.permission_manager,
            registry=self.registry,
            api_key=self.api_key,
            model=debugger_model,
            orchestrator=self
        )

        # Analyzer Agent
        analyzer_config = agent_configs.get("analyzer", {})
        analyzer_model = analyzer_config.get("model")
        self.agents["analyzer"] = CUDAAnalyzerAgent(
            working_dir=self.working_dir,
            config=self.config,
            permission_manager=self.permission_manager,
            registry=self.registry,
            api_key=self.api_key,
            model=analyzer_model,
            orchestrator=self
        )

        # General Agent
        general_config = agent_configs.get("general", {})
        general_model = general_config.get("model")
        self.agents["general"] = GeneralCUDAAgent(
            working_dir=self.working_dir,
            config=self.config,
            permission_manager=self.permission_manager,
            registry=self.registry,
            api_key=self.api_key,
            model=general_model,
            orchestrator=self
        )

    def _setup_api_key(self):
        """Prompt user for API key when needed."""
        console.print("\n[yellow]API Key Required[/yellow]")
        console.print("To use AI features, you need a FREE API key from OpenRouter.\n")
        console.print("[nvidia]Quick Setup (30 seconds):[/nvidia]")
        console.print("1. Open: [bold cyan]https://openrouter.ai[/bold cyan]")
        console.print("2. Sign up with Google/GitHub")
        console.print("3. Copy your API key\n")

        from rich.prompt import Prompt
        api_key = Prompt.ask("[nvidia]Paste your API key[/nvidia]", password=True)

        if api_key and api_key.strip():
            self.api_key = api_key.strip()
            self.cache_manager.save_api_key(self.api_key)
            self.needs_api_key = False

            # Update all agents with the new API key
            for agent in self.agents.values():
                agent.api_key = self.api_key
                agent.client.api_key = self.api_key

            show_success("API key saved! You're all set.")
            return True
        else:
            show_error("No API key provided. Get one free at https://openrouter.ai")
            return False

    def route(self, query: str) -> BaseAgent:
        """
        Route query to most appropriate agent.

        Algorithm:
        1. Check for explicit agent selection (/optimize, /debug, etc.)
        2. If auto-routing enabled, ask each agent for confidence score
        3. Select agent with highest confidence above threshold
        4. Fall back to current/general agent if no high confidence

        Args:
            query: User's query

        Returns:
            Selected agent
        """
        # Check for explicit agent commands
        query_lower = query.lower().strip()

        if query_lower.startswith("/optimize") or query_lower.startswith("/opt"):
            return self.agents["optimizer"]
        elif query_lower.startswith("/debug") or query_lower.startswith("/dbg"):
            return self.agents["debugger"]
        elif query_lower.startswith("/analyze") or query_lower.startswith("/anal"):
            return self.agents["analyzer"]
        elif query_lower.startswith("/general") or query_lower.startswith("/gen"):
            return self.agents["general"]

        # If auto-routing disabled, use current agent
        if not self.auto_routing:
            return self.current_agent

        # Get confidence scores from all agents
        scores = {}
        for name, agent in self.agents.items():
            scores[name] = agent.can_handle(query)

        # Find best agent
        best_name, best_score = max(scores.items(), key=lambda x: x[1])

        # Threshold for switching agents (0.7 = high confidence)
        if best_score >= 0.7:
            return self.agents[best_name]

        # If no high confidence, use current agent
        # (stay with current agent unless there's a clear better option)
        return self.current_agent

    def chat(self, user_message: str) -> str:
        """
        Route query to appropriate agent and execute.

        Args:
            user_message: User's message

        Returns:
            Agent's response
        """
        # Check if we need to set up API key first
        # Validate that we don't have a placeholder key
        if self.needs_api_key or self.api_key == "sk-temp-placeholder":
            if not self._setup_api_key():
                raise ValueError("API key required to use AI features. Get your free key at: https://openrouter.ai")

        # Route to best agent
        agent = self.route(user_message)

        # Show agent switch if changed
        if agent != self.current_agent:
            console.print(f"\n[dim]→ Routing to {agent.display_name}[/dim]\n")
            self.current_agent = agent
            # Update session manager's agent name
            self.session_manager.agent_name = agent.name

        # Execute on agent
        response = agent.chat(user_message)

        # Update session with conversation history
        # DISABLED: Blocking issue - session saving is too slow
        # self._update_session()

        return response

    def _update_session(self):
        """Update current session with latest conversation history."""
        try:
            # Get current conversation from active agent
            messages = self.current_agent.conversation

            # Create or update session
            if not self.session_manager.current_session:
                # Create new session
                self.session_manager.create_session(messages=messages)
            else:
                # Update existing session messages
                self.session_manager.current_session.messages = messages
                self.session_manager.current_session.metadata.update_timestamp()

            # Save session
            self.session_manager.save_current()

        except Exception as e:
            # Don't fail the chat if session save fails
            pass

    def switch_agent(self, agent_name: str) -> bool:
        """
        Manually switch to a specific agent.

        Args:
            agent_name: Name of agent to switch to

        Returns:
            True if successful, False if agent not found
        """
        if agent_name in self.agents:
            self.current_agent = self.agents[agent_name]
            # Update session manager's agent name
            self.session_manager.agent_name = self.current_agent.name
            show_agent_switch("Previous agent", self.current_agent.display_name)
            console.print(f"[nvidia_dim]{self.current_agent.description}[/nvidia_dim]\n")
            return True
        else:
            show_error(f"Unknown agent: {agent_name}\nAvailable: {', '.join(self.agents.keys())}")
            return False

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get agent by name.

        Args:
            agent_name: Agent name

        Returns:
            Agent instance or None
        """
        return self.agents.get(agent_name)

    def restore_session(self, session_id: str) -> bool:
        """
        Restore a saved session and load its conversation history.

        Args:
            session_id: Session ID to restore

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load session
            session = self.session_manager.load_session(session_id)
            if not session:
                show_error(f"Session not found: {session_id}")
                return False

            # Switch to the session's agent
            agent_name = session.metadata.agent_name
            if agent_name in self.agents:
                self.current_agent = self.agents[agent_name]
                self.session_manager.agent_name = agent_name

            # Restore conversation history to current agent
            from ..openrouter_v2 import Message
            self.current_agent.conversation = []

            # Add system prompt (always first)
            system_prompt = self.current_agent._build_enhanced_system_prompt()
            self.current_agent.conversation.append(Message(
                role="system",
                content=system_prompt
            ))

            # Add conversation history
            for msg in session.messages:
                # Convert from session format to Message format
                if hasattr(msg, 'role'):
                    message = Message(
                        role=msg.role,
                        content=msg.content if hasattr(msg, 'content') else None,
                        tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None,
                        tool_call_id=msg.tool_call_id if hasattr(msg, 'tool_call_id') else None,
                        name=msg.name if hasattr(msg, 'name') else None
                    )
                    self.current_agent.conversation.append(message)

            show_success(f"Restored session: {session.metadata.name}")
            console.print(f"[dim]Loaded {len(session.messages)} messages[/dim]\n")
            return True

        except Exception as e:
            show_error(f"Failed to restore session: {e}")
            return False

    def toggle_auto_routing(self) -> bool:
        """
        Toggle automatic agent routing.

        Returns:
            New auto_routing state
        """
        self.auto_routing = not self.auto_routing
        status = "enabled" if self.auto_routing else "disabled"
        console.print(f"\n[nvidia]Auto-routing {status}[/nvidia]\n")
        return self.auto_routing

    def show_agents(self):
        """Display all available agents with NVIDIA theme."""
        console.print("\n[nvidia]🤖 Available Agents[/nvidia]\n")

        table = themed_table()
        table.add_column("Agent", style="nvidia", width=15)
        table.add_column("Model", style="nvidia_dim", width=30)
        table.add_column("Description", style="white")
        table.add_column("Active", style="success", width=8)

        for name, agent in self.agents.items():
            active = "✓" if agent == self.current_agent else ""
            table.add_row(
                agent.display_name,
                agent.model,
                agent.description,
                active
            )

        console.print(table)
        console.print()

    def show_status(self):
        """Show orchestrator status with NVIDIA theme."""
        panel_content = (
            f"[nvidia]Agent:[/nvidia] [bold]{self.current_agent.display_name}[/bold]\n"
            f"[nvidia]Model:[/nvidia] {self.current_agent.model}\n"
            f"[nvidia]Auto-routing:[/nvidia] {'enabled' if self.auto_routing else 'disabled'}\n"
            f"[nvidia]Working directory:[/nvidia] [nvidia_dim]{self.working_dir}[/nvidia_dim]"
        )
        panel = themed_panel(panel_content, title="System Status", border_style="nvidia")
        console.print()
        console.print(panel)
        console.print()

    def get_current_agent(self) -> BaseAgent:
        """
        Get current active agent.

        Returns:
            Current agent
        """
        return self.current_agent

    def __str__(self) -> str:
        """String representation."""
        return f"AgentOrchestrator(current={self.current_agent.name}, agents={len(self.agents)})"

    def __repr__(self) -> str:
        """Debug representation."""
        return f"<AgentOrchestrator current={self.current_agent.name} routing={self.auto_routing}>"
