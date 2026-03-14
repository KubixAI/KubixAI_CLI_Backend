"""
Enhanced Base Agent - Production-ready with advanced error handling.

Handles all edge cases:
- Tool execution failures
- Invalid parameters
- Permission errors
- File not found
- Network timeouts
- Malformed responses
- Tool call loops
- Context overflow
- State recovery
"""

import os
import sys
import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console

from ..tools import ToolRegistry
from ..tools.base import ToolContext
from ..openrouter_v2 import OpenRouterClientV2, Message
from ..config.schema import Config
from ..permissions import PermissionManager
from ..ui.tool_display import execute_tool_with_display
from ..ui.theme import console


@dataclass
class AgentState:
    """Track agent state for recovery."""
    conversation_length: int = 0
    tool_calls_count: int = 0
    errors_count: int = 0
    last_error: Optional[str] = None
    last_tool: Optional[str] = None
    loop_detection: List[str] = None  # Track recent tool calls

    def __post_init__(self):
        if self.loop_detection is None:
            self.loop_detection = []


class EnhancedBaseAgent(ABC):
    """
    Production-ready base agent with advanced error handling.

    Features:
    - Robust tool execution with retries
    - Loop detection and prevention
    - Context overflow management
    - Error recovery strategies
    - State tracking and monitoring
    - Graceful degradation
    """

    # Configuration
    MAX_TOOL_RETRIES = 2
    MAX_CONSECUTIVE_ERRORS = 3
    MAX_TOOL_CALLS_PER_TURN = 10
    LOOP_DETECTION_WINDOW = 5
    MAX_CONVERSATION_LENGTH = 100

    def __init__(
        self,
        working_dir: Path,
        config: Config,
        permission_manager: PermissionManager,
        registry: ToolRegistry,
        api_key: str,
        model: Optional[str] = None,
        orchestrator: Optional[Any] = None
    ):
        """Initialize enhanced agent."""
        self.working_dir = working_dir
        self.config = config
        self.permission_manager = permission_manager
        self.registry = registry
        self.api_key = api_key
        self.orchestrator = orchestrator

        # State tracking
        self.state = AgentState()

        # Use provided model or agent's default
        self.model = model or self.default_model()

        # Conversation state
        self.conversation: List[Message] = []
        self.session_id = f"session_{int(time.time())}"

        # Setup API client
        self.client = OpenRouterClientV2(self.api_key, model=self.model)

        # Add system message with enhanced instructions
        self.conversation.append(Message(
            role="system",
            content=self._build_enhanced_system_prompt()
        ))

    # ============================================================================
    # Abstract Methods - Must be implemented by specialized agents
    # ============================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name (e.g., 'optimizer', 'debugger')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description."""
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent's expertise."""
        pass

    @abstractmethod
    def default_model(self) -> str:
        """Default model for this agent."""
        pass

    def can_handle(self, query: str) -> float:
        """Confidence score 0-1 for handling query."""
        return 0.5

    # ============================================================================
    # Enhanced System Prompt
    # ============================================================================

    def _build_enhanced_system_prompt(self) -> str:
        """Build system prompt with advanced instructions."""
        base_prompt = self.system_prompt()

        enhanced_instructions = """

**CRITICAL INSTRUCTIONS FOR RELIABLE OPERATION:**

1. **Tool Usage Best Practices**
   - Always validate file paths exist before reading/writing
   - Check tool results before proceeding to next step
   - If a tool fails, try an alternative approach
   - Don't retry the same failed operation more than once
   - Use bash tool for file operations only when necessary

2. **Error Handling**
   - If a tool returns an error, show the error message directly
   - Do NOT give long explanations or tutorials
   - Error messages already contain fix instructions
   - Just acknowledge the error and wait for user

3. **File Operations & Task Completion - CRITICAL**
   - You have up to 15 tool calls per task - use them wisely
   - **NEVER say you will do something and then not do it - ALWAYS CALL THE TOOLS**
   - If you say "I'll list files" or "I'll scan" - IMMEDIATELY call list_files tool
   - If you say "I'll read" - IMMEDIATELY call read_file tool
   - If you say "I'll write" - IMMEDIATELY call write_file tool
   - DO NOT just describe what you would do - ACTUALLY DO IT
   - ALWAYS complete the task the user requested before stopping
   - If editing a file, make ALL necessary changes before stopping
   - Don't stop in the middle of a multi-step task
   - Use read_file before write_file to understand context
   - Always use absolute paths or verify working directory
   - Check file exists before reading
   - Verify write succeeded before declaring success

4. **Compilation & Execution**
   - Check for compilation errors in output
   - Don't proceed if compilation failed
   - Show compiler errors directly - no tutorials
   - Use appropriate compiler flags

5. **Communication**
   - Be concise but complete
   - Explain what you're doing and why
   - Report success/failure clearly
   - Ask for clarification if uncertain
   - NEVER show large code blocks in responses - code is in files

6. **Loop Prevention**
   - Don't repeat the same tool call if it failed
   - Don't call tools in circles
   - If stuck, ask user for guidance
   - Recognize when you need different information

7. **Context Management**
   - Focus on current task
   - Don't include unnecessary tool calls
   - Summarize progress periodically
   - Keep responses focused and relevant

8. **User Interaction Respect**
   - AFTER writing a file, a post-edit prompt will appear for the USER
   - DO NOT call compile/analyze/profile tools automatically
   - WAIT for user to choose their action (c/a/p/b/s)
   - The system will execute their choice automatically
   - Your job is ONLY to write the file, not execute next steps
   - If user skips the prompt, RESPECT that and don't auto-execute

**TOOL CALL RULES:**

- Maximum 5-7 tool calls per response (stay focused)
- If a tool fails, don't retry immediately - analyze why
- Always check tool output before using results
- Use bash tool sparingly - prefer specific tools
- Verify file paths are valid before operations

**RESPONSE STRUCTURE:**

1. Acknowledge user request
2. Explain your approach (briefly)
3. Execute tools (with validation)
4. Report results clearly
5. Suggest next steps if applicable

**REMEMBER:**
- You are production-ready - be reliable and robust
- Handle errors gracefully - don't crash or loop
- Validate assumptions - don't assume success
- Be helpful but honest - admit when stuck
- Focus on solving the task efficiently
- **NEVER explain tool errors - they have instructions already**
"""

        return base_prompt + enhanced_instructions

    # ============================================================================
    # Enhanced Tool Execution with Error Handling
    # ============================================================================

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Execute tool with advanced error handling.

        Features:
        - Retry logic with backoff
        - Error classification
        - Loop detection
        - State tracking
        - Graceful degradation
        """
        # Check for tool call loops
        if self._detect_loop(tool_name, params):
            error_msg = f"⚠ Loop detected: Repeated {tool_name} calls. Stopping to prevent infinite loop."
            console.print(f"[yellow]{error_msg}[/yellow]")
            return error_msg

        # Track tool call
        self.state.last_tool = tool_name
        self.state.tool_calls_count += 1

        # Update loop detection
        tool_signature = f"{tool_name}:{str(params)[:50]}"
        self.state.loop_detection.append(tool_signature)
        if len(self.state.loop_detection) > self.LOOP_DETECTION_WINDOW:
            self.state.loop_detection.pop(0)

        # Check if too many tool calls
        if self.state.tool_calls_count > self.MAX_TOOL_CALLS_PER_TURN * 2:
            error_msg = "⚠ Too many tool calls. Please simplify your approach."
            console.print(f"[yellow]{error_msg}[/yellow]")
            return error_msg

        # Execute with retries
        for attempt in range(self.MAX_TOOL_RETRIES):
            try:
                result = self._execute_tool_internal(tool_name, params)

                # Reset error count on success
                self.state.errors_count = 0
                self.state.last_error = None

                return result

            except FileNotFoundError as e:
                error_msg = f"File not found: {e}"
                console.print(f"[yellow]⚠ {error_msg}[/yellow]")
                return f"Error: {error_msg}\n\nPlease verify the file path exists."

            except PermissionError as e:
                error_msg = f"Permission denied: {e}"
                console.print(f"[red]❌ {error_msg}[/red]")
                return f"Error: {error_msg}\n\nThis operation requires additional permissions."

            except TimeoutError as e:
                error_msg = f"Operation timed out: {e}"
                if attempt < self.MAX_TOOL_RETRIES - 1:
                    console.print(f"[yellow]⚠ {error_msg} - Retrying...[/yellow]")
                    time.sleep(1)
                    continue
                else:
                    console.print(f"[red]❌ {error_msg}[/red]")
                    return f"Error: {error_msg}\n\nThe operation took too long."

            except ValueError as e:
                error_msg = f"Invalid parameters: {e}"
                console.print(f"[red]❌ {error_msg}[/red]")
                return f"Error: {error_msg}\n\nPlease check the tool parameters."

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self.state.errors_count += 1
                self.state.last_error = error_msg

                # Log detailed error in development
                if os.getenv("DEBUG"):
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

                # Too many consecutive errors
                if self.state.errors_count >= self.MAX_CONSECUTIVE_ERRORS:
                    console.print(f"[red]❌ Too many consecutive errors. Please check your approach.[/red]")
                    return f"Error: Multiple tool failures. Last error: {error_msg}"

                # Retry or fail
                if attempt < self.MAX_TOOL_RETRIES - 1:
                    console.print(f"[yellow]⚠ Tool failed: {error_msg} - Retrying...[/yellow]")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    console.print(f"[red]❌ Tool failed: {error_msg}[/red]")
                    return f"Error: {error_msg}\n\nPlease try a different approach."

        return "Error: Tool execution failed after retries."

    def _execute_tool_internal(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Internal tool executor with validation."""

        def _internal_executor(tool_name: str, params: Dict[str, Any]) -> str:
            """Execute tool with async handling."""
            # Validate parameters
            validated_params = self._validate_tool_params(tool_name, params)

            # Create context
            ctx = ToolContext(
                session_id=self.session_id,
                message_id=f"msg_{len(self.conversation)}",
                agent=self.name,
                working_dir=str(self.working_dir),
                permission_manager=self.permission_manager
            )

            # Execute tool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.registry.execute(tool_name, validated_params, ctx)
                )
                return str(result)
            finally:
                loop.close()

        # Use display wrapper
        return execute_tool_with_display(tool_name, params, _internal_executor)

    def _validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize tool parameters.

        Prevents common issues:
        - Invalid file paths
        - Missing required params
        - Type mismatches
        """
        validated = params.copy()

        # File path validation
        if tool_name in ["read_file", "write_file", "analyze_cuda"]:
            path_key = "filepath" if "filepath" in params else "file_path"

            if path_key in validated:
                filepath = validated[path_key]

                # Convert to absolute path if needed
                if not os.path.isabs(filepath):
                    validated[path_key] = str(self.working_dir / filepath)

                # For read operations, verify file exists
                if tool_name == "read_file":
                    if not os.path.exists(validated[path_key]):
                        raise FileNotFoundError(f"File not found: {validated[path_key]}")

        # Bash command validation
        if tool_name == "bash":
            if "command" not in params:
                raise ValueError("bash tool requires 'command' parameter")

            # Prevent dangerous commands in production
            dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/"]
            cmd = params["command"]
            if any(d in cmd for d in dangerous):
                raise PermissionError(f"Dangerous command blocked: {cmd}")

        return validated

    def _detect_loop(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """
        Detect if we're in a tool call loop.

        Returns True if:
        - Same tool called repeatedly with same params
        - Alternating between same two tools
        """
        if len(self.state.loop_detection) < 3:
            return False

        tool_signature = f"{tool_name}:{str(params)[:50]}"

        # Check for repeated calls
        recent_calls = self.state.loop_detection[-3:]
        if all(sig == tool_signature for sig in recent_calls):
            return True

        # Check for alternating pattern (A-B-A-B)
        if len(self.state.loop_detection) >= 4:
            last_four = self.state.loop_detection[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                return True

        return False

    # ============================================================================
    # Enhanced Chat with Context Management
    # ============================================================================

    def chat(self, user_message: str) -> str:
        """
        Chat with enhanced error handling and context management.
        """
        # Check conversation length
        if len(self.conversation) > self.MAX_CONVERSATION_LENGTH:
            self._trim_conversation()

        # Reset per-turn counters
        self.state.tool_calls_count = 0
        self.state.loop_detection = []

        # Add user message
        self.conversation.append(Message(
            role="user",
            content=user_message
        ))

        console.print()  # Spacing

        try:
            # Get tools
            tools = self.registry.to_openrouter_format()

            # Chat with tools
            response = self.client.chat_with_tools(
                messages=self.conversation,
                tools=tools,
                tool_executor=self._execute_tool,
                stream=True
            )

            # Add response
            self.conversation.append(response)

            console.print()  # Spacing

            return response.content

        except Exception as e:
            error_msg = f"Chat error: {e}"
            console.print(f"[red]❌ {error_msg}[/red]")

            # Log detailed error
            if os.getenv("DEBUG"):
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

            # Add error to conversation for context
            error_response = Message(
                role="assistant",
                content=f"I encountered an error: {error_msg}\n\nPlease try rephrasing your request or asking for help."
            )
            self.conversation.append(error_response)

            return error_response.content

    def _trim_conversation(self):
        """Trim conversation to prevent context overflow."""
        # Keep system prompt and recent messages
        system_msg = self.conversation[0]
        recent_messages = self.conversation[-(self.MAX_CONVERSATION_LENGTH // 2):]

        self.conversation = [system_msg] + recent_messages

        console.print("[dim]📝 Conversation trimmed to manage context length[/dim]\n")

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def clear_conversation(self):
        """Clear conversation and reset state."""
        system_msg = self.conversation[0]
        self.conversation = [system_msg]
        self.state = AgentState()

    def get_info(self) -> Dict[str, Any]:
        """Get agent information including state."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "model": self.model,
            "conversation_length": len(self.conversation),
            "session_id": self.session_id,
            "state": {
                "tool_calls": self.state.tool_calls_count,
                "errors": self.state.errors_count,
                "last_error": self.state.last_error,
                "last_tool": self.state.last_tool
            }
        }

    def __str__(self) -> str:
        return f"{self.display_name} ({self.model})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} model={self.model}>"
