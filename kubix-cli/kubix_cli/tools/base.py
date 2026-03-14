"""
Base tool infrastructure for Kubix CLI.
Inspired by OpenCode's tool system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
from enum import Enum


class Permission(str, Enum):
    """Permission level for tool execution."""
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class ToolContext:
    """Context passed to tool execution."""
    session_id: str
    message_id: str
    agent: str
    working_dir: str
    extra: Dict[str, Any] = field(default_factory=dict)

    # Callback for metadata updates
    metadata_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # Permission manager (optional, for permission-controlled tools)
    permission_manager: Optional[Any] = None  # Type: PermissionManager (avoid circular import)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Update tool execution metadata."""
        if self.metadata_callback:
            self.metadata_callback(metadata)


@dataclass
class ToolResult:
    """Result from tool execution."""
    title: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __str__(self) -> str:
        """String representation for display."""
        if self.error:
            return f"❌ {self.title}\n{self.error}"
        return f"✓ {self.title}\n{self.output}"


class Tool(ABC):
    """
    Base class for all tools.

    Each tool must implement:
    - name: Unique identifier
    - description: What the tool does (shown to AI)
    - parameters: JSON Schema for parameters
    - execute: Async function that performs the action
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for AI model."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema definition for tool parameters.

        Example:
        {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File to read"},
                "limit": {"type": "integer", "description": "Max lines"}
            },
            "required": ["filename"]
        }
        """
        pass

    @abstractmethod
    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            params: Validated parameters matching the schema
            ctx: Execution context

        Returns:
            ToolResult with output and metadata
        """
        pass

    def to_openrouter_format(self) -> Dict[str, Any]:
        """
        Convert tool to OpenRouter function calling format.

        Returns:
            Dictionary in OpenRouter's expected format
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against schema.
        Basic validation - can be extended with jsonschema library.

        Args:
            params: Parameters to validate

        Returns:
            Validated parameters

        Raises:
            ValueError: If validation fails
        """
        schema = self.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field in required:
            if field not in params:
                raise ValueError(f"Missing required parameter: {field}")

        # Check extra fields
        for field in params:
            if field not in properties:
                raise ValueError(f"Unknown parameter: {field}")

        # Type checking (basic)
        for field, value in params.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Parameter {field} must be a string")
                elif expected_type == "integer" and not isinstance(value, int):
                    raise ValueError(f"Parameter {field} must be an integer")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Parameter {field} must be a boolean")

        return params
