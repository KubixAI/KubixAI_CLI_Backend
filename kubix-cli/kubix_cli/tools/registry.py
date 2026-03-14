"""
Tool Registry - Central registry for all available tools.
Inspired by OpenCode's tool registry system.
"""

from typing import Dict, List, Optional, Type
from .base import Tool, ToolContext, ToolResult
import importlib
import inspect
from pathlib import Path


class ToolRegistry:
    """
    Central registry for all tools.

    Automatically discovers and registers tools from the tools directory.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, tool: Tool):
        """Register a tool instance."""
        self._tools[tool.name] = tool
        self._enabled[tool.name] = True

    def register_class(self, tool_class: Type[Tool]):
        """Register a tool class (will instantiate it)."""
        tool_instance = tool_class()
        self.register(tool_instance)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_enabled(self) -> List[Tool]:
        """List only enabled tools."""
        return [tool for name, tool in self._tools.items() if self._enabled.get(name, True)]

    def enable(self, name: str):
        """Enable a tool."""
        if name in self._tools:
            self._enabled[name] = True

    def disable(self, name: str):
        """Disable a tool."""
        if name in self._tools:
            self._enabled[name] = False

    def to_openrouter_format(self, enabled_only: bool = True) -> List[Dict]:
        """
        Convert all tools to OpenRouter function calling format.

        Args:
            enabled_only: Only include enabled tools

        Returns:
            List of tool definitions in OpenRouter format
        """
        tools = self.list_enabled() if enabled_only else self.list()
        return [tool.to_openrouter_format() for tool in tools]

    async def execute(
        self,
        tool_name: str,
        params: Dict,
        ctx: ToolContext
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            ctx: Execution context

        Returns:
            ToolResult

        Raises:
            ValueError: If tool not found or disabled
        """
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        if not self._enabled.get(tool_name, True):
            raise ValueError(f"Tool is disabled: {tool_name}")

        # Validate parameters
        validated_params = tool.validate_params(params)

        # Execute
        return await tool.execute(validated_params, ctx)

    def auto_discover(self, tools_package: str = "kubix_cli.tools"):
        """
        Auto-discover and register all tools in the tools package.

        Args:
            tools_package: Package path to search for tools
        """
        try:
            # Import the tools package
            package = importlib.import_module(tools_package)
            package_path = Path(package.__file__).parent

            # Find all Python files (except __init__ and base)
            for py_file in package_path.glob("*.py"):
                if py_file.stem in ["__init__", "base", "registry"]:
                    continue

                # Import the module
                module_name = f"{tools_package}.{py_file.stem}"
                try:
                    module = importlib.import_module(module_name)

                    # Find all Tool subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Tool) and obj != Tool:
                            # Check if it's not an abstract class
                            if not inspect.isabstract(obj):
                                self.register_class(obj)

                except Exception as e:
                    # Silently skip modules that fail to import
                    pass

        except Exception as e:
            # Package not found or other error
            pass


# Global registry instance
registry = ToolRegistry()
