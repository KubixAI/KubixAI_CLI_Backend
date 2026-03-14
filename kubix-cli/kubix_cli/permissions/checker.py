"""
Permission Checker - Core permission checking logic.

Determines if operations are allowed based on configuration.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from ..config.schema import Config, ToolPermissions
from .matcher import WildcardMatcher, is_dangerous_command


PermissionLevel = Literal["allow", "ask", "deny"]


@dataclass
class PermissionResult:
    """
    Result of a permission check.

    Attributes:
        allowed: Whether operation is allowed
        should_prompt: Whether user should be prompted for confirmation
        permission_level: The permission level that was applied
        matched_pattern: For bash commands, which pattern matched
        reason: Human-readable reason for the decision
    """
    allowed: bool
    should_prompt: bool
    permission_level: PermissionLevel
    matched_pattern: Optional[str] = None
    reason: Optional[str] = None

    @staticmethod
    def allow(reason: str = "Operation is allowed") -> "PermissionResult":
        """Create an 'allow' result."""
        return PermissionResult(
            allowed=True,
            should_prompt=False,
            permission_level="allow",
            reason=reason
        )

    @staticmethod
    def ask(reason: str = "User confirmation required") -> "PermissionResult":
        """Create an 'ask' result."""
        return PermissionResult(
            allowed=False,
            should_prompt=True,
            permission_level="ask",
            reason=reason
        )

    @staticmethod
    def deny(reason: str = "Operation is denied") -> "PermissionResult":
        """Create a 'deny' result."""
        return PermissionResult(
            allowed=False,
            should_prompt=False,
            permission_level="deny",
            reason=reason
        )


class PermissionChecker:
    """
    Checks if operations are allowed based on configuration.

    Handles three permission levels:
    - allow: Operation proceeds without confirmation
    - ask: User is prompted for confirmation
    - deny: Operation is blocked
    """

    def __init__(self, config: Config):
        """
        Initialize permission checker.

        Args:
            config: Configuration object with permission settings
        """
        self.config = config

        # Get permission settings or use defaults
        self.permissions = config.permission or ToolPermissions()

    def check_edit(self, filepath: str, operation: str = "edit") -> PermissionResult:
        """
        Check if file editing is allowed.

        Args:
            filepath: Path to file being edited
            operation: Type of operation (edit, write, delete)

        Returns:
            PermissionResult
        """
        permission = self.permissions.edit

        if permission == "allow":
            return PermissionResult.allow(f"File editing is allowed")

        elif permission == "ask":
            return PermissionResult.ask(f"Confirm {operation} of {filepath}")

        else:  # deny
            return PermissionResult.deny(f"File editing is denied by policy")

    def check_bash(self, command: str) -> PermissionResult:
        """
        Check if bash command is allowed.

        Uses wildcard pattern matching to find the most specific
        permission rule for the command.

        Args:
            command: Bash command to check

        Returns:
            PermissionResult with matched pattern info
        """
        bash_permissions = self.permissions.bash

        # Find best matching pattern
        matched_pattern, permission = WildcardMatcher.find_best_match(
            command,
            bash_permissions
        )

        # No matching pattern found
        if matched_pattern is None:
            # Default to 'ask' for safety
            return PermissionResult(
                allowed=False,
                should_prompt=True,
                permission_level="ask",
                reason="No permission rule found for this command (defaulting to ask)"
            )

        # Check if command is dangerous (extra safety)
        if is_dangerous_command(command):
            # If command is dangerous and permission is "allow", downgrade to "ask"
            if permission == "allow":
                return PermissionResult(
                    allowed=False,
                    should_prompt=True,
                    permission_level="ask",
                    matched_pattern=matched_pattern,
                    reason=f"Command is potentially dangerous (matched: {matched_pattern})"
                )

        # Apply the matched permission
        if permission == "allow":
            return PermissionResult(
                allowed=True,
                should_prompt=False,
                permission_level="allow",
                matched_pattern=matched_pattern,
                reason=f"Command allowed by pattern: {matched_pattern}"
            )

        elif permission == "ask":
            return PermissionResult(
                allowed=False,
                should_prompt=True,
                permission_level="ask",
                matched_pattern=matched_pattern,
                reason=f"Confirmation required (matched: {matched_pattern})"
            )

        else:  # deny
            return PermissionResult(
                allowed=False,
                should_prompt=False,
                permission_level="deny",
                matched_pattern=matched_pattern,
                reason=f"Command denied by pattern: {matched_pattern}"
            )

    def check_webfetch(self, url: str) -> PermissionResult:
        """
        Check if web fetching is allowed.

        Args:
            url: URL to fetch

        Returns:
            PermissionResult
        """
        permission = self.permissions.webfetch

        if permission == "allow":
            return PermissionResult.allow(f"Web fetching is allowed")

        elif permission == "ask":
            return PermissionResult.ask(f"Confirm fetch from {url}")

        else:  # deny
            return PermissionResult.deny(f"Web fetching is denied by policy")

    def check_tool(self, tool_name: str, operation: Optional[str] = None) -> PermissionResult:
        """
        Generic tool permission check.

        Args:
            tool_name: Name of the tool
            operation: Optional specific operation

        Returns:
            PermissionResult
        """
        # Tool-specific checks
        if tool_name in ("write_file", "edit"):
            return self.check_edit(operation or "file", "edit")

        elif tool_name == "bash":
            if operation:
                return self.check_bash(operation)
            else:
                return PermissionResult.ask("No command specified")

        elif tool_name == "webfetch":
            return self.check_webfetch(operation or "")

        # Default: allow other tools
        return PermissionResult.allow(f"Tool {tool_name} is allowed")

    def get_bash_patterns(self) -> dict:
        """
        Get all bash permission patterns.

        Returns:
            Dict of {pattern: permission}
        """
        return self.permissions.bash

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Quick check if a tool is generally allowed (not blocked entirely).

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is not completely denied
        """
        if tool_name in ("write_file", "edit"):
            return self.permissions.edit != "deny"

        elif tool_name == "bash":
            # If all bash is denied, the "*" pattern should be "deny"
            return self.permissions.bash.get("*") != "deny"

        elif tool_name == "webfetch":
            return self.permissions.webfetch != "deny"

        return True  # Other tools default to allowed
