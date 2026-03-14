"""
Permission Manager - Coordinates permission checking and prompting.

This is the main interface for tools to check permissions.
"""

from typing import Dict, Optional, Literal
from ..config.schema import Config
from .checker import PermissionChecker, PermissionResult
from .prompter import PermissionPrompter, UserDecision


class PermissionManager:
    """
    Manages permission checking with session-based memory.

    Coordinates:
    1. Permission checking (via PermissionChecker)
    2. User prompting (via PermissionPrompter)
    3. Session memory ("always allow" / "always deny" choices)

    Example usage:
        manager = PermissionManager(config)

        # Check bash permission
        if manager.check_bash("rm -rf /tmp/file.txt"):
            # Execute command
            ...

        # Check edit permission
        if manager.check_edit("kernel.cu", "write"):
            # Write file
            ...
    """

    def __init__(self, config: Config):
        """
        Initialize permission manager.

        Args:
            config: Configuration with permission settings
        """
        self.config = config
        self.checker = PermissionChecker(config)
        self.prompter = PermissionPrompter()

        # Session memory for "always allow" / "always deny" decisions
        # Structure: {"bash": {"pattern": "allow"/"deny"}, "edit": "allow", ...}
        self.session_memory: Dict[str, any] = {
            "bash": {},  # Pattern-based memory
            "edit": None,  # Global edit permission override
            "webfetch": None,  # Global webfetch permission override
        }

    def check_bash(self, command: str) -> bool:
        """
        Check if bash command is allowed.

        Handles the full flow:
        1. Check session memory
        2. Check configured permissions
        3. Prompt user if needed
        4. Store decision if "always_*"

        Args:
            command: Bash command to execute

        Returns:
            True if command is allowed, False if denied
        """
        # Check session memory first
        if self.session_memory["bash"]:
            # Check if we have a stored decision for this command's pattern
            for pattern, decision in self.session_memory["bash"].items():
                from .matcher import WildcardMatcher
                if WildcardMatcher.matches(command, pattern):
                    if decision == "allow":
                        return True
                    elif decision == "deny":
                        return False

        # Check permissions via checker
        result = self.checker.check_bash(command)

        # If allowed, proceed
        if result.allowed:
            return True

        # If denied, block
        if not result.should_prompt:
            return False

        # Need to prompt user
        decision = self.prompter.prompt_bash(command, result.matched_pattern)

        # Handle user decision
        if decision == "allow":
            return True

        elif decision == "deny":
            return False

        elif decision == "always_allow":
            # Store in session memory
            pattern = result.matched_pattern or command
            self.session_memory["bash"][pattern] = "allow"
            return True

        elif decision == "always_deny":
            # Store in session memory
            pattern = result.matched_pattern or command
            self.session_memory["bash"][pattern] = "deny"
            return False

        # Default to deny
        return False

    def check_edit(self, filepath: str, operation: str = "edit") -> bool:
        """
        Check if file editing is allowed.

        Args:
            filepath: Path to file
            operation: Type of operation (edit, write, create, delete)

        Returns:
            True if allowed, False if denied
        """
        # Check session memory
        if self.session_memory["edit"] == "allow":
            return True
        elif self.session_memory["edit"] == "deny":
            return False

        # Check permissions
        result = self.checker.check_edit(filepath, operation)

        # If allowed, proceed
        if result.allowed:
            return True

        # If denied, block
        if not result.should_prompt:
            return False

        # Prompt user
        decision = self.prompter.prompt_edit(filepath, operation)

        # Handle decision
        if decision == "allow":
            return True

        elif decision == "deny":
            return False

        elif decision == "always_allow":
            self.session_memory["edit"] = "allow"
            return True

        elif decision == "always_deny":
            self.session_memory["edit"] = "deny"
            return False

        return False

    def check_webfetch(self, url: str) -> bool:
        """
        Check if web fetching is allowed.

        Args:
            url: URL to fetch

        Returns:
            True if allowed, False if denied
        """
        # Check session memory
        if self.session_memory["webfetch"] == "allow":
            return True
        elif self.session_memory["webfetch"] == "deny":
            return False

        # Check permissions
        result = self.checker.check_webfetch(url)

        # If allowed, proceed
        if result.allowed:
            return True

        # If denied, block
        if not result.should_prompt:
            return False

        # Prompt user
        decision = self.prompter.prompt_webfetch(url)

        # Handle decision
        if decision == "allow":
            return True

        elif decision == "deny":
            return False

        elif decision == "always_allow":
            self.session_memory["webfetch"] = "allow"
            return True

        elif decision == "always_deny":
            self.session_memory["webfetch"] = "deny"
            return False

        return False

    def check_tool(self, tool_name: str, operation_details: Optional[str] = None) -> bool:
        """
        Generic tool permission check.

        Routes to appropriate check method based on tool name.

        Args:
            tool_name: Name of the tool
            operation_details: Details about the operation (command, filepath, url, etc.)

        Returns:
            True if allowed, False if denied
        """
        if tool_name == "bash" and operation_details:
            return self.check_bash(operation_details)

        elif tool_name in ("write_file", "edit") and operation_details:
            return self.check_edit(operation_details, tool_name)

        elif tool_name == "webfetch" and operation_details:
            return self.check_webfetch(operation_details)

        # Default: allow other tools
        return True

    def is_tool_enabled(self, tool_name: str) -> bool:
        """
        Check if tool is enabled (not completely blocked).

        This is a quick check without prompting.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is not completely denied
        """
        return self.checker.is_tool_allowed(tool_name)

    def get_session_memory(self) -> Dict:
        """
        Get current session memory (for debugging/inspection).

        Returns:
            Dict of stored permission decisions
        """
        return self.session_memory.copy()

    def clear_session_memory(self):
        """
        Clear session memory (reset all "always_*" decisions).
        """
        self.session_memory = {
            "bash": {},
            "edit": None,
            "webfetch": None,
        }

    def get_permission_summary(self) -> str:
        """
        Get human-readable summary of current permissions.

        Returns:
            Formatted string describing permissions
        """
        lines = []
        lines.append("🔐 Permission Summary\n")

        # Edit permissions
        edit_perm = self.checker.permissions.edit
        session_edit = self.session_memory["edit"]
        if session_edit:
            lines.append(f"  File Editing: {session_edit} (session override)")
        else:
            lines.append(f"  File Editing: {edit_perm}")

        # Bash permissions
        bash_patterns = self.checker.permissions.bash
        lines.append(f"  Bash Commands: {len(bash_patterns)} patterns configured")
        for pattern, perm in list(bash_patterns.items())[:3]:
            lines.append(f"    • {pattern}: {perm}")
        if len(bash_patterns) > 3:
            lines.append(f"    ... and {len(bash_patterns) - 3} more")

        # Session overrides
        if self.session_memory["bash"]:
            lines.append(f"  Session overrides: {len(self.session_memory['bash'])} bash patterns")

        # Webfetch permissions
        webfetch_perm = self.checker.permissions.webfetch
        session_webfetch = self.session_memory["webfetch"]
        if session_webfetch:
            lines.append(f"  Web Fetch: {session_webfetch} (session override)")
        else:
            lines.append(f"  Web Fetch: {webfetch_perm}")

        return "\n".join(lines)
