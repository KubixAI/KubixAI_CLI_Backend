"""
Kubix CLI Permission System

Controls what operations tools can perform with granular permissions.
Inspired by OpenCode's permission system.

Features:
- Three permission levels: allow, ask, deny
- Wildcard pattern matching for bash commands
- Interactive user prompts
- Session-based permission memory
"""

from .manager import PermissionManager
from .checker import PermissionChecker

__all__ = ["PermissionManager", "PermissionChecker"]
