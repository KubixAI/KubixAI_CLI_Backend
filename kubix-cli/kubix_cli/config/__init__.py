"""
Kubix CLI Configuration System
Hierarchical config loading with deep merge support.
"""

from .manager import ConfigManager, Config

__all__ = ["ConfigManager", "Config"]
