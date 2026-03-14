"""
Config Manager - Hierarchical configuration loading and management.
Inspired by OpenCode's config system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from .schema import Config, DEFAULT_CONFIG
from .utils import deep_merge, validate_config


class ConfigManager:
    """
    Manages hierarchical configuration loading.

    Config precedence (lowest to highest):
    1. Default config (hardcoded)
    2. Global config (~/.config/kubix/config.json)
    3. Project config (.kubix/config.json in project root)
    4. Local config (kubix.json in current directory)
    5. Environment variables (KUBIX_*)
    6. CLI flags (highest priority)
    """

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self._config: Optional[Config] = None
        self._config_paths: List[Path] = []

    @property
    def global_config_dir(self) -> Path:
        """Get global config directory."""
        if os.name == 'nt':  # Windows
            base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        else:  # Unix-like
            base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

        return base / 'kubix'

    @property
    def global_config_file(self) -> Path:
        """Get global config file path."""
        return self.global_config_dir / 'config.json'

    def find_project_root(self) -> Optional[Path]:
        """
        Find project root by looking for .git directory or .kubix directory.

        Returns:
            Project root path or None if not found
        """
        current = self.working_dir

        while current != current.parent:
            if (current / '.git').exists() or (current / '.kubix').exists():
                return current
            current = current.parent

        return None

    def find_config_files(self) -> List[Path]:
        """
        Find all config files in the hierarchy.

        Returns:
            List of config file paths (in order of precedence, low to high)
        """
        config_files = []

        # 1. Global config
        if self.global_config_file.exists():
            config_files.append(self.global_config_file)

        # 2. Project config
        project_root = self.find_project_root()
        if project_root:
            project_config = project_root / '.kubix' / 'config.json'
            if project_config.exists():
                config_files.append(project_config)

        # 3. Local config
        local_configs = [
            self.working_dir / 'kubix.json',
            self.working_dir / '.kubix.json',
            self.working_dir / 'kubix.jsonc'
        ]
        for local_config in local_configs:
            if local_config.exists():
                config_files.append(local_config)
                break  # Only use first found

        return config_files

    def load_config_file(self, path: Path) -> Dict[str, Any]:
        """
        Load a single config file.

        Args:
            path: Path to config file

        Returns:
            Config dictionary

        Raises:
            ValueError: If config is invalid JSON
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Support for JSONC (comments)
                # Simple approach: remove // comments
                lines = []
                for line in content.split('\n'):
                    # Remove // comments (but not in strings - simplified)
                    if '//' in line:
                        idx = line.find('//')
                        # Very basic check - not in quotes
                        if line[:idx].count('"') % 2 == 0:
                            line = line[:idx]
                    lines.append(line)

                content = '\n'.join(lines)

                return json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config from {path}: {e}")

    def load(
        self,
        cli_overrides: Optional[Dict[str, Any]] = None,
        env_overrides: Optional[Dict[str, Any]] = None
    ) -> Config:
        """
        Load configuration from all sources and merge.

        Args:
            cli_overrides: Overrides from CLI flags
            env_overrides: Overrides from environment variables

        Returns:
            Merged Config object
        """
        # Start with default config
        config_dict = DEFAULT_CONFIG.to_dict()

        # Find and load config files
        config_files = self.find_config_files()
        self._config_paths = config_files

        for config_file in config_files:
            try:
                file_config = self.load_config_file(config_file)
                file_config = validate_config(file_config)
                config_dict = deep_merge(config_dict, file_config)
            except Exception as e:
                print(f"Warning: Failed to load {config_file}: {e}")

        # Apply environment overrides
        if env_overrides:
            config_dict = deep_merge(config_dict, env_overrides)

        # Apply CLI overrides (highest priority)
        if cli_overrides:
            config_dict = deep_merge(config_dict, cli_overrides)

        # Create Config object
        self._config = Config.from_dict(config_dict)

        return self._config

    def get(self) -> Config:
        """
        Get loaded config.

        Returns:
            Config object

        Raises:
            RuntimeError: If config not loaded yet
        """
        if self._config is None:
            self.load()

        return self._config

    def save_global(self, config: Config):
        """
        Save config to global config file.

        Args:
            config: Config to save
        """
        # Ensure directory exists
        self.global_config_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.global_config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

    def save_local(self, config: Config, filename: str = "kubix.json"):
        """
        Save config to local config file.

        Args:
            config: Config to save
            filename: Filename for config
        """
        config_path = self.working_dir / filename

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

    def init_project(self):
        """
        Initialize a new project with .kubix directory and config.
        """
        # Create .kubix directory
        kubix_dir = self.working_dir / '.kubix'
        kubix_dir.mkdir(exist_ok=True)

        # Create config file
        config_file = kubix_dir / 'config.json'

        if config_file.exists():
            print(f"Config already exists: {config_file}")
            return

        # Create default config
        default_config = Config()
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config.to_dict(), f, indent=2)

        print(f"Initialized Kubix project config: {config_file}")

        # Create .gitignore
        gitignore = kubix_dir / '.gitignore'
        if not gitignore.exists():
            with open(gitignore, 'w') as f:
                f.write("# Kubix CLI\n")
                f.write("cache/\n")
                f.write("sessions/\n")

    def get_loaded_paths(self) -> List[Path]:
        """
        Get list of config files that were loaded.

        Returns:
            List of config file paths
        """
        return self._config_paths

    def show_config_info(self):
        """
        Display information about loaded configuration.
        """
        config = self.get()

        print("📋 Kubix CLI Configuration\n")

        print("📁 Config Files Loaded:")
        if self._config_paths:
            for path in self._config_paths:
                print(f"  ✓ {path}")
        else:
            print("  (none - using defaults)")
        print()

        print("🔧 Current Settings:")
        print(f"  Model: {config.model}")
        print(f"  Theme: {config.ui.theme}")
        print(f"  Default GPU: {config.cuda.default_gpu}")
        print(f"  Auto-compile: {config.cuda.auto_compile}")
        print(f"  Streaming: {config.ui.streaming}")
        print()

        print("🤖 Agents:")
        if config.agents:
            for name, agent in config.agents.items():
                print(f"  • {name}: {agent.description or 'No description'}")
        else:
            print("  (none configured)")
        print()

        print("🔐 Permissions:")
        if config.permission:
            print(f"  Edit: {config.permission.edit}")
            print(f"  Bash: {config.permission.bash}")
            print(f"  WebFetch: {config.permission.webfetch}")
        else:
            print("  (using defaults)")
