"""
Kubix CLI - Plugin Manager

Extensible plugin system for custom backends, optimizers, and analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util
import inspect

from ..backends.base import GPUBackend
from ..exceptions import PluginError, PluginNotFoundError, PluginLoadError
from ..logger import get_logger


logger = get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str  # 'backend', 'optimizer', 'analyzer'
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class Plugin(ABC):
    """Base class for all plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class BackendPlugin(Plugin):
    """Base class for backend plugins."""

    @abstractmethod
    def get_backend(self) -> GPUBackend:
        """Return the GPU backend implementation."""
        pass


class OptimizerPlugin(Plugin):
    """Base class for optimizer plugins."""

    @abstractmethod
    def optimize(
        self,
        code: str,
        analysis: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """
        Optimize kernel code.

        Args:
            code: Original kernel code
            analysis: Kernel analysis results
            config: Optimization configuration

        Returns:
            Optimized kernel code
        """
        pass


class AnalyzerPlugin(Plugin):
    """Base class for analyzer plugins."""

    @abstractmethod
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyze kernel code.

        Args:
            code: Kernel source code

        Returns:
            Analysis results dictionary
        """
        pass


class PluginManager:
    """
    Manages plugins for Kubix CLI.

    Features:
    - Plugin discovery from directories
    - Plugin loading and initialization
    - Backend management
    - Optimizer management
    - Analyzer management
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.backends: Dict[str, BackendPlugin] = {}
        self.optimizers: Dict[str, OptimizerPlugin] = {}
        self.analyzers: Dict[str, AnalyzerPlugin] = {}
        self._plugin_dirs: List[Path] = []

    def add_plugin_directory(self, directory: Path) -> None:
        """
        Add a directory to search for plugins.

        Args:
            directory: Path to plugin directory
        """
        if directory.exists() and directory.is_dir():
            self._plugin_dirs.append(directory)
            logger.info("Added plugin directory", path=str(directory))
        else:
            logger.warning("Plugin directory not found", path=str(directory))

    def discover_plugins(self) -> List[PluginMetadata]:
        """
        Discover plugins in registered directories.

        Returns:
            List of discovered plugin metadata
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    metadata = self._load_plugin_metadata(plugin_file)
                    if metadata:
                        discovered.append(metadata)
                        logger.info(
                            "Discovered plugin",
                            name=metadata.name,
                            type=metadata.plugin_type,
                            version=metadata.version
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to load plugin metadata",
                        file=str(plugin_file),
                        error=str(e)
                    )

        return discovered

    def load_plugin(self, plugin_name: str, plugin_path: Optional[Path] = None) -> Plugin:
        """
        Load a plugin by name or path.

        Args:
            plugin_name: Name of the plugin
            plugin_path: Optional path to plugin file

        Returns:
            Loaded plugin instance

        Raises:
            PluginNotFoundError: If plugin not found
            PluginLoadError: If plugin fails to load
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]

        # Find plugin file
        if plugin_path is None:
            plugin_path = self._find_plugin_file(plugin_name)
            if plugin_path is None:
                raise PluginNotFoundError(plugin_name)

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(plugin_name, Exception("Invalid plugin file"))

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                raise PluginLoadError(
                    plugin_name,
                    Exception("No plugin class found in module")
                )

            # Instantiate plugin
            plugin = plugin_class()

            # Register plugin
            self.register_plugin(plugin_name, plugin)

            logger.info("Loaded plugin", name=plugin_name)
            return plugin

        except Exception as e:
            raise PluginLoadError(plugin_name, e)

    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register a plugin instance.

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self.plugins[name] = plugin

        metadata = plugin.get_metadata()

        # Register by type
        if isinstance(plugin, BackendPlugin):
            self.backends[name] = plugin
            logger.info("Registered backend plugin", name=name)
        elif isinstance(plugin, OptimizerPlugin):
            self.optimizers[name] = plugin
            logger.info("Registered optimizer plugin", name=name)
        elif isinstance(plugin, AnalyzerPlugin):
            self.analyzers[name] = plugin
            logger.info("Registered analyzer plugin", name=name)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self.plugins.get(name)

    def get_backend(self, name: str) -> Optional[GPUBackend]:
        """Get backend plugin by name."""
        backend_plugin = self.backends.get(name)
        if backend_plugin:
            return backend_plugin.get_backend()
        return None

    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        available = []
        for name, backend_plugin in self.backends.items():
            backend = backend_plugin.get_backend()
            if backend.is_available():
                available.append(name)
        return available

    def auto_select_backend(self) -> Optional[GPUBackend]:
        """
        Automatically select the best available backend.

        Priority: CUDA > ROCm > SYCL > Vulkan

        Returns:
            Best available backend or None
        """
        priority = ['cuda', 'rocm', 'sycl', 'vulkan', 'metal']

        for backend_name in priority:
            backend = self.get_backend(backend_name)
            if backend and backend.is_available():
                logger.info("Auto-selected backend", backend=backend_name)
                return backend

        logger.warning("No GPU backend available")
        return None

    def initialize_plugin(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Initialize a plugin with configuration.

        Args:
            name: Plugin name
            config: Configuration dictionary

        Returns:
            True if successful
        """
        plugin = self.get_plugin(name)
        if not plugin:
            logger.error("Plugin not found", name=name)
            return False

        try:
            success = plugin.initialize(config)
            if success:
                logger.info("Initialized plugin", name=name)
            else:
                logger.error("Plugin initialization failed", name=name)
            return success
        except Exception as e:
            logger.error(
                "Plugin initialization error",
                name=name,
                error=str(e)
            )
            return False

    def cleanup_all(self) -> None:
        """Clean up all loaded plugins."""
        for name, plugin in self.plugins.items():
            try:
                plugin.cleanup()
                logger.info("Cleaned up plugin", name=name)
            except Exception as e:
                logger.error(
                    "Plugin cleanup error",
                    name=name,
                    error=str(e)
                )

    def list_plugins(self) -> Dict[str, PluginMetadata]:
        """List all loaded plugins with metadata."""
        return {
            name: plugin.get_metadata()
            for name, plugin in self.plugins.items()
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _find_plugin_file(self, plugin_name: str) -> Optional[Path]:
        """Find plugin file in registered directories."""
        for plugin_dir in self._plugin_dirs:
            plugin_file = plugin_dir / f"{plugin_name}.py"
            if plugin_file.exists():
                return plugin_file
        return None

    def _load_plugin_metadata(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata without fully loading the plugin."""
        try:
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem,
                plugin_file
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for PLUGIN_METADATA constant
            if hasattr(module, 'PLUGIN_METADATA'):
                metadata_dict = module.PLUGIN_METADATA
                return PluginMetadata(**metadata_dict)

            # Try to instantiate and get metadata
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                plugin = plugin_class()
                return plugin.get_metadata()

        except Exception:
            pass

        return None

    def _find_plugin_class(self, module) -> Optional[Type[Plugin]]:
        """Find plugin class in module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj != Plugin:
                # Check for specific plugin types
                if issubclass(obj, (BackendPlugin, OptimizerPlugin, AnalyzerPlugin)):
                    return obj
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_all()
        return False


# Global plugin manager instance
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create global plugin manager."""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager
