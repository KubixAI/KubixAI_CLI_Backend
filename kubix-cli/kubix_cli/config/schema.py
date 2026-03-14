"""
Configuration Schema - Type definitions for Kubix CLI config.
Inspired by OpenCode's config system.
"""

from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Permission types
Permission = Literal["allow", "ask", "deny"]


@dataclass
class ToolPermissions:
    """Permissions for tool usage."""
    edit: Permission = "allow"
    bash: Dict[str, Permission] = field(default_factory=lambda: {"*": "allow"})
    webfetch: Permission = "allow"


@dataclass
class AgentConfig:
    """Configuration for a specific agent."""
    name: str
    description: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    tools: Dict[str, bool] = field(default_factory=dict)
    permission: Optional[ToolPermissions] = None
    mode: Literal["primary", "subagent", "all"] = "all"
    disable: bool = False


@dataclass
class CompilerPaths:
    """Compiler path configuration."""
    nvcc: Optional[str] = None
    cl: Optional[str] = None  # MSVC cl.exe
    gcc: Optional[str] = None
    ncu: Optional[str] = None  # Nsight Compute
    nsys: Optional[str] = None  # Nsight Systems


@dataclass
class CUDAConfig:
    """CUDA-specific configuration."""
    default_gpu: str = "sm_70"
    auto_compile: bool = True
    auto_benchmark: bool = False
    cache_optimizations: bool = True
    optimization_level: int = 3
    compiler_paths: CompilerPaths = field(default_factory=CompilerPaths)


@dataclass
class UIConfig:
    """UI/UX configuration."""
    theme: str = "nvidia-green"
    compact_mode: bool = True
    show_tool_calls: bool = True
    streaming: bool = True
    banner: bool = True


@dataclass
class Config:
    """
    Main configuration object.

    Supports hierarchical loading:
    1. Global: ~/.config/kubix/config.json
    2. Project: <project>/.kubix/config.json
    3. Local: <cwd>/kubix.json
    4. CLI flags (override)
    """

    # Model settings
    model: str = "deepseek/deepseek-chat"

    # API settings
    api_key: Optional[str] = None

    # Agent configurations
    agents: Dict[str, AgentConfig] = field(default_factory=dict)

    # Global permissions
    permission: Optional[ToolPermissions] = None

    # CUDA settings
    cuda: CUDAConfig = field(default_factory=CUDAConfig)

    # UI settings
    ui: UIConfig = field(default_factory=UIConfig)

    # Tool settings
    tools: Dict[str, bool] = field(default_factory=dict)

    # Session settings
    session: Dict[str, Any] = field(default_factory=dict)

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Handle nested dataclasses
        if "permission" in data and isinstance(data["permission"], dict):
            data["permission"] = ToolPermissions(**data["permission"])

        if "cuda" in data and isinstance(data["cuda"], dict):
            data["cuda"] = CUDAConfig(**data["cuda"])

        if "ui" in data and isinstance(data["ui"], dict):
            data["ui"] = UIConfig(**data["ui"])

        if "agents" in data and isinstance(data["agents"], dict):
            agents = {}
            for name, agent_data in data["agents"].items():
                if isinstance(agent_data, dict):
                    agent_data["name"] = name
                    if "permission" in agent_data and isinstance(agent_data["permission"], dict):
                        agent_data["permission"] = ToolPermissions(**agent_data["permission"])
                    agents[name] = AgentConfig(**agent_data)
                else:
                    agents[name] = agent_data
            data["agents"] = agents

        return cls(**data)


# Default configurations for different use cases
DEFAULT_CONFIG = Config()

OPTIMIZER_CONFIG = Config(
    model="anthropic/claude-3.5-sonnet",
    agents={
        "optimizer": AgentConfig(
            name="optimizer",
            description="Specialized agent for CUDA kernel optimization",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.2,
            tools={
                "bash": False  # Disable bash for safety
            }
        )
    },
    cuda=CUDAConfig(
        auto_compile=True,
        auto_benchmark=True,
        optimization_level=3
    )
)

SAFE_CONFIG = Config(
    permission=ToolPermissions(
        edit="ask",
        bash={"*": "ask", "rm *": "deny", "nvcc *": "allow"},
        webfetch="allow"
    ),
    cuda=CUDAConfig(
        auto_compile=False,
        auto_benchmark=False
    )
)
