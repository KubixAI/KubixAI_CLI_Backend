"""
Config Templates - Pre-defined configurations for common use cases.
"""

# Default Configuration
DEFAULT_TEMPLATE = {
    "model": "deepseek/deepseek-chat",
    "cuda": {
        "default_gpu": "sm_70",
        "auto_compile": True,
        "auto_benchmark": False,
        "cache_optimizations": True,
        "optimization_level": 3
    },
    "ui": {
        "theme": "nvidia-green",
        "compact_mode": True,
        "show_tool_calls": True,
        "streaming": True,
        "banner": True
    },
    "permission": {
        "edit": "allow",
        "bash": {
            "*": "allow"
        },
        "webfetch": "allow"
    }
}

# Optimizer Configuration - For production optimization tasks
OPTIMIZER_TEMPLATE = {
    "model": "anthropic/claude-3.5-sonnet",
    "agents": {
        "optimizer": {
            "description": "Specialized CUDA kernel optimizer",
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.2,
            "max_tokens": 4000,
            "tools": {
                "bash": False  # Disable bash for safety
            },
            "mode": "primary"
        }
    },
    "cuda": {
        "default_gpu": "sm_86",
        "auto_compile": True,
        "auto_benchmark": True,
        "cache_optimizations": True,
        "optimization_level": 3
    },
    "permission": {
        "edit": "allow",
        "bash": {
            "*": "deny"  # No bash in optimizer mode
        },
        "webfetch": "allow"
    }
}

# Safe Configuration - Maximum safety, always ask for confirmation
SAFE_TEMPLATE = {
    "model": "deepseek/deepseek-chat",
    "cuda": {
        "auto_compile": False,
        "auto_benchmark": False
    },
    "permission": {
        "edit": "ask",  # Always ask before editing
        "bash": {
            "*": "ask",  # Always ask for bash commands
            "rm *": "deny",  # Never allow rm
            "git push *": "ask"  # Confirm pushes
        },
        "webfetch": "ask"
    }
}

# Free/Cheap Configuration - Use only free models
FREE_TEMPLATE = {
    "model": "deepseek/deepseek-chat:free",
    "agents": {
        "general": {
            "model": "deepseek/deepseek-chat:free",
            "description": "General purpose agent with free model"
        }
    }
}

# Premium Configuration - Use best models for everything
PREMIUM_TEMPLATE = {
    "model": "anthropic/claude-3.5-sonnet",
    "agents": {
        "optimizer": {
            "model": "anthropic/claude-3.5-sonnet",
            "description": "Premium optimizer",
            "temperature": 0.2
        },
        "debugger": {
            "model": "openai/gpt-4o",
            "description": "Premium debugger",
            "temperature": 0.3
        }
    },
    "cuda": {
        "auto_compile": True,
        "auto_benchmark": True
    }
}

# Multi-Agent Configuration - Multiple specialized agents
MULTI_AGENT_TEMPLATE = {
    "model": "deepseek/deepseek-chat",
    "agents": {
        "optimizer": {
            "description": "Optimize CUDA kernels for performance",
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.2,
            "tools": {
                "bash": False
            },
            "mode": "primary"
        },
        "debugger": {
            "description": "Debug compilation and runtime errors",
            "model": "deepseek/deepseek-chat",
            "temperature": 0.3,
            "mode": "primary"
        },
        "general": {
            "description": "General CUDA development assistance",
            "model": "deepseek/deepseek-chat",
            "temperature": 0.7,
            "mode": "all"
        }
    }
}

# Research Configuration - For exploring and analyzing code
RESEARCH_TEMPLATE = {
    "model": "deepseek/deepseek-chat",
    "permission": {
        "edit": "deny",  # Read-only mode
        "bash": {
            "*": "ask",
            "git *": "allow",  # Allow git commands
            "nvcc *": "deny"  # No compilation
        },
        "webfetch": "allow"
    },
    "cuda": {
        "auto_compile": False
    }
}

# All available templates
TEMPLATES = {
    "default": DEFAULT_TEMPLATE,
    "optimizer": OPTIMIZER_TEMPLATE,
    "safe": SAFE_TEMPLATE,
    "free": FREE_TEMPLATE,
    "premium": PREMIUM_TEMPLATE,
    "multi-agent": MULTI_AGENT_TEMPLATE,
    "research": RESEARCH_TEMPLATE
}


def get_template(name: str) -> dict:
    """
    Get a config template by name.

    Args:
        name: Template name

    Returns:
        Template dictionary

    Raises:
        KeyError: If template not found
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")

    return TEMPLATES[name]


def list_templates() -> list:
    """
    List all available templates with descriptions.

    Returns:
        List of (name, description) tuples
    """
    descriptions = {
        "default": "Standard configuration with reasonable defaults",
        "optimizer": "Optimized for kernel optimization tasks, uses premium models",
        "safe": "Maximum safety, always asks for confirmation",
        "free": "Uses only free models to minimize cost",
        "premium": "Best models for everything, highest quality",
        "multi-agent": "Multiple specialized agents for different tasks",
        "research": "Read-only mode for code exploration and analysis"
    }

    return [(name, descriptions.get(name, "")) for name in TEMPLATES.keys()]
