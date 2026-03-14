"""
Kubix CLI Agents System
Multi-agent architecture for CUDA development.
"""

# Backward compatibility
from .cuda_agent import CUDAAgent

# New multi-agent system
from .base_agent import BaseAgent
from .optimizer_agent import CUDAOptimizerAgent
from .debugger_agent import CUDADebuggerAgent
from .analyzer_agent import CUDAAnalyzerAgent
from .general_agent import GeneralCUDAAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    # Backward compatibility
    "CUDAAgent",

    # Multi-agent system
    "BaseAgent",
    "CUDAOptimizerAgent",
    "CUDADebuggerAgent",
    "CUDAAnalyzerAgent",
    "GeneralCUDAAgent",
    "AgentOrchestrator",
]
