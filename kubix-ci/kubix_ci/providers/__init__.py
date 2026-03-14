"""GPU cloud providers for kubix_ci.

Supported providers:
- ssh: Direct SSH to your own GPU machines
- runpod: RunPod cloud GPUs (https://runpod.io)
- lambdalabs: Lambda Labs cloud GPUs (https://lambdalabs.com)
- vastai: Vast.ai GPU marketplace (https://vast.ai)
- fluidstack: FluidStack enterprise GPUs (https://fluidstack.io)
- brev: NVIDIA Brev cloud GPUs (https://brev.dev)
"""

from kubix_ci.providers.base import BaseProvider, ExecutionResult
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.providers.brev import BrevProvider
from kubix_ci.providers.runpod import RunPodProvider
from kubix_ci.providers.lambdalabs import LambdaLabsProvider
from kubix_ci.providers.vastai import VastAIProvider
from kubix_ci.providers.fluidstack import FluidStackProvider

__all__ = [
    "BaseProvider",
    "ExecutionResult",
    "SSHProvider",
    "BrevProvider",
    "RunPodProvider",
    "LambdaLabsProvider",
    "VastAIProvider",
    "FluidStackProvider",
]
