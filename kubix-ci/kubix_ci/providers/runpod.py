"""RunPod cloud GPU provider.

Uses the RunPod Python SDK to create and manage GPU pods.
Requires: pip install runpod
Environment: RUNPOD_API_KEY

Documentation: https://docs.runpod.io/sdks/python/apis
"""

from __future__ import annotations

import os
import time

from kubix_ci.providers.base import BaseProvider
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.config import SSHTarget
from kubix_ci.exceptions import ProviderError


class RunPodProvider(BaseProvider):
    """RunPod cloud GPU provider.

    Creates on-demand GPU pods, runs tests via SSH, then terminates.

    Usage in kubix_ci.yml:
        - name: runpod-a100
          provider: runpod
          gpu: "NVIDIA A100 80GB PCIe"
          gpu_count: 1
          image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self._pod_id: str | None = None
        self._ssh_provider: SSHProvider | None = None
        self._created_pod: bool = False

    def connect(self) -> None:
        """Create a RunPod GPU pod and establish SSH connection."""
        try:
            import runpod
        except ImportError:
            raise ProviderError(
                self.name,
                "RunPod SDK not installed. Run: pip install runpod"
            )

        # Get API key
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise ProviderError(
                self.name,
                "RUNPOD_API_KEY environment variable not set. "
                "Get your key at: https://www.runpod.io/console/user/settings"
            )

        runpod.api_key = api_key

        # Create the pod
        try:
            pod = runpod.create_pod(
                name=f"kubix_ci-{self.config.name}-{int(time.time())}",
                image_name=self.config.image,
                gpu_type_id=self.config.gpu,
                gpu_count=self.config.gpu_count,
                volume_in_gb=self.config.volume_size,
                ports="22/tcp",  # SSH port
                docker_args="",
            )

            self._pod_id = pod["id"]
            self._created_pod = True

        except Exception as e:
            raise ProviderError(self.name, f"Failed to create pod: {e}")

        # Wait for pod to be ready
        self._wait_for_ready(runpod)

        # Get SSH connection info
        pod_info = runpod.get_pod(self._pod_id)
        ssh_host, ssh_port = self._extract_ssh_info(pod_info)

        # Connect via SSH
        ssh_config = SSHTarget(
            name=f"{self.name}-ssh",
            host=ssh_host,
            port=ssh_port,
            user="root",
            gpu=self.config.gpu,
        )

        self._ssh_provider = SSHProvider(ssh_config)
        self._ssh_provider.connect()

    def _wait_for_ready(self, runpod, timeout: int = 300) -> None:
        """Wait for the pod to be in RUNNING state."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                pod = runpod.get_pod(self._pod_id)
                status = pod.get("desiredStatus", "").upper()

                if status == "RUNNING":
                    # Additional wait for SSH to be ready
                    time.sleep(10)
                    return

                if status in ("EXITED", "TERMINATED", "FAILED"):
                    raise ProviderError(
                        self.name,
                        f"Pod entered {status} state instead of RUNNING"
                    )

            except Exception as e:
                if "not found" in str(e).lower():
                    pass  # Pod still being created
                else:
                    raise

            time.sleep(5)

        raise ProviderError(
            self.name,
            f"Pod did not become ready within {timeout} seconds"
        )

    def _extract_ssh_info(self, pod_info: dict) -> tuple[str, int]:
        """Extract SSH host and port from pod info."""
        # RunPod provides SSH access via a public IP and mapped port
        runtime = pod_info.get("runtime", {})

        # Try to get from ports
        ports = runtime.get("ports", [])
        for port in ports:
            if port.get("privatePort") == 22:
                public_ip = port.get("ip")
                public_port = port.get("publicPort")
                if public_ip and public_port:
                    return public_ip, int(public_port)

        # Fallback: try machine info
        machine = pod_info.get("machine", {})
        pod_host_id = machine.get("podHostId")

        if pod_host_id:
            # Format: {pod_host_id}.runpod.io
            return f"{pod_host_id}.runpod.io", 22

        raise ProviderError(
            self.name,
            "Could not extract SSH connection info from pod"
        )

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload file via SSH."""
        if not self._ssh_provider:
            raise ProviderError(self.name, "Not connected")
        self._ssh_provider.upload(local_path, remote_path)

    def execute(self, command: str, timeout: int = 120) -> tuple[str, str, int]:
        """Execute command via SSH."""
        if not self._ssh_provider:
            raise ProviderError(self.name, "Not connected")
        return self._ssh_provider.execute(command, timeout)

    def download(self, remote_path: str, local_path: str) -> None:
        """Download file via SSH."""
        if not self._ssh_provider:
            raise ProviderError(self.name, "Not connected")
        self._ssh_provider.download(remote_path, local_path)

    def disconnect(self) -> None:
        """Disconnect SSH and terminate the pod."""
        # Disconnect SSH first
        if self._ssh_provider:
            try:
                self._ssh_provider.disconnect()
            except Exception:
                pass
            self._ssh_provider = None

        # Terminate the pod if we created it
        if self._created_pod and self._pod_id:
            try:
                import runpod

                api_key = os.environ.get("RUNPOD_API_KEY")
                if api_key:
                    runpod.api_key = api_key
                    runpod.terminate_pod(self._pod_id)
            except Exception:
                pass  # Best effort cleanup

            self._pod_id = None
            self._created_pod = False
