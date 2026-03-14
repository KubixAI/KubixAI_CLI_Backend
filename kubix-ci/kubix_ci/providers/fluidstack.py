"""FluidStack cloud GPU provider.

Uses the FluidStack REST API to create and manage GPU instances.
Requires: FLUIDSTACK_API_KEY environment variable

Documentation: https://docs.fluidstack.io/
"""

from __future__ import annotations

import os
import time

import httpx

from kubix_ci.providers.base import BaseProvider
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.config import SSHTarget
from kubix_ci.exceptions import ProviderError


# FluidStack API base URL
API_BASE = "https://api.fluidstack.io/v1"


class FluidStackProvider(BaseProvider):
    """FluidStack cloud GPU provider.

    Creates on-demand GPU instances, runs tests via SSH, then terminates.

    Usage in kubix_ci.yml:
        - name: fluidstack-h100
          provider: fluidstack
          gpu: H100_SXM_80GB
          ssh_key_name: my-key  # optional

    GPU types: RTX_A6000_48GB, H100_SXM_80GB, A100_SXM_80GB, etc.
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self._instance_id: str | None = None
        self._ssh_provider: SSHProvider | None = None
        self._created_instance: bool = False
        self._api_key: str | None = None

    def connect(self) -> None:
        """Create a FluidStack instance and establish SSH connection."""
        # Get API key
        self._api_key = os.environ.get("FLUIDSTACK_API_KEY")
        if not self._api_key:
            raise ProviderError(
                self.name,
                "FLUIDSTACK_API_KEY environment variable not set. "
                "Get your key at: https://console.fluidstack.io/account/api-keys"
            )

        # Get or create SSH key
        ssh_key_name = self.config.ssh_key_name
        if not ssh_key_name:
            ssh_keys = self._get_ssh_keys()
            if ssh_keys:
                ssh_key_name = ssh_keys[0]["name"]
            else:
                raise ProviderError(
                    self.name,
                    "No SSH keys found in FluidStack account. "
                    "Add one at: https://console.fluidstack.io/account/ssh-keys"
                )

        # Create instance
        self._instance_id = self._create_instance(ssh_key_name)
        self._created_instance = True

        # Wait for instance to be ready
        instance = self._wait_for_ready()

        # Get SSH connection info
        ssh_host = instance.get("ip_address")
        if not ssh_host:
            raise ProviderError(self.name, "Instance has no IP address")

        # Connect via SSH
        ssh_config = SSHTarget(
            name=f"{self.name}-ssh",
            host=ssh_host,
            port=22,
            user="ubuntu",  # FluidStack uses ubuntu user
            gpu=self.config.gpu,
        )

        self._ssh_provider = SSHProvider(ssh_config)
        self._ssh_provider.connect()

    def _api_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
    ) -> dict | list:
        """Make an API request to FluidStack."""
        url = f"{API_BASE}/{endpoint}"
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.request(
                    method, url, headers=headers, json=json  # type: ignore[arg-type]
                )

                if response.status_code == 401:
                    raise ProviderError(self.name, "Invalid API key")

                if response.status_code == 403:
                    raise ProviderError(self.name, "API access forbidden")

                if response.status_code >= 400:
                    try:
                        error = response.json()
                        msg = error.get("message", error.get("error", response.text))
                    except Exception:
                        msg = response.text
                    raise ProviderError(self.name, f"API error: {msg}")

                return response.json()

        except httpx.RequestError as e:
            raise ProviderError(self.name, f"API request failed: {e}")

    def _get_ssh_keys(self) -> list[dict]:
        """Get list of SSH keys from account."""
        data = self._api_request("GET", "ssh-keys")
        if isinstance(data, list):
            return data
        return data.get("ssh_keys", [])

    def _create_instance(self, ssh_key_name: str) -> str:
        """Create a new instance."""
        payload = {
            "gpu_type": self.config.gpu,
            "name": f"kubix_ci-{self.config.name}-{int(time.time())}",
            "ssh_key": ssh_key_name,
        }

        try:
            data = self._api_request("POST", "instances", json=payload)

            instance_id = data.get("id")
            if not instance_id:
                raise ProviderError(self.name, "No instance ID returned")

            return instance_id

        except ProviderError as e:
            # Check for common errors
            error_msg = str(e).lower()
            if "capacity" in error_msg or "unavailable" in error_msg:
                raise ProviderError(
                    self.name,
                    f"No {self.config.gpu} instances available. "
                    "Try a different GPU type."
                )
            raise

    def _wait_for_ready(self, timeout: int = 300) -> dict:
        """Wait for instance to be running."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                data = self._api_request("GET", f"instances/{self._instance_id}")

                if isinstance(data, dict):
                    status = data.get("status", "").lower()

                    if status == "running":
                        # Wait for SSH to be ready
                        time.sleep(15)
                        return data

                    if status in ("terminated", "error", "failed"):
                        raise ProviderError(
                            self.name,
                            f"Instance entered {status} state"
                        )

            except ProviderError:
                raise
            except Exception:
                pass  # Instance might not be visible yet

            time.sleep(10)

        raise ProviderError(
            self.name,
            f"Instance did not become ready within {timeout} seconds"
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
        """Disconnect SSH and terminate the instance."""
        # Disconnect SSH first
        if self._ssh_provider:
            try:
                self._ssh_provider.disconnect()
            except Exception:
                pass
            self._ssh_provider = None

        # Terminate the instance if we created it
        if self._created_instance and self._instance_id:
            try:
                self._api_request("DELETE", f"instances/{self._instance_id}")
            except Exception:
                pass  # Best effort cleanup

            self._instance_id = None
            self._created_instance = False
