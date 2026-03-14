"""Lambda Labs cloud GPU provider.

Uses the Lambda Cloud REST API to create and manage GPU instances.
Requires: LAMBDA_API_KEY environment variable

Documentation: https://docs.lambda.ai/
API: https://docs.lambda.ai/public-cloud/cloud-api/
"""

from __future__ import annotations

import os
import time

import httpx

from kubix_ci.providers.base import BaseProvider
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.config import SSHTarget
from kubix_ci.exceptions import ProviderError


# Lambda Cloud API base URL
API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaLabsProvider(BaseProvider):
    """Lambda Labs cloud GPU provider.

    Creates on-demand GPU instances, runs tests via SSH, then terminates.

    Usage in kubix_ci.yml:
        - name: lambda-h100
          provider: lambdalabs
          gpu: gpu_1x_h100_pcie  # instance type name
          region: us-west-1      # optional
          ssh_key_name: my-key   # optional, uses first key if not specified

    Instance types: gpu_1x_h100_pcie, gpu_1x_a100, gpu_8x_a100, etc.
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self._instance_id: str | None = None
        self._ssh_provider: SSHProvider | None = None
        self._created_instance: bool = False
        self._api_key: str | None = None

    def connect(self) -> None:
        """Create a Lambda Labs instance and establish SSH connection."""
        # Get API key
        self._api_key = os.environ.get("LAMBDA_API_KEY")
        if not self._api_key:
            raise ProviderError(
                self.name,
                "LAMBDA_API_KEY environment variable not set. "
                "Get your key at: https://cloud.lambdalabs.com/api-keys"
            )

        # Get SSH key
        ssh_key_names = self._get_ssh_key_names()
        if not ssh_key_names:
            raise ProviderError(
                self.name,
                "No SSH keys found in Lambda Labs account. "
                "Add one at: https://cloud.lambdalabs.com/ssh-keys"
            )

        ssh_key_name = self.config.ssh_key_name or ssh_key_names[0]
        if ssh_key_name not in ssh_key_names:
            raise ProviderError(
                self.name,
                f"SSH key '{ssh_key_name}' not found. Available: {ssh_key_names}"
            )

        # Launch instance
        self._instance_id = self._launch_instance(ssh_key_name)
        self._created_instance = True

        # Wait for instance to be ready
        instance = self._wait_for_ready()

        # Get SSH connection info
        ssh_host = instance.get("ip")
        if not ssh_host:
            raise ProviderError(self.name, "Instance has no IP address")

        # Connect via SSH
        ssh_config = SSHTarget(
            name=f"{self.name}-ssh",
            host=ssh_host,
            port=22,
            user="ubuntu",  # Lambda uses ubuntu user
            gpu=self.config.gpu,
        )

        self._ssh_provider = SSHProvider(ssh_config)
        self._ssh_provider.connect()

    def _api_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
    ) -> dict:
        """Make an API request to Lambda Cloud."""
        url = f"{API_BASE}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.request(method, url, headers=headers, json=json)

                if response.status_code == 401:
                    raise ProviderError(self.name, "Invalid API key")

                if response.status_code == 403:
                    raise ProviderError(self.name, "API access forbidden")

                if response.status_code >= 400:
                    error = response.json().get("error", {})
                    msg = error.get("message", response.text)
                    raise ProviderError(self.name, f"API error: {msg}")

                return response.json()

        except httpx.RequestError as e:
            raise ProviderError(self.name, f"API request failed: {e}")

    def _get_ssh_key_names(self) -> list[str]:
        """Get list of SSH key names from account."""
        data = self._api_request("GET", "ssh-keys")
        return [key["name"] for key in data.get("data", [])]

    def _launch_instance(self, ssh_key_name: str) -> str:
        """Launch a new instance."""
        payload = {
            "instance_type_name": self.config.gpu,
            "ssh_key_names": [ssh_key_name],
            "name": f"kubix_ci-{self.config.name}-{int(time.time())}",
        }

        if self.config.region:
            payload["region_name"] = self.config.region

        try:
            data = self._api_request("POST", "instance-operations/launch", json=payload)
            instance_ids = data.get("data", {}).get("instance_ids", [])

            if not instance_ids:
                raise ProviderError(self.name, "No instance ID returned")

            return instance_ids[0]

        except ProviderError as e:
            # Check if it's a capacity issue
            if "insufficient capacity" in str(e).lower():
                raise ProviderError(
                    self.name,
                    f"No {self.config.gpu} instances available. "
                    "Try a different GPU type or region."
                )
            raise

    def _wait_for_ready(self, timeout: int = 600) -> dict:
        """Wait for instance to be active."""
        start = time.time()

        while time.time() - start < timeout:
            data = self._api_request("GET", f"instances/{self._instance_id}")
            instance = data.get("data", {})
            status = instance.get("status", "").lower()

            if status == "active":
                # Additional wait for SSH
                time.sleep(30)  # Lambda instances need time for SSH setup
                return instance

            if status in ("terminated", "unhealthy", "error"):
                raise ProviderError(
                    self.name,
                    f"Instance entered {status} state"
                )

            time.sleep(10)

        raise ProviderError(
            self.name,
            f"Instance did not become active within {timeout} seconds"
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
                self._api_request(
                    "POST",
                    "instance-operations/terminate",
                    json={"instance_ids": [self._instance_id]}
                )
            except Exception:
                pass  # Best effort cleanup

            self._instance_id = None
            self._created_instance = False
