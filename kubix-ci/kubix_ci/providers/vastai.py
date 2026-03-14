"""Vast.ai cloud GPU provider.

Uses the Vast.ai Python SDK to rent GPU instances.
Requires: pip install vastai-sdk
Environment: VASTAI_API_KEY

Documentation: https://docs.vast.ai/
SDK: https://docs.vast.ai/sdk/python/quickstart
"""

from __future__ import annotations

import os
import time

from kubix_ci.providers.base import BaseProvider
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.config import SSHTarget
from kubix_ci.exceptions import ProviderError


class VastAIProvider(BaseProvider):
    """Vast.ai GPU marketplace provider.

    Searches for available GPUs, rents the best match, runs tests, then destroys.

    Usage in kubix_ci.yml:
        - name: vastai-4090
          provider: vastai
          gpu: RTX_4090
          max_price: 0.50  # max $/hour
          min_gpu_ram: 24  # minimum GPU RAM in GB
          image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self._instance_id: int | None = None
        self._ssh_provider: SSHProvider | None = None
        self._created_instance: bool = False

    def connect(self) -> None:
        """Find and rent a Vast.ai GPU instance."""
        try:
            from vastai_sdk import VastAI
        except ImportError:
            raise ProviderError(
                self.name,
                "Vast.ai SDK not installed. Run: pip install vastai-sdk"
            )

        # Get API key
        api_key = os.environ.get("VASTAI_API_KEY")
        if not api_key:
            raise ProviderError(
                self.name,
                "VASTAI_API_KEY environment variable not set. "
                "Get your key at: https://vast.ai/console/account/"
            )

        vast = VastAI(api_key=api_key)

        # Build search query
        query_parts = [
            f"gpu_name={self.config.gpu}",
            "rented=False",
            "rentable=True",
            "verified=True",  # Only verified hosts
        ]

        if self.config.min_gpu_ram > 0:
            query_parts.append(f"gpu_ram>={self.config.min_gpu_ram}")

        if self.config.max_price:
            query_parts.append(f"dph<={self.config.max_price}")

        query = " ".join(query_parts)

        # Search for offers
        try:
            offers = vast.search_offers(query=query)

            if not offers or len(offers) == 0:
                raise ProviderError(
                    self.name,
                    f"No {self.config.gpu} instances available matching criteria. "
                    f"Query: {query}"
                )

            # Sort by price and pick cheapest
            if isinstance(offers, list):
                offers.sort(key=lambda x: x.get("dph_total", float("inf")))
                best_offer = offers[0]
            else:
                best_offer = offers

            offer_id = best_offer.get("id")
            if not offer_id:
                raise ProviderError(self.name, "Offer has no ID")

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.name, f"Failed to search offers: {e}")

        # Launch instance
        try:
            result = vast.launch_instance(
                num_gpus="1",
                gpu_name=self.config.gpu,
                image=self.config.image,
            )

            # Result might be a dict or have an id attribute
            if isinstance(result, dict):
                self._instance_id = result.get("id") or result.get("new_contract")
            else:
                self._instance_id = getattr(result, "id", None)

            if not self._instance_id:
                raise ProviderError(self.name, "No instance ID returned")

            self._created_instance = True

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.name, f"Failed to launch instance: {e}")

        # Wait for instance to be ready
        ssh_info = self._wait_for_ready(vast)

        # Connect via SSH
        ssh_config = SSHTarget(
            name=f"{self.name}-ssh",
            host=ssh_info["host"],
            port=ssh_info["port"],
            user="root",
            gpu=self.config.gpu,
        )

        self._ssh_provider = SSHProvider(ssh_config)
        self._ssh_provider.connect()

    def _wait_for_ready(self, vast, timeout: int = 300) -> dict:
        """Wait for instance to be running and return SSH info."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                # Get instance status
                instances = vast.show_instances()

                if not instances:
                    time.sleep(10)
                    continue

                for instance in instances:
                    inst_id = instance.get("id")
                    if inst_id == self._instance_id:
                        status = instance.get("actual_status", "").lower()

                        if status == "running":
                            # Extract SSH info
                            ssh_host = instance.get("ssh_host")
                            ssh_port = instance.get("ssh_port", 22)

                            if ssh_host:
                                time.sleep(10)  # Extra time for SSH to start
                                return {"host": ssh_host, "port": int(ssh_port)}

                        if status in ("exited", "error", "destroyed"):
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
        """Disconnect SSH and destroy the instance."""
        # Disconnect SSH first
        if self._ssh_provider:
            try:
                self._ssh_provider.disconnect()
            except Exception:
                pass
            self._ssh_provider = None

        # Destroy the instance if we created it
        if self._created_instance and self._instance_id:
            try:
                from vastai_sdk import VastAI

                api_key = os.environ.get("VASTAI_API_KEY")
                if api_key:
                    vast = VastAI(api_key=api_key)
                    vast.destroy_instance(ID=self._instance_id)
            except Exception:
                pass  # Best effort cleanup

            self._instance_id = None
            self._created_instance = False
