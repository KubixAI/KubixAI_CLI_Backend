"""Brev.dev (NVIDIA Brev) provider.

Uses the brev CLI to manage instances, then SSHs in for file operations.
Requires: brev CLI installed and authenticated (`brev login`)

Documentation: https://docs.nvidia.com/brev/
"""

from __future__ import annotations

import json
import subprocess
import time

from kubix_ci.providers.base import BaseProvider
from kubix_ci.providers.ssh import SSHProvider
from kubix_ci.config import SSHTarget
from kubix_ci.exceptions import ProviderError


class BrevProvider(BaseProvider):
    """Brev.dev cloud GPU provider.

    Creates instances via `brev` CLI, connects via SSH.

    The workflow:
    1. Check for existing instance with matching name/GPU
    2. Create new instance if needed via `brev create`
    3. Wait for instance to be ready
    4. Extract SSH connection info
    5. Delegate to SSHProvider for file operations
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self._instance_name: str | None = None
        self._ssh_provider: SSHProvider | None = None
        self._created_instance: bool = False

    def connect(self) -> None:
        """Connect to Brev instance, creating if necessary."""
        # Verify brev CLI is available
        if not self._brev_available():
            raise ProviderError(
                self.name,
                "Brev CLI not found. Install with: "
                "curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh | bash && brev login"
            )

        # Check for existing instance
        existing = self._find_existing_instance()

        if existing:
            self._instance_name = existing["name"]
            status = existing.get("status", "").lower()

            if status != "running":
                # Start the instance
                self._brev_start(self._instance_name)
                self._wait_for_ready()
        else:
            # Create new instance
            self._instance_name = f"kubix_ci-{self.config.gpu.lower().replace(' ', '-')}"

            # Check if name already taken
            all_instances = self._brev_ls()
            if any(i["name"] == self._instance_name for i in all_instances):
                self._instance_name = f"{self._instance_name}-{int(time.time()) % 10000}"

            self._brev_create(self._instance_name)
            self._created_instance = True
            self._wait_for_ready()

        # Get SSH connection info
        ssh_info = self._get_ssh_info()

        # Create SSH provider
        ssh_target = SSHTarget(
            name=self.name,
            host=ssh_info["host"],
            user=ssh_info["user"],
            port=ssh_info.get("port", 22),
            gpu=self.gpu,
            key=ssh_info.get("key"),
        )

        self._ssh_provider = SSHProvider(ssh_target)
        self._ssh_provider.connect()

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
        """Disconnect SSH and optionally stop instance."""
        if self._ssh_provider:
            self._ssh_provider.disconnect()
            self._ssh_provider = None

        # Optionally stop instance to save costs
        # For now, leave it running for faster subsequent runs
        # TODO: Add config option to auto-stop

    def _brev_available(self) -> bool:
        """Check if brev CLI is installed."""
        try:
            result = subprocess.run(
                ["brev", "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _brev_ls(self) -> list[dict]:
        """List all brev instances."""
        try:
            result = subprocess.run(
                ["brev", "ls", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # Try without --json flag
                result = subprocess.run(
                    ["brev", "ls"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return self._parse_brev_ls_text(result.stdout)

            data: list[dict] = json.loads(result.stdout)
            return data

        except subprocess.SubprocessError as e:
            raise ProviderError(self.name, f"Failed to list instances: {e}")
        except json.JSONDecodeError:
            # Try parsing as text
            return self._parse_brev_ls_text(result.stdout)

    def _parse_brev_ls_text(self, output: str) -> list[dict]:
        """Parse text output from `brev ls` into instance list."""
        instances = []
        lines = output.strip().split('\n')

        for line in lines:
            # Skip headers and empty lines
            if not line.strip() or 'NAME' in line.upper():
                continue

            # Parse columns (format varies, try common patterns)
            parts = line.split()
            if len(parts) >= 2:
                instances.append({
                    "name": parts[0],
                    "status": parts[1] if len(parts) > 1 else "unknown",
                })

        return instances

    def _find_existing_instance(self) -> dict | None:
        """Find an existing instance matching our config."""
        instances = self._brev_ls()

        # Look for instance with matching name pattern
        config_name = self.config.instance_name
        if config_name:
            for inst in instances:
                if inst["name"] == config_name:
                    return inst

        # Look for any kubix_ci instance with matching GPU
        gpu_lower = self.config.gpu.lower().replace(' ', '-')
        for inst in instances:
            name = inst["name"].lower()
            if name.startswith("kubix_ci-") and gpu_lower in name:
                return inst

        return None

    def _brev_create(self, name: str) -> None:
        """Create a new brev instance."""
        try:
            # Build create command
            cmd = ["brev", "create", name]

            # Add GPU type if brev supports it
            # Note: brev create syntax may vary
            # --gpu flag may not be supported in all versions

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise ProviderError(
                    self.name,
                    f"Failed to create instance '{name}': {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise ProviderError(self.name, "Timed out creating instance")
        except subprocess.SubprocessError as e:
            raise ProviderError(self.name, f"Failed to create instance: {e}")

    def _brev_start(self, name: str) -> None:
        """Start a stopped brev instance."""
        try:
            result = subprocess.run(
                ["brev", "start", name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise ProviderError(
                    self.name,
                    f"Failed to start instance '{name}': {result.stderr}"
                )

        except subprocess.SubprocessError as e:
            raise ProviderError(self.name, f"Failed to start instance: {e}")

    def _wait_for_ready(self, timeout: int = 300) -> None:
        """Wait for instance to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            instances = self._brev_ls()

            for inst in instances:
                if inst["name"] == self._instance_name:
                    status = inst.get("status", "").lower()
                    if status == "running" or status == "ready":
                        return

            time.sleep(5)

        raise ProviderError(
            self.name,
            f"Timed out waiting for instance '{self._instance_name}' to be ready"
        )

    def _get_ssh_info(self) -> dict:
        """Extract SSH connection info for the instance."""
        if not self._instance_name:
            raise ProviderError(self.name, "No instance name set")

        # Try `brev ssh-config` or similar command
        try:
            # Get SSH config from brev
            result = subprocess.run(
                ["brev", "ssh-config", self._instance_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return self._parse_ssh_config(result.stdout)

        except subprocess.SubprocessError:
            pass

        # Fallback: Try to get host from `brev ls` output or environment
        # Brev instances are typically accessible via <name>.brev.dev
        return {
            "host": f"{self._instance_name}.brev.dev",
            "user": "ubuntu",
            "port": 22,
        }

    def _parse_ssh_config(self, config_output: str) -> dict:
        """Parse SSH config output into connection dict."""
        info = {
            "host": "",
            "user": "ubuntu",
            "port": 22,
        }

        for line in config_output.split('\n'):
            line = line.strip().lower()

            if line.startswith("hostname"):
                info["host"] = line.split()[-1]
            elif line.startswith("user"):
                info["user"] = line.split()[-1]
            elif line.startswith("port"):
                try:
                    info["port"] = int(line.split()[-1])
                except ValueError:
                    pass
            elif line.startswith("identityfile"):
                info["key"] = line.split()[-1]

        return info
