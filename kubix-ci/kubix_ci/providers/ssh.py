"""Direct SSH provider using paramiko."""

from __future__ import annotations

from pathlib import Path
import socket

import paramiko

from kubix_ci.providers.base import BaseProvider
from kubix_ci.exceptions import ConnectionError, TimeoutError


class SSHProvider(BaseProvider):
    """Direct SSH connection to GPU machines using paramiko.

    Requires:
    - SSH access to target machine
    - CUDA toolkit (nvcc) installed on target
    - NVIDIA GPU with drivers installed
    """

    def __init__(self, target_config):
        super().__init__(target_config)
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None

    def connect(self) -> None:
        """Establish SSH connection to the target."""
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.user,
            "timeout": 30,  # Connection timeout
        }

        # Handle SSH key
        if self.config.key:
            key_path = Path(self.config.key).expanduser()
            if not key_path.exists():
                raise ConnectionError(
                    self.name,
                    f"SSH key not found: {key_path}"
                )
            connect_kwargs["key_filename"] = str(key_path)

        try:
            self.client.connect(**connect_kwargs)
            self.sftp = self.client.open_sftp()
        except paramiko.AuthenticationException as e:
            raise ConnectionError(
                self.name,
                f"Authentication failed for {self.config.user}@{self.config.host}: {e}"
            )
        except paramiko.SSHException as e:
            raise ConnectionError(
                self.name,
                f"SSH error connecting to {self.config.host}: {e}"
            )
        except socket.timeout:
            raise ConnectionError(
                self.name,
                f"Connection timed out to {self.config.host}:{self.config.port}"
            )
        except socket.error as e:
            raise ConnectionError(
                self.name,
                f"Network error connecting to {self.config.host}: {e}"
            )

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote machine via SFTP."""
        if not self.sftp:
            raise ConnectionError(self.name, "Not connected")

        try:
            self.sftp.put(local_path, remote_path)
        except IOError as e:
            raise ConnectionError(
                self.name,
                f"Failed to upload {local_path} to {remote_path}: {e}"
            )

    def execute(self, command: str, timeout: int = 120) -> tuple[str, str, int]:
        """Execute a command remotely.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds

        Returns:
            tuple: (stdout, stderr, exit_code)
        """
        if not self.client:
            raise ConnectionError(self.name, "Not connected")

        try:
            stdin, stdout, stderr = self.client.exec_command(
                command,
                timeout=timeout,
            )

            # Wait for command to complete
            exit_code = stdout.channel.recv_exit_status()

            stdout_str = stdout.read().decode('utf-8', errors='replace')
            stderr_str = stderr.read().decode('utf-8', errors='replace')

            return stdout_str, stderr_str, exit_code

        except socket.timeout:
            raise TimeoutError(
                self.name,
                timeout,
                "Command execution"
            )
        except paramiko.SSHException as e:
            raise ConnectionError(
                self.name,
                f"SSH error during command execution: {e}"
            )

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote machine via SFTP."""
        if not self.sftp:
            raise ConnectionError(self.name, "Not connected")

        try:
            self.sftp.get(remote_path, local_path)
        except IOError as e:
            raise ConnectionError(
                self.name,
                f"Failed to download {remote_path} to {local_path}: {e}"
            )

    def disconnect(self) -> None:
        """Close SSH connection and clean up."""
        if self.sftp:
            try:
                self.sftp.close()
            except Exception:
                pass
            self.sftp = None

        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
