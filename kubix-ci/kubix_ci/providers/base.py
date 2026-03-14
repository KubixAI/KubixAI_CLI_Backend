"""Abstract base provider interface for kubix_ci."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class ExecutionResult:
    """Result from running a kernel on a remote target."""

    target_name: str
    gpu_name: str
    status: str  # "success", "compile_error", "runtime_error", "timeout", "connection_error"

    # Output
    stdout: str = ""
    stderr: str = ""

    # Timing (from CUDA events)
    kernel_time_ms: float | None = None  # median time
    mean_time_ms: float | None = None
    min_time_ms: float | None = None
    max_time_ms: float | None = None
    all_times: list[float] = field(default_factory=list)

    # Build info
    compile_time_s: float | None = None
    nvcc_version: str | None = None

    # GPU info (from runtime)
    device_name: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    compute_capability: str | None = None

    # Error details
    error: str | None = None


class BaseProvider(ABC):
    """Abstract interface for GPU providers.

    All providers must implement these methods:
    - connect(): Establish connection to target
    - upload(): Copy file to remote
    - execute(): Run command remotely
    - download(): Copy file from remote
    - disconnect(): Clean up connection

    The compile_and_run() method provides the shared workflow
    for compiling and running CUDA kernels with timing.
    """

    def __init__(self, target_config):
        self.config = target_config
        self.name = target_config.name
        self.gpu = target_config.gpu

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the target machine."""
        pass

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote machine via SFTP."""
        pass

    @abstractmethod
    def execute(self, command: str, timeout: int = 120) -> tuple[str, str, int]:
        """Execute a command remotely.

        Returns:
            tuple: (stdout, stderr, exit_code)
        """
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote machine."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and clean up resources."""
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def compile_and_run(
        self,
        wrapped_source: str,
        nvcc_flags: list[str],
        timeout: int = 120,
    ) -> ExecutionResult:
        """Full workflow: upload, compile, run, parse results.

        This is the shared implementation used by all providers.

        Args:
            wrapped_source: CUDA source with timing wrapper
            nvcc_flags: Additional nvcc compilation flags
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with timing and status
        """
        from kubix_ci.timing import parse_timing_output

        result = ExecutionResult(
            target_name=self.name,
            gpu_name=self.gpu,
            status="error",
        )

        # Create unique temp directory on remote
        run_id = str(uuid.uuid4())[:8]
        remote_dir = f"/tmp/kubix_ci_{run_id}"
        remote_source = f"{remote_dir}/kernel.cu"
        remote_binary = f"{remote_dir}/kernel"

        try:
            # Create temp directory
            stdout, stderr, code = self.execute(f"mkdir -p {remote_dir}")
            if code != 0:
                result.error = f"Failed to create temp directory: {stderr}"
                return result

            # Write source to temp file locally, then upload
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(wrapped_source)
                local_temp = f.name

            try:
                self.upload(local_temp, remote_source)
            finally:
                os.unlink(local_temp)

            # Get nvcc version
            stdout, stderr, code = self.execute("nvcc --version")
            if code == 0:
                # Parse version from output
                for line in stdout.split('\n'):
                    if 'release' in line.lower():
                        result.nvcc_version = line.strip()
                        break

            # Detect GPU architecture for optimal compilation
            arch_flag = self._detect_gpu_arch()

            # Compile
            flags_str = ' '.join(nvcc_flags)
            compile_cmd = f"nvcc {flags_str} {arch_flag} -o {remote_binary} {remote_source} 2>&1"

            compile_start = time.time()
            stdout, stderr, code = self.execute(compile_cmd, timeout=timeout)
            compile_time = time.time() - compile_start
            result.compile_time_s = compile_time

            if code != 0:
                result.status = "compile_error"
                result.error = stdout + stderr
                result.stderr = stdout + stderr
                return result

            # Run the binary
            stdout, stderr, code = self.execute(remote_binary, timeout=timeout)
            result.stdout = stdout
            result.stderr = stderr

            if code != 0:
                result.status = "runtime_error"
                result.error = f"Exit code {code}: {stderr}"
                # Still try to parse any output
                timing = parse_timing_output(stdout)
                if timing.error:
                    result.error = timing.error
                return result

            # Parse timing output
            timing = parse_timing_output(stdout)

            result.status = timing.status
            result.error = timing.error
            result.device_name = timing.device_name
            result.compute_capability = timing.compute_capability
            result.cuda_version = timing.runtime_version
            result.driver_version = timing.driver_version
            result.kernel_time_ms = timing.median_ms
            result.mean_time_ms = timing.mean_ms
            result.min_time_ms = timing.min_ms
            result.max_time_ms = timing.max_ms
            result.all_times = timing.all_times or []

            # Update GPU name if we got it from the device
            if timing.device_name:
                result.gpu_name = timing.device_name

        except Exception as e:
            result.status = "error"
            result.error = str(e)

        finally:
            # Cleanup temp directory
            try:
                self.execute(f"rm -rf {remote_dir}")
            except Exception:
                pass  # Best effort cleanup

        return result

    def _detect_gpu_arch(self) -> str:
        """Detect GPU compute capability and return appropriate nvcc arch flag."""
        try:
            stdout, stderr, code = self.execute(
                "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1"
            )
            if code == 0 and stdout.strip():
                # Format: "8.9" -> "sm_89"
                cap = stdout.strip().replace('.', '')
                return f"-arch=sm_{cap}"
        except Exception:
            pass

        # Fallback to a reasonable default (Ampere)
        return "-arch=sm_80"
