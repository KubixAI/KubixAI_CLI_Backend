"""Custom exception hierarchy for kubix_ci."""


class KUBIXCIError(Exception):
    """Base exception for kubix_ci."""

    pass


class ConfigError(KUBIXCIError):
    """Configuration file errors."""

    pass


class ProviderError(KUBIXCIError):
    """Provider connection/execution errors."""

    def __init__(self, provider_name: str, message: str):
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class CompilationError(KUBIXCIError):
    """Remote CUDA compilation failures."""

    def __init__(self, target_name: str, stderr: str):
        self.target_name = target_name
        self.stderr = stderr
        super().__init__(f"Compilation failed on {target_name}: {stderr[:500]}")


class ExecutionError(KUBIXCIError):
    """Remote kernel execution failures."""

    def __init__(self, target_name: str, message: str):
        self.target_name = target_name
        super().__init__(f"Execution failed on {target_name}: {message}")


class ConnectionError(KUBIXCIError):
    """SSH/provider connection failures."""

    def __init__(self, target_name: str, message: str):
        self.target_name = target_name
        super().__init__(f"Connection failed to {target_name}: {message}")


class TimeoutError(KUBIXCIError):
    """Operation timed out."""

    def __init__(self, target_name: str, timeout_seconds: int, operation: str = "operation"):
        self.target_name = target_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{operation.capitalize()} timed out on {target_name} after {timeout_seconds}s")
