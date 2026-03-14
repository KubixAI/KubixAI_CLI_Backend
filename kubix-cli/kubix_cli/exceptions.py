"""
Kubix CLI - Exception Hierarchy

Comprehensive exception system for better error handling and debugging.
"""

from typing import List, Optional


class KubixError(Exception):
    """Base exception for all Kubix CLI errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================================================
# Backend Errors
# ============================================================================


class BackendError(KubixError):
    """Base class for GPU backend errors."""
    pass


class BackendNotAvailableError(BackendError):
    """Requested GPU backend is not available on this system."""

    def __init__(self, backend_name: str, reason: Optional[str] = None):
        message = f"Backend '{backend_name}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"backend": backend_name, "reason": reason})


class BackendInitializationError(BackendError):
    """Failed to initialize GPU backend."""

    def __init__(self, backend_name: str, error: Exception):
        message = f"Failed to initialize backend '{backend_name}'"
        super().__init__(message, {"backend": backend_name, "error": str(error)})


class DeviceNotFoundError(BackendError):
    """No GPU devices found for the specified backend."""

    def __init__(self, backend_name: str):
        message = f"No GPU devices found for backend '{backend_name}'"
        super().__init__(message, {"backend": backend_name})


# ============================================================================
# Compilation Errors
# ============================================================================


class CompilationError(KubixError):
    """Kernel compilation failed."""

    def __init__(self, code: str, errors: List[str], file_path: Optional[str] = None):
        self.code = code
        self.errors = errors
        self.file_path = file_path

        message = f"Kernel compilation failed with {len(errors)} error(s)"
        if file_path:
            message += f" in {file_path}"

        super().__init__(message, {"error_count": len(errors), "file": file_path})

    def get_formatted_errors(self) -> str:
        """Get formatted error messages."""
        return "\n".join(f"  - {error}" for error in self.errors)


class KernelSyntaxError(CompilationError):
    """Kernel has syntax errors."""
    pass


class KernelLinkError(CompilationError):
    """Failed to link kernel modules."""
    pass


class PTXGenerationError(CompilationError):
    """Failed to generate PTX code."""
    pass


# ============================================================================
# API Errors
# ============================================================================


class APIError(KubixError):
    """Base class for API-related errors."""
    pass


class OpenRouterAPIError(APIError):
    """OpenRouter API error."""

    def __init__(self, status_code: int, message: str, response_body: Optional[str] = None):
        self.status_code = status_code
        self.response_body = response_body

        error_message = f"OpenRouter API error {status_code}: {message}"
        super().__init__(error_message, {"status_code": status_code})


class APIKeyError(APIError):
    """Invalid or missing API key."""

    def __init__(self, message: str = "Invalid or missing OpenRouter API key"):
        super().__init__(message)


class APIRateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        message = "API rate limit exceeded"
        details = {}
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            details["retry_after"] = retry_after
        super().__init__(message, details)


class APITimeoutError(APIError):
    """API request timed out."""

    def __init__(self, timeout_seconds: int):
        message = f"API request timed out after {timeout_seconds} seconds"
        super().__init__(message, {"timeout": timeout_seconds})


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(KubixError):
    """Input validation failed."""
    pass


class ConfigurationError(ValidationError):
    """Invalid configuration."""

    def __init__(self, field: str, value: any, reason: str):
        message = f"Invalid configuration for '{field}': {reason}"
        super().__init__(message, {"field": field, "value": value})


class ConstraintViolationError(ValidationError):
    """Kernel constraints violation."""

    def __init__(self, constraint: str, value: any, limit: any):
        message = f"Constraint violation: {constraint} = {value} exceeds limit {limit}"
        super().__init__(message, {"constraint": constraint, "value": value, "limit": limit})


class InvalidKernelError(ValidationError):
    """Kernel code is invalid."""

    def __init__(self, reason: str):
        message = f"Invalid kernel: {reason}"
        super().__init__(message)


# ============================================================================
# Benchmarking Errors
# ============================================================================


class BenchmarkError(KubixError):
    """Base class for benchmarking errors."""
    pass


class BenchmarkTimeoutError(BenchmarkError):
    """Benchmark took too long to complete."""

    def __init__(self, timeout_seconds: int):
        message = f"Benchmark timed out after {timeout_seconds} seconds"
        super().__init__(message, {"timeout": timeout_seconds})


class BenchmarkFailedError(BenchmarkError):
    """Benchmark execution failed."""

    def __init__(self, kernel_name: str, error: Exception):
        message = f"Benchmark failed for kernel '{kernel_name}'"
        super().__init__(message, {"kernel": kernel_name, "error": str(error)})


class MemoryError(BenchmarkError):
    """Insufficient memory for benchmark."""

    def __init__(self, required_mb: float, available_mb: float):
        message = f"Insufficient memory: required {required_mb:.1f} MB, available {available_mb:.1f} MB"
        super().__init__(message, {"required_mb": required_mb, "available_mb": available_mb})


# ============================================================================
# Cache Errors
# ============================================================================


class CacheError(KubixError):
    """Base class for cache-related errors."""
    pass


class CacheCorruptedError(CacheError):
    """Cache data is corrupted."""

    def __init__(self, cache_path: str):
        message = f"Cache is corrupted: {cache_path}"
        super().__init__(message, {"cache_path": cache_path})


class CacheWriteError(CacheError):
    """Failed to write to cache."""

    def __init__(self, cache_path: str, error: Exception):
        message = f"Failed to write to cache: {cache_path}"
        super().__init__(message, {"cache_path": cache_path, "error": str(error)})


# ============================================================================
# Optimization Errors
# ============================================================================


class OptimizationError(KubixError):
    """Base class for optimization errors."""
    pass


class NoValidVariantsError(OptimizationError):
    """No valid optimization variants were generated."""

    def __init__(self, attempted: int):
        message = f"No valid variants generated from {attempted} attempts"
        super().__init__(message, {"attempted": attempted})


class OptimizationTimeoutError(OptimizationError):
    """Optimization process timed out."""

    def __init__(self, timeout_seconds: int):
        message = f"Optimization timed out after {timeout_seconds} seconds"
        super().__init__(message, {"timeout": timeout_seconds})


# ============================================================================
# Plugin Errors
# ============================================================================


class PluginError(KubixError):
    """Base class for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Plugin not found."""

    def __init__(self, plugin_name: str):
        message = f"Plugin '{plugin_name}' not found"
        super().__init__(message, {"plugin": plugin_name})


class PluginLoadError(PluginError):
    """Failed to load plugin."""

    def __init__(self, plugin_name: str, error: Exception):
        message = f"Failed to load plugin '{plugin_name}'"
        super().__init__(message, {"plugin": plugin_name, "error": str(error)})
