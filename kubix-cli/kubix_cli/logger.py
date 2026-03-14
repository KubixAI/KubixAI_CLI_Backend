"""
Kubix CLI - Structured Logging

Provides structured, contextual logging with rich formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from rich.console import Console
from rich.logging import RichHandler


# Global console for rich output
console = Console()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """
    Setup structured logging with rich formatting.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        verbose: Enable verbose output
    """

    # Configure structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if sys.stderr.isatty():
        # Use rich formatting for console
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        # Use JSON for non-TTY (CI/CD, etc.)
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup standard library logging
    handlers = []

    # Console handler with rich formatting
    if sys.stderr.isatty():
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
            markup=True
        )
        handlers.append(rich_handler)
    else:
        # Plain handler for non-TTY
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(stream_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(message)s"
    )

    # Set log level for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding context to logs.

    Example:
        with LogContext(kernel_name="matmul", variant=3):
            logger.info("compiling kernel")
            # Logs will include kernel_name and variant
    """

    def __init__(self, **kwargs):
        self.context = kwargs

    def __enter__(self):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()
        return False


class PerformanceLogger:
    """
    Logger for tracking performance metrics.

    Example:
        perf = PerformanceLogger("kernel_optimization")
        with perf:
            # ... do work ...
        perf.log_metrics()
    """

    def __init__(self, operation: str, logger: Optional[structlog.BoundLogger] = None):
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
        self.metrics: Dict[str, Any] = {}

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            f"{self.operation} started",
            operation=self.operation,
            start_time=self.start_time.isoformat()
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if exc_type is None:
            self.logger.info(
                f"{self.operation} completed",
                operation=self.operation,
                duration_ms=duration_ms,
                success=True,
                **self.metrics
            )
        else:
            self.logger.error(
                f"{self.operation} failed",
                operation=self.operation,
                duration_ms=duration_ms,
                success=False,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.metrics
            )

        return False

    def add_metric(self, key: str, value: Any):
        """Add a metric to be logged."""
        self.metrics[key] = value

    def log_metrics(self):
        """Manually log metrics."""
        if self.start_time and self.end_time:
            duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
            self.logger.info(
                f"{self.operation} metrics",
                operation=self.operation,
                duration_ms=duration_ms,
                **self.metrics
            )


# Convenience functions for common log patterns
def log_optimization_start(
    logger: structlog.BoundLogger,
    kernel_name: str,
    variants: int
):
    """Log optimization start."""
    logger.info(
        "Starting kernel optimization",
        kernel_name=kernel_name,
        variants=variants,
        status="started"
    )


def log_optimization_complete(
    logger: structlog.BoundLogger,
    kernel_name: str,
    speedup: float,
    duration_ms: float
):
    """Log optimization completion."""
    logger.info(
        "Kernel optimization completed",
        kernel_name=kernel_name,
        speedup=speedup,
        duration_ms=duration_ms,
        status="completed"
    )


def log_compilation_error(
    logger: structlog.BoundLogger,
    kernel_name: str,
    errors: list
):
    """Log compilation errors."""
    logger.error(
        "Kernel compilation failed",
        kernel_name=kernel_name,
        error_count=len(errors),
        errors=errors[:5],  # First 5 errors
        status="failed"
    )


def log_api_request(
    logger: structlog.BoundLogger,
    model: str,
    prompt_length: int,
    variant: int
):
    """Log API request."""
    logger.debug(
        "Making API request",
        model=model,
        prompt_length=prompt_length,
        variant=variant
    )


def log_api_response(
    logger: structlog.BoundLogger,
    model: str,
    response_length: int,
    duration_ms: float
):
    """Log API response."""
    logger.debug(
        "Received API response",
        model=model,
        response_length=response_length,
        duration_ms=duration_ms
    )


def log_backend_info(
    logger: structlog.BoundLogger,
    backend_type: str,
    device_count: int,
    device_names: list
):
    """Log backend information."""
    logger.info(
        "GPU backend initialized",
        backend_type=backend_type,
        device_count=device_count,
        devices=device_names
    )


def log_cache_hit(
    logger: structlog.BoundLogger,
    kernel_name: str,
    cache_key: str
):
    """Log cache hit."""
    logger.debug(
        "Cache hit",
        kernel_name=kernel_name,
        cache_key=cache_key,
        cache_status="hit"
    )


def log_cache_miss(
    logger: structlog.BoundLogger,
    kernel_name: str,
    cache_key: str
):
    """Log cache miss."""
    logger.debug(
        "Cache miss",
        kernel_name=kernel_name,
        cache_key=cache_key,
        cache_status="miss"
    )
