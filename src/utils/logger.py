"""
Simple Structured Logging Utility
JSON-based logging for server environments
"""

import logging
import time
import sys
from contextlib import contextmanager
from typing import Optional

import structlog

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ECGLogger:
    """Simple structured logger for ECG Audio Analysis"""

    def __init__(self, name: str = "ecg-audio-analyzer", level: str = "INFO"):
        self.name = name

        # Configure structlog with minimal processors
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.WriteLoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # Set up standard logging
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.logger = structlog.get_logger(name)

    def bind_context(self, **kwargs):
        """Bind context variables for structured logging"""
        self.logger = self.logger.bind(**kwargs)
        return self

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def log_gpu_memory(self, stage: str):
        """Log current GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB

            self.logger.info(
                "gpu_memory_status",
                stage=stage,
                allocated_gb=round(allocated, 2),
                reserved_gb=round(reserved, 2),
            )
        except Exception as e:
            self.logger.warning("gpu_memory_log_failed", error=str(e))

    @contextmanager
    def timer(self, stage: str):
        """Simple timer context manager"""
        start_time = time.time()

        try:
            self.logger.info("stage_started", stage=stage)
            yield
            self.logger.info("stage_completed", stage=stage)
        except Exception as e:
            self.logger.error("stage_failed", stage=stage, error=str(e))
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(
                "stage_duration", stage=stage, duration_seconds=round(duration, 2)
            )


# Global logger instance
_global_logger: Optional[ECGLogger] = None


def get_logger(name: str = "ecg-audio-analyzer", **kwargs) -> ECGLogger:
    """Get or create global logger instance"""
    global _global_logger

    if _global_logger is None:
        _global_logger = ECGLogger(name=name, **kwargs)

    return _global_logger


def setup_logging(level: str = "INFO") -> ECGLogger:
    """Setup global logging configuration"""
    global _global_logger

    _global_logger = ECGLogger(level=level)
    return _global_logger
