"""
Resource Manager - GPU/CPU resource management and monitoring
Single Responsibility: Manage system resources efficiently
"""

import asyncio
import psutil
import torch
from contextlib import asynccontextmanager
from dataclasses import dataclass
from ..utils.logger import get_logger


@dataclass
class ResourceUsage:
    """System resource usage tracking"""

    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0


class ResourceManager:
    """Manage GPU/CPU resources and monitor system usage"""

    def __init__(self, max_concurrent_gpu_tasks: int = 2):
        self.max_concurrent_gpu_tasks = max_concurrent_gpu_tasks
        self.gpu_semaphore = asyncio.Semaphore(max_concurrent_gpu_tasks)
        self.logger = get_logger().bind_context(component="resource_manager")
        self.resource_usage = ResourceUsage()

        # Check GPU availability once
        self.gpu_available = torch.cuda.is_available()

    @asynccontextmanager
    async def acquire_gpu_resource(self):
        """Context manager for GPU resource acquisition"""
        async with self.gpu_semaphore:
            self.logger.debug("gpu_resource_acquired")

            try:
                yield
            finally:
                self.logger.debug("gpu_resource_released")

    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory allocation in MB"""
        if not self.gpu_available:
            return 0.0

        try:
            device = torch.cuda.current_device()
            return torch.cuda.memory_allocated(device) / 1024 / 1024
        except Exception:
            return 0.0

    def update_resource_usage(self):
        """Update resource usage statistics"""
        try:
            # System resources
            process = psutil.Process()
            self.resource_usage.cpu_percent = process.cpu_percent()

            # Memory usage
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            self.resource_usage.memory_mb = current_memory

            # Update peak memory
            if current_memory > self.resource_usage.peak_memory_mb:
                self.resource_usage.peak_memory_mb = current_memory

            # GPU memory
            self.resource_usage.gpu_memory_mb = self.get_gpu_memory_mb()

        except Exception as e:
            self.logger.warning("resource_monitoring_failed", error=str(e))

    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        self.update_resource_usage()
        return self.resource_usage

    def cleanup_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            self.logger.debug("gpu_memory_cleared")

    def log_resource_summary(self):
        """Log current resource usage summary"""
        usage = self.get_resource_usage()
        self.logger.info(
            "resource_summary",
            cpu_percent=round(usage.cpu_percent, 1),
            memory_mb=round(usage.memory_mb, 1),
            gpu_memory_mb=round(usage.gpu_memory_mb, 1),
            peak_memory_mb=round(usage.peak_memory_mb, 1),
        )
