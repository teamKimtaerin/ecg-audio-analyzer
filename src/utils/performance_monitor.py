"""
Performance Monitoring Utilities
- Refactored for better separation of concerns, testability, and maintainability.
"""

import time
import threading
import psutil
import platform
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from collections import deque

# Assume these are correctly implemented in your project structure
from .logger import get_logger
from .gpu_optimizer import get_gpu_optimizer, GPUMemoryStats


# --- Data Structures (No major changes, already well-designed) ---


@dataclass
class SystemResources:
    """System resource snapshot."""

    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    gpu_stats: Optional[GPUMemoryStats] = None

    @property
    def memory_available_mb(self) -> float:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def disk_available_gb(self) -> float:
        return self.disk_total_gb - self.disk_used_gb


@dataclass
class ProcessingBenchmark:
    """Processing benchmark results."""

    operation_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    items_processed: int
    throughput_items_per_second: float
    cpu_usage_during: float
    memory_peak_mb: float
    gpu_memory_peak_mb: Optional[float]
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation_name,
            "duration_seconds": self.duration_seconds,
            "items_processed": self.items_processed,
            "throughput_items_per_second": self.throughput_items_per_second,
            "cpu_usage_during": self.cpu_usage_during,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "success": self.success,
            "error": self.error_message,
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""

    system_info: Dict[str, Any]
    resource_utilization: Dict[str, Any]
    benchmarks: List[ProcessingBenchmark]
    bottlenecks: List[str]
    recommendations: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_info": self.system_info,
            "resource_utilization": self.resource_utilization,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "optimization_opportunities": self.optimization_opportunities,
            "timestamp": self.timestamp,
        }

    def save_to_file(self, output_path: Path):
        """Save performance report to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# --- NEW: Centralized Configuration ---


@dataclass
class ProfilerConfig:
    """Configuration for performance analysis thresholds."""

    cpu_warning_threshold: float = 80.0
    cpu_severe_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_severe_threshold: float = 95.0
    gpu_memory_warning_threshold: float = 90.0
    gpu_memory_severe_threshold: float = 95.0
    disk_space_warning_threshold: float = 90.0
    low_throughput_threshold: float = 0.1
    low_gpu_util_threshold: float = 50.0
    underutilized_gpu_mem_threshold: float = 30.0


# --- Core Components: Refactored for Single Responsibility ---


class ResourceTracker:
    """Tracks system resources over time in a background thread."""

    def __init__(self, max_samples: int = 1000):
        self.samples: deque = deque(maxlen=max_samples)
        self.logger = get_logger().bind_context(component="resource_tracker")
        self._tracking = False
        self._track_thread: Optional[threading.Thread] = None

    def start_tracking(self, interval: float = 1.0):
        if self._tracking:
            return
        self._tracking = True
        self._track_thread = threading.Thread(
            target=self._tracking_loop, args=(interval,), daemon=True
        )
        self._track_thread.start()
        self.logger.info("Resource tracking started.", interval=interval)

    def stop_tracking(self):
        self._tracking = False
        if self._track_thread and self._track_thread.is_alive():
            self._track_thread.join(timeout=2.0)
        self.logger.info("Resource tracking stopped.")

    def _tracking_loop(self, interval: float):
        while self._tracking:
            try:
                snapshot = self.get_resource_snapshot()
                self.samples.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                self.logger.warning("Resource tracking error.", error=str(e))
                time.sleep(interval)  # Prevent rapid-fire errors

    @staticmethod
    def get_resource_snapshot() -> SystemResources:
        """Gets a single snapshot of current system resources."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net = psutil.net_io_counters()

        gpu_stats = None
        try:
            # Encapsulate optional dependency
            gpu_optimizer = get_gpu_optimizer()
            gpu_stats = gpu_optimizer.monitor.get_current_stats(0)
        except Exception:
            pass  # Gracefully handle cases where GPU is not available or nvidia-smi fails

        return SystemResources(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_used_mb=mem.used / (1024**2),
            memory_total_mb=mem.total / (1024**2),
            memory_percent=mem.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            disk_percent=disk.percent,
            network_bytes_sent=net.bytes_sent,
            network_bytes_recv=net.bytes_recv,
            process_count=len(psutil.pids()),
            thread_count=psutil.Process().num_threads(),
            gpu_stats=gpu_stats,
        )

    def get_resource_summary(self, last_n_samples: int = 60) -> Dict[str, Any]:
        if not self.samples:
            return {}

        recent_samples = list(self.samples)[-last_n_samples:]
        if not recent_samples:
            return {}

        def _calculate_stats(data: List[float]) -> Tuple[float, float]:
            return (sum(data) / len(data), max(data)) if data else (0.0, 0.0)

        avg_cpu, peak_cpu = _calculate_stats([s.cpu_percent for s in recent_samples])
        avg_mem, peak_mem = _calculate_stats([s.memory_percent for s in recent_samples])
        avg_disk, _ = _calculate_stats([s.disk_percent for s in recent_samples])

        gpu_samples = [s.gpu_stats for s in recent_samples if s.gpu_stats]
        avg_gpu_mem, peak_gpu_mem = (0.0, 0.0)
        avg_gpu_util = 0.0
        if gpu_samples:
            avg_gpu_mem, peak_gpu_mem = _calculate_stats(
                [s.utilization_percent for s in gpu_samples]
            )
            avg_gpu_util, _ = _calculate_stats(
                [s.allocated_mb / s.total_mb * 100 for s in gpu_samples]
            )

        return {
            "cpu": {"average_percent": avg_cpu, "peak_percent": peak_cpu},
            "memory": {
                "average_percent": avg_mem,
                "peak_percent": peak_mem,
                "total_mb": recent_samples[-1].memory_total_mb,
            },
            "disk": {
                "average_percent": avg_disk,
                "total_gb": recent_samples[-1].disk_total_gb,
            },
            "gpu": {
                "available": bool(gpu_samples),
                "average_memory_percent": avg_gpu_mem,
                "peak_memory_percent": peak_gpu_mem,
                "average_utilization": avg_gpu_util,
            },
            "samples_analyzed": len(recent_samples),
        }


class PerformanceAnalyzer:
    """Analyzes collected data to identify bottlenecks and generate recommendations."""

    def __init__(self, config: ProfilerConfig, system_info: Dict[str, Any]):
        self.config = config
        self.system_info = system_info

    def run_analysis(
        self, benchmarks: List[ProcessingBenchmark], resource_summary: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Runs all analysis steps and returns results."""
        bottlenecks = self._analyze_bottlenecks(benchmarks, resource_summary)
        recommendations = self._generate_recommendations(resource_summary, bottlenecks)
        opportunities = self._identify_optimization_opportunities(benchmarks)
        return bottlenecks, recommendations, opportunities

    def _analyze_bottlenecks(
        self, benchmarks: List[ProcessingBenchmark], res: Dict[str, Any]
    ) -> List[str]:
        bottlenecks = []
        # CPU
        if res.get("cpu", {}).get("peak_percent", 0) > self.config.cpu_severe_threshold:
            bottlenecks.append(
                f"Severe CPU bottleneck detected (peak > {self.config.cpu_severe_threshold}%)"
            )
        elif (
            res.get("cpu", {}).get("average_percent", 0)
            > self.config.cpu_warning_threshold
        ):
            bottlenecks.append(
                f"High CPU utilization (avg > {self.config.cpu_warning_threshold}%)"
            )
        # Memory
        if (
            res.get("memory", {}).get("peak_percent", 0)
            > self.config.memory_severe_threshold
        ):
            bottlenecks.append(
                f"Severe memory bottleneck (peak > {self.config.memory_severe_threshold}%)"
            )
        elif (
            res.get("memory", {}).get("average_percent", 0)
            > self.config.memory_warning_threshold
        ):
            bottlenecks.append(
                f"High memory utilization (avg > {self.config.memory_warning_threshold}%)"
            )
        # GPU
        if res.get("gpu", {}).get("available"):
            if (
                res["gpu"].get("peak_memory_percent", 0)
                > self.config.gpu_memory_severe_threshold
            ):
                bottlenecks.append(
                    f"Severe GPU memory bottleneck (peak > {self.config.gpu_memory_severe_threshold}%)"
                )
            elif (
                res["gpu"].get("average_memory_percent", 0)
                > self.config.gpu_memory_warning_threshold
            ):
                bottlenecks.append(
                    f"High GPU memory utilization (avg > {self.config.gpu_memory_warning_threshold}%)"
                )
        # Disk
        if (
            res.get("disk", {}).get("average_percent", 0)
            > self.config.disk_space_warning_threshold
        ):
            bottlenecks.append(
                f"High disk space usage (> {self.config.disk_space_warning_threshold}%)"
            )
        # Throughput
        slow_ops = [
            b
            for b in benchmarks
            if b.success
            and b.throughput_items_per_second < self.config.low_throughput_threshold
        ]
        if slow_ops:
            bottlenecks.append(
                f"Slow processing detected in {len(slow_ops)} operation(s)."
            )
        return bottlenecks

    def _generate_recommendations(
        self, res: Dict[str, Any], bottlenecks: List[str]
    ) -> List[str]:
        recommendations = []
        if any("CPU" in b for b in bottlenecks):
            recommendations.extend(
                [
                    "Consider optimizing CPU-bound tasks, increasing parallel workers, or upgrading the CPU.",
                    "Explore batch processing to improve CPU efficiency for repetitive tasks.",
                ]
            )
        if any("memory" in b for b in bottlenecks):
            recommendations.extend(
                [
                    "Memory usage is high. Profile memory usage to find leaks or consider upgrading RAM.",
                    "Try using more memory-efficient data structures or reducing batch sizes.",
                ]
            )
        if res.get("gpu", {}).get("available"):
            if any("GPU" in b for b in bottlenecks):
                recommendations.append(
                    "GPU memory is a bottleneck. Consider model quantization, gradient accumulation, or using a larger VRAM GPU."
                )
            if (
                res["gpu"].get("average_utilization", 100)
                < self.config.low_gpu_util_threshold
            ):
                recommendations.append(
                    "GPU utilization is low. Increase batch sizes or use data prefetching to ensure the GPU is not waiting for data."
                )
        else:
            recommendations.append(
                "No GPU detected. For deep learning or parallel computation, using a GPU can provide significant performance gains."
            )
        if "AWS" in self.system_info.get("platform", ""):
            recommendations.append(
                "On AWS: Evaluate if the current EC2 instance type (CPU, memory, GPU) is optimal for the workload. Consider Graviton instances for better price/performance."
            )
        return recommendations

    def _identify_optimization_opportunities(
        self, benchmarks: List[ProcessingBenchmark]
    ) -> List[Dict[str, Any]]:
        opportunities = []
        # GPU Optimizations
        try:
            gpu_optimizer = get_gpu_optimizer()
            if gpu_optimizer.is_available():
                summary = gpu_optimizer.get_memory_summary()
                util = summary.get("utilization_percent", 100)
                if util < self.config.underutilized_gpu_mem_threshold:
                    opportunities.append(
                        {
                            "type": "gpu_memory_underutilization",
                            "description": f"GPU memory is underutilized ({util:.1f}%).",
                            "recommendation": "Increase batch size or run models in parallel.",
                        }
                    )
                if not summary.get("optimizations", {}).get("amp_enabled"):
                    opportunities.append(
                        {
                            "type": "mixed_precision",
                            "description": "Automatic Mixed Precision (AMP) is not enabled.",
                            "recommendation": "Enable AMP for potential 1.5-2x speed-up on compatible GPUs.",
                        }
                    )
        except Exception:
            pass
        # Parallel Processing
        cpu_count = self.system_info.get("cpu_count", 1)
        if cpu_count > 4:
            opportunities.append(
                {
                    "type": "parallel_processing",
                    "description": f"Multi-core CPU ({cpu_count} cores) is available.",
                    "recommendation": "Ensure your application is leveraging multiple cores for parallelizable tasks.",
                }
            )
        # Batch Processing
        if (
            benchmarks
            and len([b for b in benchmarks if b.items_processed == 1])
            > len(benchmarks) / 2
        ):
            opportunities.append(
                {
                    "type": "batch_processing",
                    "description": "High number of single-item operations detected.",
                    "recommendation": "Implement batch processing to reduce overhead and improve throughput.",
                }
            )
        return opportunities


class SystemInfo:
    """Collects and holds static system information."""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """Collects comprehensive system hardware and software information."""
        logger = get_logger().bind_context(component="system_info")
        try:
            info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": platform.python_version(),
            }
            # Optional: GPU information
            try:
                import torch

                if torch.cuda.is_available():
                    info["cuda_available"] = True
                    info["cuda_version"] = torch.version.cuda
                    info["gpu_count"] = torch.cuda.device_count()
                    info["gpu_devices"] = [
                        {
                            "name": props.name,
                            "memory_gb": props.total_memory / (1024**3),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                        for i in range(torch.cuda.device_count())
                        if (props := torch.cuda.get_device_properties(i))
                    ]
                else:
                    info["cuda_available"] = False
            except ImportError:
                info["cuda_available"] = False
            return info
        except Exception as e:
            logger.warning("System info collection failed.", error=str(e))
            return {"error": str(e)}


# --- Main Orchestrator ---


class PerformanceProfiler:
    """Orchestrates performance profiling, data collection, and report generation."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.logger = get_logger().bind_context(component="performance_profiler")
        self.config = config or ProfilerConfig()

        self.system_info = SystemInfo.collect()
        self.resource_tracker = ResourceTracker()
        self.analyzer = PerformanceAnalyzer(self.config, self.system_info)

        self.benchmarks: List[ProcessingBenchmark] = []

    @contextmanager
    def profile_operation(self, operation_name: str, items_count: int = 1):
        """Context manager to profile a block of code."""
        if not self.resource_tracker._tracking:
            self.resource_tracker.start_tracking(interval=0.5)

        start_time = time.time()
        start_res = self.resource_tracker.get_resource_snapshot()
        success, error_message = True, None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(
                "Profiled operation failed.",
                operation=operation_name,
                error=error_message,
            )
            raise
        finally:
            end_time = time.time()
            end_res = self.resource_tracker.get_resource_snapshot()
            duration = end_time - start_time

            benchmark = ProcessingBenchmark(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                items_processed=items_count,
                throughput_items_per_second=(
                    items_count / duration if duration > 0 else 0
                ),
                cpu_usage_during=max(start_res.cpu_percent, end_res.cpu_percent),
                memory_peak_mb=max(start_res.memory_used_mb, end_res.memory_used_mb),
                gpu_memory_peak_mb=(
                    max(
                        start_res.gpu_stats.allocated_mb, end_res.gpu_stats.allocated_mb
                    )
                    if start_res.gpu_stats and end_res.gpu_stats
                    else None
                ),
                success=success,
                error_message=error_message,
            )
            self.benchmarks.append(benchmark)
            self.logger.info(
                "Operation profiled.",
                operation=operation_name,
                duration=f"{duration:.2f}s",
                success=success,
            )

    def generate_report(self) -> PerformanceReport:
        """Generates a comprehensive performance report."""
        resource_summary = self.resource_tracker.get_resource_summary()
        bottlenecks, recommendations, opportunities = self.analyzer.run_analysis(
            self.benchmarks, resource_summary
        )

        return PerformanceReport(
            system_info=self.system_info,
            resource_utilization=resource_summary,
            benchmarks=self.benchmarks,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            optimization_opportunities=opportunities,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

    def cleanup(self):
        """Stops background threads and cleans up resources."""
        self.resource_tracker.stop_tracking()
        self.logger.info("Performance profiler cleanup completed.")


# --- Global Singleton Accessor ---

_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Gets the global performance profiler instance, creating it if necessary."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_operation(operation_name: str, items_count: int = 1):
    """Decorator/context manager for easy profiling of operations."""
    return get_performance_profiler().profile_operation(operation_name, items_count)
