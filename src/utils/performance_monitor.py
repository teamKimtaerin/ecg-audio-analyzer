"""
Performance Monitoring Utilities
Comprehensive system performance monitoring and optimization recommendations
"""

import time
import threading
import psutil
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from contextlib import contextmanager
from collections import deque
import json

from .logger import get_logger
from .gpu_optimizer import get_gpu_optimizer, GPUMemoryStats


@dataclass
class SystemResources:
    """System resource snapshot"""
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
    """Processing benchmark results"""
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
            'operation': self.operation_name,
            'duration_seconds': self.duration_seconds,
            'items_processed': self.items_processed,
            'throughput_items_per_second': self.throughput_items_per_second,
            'cpu_usage_during': self.cpu_usage_during,
            'memory_peak_mb': self.memory_peak_mb,
            'gpu_memory_peak_mb': self.gpu_memory_peak_mb,
            'success': self.success,
            'error': self.error_message
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    system_info: Dict[str, Any]
    resource_utilization: Dict[str, float]
    benchmarks: List[ProcessingBenchmark]
    bottlenecks: List[str]
    recommendations: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system_info': self.system_info,
            'resource_utilization': self.resource_utilization,
            'benchmarks': [b.to_dict() for b in self.benchmarks],
            'bottlenecks': self.bottlenecks,
            'recommendations': self.recommendations,
            'optimization_opportunities': self.optimization_opportunities,
            'timestamp': self.timestamp
        }
    
    def save_to_file(self, output_path: Path):
        """Save performance report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ResourceTracker:
    """Track system resources over time"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.logger = get_logger().bind_context(component="resource_tracker")
        self.tracking = False
        self.track_thread: Optional[threading.Thread] = None
        
    def start_tracking(self, interval: float = 1.0):
        """Start background resource tracking"""
        if self.tracking:
            return
        
        self.tracking = True
        self.track_thread = threading.Thread(
            target=self._tracking_loop,
            args=(interval,),
            daemon=True
        )
        self.track_thread.start()
        self.logger.info("resource_tracking_started", interval=interval)
    
    def stop_tracking(self):
        """Stop background tracking"""
        self.tracking = False
        if self.track_thread and self.track_thread.is_alive():
            self.track_thread.join(timeout=2.0)
        self.logger.info("resource_tracking_stopped")
    
    def _tracking_loop(self, interval: float):
        """Background tracking loop"""
        while self.tracking:
            try:
                snapshot = self._get_resource_snapshot()
                self.samples.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                self.logger.warning("resource_tracking_error", error=str(e))
                time.sleep(interval)
    
    def _get_resource_snapshot(self) -> SystemResources:
        """Get current system resource snapshot"""
        # System resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Process information
        process_count = len(psutil.pids())
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        # GPU stats if available
        gpu_stats = None
        try:
            gpu_optimizer = get_gpu_optimizer()
            gpu_stats = gpu_optimizer.monitor.get_current_stats(0)
        except Exception:
            pass
        
        return SystemResources(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_used_mb=memory.used / 1024**2,
            memory_total_mb=memory.total / 1024**2,
            memory_percent=memory.percent,
            disk_used_gb=disk.used / 1024**3,
            disk_total_gb=disk.total / 1024**3,
            disk_percent=(disk.used / disk.total) * 100,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=process_count,
            thread_count=thread_count,
            gpu_stats=gpu_stats
        )
    
    def get_resource_summary(self, last_n_samples: int = 60) -> Dict[str, Any]:
        """Get resource utilization summary"""
        if not self.samples:
            return {}
        
        recent_samples = list(self.samples)[-last_n_samples:]
        
        if not recent_samples:
            return {}
        
        # Calculate averages and peaks
        avg_cpu = sum(s.cpu_percent for s in recent_samples) / len(recent_samples)
        peak_cpu = max(s.cpu_percent for s in recent_samples)
        
        avg_memory = sum(s.memory_percent for s in recent_samples) / len(recent_samples)
        peak_memory = max(s.memory_percent for s in recent_samples)
        
        avg_disk = sum(s.disk_percent for s in recent_samples) / len(recent_samples)
        
        # GPU stats if available
        gpu_samples = [s for s in recent_samples if s.gpu_stats]
        avg_gpu_memory = 0.0
        peak_gpu_memory = 0.0
        avg_gpu_util = 0.0
        
        if gpu_samples:
            avg_gpu_memory = sum(s.gpu_stats.utilization_percent for s in gpu_samples) / len(gpu_samples)
            peak_gpu_memory = max(s.gpu_stats.utilization_percent for s in gpu_samples)
            avg_gpu_util = sum(s.gpu_stats.allocated_mb / s.gpu_stats.total_mb * 100 for s in gpu_samples) / len(gpu_samples)
        
        return {
            'cpu': {
                'average_percent': avg_cpu,
                'peak_percent': peak_cpu
            },
            'memory': {
                'average_percent': avg_memory,
                'peak_percent': peak_memory,
                'total_mb': recent_samples[-1].memory_total_mb
            },
            'disk': {
                'average_percent': avg_disk,
                'total_gb': recent_samples[-1].disk_total_gb
            },
            'gpu': {
                'available': len(gpu_samples) > 0,
                'average_memory_percent': avg_gpu_memory,
                'peak_memory_percent': peak_gpu_memory,
                'average_utilization': avg_gpu_util
            },
            'samples_analyzed': len(recent_samples)
        }


class PerformanceProfiler:
    """Profile performance of operations and provide optimization recommendations"""
    
    def __init__(self):
        self.logger = get_logger().bind_context(component="performance_profiler")
        self.resource_tracker = ResourceTracker()
        self.benchmarks: List[ProcessingBenchmark] = []
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Performance thresholds
        self.cpu_warning_threshold = 80.0
        self.memory_warning_threshold = 85.0
        self.gpu_memory_warning_threshold = 90.0
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        try:
            info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': platform.python_version(),
            }
            
            # GPU information
            try:
                import torch
                if torch.cuda.is_available():
                    info['cuda_available'] = True
                    info['cuda_version'] = torch.version.cuda
                    info['gpu_count'] = torch.cuda.device_count()
                    info['gpu_devices'] = []
                    
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        info['gpu_devices'].append({
                            'name': props.name,
                            'memory_gb': props.total_memory / 1024**3,
                            'compute_capability': f"{props.major}.{props.minor}"
                        })
                else:
                    info['cuda_available'] = False
            except ImportError:
                info['cuda_available'] = False
            
            return info
            
        except Exception as e:
            self.logger.warning("system_info_collection_failed", error=str(e))
            return {'error': str(e)}
    
    @contextmanager
    def profile_operation(self, operation_name: str, items_count: int = 1):
        """Context manager to profile an operation"""
        
        # Start resource tracking if not already started
        if not self.resource_tracker.tracking:
            self.resource_tracker.start_tracking(interval=0.5)
        
        start_time = time.time()
        start_resources = self.resource_tracker._get_resource_snapshot()
        
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_resources = self.resource_tracker._get_resource_snapshot()
            
            # Calculate metrics
            duration = end_time - start_time
            throughput = items_count / duration if duration > 0 else 0
            
            # Resource usage during operation
            cpu_usage = max(start_resources.cpu_percent, end_resources.cpu_percent)
            memory_peak = max(start_resources.memory_used_mb, end_resources.memory_used_mb)
            
            gpu_memory_peak = None
            if (start_resources.gpu_stats and end_resources.gpu_stats):
                gpu_memory_peak = max(
                    start_resources.gpu_stats.allocated_mb,
                    end_resources.gpu_stats.allocated_mb
                )
            
            # Create benchmark record
            benchmark = ProcessingBenchmark(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                items_processed=items_count,
                throughput_items_per_second=throughput,
                cpu_usage_during=cpu_usage,
                memory_peak_mb=memory_peak,
                gpu_memory_peak_mb=gpu_memory_peak,
                success=success,
                error_message=error_message
            )
            
            self.benchmarks.append(benchmark)
            
            self.logger.info("operation_profiled",
                           operation=operation_name,
                           duration=duration,
                           throughput=throughput,
                           success=success)
    
    def analyze_bottlenecks(self, resource_summary: Dict[str, Any]) -> List[str]:
        """Analyze system bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_avg = resource_summary.get('cpu', {}).get('average_percent', 0)
        cpu_peak = resource_summary.get('cpu', {}).get('peak_percent', 0)
        
        if cpu_peak > 95:
            bottlenecks.append("Severe CPU bottleneck detected (peak >95%)")
        elif cpu_avg > self.cpu_warning_threshold:
            bottlenecks.append(f"High CPU utilization (avg {cpu_avg:.1f}%)")
        
        # Memory bottleneck
        memory_avg = resource_summary.get('memory', {}).get('average_percent', 0)
        memory_peak = resource_summary.get('memory', {}).get('peak_percent', 0)
        
        if memory_peak > 95:
            bottlenecks.append("Severe memory bottleneck detected (peak >95%)")
        elif memory_avg > self.memory_warning_threshold:
            bottlenecks.append(f"High memory utilization (avg {memory_avg:.1f}%)")
        
        # GPU bottleneck
        if resource_summary.get('gpu', {}).get('available', False):
            gpu_memory_avg = resource_summary['gpu'].get('average_memory_percent', 0)
            gpu_memory_peak = resource_summary['gpu'].get('peak_memory_percent', 0)
            
            if gpu_memory_peak > 95:
                bottlenecks.append("Severe GPU memory bottleneck detected (peak >95%)")
            elif gpu_memory_avg > self.gpu_memory_warning_threshold:
                bottlenecks.append(f"High GPU memory utilization (avg {gpu_memory_avg:.1f}%)")
        
        # Disk I/O bottleneck
        disk_avg = resource_summary.get('disk', {}).get('average_percent', 0)
        if disk_avg > 90:
            bottlenecks.append(f"Disk space bottleneck (avg {disk_avg:.1f}% used)")
        
        # Performance bottlenecks from benchmarks
        if self.benchmarks:
            slow_operations = [b for b in self.benchmarks if b.throughput_items_per_second < 0.1]
            if slow_operations:
                bottlenecks.append(f"Slow processing detected in {len(slow_operations)} operations")
        
        return bottlenecks
    
    def generate_recommendations(self, 
                               resource_summary: Dict[str, Any], 
                               bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        cpu_avg = resource_summary.get('cpu', {}).get('average_percent', 0)
        if cpu_avg > self.cpu_warning_threshold:
            recommendations.append(
                "Consider increasing parallel processing workers or upgrading to a higher-CPU instance"
            )
            recommendations.append(
                "Enable batch processing to improve CPU efficiency"
            )
        
        # Memory recommendations
        memory_avg = resource_summary.get('memory', {}).get('average_percent', 0)
        memory_total = resource_summary.get('memory', {}).get('total_mb', 0)
        
        if memory_avg > self.memory_warning_threshold:
            recommendations.append(
                f"Current memory usage is high ({memory_avg:.1f}%). Consider upgrading to instance with more than {memory_total/1024:.1f}GB RAM"
            )
            recommendations.append(
                "Enable memory-efficient processing modes and reduce batch sizes"
            )
        
        # GPU recommendations
        if resource_summary.get('gpu', {}).get('available', False):
            gpu_memory_avg = resource_summary['gpu'].get('average_memory_percent', 0)
            gpu_util_avg = resource_summary['gpu'].get('average_utilization', 0)
            
            if gpu_memory_avg > self.gpu_memory_warning_threshold:
                recommendations.append(
                    "GPU memory usage is high. Consider model optimization or larger GPU instance"
                )
            
            if gpu_util_avg < 50:
                recommendations.append(
                    "GPU utilization is low. Consider increasing batch sizes or enabling mixed precision"
                )
        else:
            recommendations.append(
                "No GPU detected. Consider using GPU-enabled instance for significant performance improvements"
            )
        
        # Processing efficiency recommendations
        if self.benchmarks:
            avg_throughput = sum(b.throughput_items_per_second for b in self.benchmarks if b.success) / max(1, len(self.benchmarks))
            
            if avg_throughput < 1.0:
                recommendations.append(
                    "Low processing throughput detected. Consider optimizing algorithms or using faster hardware"
                )
        
        # AWS-specific recommendations
        if 'AWS' in str(self.system_info.get('platform', '')):
            recommendations.append(
                "Running on AWS: Consider using spot instances for cost optimization"
            )
            recommendations.append(
                "Enable enhanced networking for better I/O performance"
            )
        
        return recommendations
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # GPU optimization opportunities
        try:
            gpu_optimizer = get_gpu_optimizer()
            gpu_summary = gpu_optimizer.get_memory_summary()
            
            if gpu_summary.get('gpu_available', False):
                current_memory = gpu_summary.get('current_stats', {}).get('allocated_mb', 0)
                total_memory = gpu_summary.get('primary_device', {}).get('total_memory_mb', 0)
                
                if total_memory > 0:
                    utilization = (current_memory / total_memory) * 100
                    
                    if utilization < 30:
                        opportunities.append({
                            'type': 'gpu_memory_underutilization',
                            'description': f'GPU memory underutilized ({utilization:.1f}%)',
                            'recommendation': 'Increase batch sizes or load more models in memory',
                            'potential_improvement': '2-4x throughput increase'
                        })
                    
                    if not gpu_summary.get('optimizations', {}).get('amp_enabled', False):
                        opportunities.append({
                            'type': 'mixed_precision',
                            'description': 'Automatic Mixed Precision not enabled',
                            'recommendation': 'Enable AMP for 1.5-2x speed improvement',
                            'potential_improvement': '1.5-2x speed increase'
                        })
        except Exception:
            pass
        
        # Parallel processing opportunities
        cpu_count = self.system_info.get('cpu_count', 1)
        if cpu_count > 4:
            opportunities.append({
                'type': 'parallel_processing',
                'description': f'Multi-core CPU available ({cpu_count} cores)',
                'recommendation': 'Increase worker count for parallel processing',
                'potential_improvement': f'Up to {cpu_count//2}x speedup for parallel tasks'
            })
        
        # Batch processing opportunities
        if self.benchmarks:
            single_item_ops = [b for b in self.benchmarks if b.items_processed == 1]
            if len(single_item_ops) > 3:
                opportunities.append({
                    'type': 'batch_processing',
                    'description': 'Many single-item operations detected',
                    'recommendation': 'Implement batch processing for better efficiency',
                    'potential_improvement': '2-5x throughput improvement'
                })
        
        return opportunities
    
    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Get resource summary
        resource_summary = self.resource_tracker.get_resource_summary()
        
        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks(resource_summary)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(resource_summary, bottlenecks)
        
        # Identify optimization opportunities
        optimization_opportunities = self.identify_optimization_opportunities()
        
        return PerformanceReport(
            system_info=self.system_info,
            resource_utilization=resource_summary,
            benchmarks=self.benchmarks,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            optimization_opportunities=optimization_opportunities,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        )
    
    def cleanup(self):
        """Clean up profiler resources"""
        self.resource_tracker.stop_tracking()
        self.logger.info("performance_profiler_cleanup_completed")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get or create global performance profiler"""
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    
    return _global_profiler


def profile_operation(operation_name: str, items_count: int = 1):
    """Decorator/context manager for profiling operations"""
    profiler = get_performance_profiler()
    return profiler.profile_operation(operation_name, items_count)