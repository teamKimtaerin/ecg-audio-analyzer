"""
Structured Logging Utility
High-performance logging with AWS CloudWatch integration and GPU monitoring
"""

import logging
import time
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import psutil
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    process_id: int
    stage: str
    duration_seconds: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    throughput_items_per_sec: Optional[float] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)


class PerformanceMonitor:
    """Monitor system and GPU performance metrics"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        self.process = psutil.Process()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self.process.memory_percent()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "memory_percent": 0.0}
    
    def get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get GPU performance metrics"""
        if not self.enable_gpu_monitoring or not torch.cuda.is_available():
            return {"gpu_utilization": None, "gpu_memory_mb": None, "gpu_memory_percent": None}
        
        try:
            device = torch.cuda.current_device()
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024 / 1024
            gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
            
            # Try to get GPU utilization using nvidia-ml-py3 if available
            gpu_utilization = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except ImportError:
                pass
            
            return {
                "gpu_utilization": gpu_utilization,
                "gpu_memory_mb": gpu_memory_allocated,
                "gpu_memory_percent": gpu_memory_percent
            }
        except Exception:
            return {"gpu_utilization": None, "gpu_memory_mb": None, "gpu_memory_percent": None}
    
    def create_metrics(self, stage: str, duration: float, 
                      throughput: Optional[float] = None,
                      error_count: int = 0) -> PerformanceMetrics:
        """Create comprehensive performance metrics"""
        system_metrics = self.get_system_metrics()
        gpu_metrics = self.get_gpu_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            process_id=os.getpid(),
            stage=stage,
            duration_seconds=duration,
            throughput_items_per_sec=throughput,
            error_count=error_count,
            **system_metrics,
            **gpu_metrics
        )


class CloudWatchHandler(logging.Handler):
    """Custom handler for AWS CloudWatch logs"""
    
    def __init__(self, log_group: str, log_stream: str, region: str = "us-east-1"):
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self.client = None
        
        if BOTO3_AVAILABLE:
            try:
                self.client = boto3.client('logs', region_name=region)
                self._ensure_log_group_exists()
            except Exception as e:
                print(f"Failed to initialize CloudWatch client: {e}")
    
    def _ensure_log_group_exists(self):
        """Ensure the CloudWatch log group exists"""
        if not self.client:
            return
            
        try:
            self.client.create_log_group(logGroupName=self.log_group)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            print(f"Failed to create log group: {e}")
    
    def emit(self, record):
        """Emit log record to CloudWatch"""
        if not self.client:
            return
            
        try:
            log_event = {
                'timestamp': int(time.time() * 1000),
                'message': self.format(record)
            }
            
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[log_event]
            )
        except Exception:
            # Silently fail - don't break the application for logging issues
            pass


class ECGLogger:
    """Enhanced structured logger for ECG Audio Analysis"""
    
    def __init__(self, 
                 name: str = "ecg-audio-analyzer",
                 level: str = "INFO",
                 enable_performance_monitoring: bool = True,
                 enable_cloudwatch: bool = False,
                 cloudwatch_log_group: str = "/aws/ec2/ecg-audio-analysis",
                 region: str = "us-east-1"):
        
        self.name = name
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.WriteLoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        
        # Set up standard logging
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        # Add CloudWatch handler if enabled
        if enable_cloudwatch and BOTO3_AVAILABLE:
            cloudwatch_handler = CloudWatchHandler(
                log_group=cloudwatch_log_group,
                log_stream=f"{name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                region=region
            )
            cloudwatch_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(cloudwatch_handler)
        
        self.logger = structlog.get_logger(name)
    
    def bind_context(self, **kwargs):
        """Bind context variables for structured logging"""
        bind_contextvars(**kwargs)
        return self
    
    def clear_context(self):
        """Clear all context variables"""
        clear_contextvars()
        return self
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self.logger.debug(message, **kwargs)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        if self.enable_performance_monitoring:
            self.logger.info("performance_metrics", **metrics.to_dict())
    
    def log_gpu_memory(self, stage: str, device: Optional[str] = None):
        """Log current GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            if device is None:
                device = torch.cuda.current_device()
            
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            
            self.logger.info(
                "gpu_memory_status",
                stage=stage,
                device=device,
                allocated_gb=round(allocated, 2),
                reserved_gb=round(reserved, 2)
            )
        except Exception as e:
            self.logger.warning("failed_to_log_gpu_memory", error=str(e))
    
    @contextmanager
    def performance_timer(self, stage: str, items_count: Optional[int] = None):
        """Context manager for timing operations with performance logging"""
        if not self.enable_performance_monitoring:
            yield
            return
        
        start_time = time.time()
        error_count = 0
        
        try:
            self.logger.info("stage_started", stage=stage, items_count=items_count)
            yield
            self.logger.info("stage_completed", stage=stage)
        except Exception as e:
            error_count = 1
            self.logger.error("stage_failed", stage=stage, error=str(e))
            raise
        finally:
            duration = time.time() - start_time
            throughput = items_count / duration if items_count and duration > 0 else None
            
            if self.performance_monitor:
                metrics = self.performance_monitor.create_metrics(
                    stage=stage,
                    duration=duration,
                    throughput=throughput,
                    error_count=error_count
                )
                self.log_performance(metrics)


# Global logger instance
_global_logger: Optional[ECGLogger] = None


def get_logger(name: str = "ecg-audio-analyzer", **kwargs) -> ECGLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ECGLogger(name=name, **kwargs)
    
    return _global_logger


def setup_logging(
    level: str = "INFO",
    enable_performance_monitoring: bool = True,
    enable_cloudwatch: bool = False,
    cloudwatch_log_group: str = "/aws/ec2/ecg-audio-analysis"
) -> ECGLogger:
    """Setup global logging configuration"""
    global _global_logger
    
    _global_logger = ECGLogger(
        level=level,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_cloudwatch=enable_cloudwatch,
        cloudwatch_log_group=cloudwatch_log_group
    )
    
    return _global_logger