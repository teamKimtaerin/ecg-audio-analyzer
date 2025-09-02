"""
GPU Optimization Utilities
Advanced CUDA optimizations and GPU memory management for high-performance processing
"""

import os
import gc
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import warnings

import torch
import torch.cuda
import torch.backends.cudnn as cudnn

from .logger import get_logger

# Suppress CUDA warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")


@dataclass
class GPUMemoryStats:
    """GPU memory statistics"""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float


@dataclass
class GPUDevice:
    """GPU device information"""
    device_id: int
    name: str
    compute_capability: tuple
    total_memory_mb: float
    is_available: bool
    is_primary: bool = False
    current_memory_stats: Optional[GPUMemoryStats] = None


@dataclass
class OptimizationConfig:
    """GPU optimization configuration"""
    enable_amp: bool = True                    # Automatic Mixed Precision
    enable_tf32: bool = True                   # TensorFloat-32 on Ampere GPUs
    enable_cudnn_benchmark: bool = True        # CuDNN optimization
    enable_memory_pool: bool = True            # Memory pool optimization
    memory_fraction: float = 0.9               # Max GPU memory to use
    enable_gradient_checkpointing: bool = False # Memory vs compute tradeoff
    enable_compile_optimization: bool = True    # PyTorch 2.0 compile
    cache_models: bool = True                  # Keep models in GPU memory
    enable_flash_attention: bool = True        # Flash Attention if available
    batch_size_optimization: bool = True       # Dynamic batch sizing


class GPUMonitor:
    """Real-time GPU monitoring and statistics"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.logger = get_logger().bind_context(component="gpu_monitor")
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats_history: List[GPUMemoryStats] = []
        self.max_history_size = 1000
        
        # Monitoring callbacks
        self.memory_warning_threshold = 0.85
        self.memory_critical_threshold = 0.95
        self.warning_callbacks: List[Callable] = []
    
    def add_warning_callback(self, callback: Callable[[GPUMemoryStats], None]):
        """Add callback for memory warnings"""
        self.warning_callbacks.append(callback)
    
    def get_current_stats(self, device_id: int = 0) -> Optional[GPUMemoryStats]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return None
        
        try:
            # Get memory info
            allocated = torch.cuda.memory_allocated(device_id) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(device_id) / 1024**2    # MB
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**2
            max_reserved = torch.cuda.max_memory_reserved(device_id) / 1024**2
            
            # Get free and total memory more accurately
            if hasattr(torch.cuda, 'mem_get_info'):
                free, total = torch.cuda.mem_get_info(device_id)
                free /= 1024**2
                total /= 1024**2
            else:
                props = torch.cuda.get_device_properties(device_id)
                total = props.total_memory / 1024**2
                free = total - reserved
            
            # Get utilization if nvidia-ml-py3 available (GPU compute utilization)
            utilization = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
            except ImportError:
                # Fallback to memory utilization as proxy
                utilization = (allocated / total * 100) if total > 0 else 0.0
            
            return GPUMemoryStats(
                allocated_mb=allocated,
                reserved_mb=reserved,
                max_allocated_mb=max_allocated,
                max_reserved_mb=max_reserved,
                free_mb=free,
                total_mb=total,
                utilization_percent=utilization
            )
            
        except Exception as e:
            self.logger.warning("gpu_stats_collection_failed", device_id=device_id, error=str(e))
            return None
    
    def start_monitoring(self):
        """Start background GPU monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("gpu_monitoring_started", update_interval=self.update_interval)
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("gpu_monitoring_stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_current_stats(0)
                if stats:
                    # Add to history
                    self.stats_history.append(stats)
                    if len(self.stats_history) > self.max_history_size:
                        self.stats_history.pop(0)
                    
                    # Check for warnings
                    memory_usage_ratio = stats.allocated_mb / stats.total_mb
                    
                    if memory_usage_ratio > self.memory_critical_threshold:
                        self.logger.error("gpu_memory_critical",
                                        allocated_mb=stats.allocated_mb,
                                        total_mb=stats.total_mb,
                                        usage_percent=memory_usage_ratio * 100)
                        
                        # Trigger callbacks
                        for callback in self.warning_callbacks:
                            try:
                                callback(stats)
                            except Exception as e:
                                self.logger.warning("warning_callback_failed", error=str(e))
                    
                    elif memory_usage_ratio > self.memory_warning_threshold:
                        self.logger.warning("gpu_memory_high",
                                          allocated_mb=stats.allocated_mb,
                                          total_mb=stats.total_mb,
                                          usage_percent=memory_usage_ratio * 100)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error("monitoring_loop_error", error=str(e))
                time.sleep(self.update_interval)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get monitoring statistics summary"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-10:]  # Last 10 samples
        
        return {
            'current_allocated_mb': recent_stats[-1].allocated_mb,
            'current_utilization': recent_stats[-1].utilization_percent,
            'peak_allocated_mb': max(s.allocated_mb for s in recent_stats),
            'avg_allocated_mb': sum(s.allocated_mb for s in recent_stats) / len(recent_stats),
            'peak_utilization': max(s.utilization_percent for s in recent_stats),
            'avg_utilization': sum(s.utilization_percent for s in recent_stats) / len(recent_stats),
            'samples_collected': len(self.stats_history)
        }


class GPUOptimizer:
    """
    Advanced GPU optimization and memory management.
    
    Provides comprehensive GPU optimization including memory management,
    model optimization, and performance monitoring for AWS GPU instances.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger().bind_context(component="gpu_optimizer")
        
        # GPU information
        self.available_devices: List[GPUDevice] = []
        self.primary_device: Optional[GPUDevice] = None
        
        # Monitoring
        self.monitor = GPUMonitor()
        
        # Model cache
        self.model_cache: Dict[str, torch.nn.Module] = {}
        self.cache_memory_limit_mb = 4000  # 4GB cache limit
        
        # Optimization state
        self.optimization_applied = False
        
        # Initialize GPU information
        self._discover_gpus()
        
        # Apply initial optimizations
        self.apply_optimizations()
        
        self.logger.info("gpu_optimizer_initialized",
                        available_gpus=len(self.available_devices),
                        primary_device=self.primary_device.name if self.primary_device else "none",
                        amp_enabled=config.enable_amp,
                        tf32_enabled=config.enable_tf32)
    
    def _discover_gpus(self):
        """Discover available GPU devices"""
        if not torch.cuda.is_available():
            self.logger.warning("cuda_not_available")
            return
        
        device_count = torch.cuda.device_count()
        self.available_devices = []
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            
            device = GPUDevice(
                device_id=i,
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_mb=props.total_memory / 1024**2,
                is_available=True,
                is_primary=(i == 0)
            )
            
            self.available_devices.append(device)
            
            if i == 0:
                self.primary_device = device
        
        self.logger.info("gpu_discovery_completed",
                        device_count=device_count,
                        devices=[d.name for d in self.available_devices])
    
    def apply_optimizations(self):
        """Apply GPU optimizations"""
        if self.optimization_applied or not self.available_devices:
            return
        
        self.logger.info("applying_gpu_optimizations")
        
        # Set memory fraction
        if self.config.memory_fraction < 1.0:
            try:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                self.logger.info("gpu_memory_fraction_set", fraction=self.config.memory_fraction)
            except Exception as e:
                self.logger.warning("memory_fraction_failed", error=str(e))
        
        # Enable TensorFloat-32 on Ampere GPUs
        if self.config.enable_tf32 and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("tf32_enabled")
        
        # CuDNN optimization
        if self.config.enable_cudnn_benchmark:
            cudnn.benchmark = True
            cudnn.deterministic = False  # For better performance
            self.logger.info("cudnn_benchmark_enabled")
        
        # Memory pool optimization
        if self.config.enable_memory_pool:
            try:
                # Use expandable memory pool
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                self.logger.info("memory_pool_optimization_enabled")
            except Exception as e:
                self.logger.warning("memory_pool_setup_failed", error=str(e))
        
        # Enable compilation optimization
        if self.config.enable_compile_optimization and hasattr(torch, 'compile'):
            self.logger.info("torch_compile_available")
        
        self.optimization_applied = True
        self.logger.info("gpu_optimizations_applied")
    
    @contextmanager
    def optimized_inference(self, model: torch.nn.Module, enable_amp: bool = None):
        """Context manager for optimized inference"""
        
        if enable_amp is None:
            enable_amp = self.config.enable_amp
        
        # Store original states
        original_training = model.training
        
        try:
            # Set model to eval mode
            model.eval()
            
            # Disable gradient computation
            with torch.no_grad():
                # Use automatic mixed precision if enabled
                if enable_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        yield model
                else:
                    yield model
                    
        finally:
            # Restore original training state
            model.train(original_training)
    
    def optimize_model(self, 
                      model: torch.nn.Module, 
                      model_name: str,
                      cache_model: bool = None) -> torch.nn.Module:
        """Optimize model for GPU inference"""
        
        if cache_model is None:
            cache_model = self.config.cache_models
        
        # Check if model is already in cache
        if cache_model and model_name in self.model_cache:
            self.logger.info("model_loaded_from_cache", model_name=model_name)
            return self.model_cache[model_name]
        
        optimized_model = model
        
        try:
            # Move to GPU if available
            if self.primary_device:
                optimized_model = optimized_model.to(f"cuda:{self.primary_device.device_id}")
                self.logger.info("model_moved_to_gpu", 
                               model_name=model_name,
                               device=f"cuda:{self.primary_device.device_id}")
            
            # Enable half precision if supported
            if (self.config.enable_amp and 
                self.primary_device and 
                self.primary_device.compute_capability[0] >= 7):  # Volta and newer
                
                optimized_model = optimized_model.half()
                self.logger.info("model_converted_to_half", model_name=model_name)
            
            # Apply torch.compile optimization if available
            if (self.config.enable_compile_optimization and 
                hasattr(torch, 'compile') and 
                torch.cuda.is_available()):
                
                try:
                    optimized_model = torch.compile(optimized_model)
                    self.logger.info("model_compiled", model_name=model_name)
                except Exception as e:
                    self.logger.warning("model_compilation_failed", 
                                      model_name=model_name, 
                                      error=str(e))
            
            # Cache the model if enabled
            if cache_model:
                self._cache_model(model_name, optimized_model)
            
            return optimized_model
            
        except Exception as e:
            self.logger.error("model_optimization_failed", 
                            model_name=model_name, 
                            error=str(e))
            return model
    
    def _cache_model(self, model_name: str, model: torch.nn.Module):
        """Cache optimized model in GPU memory"""
        try:
            # Check cache memory usage
            current_cache_memory = self._estimate_cache_memory()
            
            if current_cache_memory > self.cache_memory_limit_mb:
                self._evict_cached_models()
            
            self.model_cache[model_name] = model
            self.logger.info("model_cached", 
                           model_name=model_name,
                           cache_size=len(self.model_cache))
            
        except Exception as e:
            self.logger.warning("model_caching_failed", 
                              model_name=model_name, 
                              error=str(e))
    
    def _estimate_cache_memory(self) -> float:
        """Estimate current cache memory usage in MB"""
        total_bytes = 0
        for model in self.model_cache.values():
            try:
                # Parameters
                for p in model.parameters():
                    total_bytes += p.numel() * p.element_size()
                # Buffers
                for b in model.buffers():
                    total_bytes += b.numel() * b.element_size()
            except Exception:
                continue
        
        estimated_mb = total_bytes / (1024**2)
        return estimated_mb
    
    def _evict_cached_models(self):
        """Evict oldest cached models to free memory"""
        if not self.model_cache:
            return
        
        # Simple LRU: remove half of cached models
        models_to_remove = list(self.model_cache.keys())[:len(self.model_cache)//2]
        
        for model_name in models_to_remove:
            del self.model_cache[model_name]
        
        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("cached_models_evicted", 
                       evicted_count=len(models_to_remove),
                       remaining_count=len(self.model_cache))
    
    def optimize_batch_size(self, 
                           base_batch_size: int, 
                           model_memory_mb: float,
                           safety_factor: float = 0.8) -> int:
        """Optimize batch size based on available GPU memory"""
        
        if not self.primary_device or not self.config.batch_size_optimization:
            return base_batch_size
        
        try:
            stats = self.monitor.get_current_stats(self.primary_device.device_id)
            if not stats:
                return base_batch_size
            
            # Calculate available memory
            available_mb = stats.free_mb * safety_factor
            
            # Estimate batch size that fits in memory
            memory_per_sample = model_memory_mb / base_batch_size
            optimal_batch_size = int(available_mb / memory_per_sample)
            
            # Keep within reasonable bounds
            optimal_batch_size = max(1, min(optimal_batch_size, base_batch_size * 4))
            
            if optimal_batch_size != base_batch_size:
                self.logger.info("batch_size_optimized",
                               original=base_batch_size,
                               optimized=optimal_batch_size,
                               available_memory_mb=available_mb)
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.warning("batch_size_optimization_failed", error=str(e))
            return base_batch_size
    
    def clear_cache(self, clear_model_cache: bool = True):
        """Clear GPU memory caches"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear model cache if requested
            if clear_model_cache:
                self.model_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("gpu_cache_cleared", model_cache_cleared=clear_model_cache)
            
        except Exception as e:
            self.logger.error("cache_clearing_failed", error=str(e))
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive GPU memory summary"""
        if not self.primary_device:
            return {"gpu_available": False}
        
        stats = self.monitor.get_current_stats(self.primary_device.device_id)
        monitor_summary = self.monitor.get_stats_summary()
        
        summary = {
            "gpu_available": True,
            "primary_device": {
                "name": self.primary_device.name,
                "compute_capability": self.primary_device.compute_capability,
                "total_memory_mb": self.primary_device.total_memory_mb
            },
            "current_stats": stats.__dict__ if stats else {},
            "cached_models": len(self.model_cache),
            "cache_memory_estimate_mb": self._estimate_cache_memory(),
            "optimizations": {
                "amp_enabled": self.config.enable_amp,
                "tf32_enabled": self.config.enable_tf32,
                "cudnn_benchmark": self.config.enable_cudnn_benchmark,
                "memory_fraction": self.config.memory_fraction
            }
        }
        
        summary.update(monitor_summary)
        return summary
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitor.stop_monitoring()
    
    def cleanup(self):
        """Clean up GPU optimizer resources"""
        try:
            self.stop_monitoring()
            self.clear_cache(clear_model_cache=True)
            self.logger.info("gpu_optimizer_cleanup_completed")
        except Exception as e:
            self.logger.error("gpu_optimizer_cleanup_failed", error=str(e))
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Global GPU optimizer instance
_global_optimizer: Optional[GPUOptimizer] = None


def get_gpu_optimizer(config: Optional[OptimizationConfig] = None) -> GPUOptimizer:
    """Get or create global GPU optimizer instance"""
    global _global_optimizer
    
    if _global_optimizer is None:
        if config is None:
            config = OptimizationConfig()
        _global_optimizer = GPUOptimizer(config)
    
    return _global_optimizer


def optimize_for_inference(model: torch.nn.Module, 
                          model_name: str,
                          config: Optional[OptimizationConfig] = None) -> torch.nn.Module:
    """Convenience function to optimize model for inference"""
    optimizer = get_gpu_optimizer(config)
    return optimizer.optimize_model(model, model_name)
