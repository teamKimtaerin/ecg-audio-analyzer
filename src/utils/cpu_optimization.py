"""
CPU Optimization Utilities
Configure BLAS/NumPy for optimal multicore performance
"""

import os
import psutil
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def optimize_cpu_environment(max_threads: Optional[int] = None) -> dict:
    """
    Configure CPU environment variables for optimal performance
    
    Args:
        max_threads: Maximum number of threads to use (None for auto-detection)
        
    Returns:
        Dict of environment variables set
    """
    if max_threads is None:
        # Use physical cores, not logical (avoid hyperthreading issues)
        max_threads = psutil.cpu_count(logical=False) or 4
    
    # Limit to reasonable range
    max_threads = min(max_threads, 8)
    max_threads = max(max_threads, 1)
    
    env_vars = {
        # OpenMP threads for NumPy operations
        'OMP_NUM_THREADS': str(max_threads),
        
        # MKL threads (Intel Math Kernel Library)
        'MKL_NUM_THREADS': str(max_threads),
        
        # OpenBLAS threads
        'OPENBLAS_NUM_THREADS': str(max_threads),
        
        # BLIS threads  
        'BLIS_NUM_THREADS': str(max_threads),
        
        # NumExpr threads
        'NUMEXPR_NUM_THREADS': str(max_threads),
        
        # Disable tokenizers parallelism to avoid conflicts
        'TOKENIZERS_PARALLELISM': 'false',
        
        # Optimize NumPy for CPU
        'NPY_NUM_BUILD_JOBS': str(max_threads),
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.debug(f"Set {key}={value}")
    
    logger.info("cpu_optimization_applied", 
               max_threads=max_threads,
               physical_cores=psutil.cpu_count(logical=False),
               logical_cores=psutil.cpu_count(logical=True))
    
    return env_vars


def configure_torch_threads(num_threads: Optional[int] = None) -> None:
    """
    Configure PyTorch thread settings for CPU optimization
    
    Args:
        num_threads: Number of threads for PyTorch (None for auto)
    """
    try:
        import torch
        
        if num_threads is None:
            num_threads = psutil.cpu_count(logical=False) or 4
        
        num_threads = min(num_threads, 6)  # Limit for stability
        
        # Only set threads if they haven't been set before
        try:
            # Set intra-op parallelism (within operations)
            torch.set_num_threads(num_threads)
            
            # Set inter-op parallelism (between operations) - only if not already set
            current_interop = torch.get_num_interop_threads()
            if current_interop == 0:  # 0 means not set yet
                torch.set_num_interop_threads(max(1, num_threads // 2))
            
            logger.info("torch_threads_configured", 
                       intra_threads=num_threads,
                       inter_threads=torch.get_num_interop_threads())
        
        except RuntimeError as e:
            logger.warning("torch_threads_already_configured", error=str(e))
        
    except ImportError:
        logger.warning("torch_not_available_for_thread_configuration")


def get_cpu_optimization_info() -> dict:
    """
    Get information about current CPU optimization settings
    
    Returns:
        Dict with CPU optimization information
    """
    return {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'omp_threads': os.environ.get('OMP_NUM_THREADS', 'unset'),
        'mkl_threads': os.environ.get('MKL_NUM_THREADS', 'unset'),
        'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
        'tokenizers_parallelism': os.environ.get('TOKENIZERS_PARALLELISM', 'unset'),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1)
    }


# Auto-optimize on import
if __name__ != "__main__":
    optimize_cpu_environment()
    configure_torch_threads()