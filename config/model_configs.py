"""
Model Configuration Settings
GPU-optimized model parameters for high-performance audio analysis
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeakerDiarizationConfig:
    """Configuration for pyannote-audio speaker diarization"""

    # Model selection (matches actual implementation)
    model_name: str = "pyannote/speaker-diarization-3.1"

    # Processing parameters (aligned with actual usage)
    min_speakers: int = 2
    max_speakers: int = 8

    # GPU optimization
    device: str = "cuda:0"
    batch_size: int = 8
    enable_fp16: bool = True


@dataclass
class WhisperXConfig:
    """Configuration for WhisperX speech recognition and alignment"""

    # Model settings (matches actual implementation)
    model_size: str = "large-v3"  # Upgraded for better word alignment
    language: Optional[str] = "en"
    compute_type: str = "float16"

    # Audio processing
    sample_rate: int = 16000

    # GPU optimization
    device: Optional[str] = None  # Auto-detect
    batch_size: int = 16


@dataclass
class AcousticAnalysisConfig:
    """Configuration for acoustic feature extraction"""

    # Feature extraction settings
    feature_set: str = "ComParE_2016"  # OpenSMILE feature set
    frame_size_ms: int = 25  # 25ms frames
    frame_step_ms: int = 10  # 10ms steps

    # Audio features
    extract_mfcc: bool = True
    mfcc_coefficients: int = 13
    extract_pitch: bool = True
    extract_energy: bool = True
    extract_spectral: bool = True
    extract_prosody: bool = True

    # Processing parameters
    pre_emphasis: float = 0.97
    window_type: str = "hamming"
    fft_size: int = 512

    # Pitch analysis
    pitch_method: str = "autocorrelation"
    pitch_floor: float = 75.0  # Hz
    pitch_ceiling: float = 600.0  # Hz

    # Energy analysis
    energy_floor: float = 1e-10
    energy_normalization: bool = True

    # Performance optimization
    enable_gpu_acceleration: bool = True
    parallel_processing: bool = True
    chunk_processing: bool = True


@dataclass
class ModelCacheConfig:
    """Configuration for model loading and caching"""

    # Cache settings
    enable_model_caching: bool = True
    cache_directory: str = "/tmp/model_cache"
    max_cache_size_gb: float = 10.0

    # Model loading optimization
    lazy_loading: bool = False  # Load all models at startup
    model_warmup: bool = True  # Warm up models with dummy data
    preload_weights: bool = True  # Preload weights to GPU memory

    # Memory management
    enable_model_offloading: bool = False  # Keep models in GPU memory
    cpu_fallback: bool = True  # Fallback to CPU if GPU unavailable
    memory_mapping: bool = True  # Use memory mapping for large models

    # Download settings
    use_auth_token: Optional[str] = None
    use_local_files_only: bool = False
    force_download: bool = False


@dataclass
class ModelPerformanceConfig:
    """Performance optimization settings for all models"""

    # General optimization
    enable_torch_compile: bool = True  # PyTorch 2.0 compilation
    enable_gradient_checkpointing: bool = False  # Inference only
    enable_attention_slicing: bool = True

    # Memory optimization
    enable_cpu_offload: bool = False  # Keep in GPU memory
    enable_sequential_cpu_offload: bool = False
    low_cpu_mem_usage: bool = True

    # Precision settings
    torch_dtype: str = "float16"  # Use FP16 for all models
    enable_autocast: bool = True  # Automatic mixed precision

    # Batch processing
    dynamic_batching: bool = True
    max_batch_size: int = 32
    optimal_batch_size: int = 8

    # Hardware optimization
    enable_tf32: bool = True  # Enable TensorFloat-32 on A100
    enable_flash_attention: bool = True  # If available
    cudnn_benchmark: bool = True  # Optimize cuDNN for consistent inputs

    # Profiling (development only)
    enable_profiling: bool = False
    profile_memory: bool = False
    profile_with_stack: bool = False
