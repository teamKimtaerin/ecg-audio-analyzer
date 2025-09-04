"""
Model Configuration Settings
GPU-optimized model parameters for high-performance audio analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpeakerDiarizationConfig:
    """Configuration for pyannote-audio speaker diarization"""

    # Model selection
    model_name: str = "pyannote/speaker-diarization"
    model_version: str = "3.1"
    pipeline_config: Optional[str] = None

    # Processing parameters
    min_speakers: int = 1
    max_speakers: int = (
        4  # Increased for better 3-speaker detection (slight flexibility)
    )
    segmentation_onset: float = 0.65  # More selective speech detection
    segmentation_offset: float = 0.65  # More selective speech detection

    # GPU optimization
    device: str = "cuda:0"
    batch_size: int = 8
    enable_fp16: bool = True  # Half precision for speed

    # Performance tuning (balanced accuracy + speed)
    clustering_method: str = "centroid"  # Faster method for speed optimization
    min_segment_duration: float = 1.5  # Reduced for faster processing
    min_speakers_count: int = 1
    max_speakers_count: int = 3
    min_cluster_size: int = 2  # Prevent single-segment speakers
    min_speaker_duration: float = 5.0  # Reduced threshold for faster processing

    # Speed optimization flags
    enable_fast_mode: bool = True  # Enable faster but slightly less accurate processing
    chunk_size_seconds: float = 30.0  # Process in smaller chunks

    # Enhanced mode specific parameters (optimized for 3-speaker detection)
    enhanced_mode_similarity_threshold: float = (
        0.7  # Lower threshold for better 3-speaker separation
    )
    enhanced_mode_min_segment_duration: float = (
        1.0  # Shorter segments for better resolution
    )
    enhanced_mode_clustering_threshold: float = (
        0.6  # More aggressive clustering for 3 speakers
    )
    enhanced_mode_noise_reduction: float = 0.05  # 5% threshold for enhanced accuracy

    # Quality settings
    voice_activity_detection: bool = True
    overlapped_speech_detection: bool = True
    speaker_change_detection_threshold: float = 0.5


@dataclass
class EmotionAnalysisConfig:
    """Configuration for emotion analysis models"""

    # Model selection
    model_name: str = "facebook/wav2vec2-large-960h-lv60-self"
    emotion_classifier: str = "j-hartmann/emotion-english-distilroberta-base"

    # Emotion categories
    emotion_labels: List[str] = field(
        default_factory=lambda: [
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "neutral",
        ]
    )

    # Processing parameters
    sample_rate: int = 16000
    window_length: float = 2.0  # 2-second windows
    window_overlap: float = 0.5  # 50% overlap

    # GPU optimization
    device: str = "cuda:0"
    batch_size: int = 16
    enable_fp16: bool = True
    max_sequence_length: int = 512

    # Classification thresholds
    confidence_threshold: float = 0.6
    emotion_intensity_scaling: bool = True
    enable_valence_arousal: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_embeddings: bool = True


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
