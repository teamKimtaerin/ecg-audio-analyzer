"""
Base Configuration Settings
Performance-optimized configuration for ECG Audio Analysis Pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BaseConfig:
    """Base configuration for the audio analysis pipeline"""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/ecg-audio-analysis"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Performance settings
    max_workers: int = 4
    chunk_size_seconds: int = 30  # Process audio in 30-second chunks
    max_memory_gb: float = 8.0  # Maximum memory usage per process

    # Audio processing settings
    sample_rate: int = 16000  # Standard sample rate for speech processing
    audio_format: str = "wav"
    bit_depth: int = 16
    channels: int = 1  # Mono audio for speech analysis

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_performance_logging: bool = True

    # File handling
    supported_formats: List[str] = field(
        default_factory=lambda: [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",  # Video formats
            ".wav",
            ".mp3",
            ".flac",
            ".aac",
            ".m4a",  # Audio formats
        ]
    )
    max_file_size_gb: float = 5.0

    # Pipeline settings
    pipeline_timeout_minutes: int = 30
    retry_attempts: int = 3
    enable_cleanup: bool = True  # Clean up temp files after processing

    def __post_init__(self):
        """Create necessary directories on initialization"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessingConfig:
    """Configuration for processing optimization"""

    # Parallel processing
    enable_multiprocessing: bool = True
    enable_async_io: bool = True
    io_timeout_seconds: int = 300

    # Memory management
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.8  # Clean up when 80% memory used
    garbage_collection_interval: int = 100  # Collect garbage every N operations

    # Error handling
    continue_on_error: bool = True
    error_recovery_attempts: int = 2

    # Performance monitoring
    track_processing_time: bool = True
    track_memory_usage: bool = True
    enable_profiling: bool = False  # Enable for development only


@dataclass
class ValidationConfig:
    """Configuration for input validation and quality checks"""

    # Audio quality validation
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 14400  # 4 hours max
    min_sample_rate: int = 8000
    max_sample_rate: int = 48000

    # Content validation
    min_speech_ratio: float = 0.1  # At least 10% should contain speech
    max_silence_ratio: float = 0.9  # Maximum 90% silence allowed

    # Quality thresholds
    min_audio_quality_score: float = 0.3
    enable_quality_checks: bool = True
