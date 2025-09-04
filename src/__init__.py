"""
ECG Audio Analyzer - High-Performance Audio Analysis Library

A comprehensive library for audio analysis with speaker diarization,
emotion detection, and acoustic feature extraction.

Example usage:
    from ecg_audio_analyzer import analyze_audio, AnalysisConfig

    # Simple analysis
    result = await analyze_audio("path/to/video.mp4")

    # Advanced analysis with custom configuration
    config = AnalysisConfig(
        enable_gpu=True,
        language="en",
        emotion_detection=True,
        detailed_features=True
    )
    result = await analyze_audio("path/to/video.mp4", config=config)
"""

__version__ = "1.0.0"
__author__ = "ECG Audio Analysis Team"
__email__ = "team@ecg-audio.ai"
__description__ = "High-performance audio analysis for dynamic subtitle generation"

# Import main API functions
from .api import (
    analyze_audio,
    analyze_audio_sync,
    AnalysisConfig,
    AnalysisResult,
    AudioSegment,
    SpeakerInfo,
    EmotionInfo,
    AcousticFeatures,
)

# Import utility functions
from .utils.logger import get_logger

# Version information
VERSION_INFO = {
    "version": __version__,
    "python_requires": ">=3.9",
    "features": [
        "Speaker Diarization",
        "Emotion Analysis",
        "Acoustic Feature Extraction",
        "GPU Acceleration",
        "AWS Integration",
        "High-Performance Processing",
    ],
    "supported_formats": ["MP4", "WAV", "YouTube URLs"],
    "deployment_options": ["Docker", "AWS", "Local"],
}

__all__ = [
    # Main API
    "analyze_audio",
    "analyze_audio_sync",
    "AnalysisConfig",
    "AnalysisResult",
    "AudioSegment",
    "SpeakerInfo",
    "EmotionInfo",
    "AcousticFeatures",
    # Utilities
    "get_logger",
    # Metadata
    "__version__",
    "VERSION_INFO",
]


def get_version():
    """Get the current version of ECG Audio Analyzer."""
    return __version__


def get_supported_formats():
    """Get list of supported audio/video formats."""
    return VERSION_INFO["supported_formats"]


def get_system_info():
    """Get system and package information for debugging."""
    import sys
    import platform
    import torch

    try:
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except:
        gpu_available = False
        gpu_count = 0

    return {
        "ecg_audio_analyzer_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "supported_formats": get_supported_formats(),
    }
