"""
ECG Audio Analyzer - ML Models Package

Real machine learning models for speaker diarization and emotion analysis.
"""

from .model_manager import ModelManager
from .emotion_analyzer import EmotionAnalyzer
from .speech_recognizer import WhisperXPipeline, SpeechRecognizer

__all__ = [
    "ModelManager",
    "EmotionAnalyzer",
    "WhisperXPipeline",
    "SpeechRecognizer"
]