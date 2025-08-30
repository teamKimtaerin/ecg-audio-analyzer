"""
ECG Audio Analyzer - ML Models Package

Real machine learning models for speaker diarization and emotion analysis.
"""

from .model_manager import ModelManager
from .speaker_diarizer import SpeakerDiarizer
from .emotion_analyzer import EmotionAnalyzer

__all__ = [
    "ModelManager",
    "SpeakerDiarizer", 
    "EmotionAnalyzer"
]