"""
Emotion Analysis Service - Refactored Version
Single Responsibility: Classify emotions from speech segments using Wav2Vec2
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa

from ..utils.logger import get_logger
from config.model_configs import EmotionAnalysisConfig
from ..models.output_models import EmotionType, EmotionAnalysis, EmotionScores

warnings.filterwarnings("ignore", category=UserWarning)


# ============== Data Models ==============

@dataclass
class AudioSegment:
    """Simple audio segment container"""
    audio_data: torch.Tensor
    sample_rate: int
    start_time: float
    end_time: float
    speaker_id: str
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EmotionMetrics:
    """Emotion metrics container"""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    intensity: float  # 0 to 1


# ============== Core Components ==============

class AudioProcessor:
    """Handles audio loading and preprocessing"""
    
    TARGET_SAMPLE_RATE = 16000
    
    @staticmethod
    def load_segment(audio_path: Path, 
                    start_time: float, 
                    end_time: float) -> torch.Tensor:
        """Load and preprocess audio segment"""
        frame_offset = int(start_time * AudioProcessor.TARGET_SAMPLE_RATE)
        num_frames = int((end_time - start_time) * AudioProcessor.TARGET_SAMPLE_RATE)
        
        waveform, original_sr = torchaudio.load(
            str(audio_path),
            frame_offset=frame_offset,
            num_frames=num_frames
        )
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if original_sr != AudioProcessor.TARGET_SAMPLE_RATE:
            waveform = AudioProcessor._resample(waveform, original_sr)
        
        return waveform
    
    @staticmethod
    def _resample(waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        resampler = torchaudio.transforms.Resample(
            original_sr, 
            AudioProcessor.TARGET_SAMPLE_RATE
        )
        return resampler(waveform)


class EmotionMetricsCalculator:
    """Calculates emotion metrics from probabilities"""
    
    # Emotion to valence/arousal mapping (circumplex model)
    EMOTION_MAPPING = {
        'joy': (0.8, 0.7),
        'happiness': (0.8, 0.7),
        'surprise': (0.3, 0.8),
        'anger': (-0.6, 0.8),
        'fear': (-0.7, 0.7),
        'sadness': (-0.7, 0.3),
        'disgust': (-0.6, 0.4),
        'neutral': (0.0, 0.3)
    }
    
    @staticmethod
    def calculate(emotion_probs: Dict[str, float]) -> EmotionMetrics:
        """Calculate emotion metrics from probability distribution"""
        valence, arousal = EmotionMetricsCalculator._calculate_valence_arousal(emotion_probs)
        intensity = EmotionMetricsCalculator._calculate_intensity(emotion_probs)
        
        return EmotionMetrics(
            valence=valence,
            arousal=arousal,
            intensity=intensity
        )
    
    @staticmethod
    def _calculate_valence_arousal(emotion_probs: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valence and arousal from emotion probabilities"""
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in EmotionMetricsCalculator.EMOTION_MAPPING:
                v, a = EmotionMetricsCalculator.EMOTION_MAPPING[emotion]
                valence += prob * v
                arousal += prob * a
        
        return (
            np.clip(valence, -1.0, 1.0),
            np.clip(arousal, 0.0, 1.0)
        )
    
    @staticmethod
    def _calculate_intensity(emotion_probs: Dict[str, float]) -> float:
        """Calculate emotion intensity from probability distribution"""
        sorted_probs = sorted(emotion_probs.values(), reverse=True)
        if len(sorted_probs) > 1:
            intensity = sorted_probs[0] - sorted_probs[1]
        else:
            intensity = sorted_probs[0] if sorted_probs else 0.0
        return np.clip(intensity, 0.0, 1.0)


class ConfidenceCalibrator:
    """Calibrates model confidence scores"""
    
    @staticmethod
    def calibrate(raw_confidence: float, 
                 emotion_probs: Dict[str, float],
                 min_threshold: float = 0.3) -> float:
        """Calibrate confidence based on probability distribution entropy"""
        # Calculate entropy-based uncertainty
        probs = list(emotion_probs.values())
        entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
        max_entropy = np.log(len(probs))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        
        # Adjust confidence based on uncertainty
        calibrated = raw_confidence * (1.0 - uncertainty * 0.3)
        
        # Apply minimum threshold
        calibrated = max(min_threshold * 0.5, calibrated)
        
        return min(1.0, calibrated)


class EmotionModel:
    """Simplified emotion classification model wrapper"""
    
    def __init__(self, config: EmotionAnalysisConfig, device: str):
        self.config = config
        self.device = device
        self.logger = get_logger().bind_context(component="emotion_model")
        
        self.processor = None
        self.model = None
        
    def load(self):
        """Load the emotion classification model"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.config.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.config.emotion_labels)
            )
            
            if self.device.startswith('cuda'):
                self.model = self.model.to(self.device)
                if self.config.enable_fp16:
                    self.model = self.model.half()
            
            self.model.eval()
            self.logger.info("model_loaded_successfully")
            
        except Exception as e:
            self.logger.error("model_loading_failed", error=str(e))
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, audio_tensor: torch.Tensor) -> Dict[str, float]:
        """Predict emotion probabilities from audio"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare input
        inputs = self.processor(
            audio_tensor.squeeze().numpy(),
            sampling_rate=AudioProcessor.TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        if self.device.startswith('cuda'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            if self.config.enable_fp16 and self.device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            logits = outputs.logits
        
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Create emotion probability dictionary
        emotion_probs = {}
        for i, emotion in enumerate(self.config.emotion_labels):
            emotion_probs[emotion] = float(probabilities[i])
        
        return emotion_probs


# ============== Main Service ==============

class EmotionAnalyzer:
    """
    Simplified emotion analysis service.
    Single Responsibility: Classify emotions from audio segments.
    """
    
    def __init__(self, config: EmotionAnalysisConfig, device: Optional[str] = None):
        self.config = config
        self.device = self._validate_device(device or config.device)
        self.logger = get_logger().bind_context(service="emotion_analyzer")
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.emotion_model = EmotionModel(config, self.device)
        self.metrics_calculator = EmotionMetricsCalculator()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Statistics
        self.stats = {
            'total_segments': 0,
            'total_time': 0.0
        }
        
        self.logger.info("emotion_analyzer_initialized", device=self.device)
    
    def _validate_device(self, device: str) -> str:
        """Validate and return appropriate device"""
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning("cuda_unavailable_using_cpu")
            return "cpu"
        return device
    
    def load_models(self):
        """Load the emotion classification model"""
        self.emotion_model.load()
    
    def analyze_segment(self,
                       audio_path: Path,
                       start_time: float,
                       end_time: float,
                       speaker_id: str) -> EmotionAnalysis:
        """
        Analyze emotion for a single audio segment.
        
        Args:
            audio_path: Path to audio file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            speaker_id: Speaker identifier
            
        Returns:
            EmotionAnalysis result
        """
        start = time.time()
        
        try:
            # Load audio segment
            audio_tensor = self.audio_processor.load_segment(
                audio_path, start_time, end_time
            )
            
            # Predict emotions
            emotion_probs = self.emotion_model.predict(audio_tensor)
            
            # Find primary emotion
            primary_emotion_str = max(emotion_probs, key=emotion_probs.get)
            primary_emotion = EmotionType(primary_emotion_str)
            raw_confidence = emotion_probs[primary_emotion_str]
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate(emotion_probs)
            
            # Calibrate confidence
            calibrated_confidence = self.confidence_calibrator.calibrate(
                raw_confidence, 
                emotion_probs,
                self.config.confidence_threshold
            )
            
            # Update statistics
            self.stats['total_segments'] += 1
            self.stats['total_time'] += time.time() - start
            
            # Create result
            return EmotionAnalysis(
                primary=primary_emotion,
                confidence=calibrated_confidence,
                intensity=metrics.intensity,
                valence=metrics.valence,
                arousal=metrics.arousal,
                all_scores=EmotionScores(**emotion_probs)
            )
            
        except Exception as e:
            self.logger.error("emotion_analysis_failed",
                            speaker_id=speaker_id,
                            error=str(e))
            return self._get_fallback_emotion()
    
    def analyze_batch(self,
                     audio_path: Path,
                     segments: List[Tuple[float, float, str]]) -> List[EmotionAnalysis]:
        """
        Analyze emotions for multiple segments.
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time, speaker_id) tuples
            
        Returns:
            List of EmotionAnalysis results
        """
        results = []
        
        for start_time, end_time, speaker_id in segments:
            result = self.analyze_segment(
                audio_path, start_time, end_time, speaker_id
            )
            results.append(result)
        
        return results
    
    def _get_fallback_emotion(self) -> EmotionAnalysis:
        """Get fallback neutral emotion"""
        fallback_scores = {emotion: 0.1 for emotion in self.config.emotion_labels}
        fallback_scores['neutral'] = 0.7
        
        return EmotionAnalysis(
            primary=EmotionType.NEUTRAL,
            confidence=0.3,
            intensity=0.3,
            valence=0.0,
            arousal=0.3,
            all_scores=EmotionScores(**fallback_scores)
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get processing statistics"""
        total_segments = max(1, self.stats['total_segments'])
        avg_time = self.stats['total_time'] / total_segments
        
        return {
            'total_segments_processed': self.stats['total_segments'],
            'total_processing_time': self.stats['total_time'],
            'average_time_per_segment': avg_time,
            'segments_per_second': 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("cleanup_completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()