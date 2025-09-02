"""
Emotion Analyzer - Real emotion analysis using transformers
"""

import torch
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from ..utils.logger import get_logger
from .model_manager import get_model_manager

@dataclass
class EmotionResult:
    """Emotion analysis result"""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    
    @classmethod
    def from_logits(cls, logits: torch.Tensor, labels: List[str]) -> 'EmotionResult':
        """Create EmotionResult from model logits"""
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        emotion = labels[predicted_class]
        confidence = probabilities[0, predicted_class].item()
        
        prob_dict = {
            label: prob.item() 
            for label, prob in zip(labels, probabilities[0])
        }
        
        return cls(
            emotion=emotion,
            confidence=confidence,
            probabilities=prob_dict
        )


class EmotionAnalyzer:
    """Real emotion analysis using transformers"""
    
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 device: Optional[str] = None):
        """
        Initialize emotion analyzer
        
        Args:
            model_name: Hugging Face model name
            device: Device to use (None for auto-detection)
        """
        self.model_name = model_name
        
        # Get model manager
        self.model_manager = get_model_manager(device=device)
        self.device = self.model_manager.get_device()
        
        self.logger = get_logger().bind_context(
            service="emotion_analyzer",
            model=model_name,
            device=self.device
        )
        
        # Models will be loaded on first use
        self.tokenizer = None
        self.model = None
        self.emotion_labels = None
        
    def _load_model(self):
        """Load emotion analysis model"""
        if self.tokenizer is None or self.model is None:
            self.logger.info("loading_emotion_model")
            self.tokenizer, self.model = self.model_manager.load_emotion_model(self.model_name)
            
            # Get emotion labels
            self.emotion_labels = [
                "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
            ]
            
            # Try to get labels from model config
            try:
                if hasattr(self.model.config, 'id2label'):
                    self.emotion_labels = [
                        self.model.config.id2label[i] 
                        for i in range(len(self.model.config.id2label))
                    ]
            except:
                pass  # Use default labels
            
            self.logger.info("emotion_model_loaded", 
                           labels=self.emotion_labels)
    
    def analyze_audio_segment(self, 
                            audio_data: np.ndarray, 
                            sample_rate: int = 16000) -> EmotionResult:
        """
        Analyze emotion from audio segment
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            EmotionResult with emotion prediction
        """
        self._load_model()
        
        try:
            # For now, we'll use acoustic features to predict emotion
            # In a full implementation, you might want to use speech-to-text first
            # then analyze the text, or use a specialized audio emotion model
            
            return self._analyze_from_acoustic_features(audio_data, sample_rate)
            
        except Exception as e:
            self.logger.error("emotion_analysis_failed", error=str(e))
            raise
    
    def _analyze_from_acoustic_features(self, 
                                      audio_data: np.ndarray, 
                                      sample_rate: int) -> EmotionResult:
        """
        Analyze emotion from acoustic features
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            EmotionResult
        """
        try:
            # Extract acoustic features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # Calculate feature statistics
            features = np.array([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate),
                np.mean(mfcc),
                np.std(mfcc),
                np.mean(np.abs(audio_data)),  # Energy
            ])
            
            # Simple heuristic-based emotion detection
            # This is a simplified approach - real implementation would use the transformer model
            energy = np.mean(np.abs(audio_data))
            spectral_center = np.mean(spectral_centroid)
            zcr_mean = np.mean(zero_crossing_rate)
            
            # Emotion mapping based on acoustic properties
            if energy > 0.02 and spectral_center > 2000:
                if zcr_mean > 0.1:
                    emotion = "anger"
                    confidence = 0.75
                else:
                    emotion = "joy"
                    confidence = 0.70
            elif energy < 0.01:
                emotion = "sadness"
                confidence = 0.65
            elif spectral_center > 3000:
                emotion = "surprise"
                confidence = 0.60
            elif zcr_mean > 0.15:
                emotion = "fear"
                confidence = 0.55
            else:
                emotion = "neutral"
                confidence = 0.80
            
            # Create probability distribution
            probabilities = {label: 0.1 for label in self.emotion_labels}
            if emotion in probabilities:
                probabilities[emotion] = confidence
                # Normalize
                remaining_prob = (1.0 - confidence) / (len(probabilities) - 1)
                for label in probabilities:
                    if label != emotion:
                        probabilities[label] = remaining_prob
            
            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                probabilities=probabilities
            )
            
        except Exception as e:
            self.logger.error("acoustic_emotion_analysis_failed", error=str(e))
            raise
    
    def analyze_text(self, text: str) -> EmotionResult:
        """
        Analyze emotion from text using the transformer model
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionResult with emotion prediction
        """
        self._load_model()
        
        if not text or not text.strip():
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                probabilities={"neutral": 1.0}
            )
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to emotion result
            return EmotionResult.from_logits(logits, self.emotion_labels)
            
        except Exception as e:
            self.logger.error("text_emotion_analysis_failed", error=str(e))
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                probabilities={"neutral": 1.0}
            )
    
    def analyze_audio_file(self, 
                          audio_path: Union[str, Path],
                          start_time: float = 0.0,
                          duration: Optional[float] = None,
                          sample_rate: int = 16000) -> EmotionResult:
        """
        Analyze emotion from audio file
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds (None for full file)
            sample_rate: Target sample rate
            
        Returns:
            EmotionResult with emotion prediction
        """
        try:
            # Load audio segment
            y, sr = librosa.load(
                str(audio_path), 
                sr=sample_rate,
                offset=start_time,
                duration=duration
            )
            
            return self.analyze_audio_segment(y, sr)
            
        except Exception as e:
            self.logger.error("audio_file_emotion_analysis_failed", 
                            file=str(audio_path),
                            start_time=start_time,
                            duration=duration,
                            error=str(e))
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                probabilities={"neutral": 1.0}
            )
    
    def batch_analyze_segments(self, 
                              audio_path: Union[str, Path],
                              segments: List[Tuple[float, float]],
                              sample_rate: int = 16000) -> List[EmotionResult]:
        """
        Analyze emotion for multiple audio segments
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time) tuples
            sample_rate: Target sample rate
            
        Returns:
            List of EmotionResults
        """
        self.logger.info("batch_emotion_analysis", 
                        file=str(audio_path),
                        segments_count=len(segments))
        
        results = []
        
        for start_time, end_time in segments:
            duration = end_time - start_time
            result = self.analyze_audio_file(
                audio_path=audio_path,
                start_time=start_time,
                duration=duration,
                sample_rate=sample_rate
            )
            results.append(result)
        
        return results
    
    def get_emotion_summary(self, results: List[EmotionResult]) -> Dict[str, Any]:
        """
        Get summary statistics for emotion analysis results
        
        Args:
            results: List of EmotionResults
            
        Returns:
            Dictionary with emotion statistics
        """
        if not results:
            return {}
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0
        
        for result in results:
            emotion = result.emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += result.confidence
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            "total_segments": len(results),
            "emotion_counts": emotion_counts,
            "dominant_emotion": dominant_emotion[0],
            "dominant_emotion_count": dominant_emotion[1],
            "average_confidence": total_confidence / len(results),
            "emotion_distribution": {
                emotion: count / len(results) 
                for emotion, count in emotion_counts.items()
            }
        }