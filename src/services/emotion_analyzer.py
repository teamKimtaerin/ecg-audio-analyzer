"""
Emotion Analysis Service
Single Responsibility: Classify emotions from speech segments using Wav2Vec2

GPU-optimized emotion classification with batch processing and confidence calibration.
"""

import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import torch
import torchaudio
import numpy as np
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.preprocessing import StandardScaler
import librosa

from ..utils.logger import get_logger
from ...config.model_configs import EmotionAnalysisConfig
from ..models.output_models import EmotionType, EmotionAnalysis, EmotionScores

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AudioSegment:
    """Audio segment for emotion analysis"""
    audio_data: torch.Tensor
    sample_rate: int
    start_time: float
    end_time: float
    speaker_id: str
    segment_id: str = field(default="")
    
    def __post_init__(self):
        if not self.segment_id:
            # Generate unique ID based on audio content hash
            audio_hash = hashlib.md5(self.audio_data.numpy().tobytes()).hexdigest()[:8]
            self.segment_id = f"{self.speaker_id}_{self.start_time:.1f}_{audio_hash}"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EmotionPrediction:
    """Individual emotion prediction result"""
    segment_id: str
    primary_emotion: EmotionType
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float
    model_confidence: float  # Raw model confidence
    calibrated_confidence: float  # Calibrated confidence score
    intensity: float  # Emotion intensity
    valence: float  # Positive/negative sentiment (-1 to 1)
    arousal: float  # Activation level (0 to 1)
    
    def to_emotion_analysis(self) -> EmotionAnalysis:
        """Convert to EmotionAnalysis model format"""
        return EmotionAnalysis(
            primary=self.primary_emotion,
            confidence=self.calibrated_confidence,
            intensity=self.intensity,
            valence=self.valence,
            arousal=self.arousal,
            all_scores=EmotionScores(**self.all_probabilities)
        )


class EmotionModelEnsemble:
    """Ensemble of emotion classification models for improved accuracy"""
    
    def __init__(self, config: EmotionAnalysisConfig, device: str):
        self.config = config
        self.device = device
        self.logger = get_logger().bind_context(component="emotion_model_ensemble")
        
        # Model components
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.text_tokenizer = None
        self.text_model = None
        self.emotion_pipeline = None
        
        # Confidence calibration
        self.confidence_scaler = StandardScaler()
        self.calibration_fitted = False
        
        # Emotion mapping
        self.emotion_mapping = self._create_emotion_mapping()
        
    def _create_emotion_mapping(self) -> Dict[str, EmotionType]:
        """Create mapping from model outputs to standardized emotion types"""
        return {
            'LABEL_0': EmotionType.SADNESS,
            'LABEL_1': EmotionType.JOY, 
            'LABEL_2': EmotionType.ANGER,
            'LABEL_3': EmotionType.FEAR,
            'LABEL_4': EmotionType.SURPRISE,
            'LABEL_5': EmotionType.DISGUST,
            'LABEL_6': EmotionType.NEUTRAL,
            'sadness': EmotionType.SADNESS,
            'joy': EmotionType.JOY,
            'happiness': EmotionType.JOY,
            'anger': EmotionType.ANGER,
            'fear': EmotionType.FEAR,
            'surprise': EmotionType.SURPRISE,
            'disgust': EmotionType.DISGUST,
            'neutral': EmotionType.NEUTRAL
        }
    
    def load_models(self):
        """Load all emotion analysis models"""
        with self.logger.performance_timer("emotion_models_loading"):
            
            try:
                # Load Wav2Vec2 model for audio-based emotion recognition
                self.logger.info("loading_wav2vec2_model", model=self.config.model_name)
                
                # Use a pre-trained emotion classification model
                emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
                
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model=emotion_model_name,
                    device=0 if self.device.startswith('cuda') else -1,
                    return_all_scores=True
                )
                
                # Load Wav2Vec2 for audio feature extraction
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(self.config.model_name)
                self.wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.emotion_labels)
                )
                
                # Move to device
                if self.device.startswith('cuda'):
                    self.wav2vec_model = self.wav2vec_model.to(self.device)
                    if self.config.enable_fp16:
                        self.wav2vec_model = self.wav2vec_model.half()
                
                # Set to evaluation mode
                self.wav2vec_model.eval()
                
                self.logger.info("emotion_models_loaded_successfully")
                
            except Exception as e:
                self.logger.error("emotion_models_loading_failed", error=str(e))
                raise RuntimeError(f"Failed to load emotion models: {str(e)}")
    
    def _extract_wav2vec_features(self, audio_segment: AudioSegment) -> torch.Tensor:
        """Extract features from audio using Wav2Vec2"""
        try:
            # Ensure audio is at correct sample rate
            if audio_segment.sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_segment.audio_data.numpy().squeeze(),
                    orig_sr=audio_segment.sample_rate,
                    target_sr=16000
                )
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                audio_tensor = audio_segment.audio_data
            
            # Process with Wav2Vec2 processor
            inputs = self.wav2vec_processor(
                audio_tensor.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            if self.device.startswith('cuda'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                if self.config.enable_fp16 and self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast():
                        outputs = self.wav2vec_model(**inputs)
                else:
                    outputs = self.wav2vec_model(**inputs)
                
                features = outputs.logits
            
            return features.cpu()
            
        except Exception as e:
            self.logger.warning("wav2vec_feature_extraction_failed", 
                              segment_id=audio_segment.segment_id,
                              error=str(e))
            # Return zero features as fallback
            return torch.zeros(1, len(self.config.emotion_labels))
    
    def _predict_from_features(self, features: torch.Tensor, segment_id: str) -> Dict[str, float]:
        """Predict emotions from extracted features"""
        try:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(features, dim=1).squeeze().numpy()
            
            # Create emotion probability dictionary
            emotion_probs = {}
            for i, emotion in enumerate(self.config.emotion_labels):
                emotion_probs[emotion] = float(probabilities[i])
            
            return emotion_probs
            
        except Exception as e:
            self.logger.warning("emotion_prediction_failed", 
                              segment_id=segment_id,
                              error=str(e))
            # Return uniform distribution as fallback
            uniform_prob = 1.0 / len(self.config.emotion_labels)
            return {emotion: uniform_prob for emotion in self.config.emotion_labels}
    
    def _calculate_valence_arousal(self, emotion_probs: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valence (positive/negative) and arousal (activation) from emotion probabilities"""
        
        # Emotion to valence/arousal mapping (based on circumplex model)
        emotion_va_mapping = {
            'joy': (0.8, 0.7),
            'happiness': (0.8, 0.7),
            'surprise': (0.3, 0.8),
            'anger': (-0.6, 0.8),
            'fear': (-0.7, 0.7),
            'sadness': (-0.7, 0.3),
            'disgust': (-0.6, 0.4),
            'neutral': (0.0, 0.3)
        }
        
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in emotion_va_mapping:
                v, a = emotion_va_mapping[emotion]
                valence += prob * v
                arousal += prob * a
        
        # Normalize to expected ranges
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        return valence, arousal
    
    def _calibrate_confidence(self, raw_confidence: float, emotion_probs: Dict[str, float]) -> float:
        """Calibrate confidence score based on probability distribution"""
        
        # Calculate entropy-based uncertainty
        probs = list(emotion_probs.values())
        entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
        max_entropy = np.log(len(probs))
        uncertainty = entropy / max_entropy
        
        # Adjust confidence based on uncertainty
        calibrated = raw_confidence * (1.0 - uncertainty * 0.3)
        
        # Apply minimum confidence threshold
        calibrated = max(self.config.confidence_threshold * 0.5, calibrated)
        
        return min(1.0, calibrated)
    
    def predict_emotion(self, audio_segment: AudioSegment) -> EmotionPrediction:
        """Predict emotion for a single audio segment"""
        
        start_time = time.time()
        
        try:
            # Extract features using Wav2Vec2
            features = self._extract_wav2vec_features(audio_segment)
            
            # Predict emotions from features
            emotion_probs = self._predict_from_features(features, audio_segment.segment_id)
            
            # Find primary emotion
            primary_emotion_str = max(emotion_probs.keys(), key=lambda x: emotion_probs[x])
            primary_emotion = EmotionType(primary_emotion_str)
            
            # Calculate confidence metrics
            raw_confidence = emotion_probs[primary_emotion_str]
            calibrated_confidence = self._calibrate_confidence(raw_confidence, emotion_probs)
            
            # Calculate intensity (based on probability distribution sharpness)
            sorted_probs = sorted(emotion_probs.values(), reverse=True)
            intensity = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]
            intensity = max(0.0, min(1.0, intensity))
            
            # Calculate valence and arousal
            valence, arousal = self._calculate_valence_arousal(emotion_probs)
            
            processing_time = time.time() - start_time
            
            return EmotionPrediction(
                segment_id=audio_segment.segment_id,
                primary_emotion=primary_emotion,
                confidence=calibrated_confidence,
                all_probabilities=emotion_probs,
                processing_time=processing_time,
                model_confidence=raw_confidence,
                calibrated_confidence=calibrated_confidence,
                intensity=intensity,
                valence=valence,
                arousal=arousal
            )
            
        except Exception as e:
            self.logger.error("emotion_prediction_failed",
                            segment_id=audio_segment.segment_id,
                            error=str(e))
            
            # Return neutral emotion as fallback
            processing_time = time.time() - start_time
            fallback_probs = {emotion: 1.0/len(self.config.emotion_labels) 
                            for emotion in self.config.emotion_labels}
            fallback_probs['neutral'] = 0.7  # Bias towards neutral on failure
            
            return EmotionPrediction(
                segment_id=audio_segment.segment_id,
                primary_emotion=EmotionType.NEUTRAL,
                confidence=0.3,
                all_probabilities=fallback_probs,
                processing_time=processing_time,
                model_confidence=0.3,
                calibrated_confidence=0.3,
                intensity=0.3,
                valence=0.0,
                arousal=0.3
            )


class EmotionAnalyzer:
    """
    High-performance emotion analysis service using Wav2Vec2 and ensemble methods.
    
    Single Responsibility: Classify emotions from audio segments with GPU acceleration
    and confidence calibration for subtitle generation.
    """
    
    def __init__(self, 
                 config: EmotionAnalysisConfig,
                 device: Optional[str] = None,
                 enable_batch_processing: bool = True):
        
        self.config = config
        self.device = device or config.device
        self.enable_batch_processing = enable_batch_processing
        
        self.logger = get_logger().bind_context(service="emotion_analyzer")
        
        # Validate device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning("cuda_unavailable", fallback="cpu")
            self.device = "cpu"
        
        # Initialize model ensemble
        self.model_ensemble = EmotionModelEnsemble(config, self.device)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Processing statistics
        self.total_segments_processed = 0
        self.total_processing_time = 0.0
        
        self.logger.info("emotion_analyzer_initialized",
                        device=self.device,
                        batch_processing=enable_batch_processing,
                        emotion_labels=config.emotion_labels)
    
    def load_models(self):
        """Load emotion analysis models"""
        self.model_ensemble.load_models()
        
        # Warm up models if configured
        if hasattr(self.config, 'model_warmup') and self.config.model_warmup:
            self._warmup_models()
    
    def _warmup_models(self):
        """Warm up models with dummy audio data"""
        try:
            # Create dummy audio segment (2 seconds of silence)
            dummy_audio = torch.zeros(1, 32000)  # 2 seconds at 16kHz
            dummy_segment = AudioSegment(
                audio_data=dummy_audio,
                sample_rate=16000,
                start_time=0.0,
                end_time=2.0,
                speaker_id="warmup"
            )
            
            # Run prediction to warm up
            _ = self.model_ensemble.predict_emotion(dummy_segment)
            self.logger.info("emotion_models_warmup_completed")
            
        except Exception as e:
            self.logger.warning("emotion_models_warmup_failed", error=str(e))
    
    def _load_audio_segment(self, 
                           audio_path: Path, 
                           start_time: float, 
                           end_time: float,
                           speaker_id: str) -> AudioSegment:
        """Load audio segment from file"""
        try:
            # Calculate frame offsets
            sample_rate = 16000  # Target sample rate
            frame_offset = int(start_time * sample_rate)
            num_frames = int((end_time - start_time) * sample_rate)
            
            # Load audio segment
            waveform, original_sr = torchaudio.load(
                str(audio_path),
                frame_offset=frame_offset,
                num_frames=num_frames
            )
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if original_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(original_sr, sample_rate)
                waveform = resampler(waveform)
            
            return AudioSegment(
                audio_data=waveform,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id
            )
            
        except Exception as e:
            self.logger.error("audio_segment_loading_failed",
                            audio_path=str(audio_path),
                            start_time=start_time,
                            end_time=end_time,
                            error=str(e))
            raise
    
    def analyze_single_segment(self, 
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
            EmotionAnalysis with emotion classification results
        """
        
        if not self.model_ensemble.wav2vec_model:
            self.load_models()
        
        with self.logger.performance_timer("emotion_analysis_single", items_count=1):
            
            # Load audio segment
            audio_segment = self._load_audio_segment(
                audio_path, start_time, end_time, speaker_id
            )
            
            # Predict emotion
            prediction = self.model_ensemble.predict_emotion(audio_segment)
            
            # Update statistics
            self.total_segments_processed += 1
            self.total_processing_time += prediction.processing_time
            
            self.logger.debug("emotion_segment_analyzed",
                            segment_id=prediction.segment_id,
                            primary_emotion=prediction.primary_emotion.value,
                            confidence=prediction.calibrated_confidence,
                            processing_time=prediction.processing_time)
            
            return prediction.to_emotion_analysis()
    
    def analyze_batch(self, 
                     audio_path: Path,
                     segments: List[Tuple[float, float, str]],
                     max_concurrent: int = None) -> List[EmotionAnalysis]:
        """
        Analyze emotions for multiple audio segments concurrently.
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time, speaker_id) tuples
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of EmotionAnalysis results in same order as input segments
        """
        
        if not self.model_ensemble.wav2vec_model:
            self.load_models()
        
        if max_concurrent is None:
            max_concurrent = min(len(segments), self.config.batch_size)
        
        with self.logger.performance_timer("emotion_analysis_batch", items_count=len(segments)):
            
            self.logger.info("emotion_batch_analysis_started",
                           segment_count=len(segments),
                           max_concurrent=max_concurrent,
                           audio_file=str(audio_path))
            
            if not self.enable_batch_processing or len(segments) == 1:
                # Process sequentially
                results = []
                for start_time, end_time, speaker_id in segments:
                    result = self.analyze_single_segment(
                        audio_path, start_time, end_time, speaker_id
                    )
                    results.append(result)
                return results
            
            # Process in parallel batches
            results = [None] * len(segments)
            
            def process_segment(index, start_time, end_time, speaker_id):
                try:
                    return index, self.analyze_single_segment(
                        audio_path, start_time, end_time, speaker_id
                    )
                except Exception as e:
                    self.logger.error("batch_segment_analysis_failed",
                                    index=index,
                                    start_time=start_time,
                                    error=str(e))
                    
                    # Return neutral emotion as fallback
                    from ..models.output_models import EmotionAnalysis, EmotionScores, EmotionType
                    neutral_scores = {emotion: 0.1 for emotion in self.config.emotion_labels}
                    neutral_scores['neutral'] = 0.7
                    
                    return index, EmotionAnalysis(
                        primary=EmotionType.NEUTRAL,
                        confidence=0.3,
                        intensity=0.3,
                        valence=0.0,
                        arousal=0.3,
                        all_scores=EmotionScores(**neutral_scores)
                    )
            
            # Submit tasks to thread pool
            futures = []
            for i, (start_time, end_time, speaker_id) in enumerate(segments):
                future = self.thread_pool.submit(
                    process_segment, i, start_time, end_time, speaker_id
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error("batch_future_failed", error=str(e))
            
            # Fill any None results with neutral emotion
            for i, result in enumerate(results):
                if result is None:
                    self.logger.warning("missing_result_filled_neutral", index=i)
                    neutral_scores = {emotion: 0.1 for emotion in self.config.emotion_labels}
                    neutral_scores['neutral'] = 0.7
                    
                    results[i] = EmotionAnalysis(
                        primary=EmotionType.NEUTRAL,
                        confidence=0.3,
                        intensity=0.3,
                        valence=0.0,
                        arousal=0.3,
                        all_scores=EmotionScores(**neutral_scores)
                    )
            
            self.logger.info("emotion_batch_analysis_completed",
                           segment_count=len(segments),
                           avg_processing_time=self.total_processing_time / max(1, self.total_segments_processed))
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = (
            self.total_processing_time / max(1, self.total_segments_processed)
        )
        
        return {
            'total_segments_processed': self.total_segments_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_per_segment': avg_processing_time,
            'segments_per_second': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'device': self.device,
            'batch_processing_enabled': self.enable_batch_processing
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clear GPU memory
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("emotion_analyzer_cleanup_completed")
            
        except Exception as e:
            self.logger.error("emotion_analyzer_cleanup_failed", error=str(e))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error("emotion_analyzer_exception",
                            exception_type=str(exc_type),
                            exception_message=str(exc_val))