"""
Speech Recognizer - Refactored speech-to-text using WhisperX
Fixed issues that were causing speaker_diarizer.py failures
"""

import whisperx
import torch
import numpy as np
import librosa
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings
from ..utils.logger import get_logger
from .model_manager import get_model_manager

# Suppress warnings that can clutter logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="whisperx")


@dataclass
class SpeechResult:
    """Speech recognition result with enhanced error handling"""
    text: str
    confidence: float
    language: Optional[str] = None
    word_segments: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    
    @classmethod
    def from_whisperx_result(cls, result: Dict[str, Any]) -> 'SpeechResult':
        """Create SpeechResult from WhisperX output with robust error handling"""
        try:
            # Handle both dict and list formats from WhisperX
            if isinstance(result, dict):
                segments = result.get("segments", [])
            elif isinstance(result, list):
                segments = result
                result = {"segments": segments}
            else:
                return cls(text="", confidence=0.0, error="Invalid result format")
            
            # Safely extract text from segments
            text_parts = []
            confidences = []
            word_segments = []
            
            for seg in segments:
                # Handle segment text
                seg_text = seg.get("text", "").strip()
                if seg_text:
                    text_parts.append(seg_text)
                
                # Handle confidence
                if "confidence" in seg and seg["confidence"] is not None:
                    confidences.append(float(seg["confidence"]))
                
                # Extract word-level information safely
                words = seg.get("words", [])
                for word in words:
                    if isinstance(word, dict):
                        word_info = {
                            "word": str(word.get("word", "")).strip(),
                            "start": float(word.get("start", 0.0)),
                            "end": float(word.get("end", 0.0)),
                            "confidence": float(word.get("confidence", 0.0))
                        }
                        # Only add valid words
                        if word_info["word"] and word_info["end"] > word_info["start"]:
                            word_segments.append(word_info)
            
            # Combine text
            full_text = " ".join(text_parts).strip()
            
            # Calculate average confidence
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            
            # Get language
            language = result.get("language")
            if language and not isinstance(language, str):
                language = str(language)
            
            return cls(
                text=full_text,
                confidence=avg_confidence,
                language=language,
                word_segments=word_segments if word_segments else None
            )
            
        except Exception as e:
            return cls(
                text="",
                confidence=0.0,
                error=f"Failed to parse result: {str(e)}"
            )
    
    @classmethod
    def empty(cls, error: Optional[str] = None) -> 'SpeechResult':
        """Create an empty result with optional error message"""
        return cls(text="", confidence=0.0, error=error)


class SpeechRecognizer:
    """Refactored speech-to-text using WhisperX with improved stability"""
    
    # Model size mappings for validation
    VALID_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    def __init__(self, 
                 model_size: str = "base",
                 device: Optional[str] = None,
                 compute_type: str = "float16",
                 language: Optional[str] = None,
                 enable_alignment: bool = True,
                 enable_acoustics: bool = False):  # Disabled by default for stability
        """
        Initialize speech recognizer with validation
        
        Args:
            model_size: WhisperX model size
            device: Device to use (None for auto-detection)
            compute_type: Compute type
            language: Language code (None for auto-detection)
            enable_alignment: Whether to use alignment model
            enable_acoustics: Whether to perform acoustic analysis (can cause issues)
        """
        # Validate model size
        if model_size not in self.VALID_MODEL_SIZES:
            model_size = "base"
        
        self.model_size = model_size
        self.language = language
        self.enable_alignment = enable_alignment
        self.enable_acoustics = enable_acoustics
        
        # Initialize model manager and device
        self.model_manager = get_model_manager(device=device)
        self.device = self._get_safe_device()
        
        # Set compute type based on device
        self.compute_type = self._get_safe_compute_type(compute_type)
        
        # Initialize logger
        self.logger = get_logger().bind_context(
            service="speech_recognizer",
            model=f"whisperx-{model_size}",
            device=self.device,
            compute_type=self.compute_type
        )
        
        # Model state
        self.whisper_model = None
        self.alignment_models = {}  # Cache per language
        self._model_lock = False  # Prevent concurrent model loading
    
    def _get_safe_device(self) -> str:
        """Get device with fallback to CPU if GPU fails"""
        try:
            device = self.model_manager.get_device()
            if device == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return device
        except Exception:
            return "cpu"
    
    def _get_safe_compute_type(self, requested_type: str) -> str:
        """Get appropriate compute type for device"""
        if self.device == "cpu":
            return "int8"  # Better CPU performance
        elif self.device == "cuda":
            # Check GPU capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:  # Volta or newer
                    return "float16"
                else:
                    return "float32"
        return "float32"
    
    @contextmanager
    def _model_loading_lock(self):
        """Prevent concurrent model loading"""
        if self._model_lock:
            yield False
        else:
            self._model_lock = True
            try:
                yield True
            finally:
                self._model_lock = False
    
    def _load_whisper_model(self) -> bool:
        """Load WhisperX model with error recovery"""
        if self.whisper_model is not None:
            return True
        
        with self._model_loading_lock() as acquired:
            if not acquired:
                self.logger.warning("model_loading_in_progress")
                return False
            
            try:
                self.logger.info("loading_whisperx_model", 
                               model_size=self.model_size,
                               device=self.device,
                               compute_type=self.compute_type)
                
                # Load with error handling
                self.whisper_model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language,
                    threads=4 if self.device == "cpu" else 1
                )
                
                self.logger.info("whisperx_model_loaded")
                return True
                
            except Exception as e:
                self.logger.error("whisperx_model_loading_failed", error=str(e))
                
                # Try fallback to CPU
                if self.device != "cpu":
                    self.logger.info("falling_back_to_cpu")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    
                    try:
                        self.whisper_model = whisperx.load_model(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                            language=self.language,
                            threads=4
                        )
                        self.logger.info("whisperx_model_loaded_cpu_fallback")
                        return True
                    except Exception as e2:
                        self.logger.error("cpu_fallback_failed", error=str(e2))
                
                return False
    
    def _load_alignment_model(self, language_code: str) -> bool:
        """Load alignment model for a specific language"""
        if not self.enable_alignment:
            return False
        
        # Check cache
        if language_code in self.alignment_models:
            return self.alignment_models[language_code] is not None
        
        try:
            self.logger.info("loading_alignment_model", language=language_code)
            
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            
            self.alignment_models[language_code] = (model, metadata)
            self.logger.info("alignment_model_loaded", language=language_code)
            return True
            
        except Exception as e:
            self.logger.warning("alignment_model_failed", 
                              language=language_code, 
                              error=str(e))
            self.alignment_models[language_code] = None
            return False
    
    def transcribe_audio_segment(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int = 16000) -> SpeechResult:
        """
        Transcribe audio segment with improved error handling
        """
        # Validate input
        if audio_data is None or len(audio_data) == 0:
            return SpeechResult.empty("Empty audio data")
        
        # Load model if needed
        if not self._load_whisper_model():
            return SpeechResult.empty("Failed to load model")
        
        try:
            # Prepare audio
            audio_data = self._prepare_audio(audio_data, sample_rate)
            
            # Transcribe with WhisperX
            result = self._transcribe_with_whisperx(audio_data)
            
            if result is None:
                return SpeechResult.empty("Transcription failed")
            
            # Align if enabled and available
            if self.enable_alignment:
                result = self._align_transcription(result, audio_data)
            
            # Convert to SpeechResult
            speech_result = SpeechResult.from_whisperx_result(result)
            
            # Add acoustic analysis if enabled (and safe)
            if self.enable_acoustics and speech_result.word_segments:
                speech_result = self._add_acoustic_features_safe(
                    speech_result, audio_data, 16000
                )
            
            return speech_result
            
        except Exception as e:
            self.logger.error("transcription_failed", error=str(e))
            return SpeechResult.empty(f"Transcription error: {str(e)}")
    
    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio for transcription"""
        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
        
        return audio_data
    
    def _transcribe_with_whisperx(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform WhisperX transcription with error handling"""
        try:
            # Adaptive batch size
            if self.device == "cpu":
                batch_size = 8
            else:
                # Check available GPU memory
                if torch.cuda.is_available():
                    mem_free = torch.cuda.get_device_properties(0).total_memory
                    if mem_free > 8 * 1024**3:  # 8GB+
                        batch_size = 32
                    else:
                        batch_size = 16
                else:
                    batch_size = 16
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_data,
                batch_size=batch_size
            )
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            self.logger.warning("gpu_oom_during_transcription")
            torch.cuda.empty_cache()
            
            # Retry with smaller batch
            try:
                result = self.whisper_model.transcribe(
                    audio_data,
                    batch_size=4
                )
                return result
            except Exception as e:
                self.logger.error("transcription_retry_failed", error=str(e))
                return None
                
        except Exception as e:
            self.logger.error("whisperx_transcription_error", error=str(e))
            return None
    
    def _align_transcription(self, result: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Align transcription with audio"""
        try:
            language = result.get("language", "en")
            
            if not self._load_alignment_model(language):
                return result
            
            model, metadata = self.alignment_models[language]
            
            aligned_result = whisperx.align(
                result["segments"],
                model,
                metadata,
                audio_data,
                self.device,
                return_char_alignments=False
            )
            
            # Preserve language info
            aligned_result["language"] = language
            return aligned_result
            
        except Exception as e:
            self.logger.warning("alignment_failed", error=str(e))
            return result
    
    def _add_acoustic_features_safe(self, 
                                   speech_result: SpeechResult,
                                   audio_data: np.ndarray,
                                   sample_rate: int) -> SpeechResult:
        """Add acoustic features with safety checks"""
        try:
            # Only process if we have valid word segments
            if not speech_result.word_segments:
                return speech_result
            
            # Limit processing to prevent memory issues
            max_words = 100
            if len(speech_result.word_segments) > max_words:
                self.logger.warning("too_many_words_for_acoustics", 
                                  count=len(speech_result.word_segments))
                return speech_result
            
            # Process acoustics
            enhanced_segments = []
            for word in speech_result.word_segments[:max_words]:
                enhanced_word = self._analyze_word_safe(
                    word, audio_data, sample_rate
                )
                enhanced_segments.append(enhanced_word)
            
            speech_result.word_segments = enhanced_segments
            return speech_result
            
        except Exception as e:
            self.logger.warning("acoustic_analysis_skipped", error=str(e))
            return speech_result
    
    def _analyze_word_safe(self, 
                          word_data: Dict[str, Any],
                          audio_data: np.ndarray,
                          sample_rate: int) -> Dict[str, Any]:
        """Analyze single word with safety checks"""
        enhanced_word = word_data.copy()
        
        try:
            start = word_data.get("start", 0.0)
            end = word_data.get("end", 0.0)
            
            if end <= start:
                # Invalid timing
                enhanced_word.update({
                    "volume_db": -60.0,
                    "pitch_hz": 0.0
                })
                return enhanced_word
            
            # Extract word audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Bounds check
            start_sample = max(0, min(start_sample, len(audio_data) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio_data)))
            
            word_audio = audio_data[start_sample:end_sample]
            
            # Simple acoustic features
            if len(word_audio) > 0:
                # Volume (RMS in dB)
                rms = np.sqrt(np.mean(word_audio ** 2))
                volume_db = 20 * np.log10(max(rms, 1e-8))
                
                # Simple pitch estimation (zero-crossing rate)
                zcr = np.sum(np.abs(np.diff(np.sign(word_audio)))) / (2 * len(word_audio))
                pitch_hz = zcr * sample_rate / 2  # Rough approximation
                
                enhanced_word.update({
                    "volume_db": float(volume_db),
                    "pitch_hz": float(pitch_hz)
                })
            else:
                enhanced_word.update({
                    "volume_db": -60.0,
                    "pitch_hz": 0.0
                })
                
        except Exception:
            enhanced_word.update({
                "volume_db": -60.0,
                "pitch_hz": 0.0
            })
        
        return enhanced_word
    
    def batch_transcribe_segments(self,
                                 audio_path: Union[str, Path],
                                 segments: List[Tuple[float, float]],
                                 sample_rate: int = 16000) -> List[SpeechResult]:
        """
        Optimized batch transcription with proper error handling
        """
        if not segments:
            return []
        
        self.logger.info("batch_transcription_start", 
                        segments_count=len(segments))
        
        try:
            # Load and prepare audio once
            audio_data, sr = librosa.load(str(audio_path), sr=sample_rate)
            audio_data = self._prepare_audio(audio_data, sr)
            
            # Choose strategy based on segment count
            if len(segments) < 5:
                # For few segments, process individually
                return self._process_segments_individually(
                    audio_data, segments, sample_rate
                )
            else:
                # For many segments, use full transcription approach
                return self._process_segments_batch(
                    audio_data, segments, sample_rate
                )
                
        except Exception as e:
            self.logger.error("batch_transcription_failed", error=str(e))
            # Return empty results
            return [SpeechResult.empty(f"Batch error: {str(e)}") 
                   for _ in segments]
    
    def _process_segments_individually(self,
                                      audio_data: np.ndarray,
                                      segments: List[Tuple[float, float]],
                                      sample_rate: int) -> List[SpeechResult]:
        """Process segments one by one"""
        results = []
        
        for start_time, end_time in segments:
            # Extract segment audio
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Bounds check
            start_sample = max(0, min(start_sample, len(audio_data) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio_data)))
            
            segment_audio = audio_data[start_sample:end_sample]
            
            # Transcribe
            result = self.transcribe_audio_segment(segment_audio, sample_rate)
            results.append(result)
        
        return results
    
    def _process_segments_batch(self,
                               audio_data: np.ndarray,
                               segments: List[Tuple[float, float]],
                               sample_rate: int) -> List[SpeechResult]:
        """Process all segments using full transcription"""
        # Load model first
        if not self._load_whisper_model():
            return [SpeechResult.empty("Model loading failed") for _ in segments]
        
        try:
            # Transcribe full audio
            full_result = self._transcribe_with_whisperx(audio_data)
            
            if full_result is None:
                return [SpeechResult.empty("Full transcription failed") 
                       for _ in segments]
            
            # Align if enabled
            if self.enable_alignment:
                full_result = self._align_transcription(full_result, audio_data)
            
            # Map to segments
            return self._map_to_segments(full_result, segments)
            
        except Exception as e:
            self.logger.error("batch_processing_error", error=str(e))
            return [SpeechResult.empty(f"Batch error: {str(e)}") 
                   for _ in segments]
    
    def _map_to_segments(self,
                        full_result: Dict[str, Any],
                        segments: List[Tuple[float, float]]) -> List[SpeechResult]:
        """Map full transcription to time segments"""
        # Extract all words with timing
        all_words = []
        
        for segment in full_result.get("segments", []):
            words = segment.get("words", [])
            for word in words:
                if isinstance(word, dict) and "start" in word and "end" in word:
                    all_words.append({
                        "word": str(word.get("word", "")).strip(),
                        "start": float(word.get("start", 0.0)),
                        "end": float(word.get("end", 0.0)),
                        "confidence": float(word.get("confidence", 0.0))
                    })
        
        # Map words to segments
        results = []
        language = full_result.get("language", "en")
        
        for start_time, end_time in segments:
            # Find overlapping words
            segment_words = []
            
            for word in all_words:
                word_mid = (word["start"] + word["end"]) / 2
                # Check if word center is within segment
                if start_time <= word_mid <= end_time:
                    segment_words.append(word)
            
            # Create result
            if segment_words:
                text = " ".join([w["word"] for w in segment_words])
                confidence = np.mean([w["confidence"] for w in segment_words])
                
                result = SpeechResult(
                    text=text,
                    confidence=float(confidence),
                    language=language,
                    word_segments=segment_words
                )
            else:
                result = SpeechResult.empty()
            
            results.append(result)
        
        return results
    
    def get_transcription_summary(self, results: List[SpeechResult]) -> Dict[str, Any]:
        """Get summary statistics with error handling"""
        if not results:
            return {
                "total_segments": 0,
                "transcribed_segments": 0,
                "empty_segments": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "total_words": 0,
                "errors": []
            }
        
        # Collect statistics
        total = len(results)
        transcribed = [r for r in results if r.text.strip()]
        errors = [r.error for r in results if r.error]
        
        # Calculate metrics
        success_rate = len(transcribed) / total if total > 0 else 0.0
        avg_confidence = np.mean([r.confidence for r in transcribed]) if transcribed else 0.0
        total_words = sum(len(r.text.split()) for r in transcribed)
        
        # Language statistics
        languages = [r.language for r in results if r.language]
        language_dist = {}
        if languages:
            for lang in set(languages):
                language_dist[lang] = languages.count(lang)
        
        return {
            "total_segments": total,
            "transcribed_segments": len(transcribed),
            "empty_segments": total - len(transcribed),
            "success_rate": float(success_rate),
            "average_confidence": float(avg_confidence),
            "total_words": total_words,
            "dominant_language": max(language_dist, key=language_dist.get) if language_dist else None,
            "language_distribution": language_dist,
            "errors": errors[:10]  # Limit error list
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear model references
            self.whisper_model = None
            self.alignment_models.clear()
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("cleanup_completed")
            
        except Exception as e:
            self.logger.error("cleanup_failed", error=str(e))