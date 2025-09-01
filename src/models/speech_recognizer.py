"""
Speech Recognizer - Real speech-to-text using WhisperX
"""

import whisperx
import torch
import torchaudio
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from ..utils.logger import get_logger
from .model_manager import get_model_manager

@dataclass
class SpeechResult:
    """Speech recognition result"""
    text: str
    confidence: float
    language: Optional[str] = None
    word_segments: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_whisperx_result(cls, result: Dict[str, Any]) -> 'SpeechResult':
        """Create SpeechResult from WhisperX output"""
        # WhisperX returns segments with word-level timestamps
        segments = result.get("segments", [])
        
        # Combine all text from segments
        full_text = " ".join([seg.get("text", "").strip() for seg in segments]).strip()
        
        # Calculate average confidence if available
        confidences = []
        word_segments = []
        
        for seg in segments:
            # Segment-level confidence
            if "confidence" in seg:
                confidences.append(seg["confidence"])
            
            # Word-level information
            words = seg.get("words", [])
            for word in words:
                word_segments.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                    "confidence": word.get("confidence", 0.0)
                })
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        language = result.get("language", None)
        
        return cls(
            text=full_text,
            confidence=float(avg_confidence),
            language=language,
            word_segments=word_segments if word_segments else None
        )


class SpeechRecognizer:
    """Real speech-to-text using WhisperX"""
    
    def __init__(self, 
                 model_size: str = "base",
                 device: Optional[str] = None,
                 compute_type: str = "float16",
                 language: Optional[str] = None):
        """
        Initialize speech recognizer
        
        Args:
            model_size: WhisperX model size (tiny, base, small, medium, large-v2)
            device: Device to use (None for auto-detection)
            compute_type: Compute type (float16, float16, int8)
            language: Language code (None for auto-detection)
        """
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        
        # Get model manager
        self.model_manager = get_model_manager(device=device)
        self.device = self.model_manager.get_device()
        
        # Use CPU compute type if on CPU
        if self.device == "cpu":
            self.compute_type = "float32"
        
        self.logger = get_logger().bind_context(
            service="speech_recognizer",
            model=f"whisperx-{model_size}",
            device=self.device
        )
        
        # Models will be loaded on first use
        self.whisper_model = None
        self.alignment_model = None
        self.alignment_metadata = None
        
    def _load_models(self):
        """Load WhisperX models"""
        if self.whisper_model is None:
            self.logger.info("loading_whisperx_model")
            
            try:
                # Load WhisperX model
                self.whisper_model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language
                )
                
                self.logger.info("whisperx_model_loaded", 
                               model_size=self.model_size,
                               device=self.device,
                               compute_type=self.compute_type)
                
            except Exception as e:
                self.logger.error("failed_to_load_whisperx_model", error=str(e))
                raise
    
    def _load_alignment_model(self, language_code: str):
        """Load alignment model for better word-level timestamps"""
        try:
            if self.alignment_model is None or getattr(self, '_last_language', None) != language_code:
                self.logger.info("loading_alignment_model", language=language_code)
                
                self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self._last_language = language_code
                
                self.logger.info("alignment_model_loaded", language=language_code)
                
        except Exception as e:
            self.logger.warning("failed_to_load_alignment_model", 
                              language=language_code,
                              error=str(e))
            self.alignment_model = None
            self.alignment_metadata = None
    
    def transcribe_audio_segment(self, 
                               audio_data: np.ndarray, 
                               sample_rate: int = 16000) -> SpeechResult:
        """
        Transcribe audio segment to text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            SpeechResult with transcription
        """
        self._load_models()
        
        try:
            # Ensure audio is float32 and correct sample rate
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # WhisperX expects 16kHz audio
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe with WhisperX - optimized batch size for CPU
            batch_size = 32 if self.device != "cpu" else 16  # Larger batch for CPU with int8
            result = self.whisper_model.transcribe(
                audio_data,
                batch_size=batch_size
            )
            
            # Detect language if not specified
            detected_language = result.get("language", "en")
            
            # Load alignment model for better word-level timestamps
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)
            
            # Align whisper output for better word-level timestamps
            if self.alignment_model is not None:
                try:
                    result = whisperx.align(
                        result["segments"], 
                        self.alignment_model, 
                        self.alignment_metadata,
                        audio_data, 
                        self.device,
                        return_char_alignments=False
                    )
                except Exception as e:
                    self.logger.warning("alignment_failed", error=str(e))
                    # Continue with unaligned result
            
            # Convert to SpeechResult
            speech_result = SpeechResult.from_whisperx_result(result)
            speech_result.language = detected_language
            
            # Perform acoustic analysis on word segments
            if speech_result.word_segments and audio_data is not None:
                self.logger.info("starting_acoustic_analysis", word_count=len(speech_result.word_segments))
                speech_result.word_segments = self._analyze_words_acoustics(
                    speech_result.word_segments, audio_data, sample_rate
                )
                self.logger.info("acoustic_analysis_completed", 
                               processed_words=len(speech_result.word_segments))
            
            return speech_result
            
        except Exception as e:
            self.logger.error("speech_recognition_failed", error=str(e))
            # Return empty result as fallback
            return SpeechResult(
                text="",
                confidence=0.0,
                language=self.language or "en"
            )
    
    def transcribe_audio_file(self, 
                            audio_path: Union[str, Path],
                            start_time: float = 0.0,
                            duration: Optional[float] = None,
                            sample_rate: int = 16000) -> SpeechResult:
        """
        Transcribe audio file segment to text
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds (None for full file)
            sample_rate: Target sample rate
            
        Returns:
            SpeechResult with transcription
        """
        try:
            # Load audio segment
            y, sr = librosa.load(
                str(audio_path), 
                sr=sample_rate,
                offset=start_time,
                duration=duration
            )
            
            return self.transcribe_audio_segment(y, sr)
            
        except Exception as e:
            self.logger.error("audio_file_transcription_failed", 
                            file=str(audio_path),
                            start_time=start_time,
                            duration=duration,
                            error=str(e))
            return SpeechResult(
                text="",
                confidence=0.0,
                language=self.language or "en"
            )
    
    def batch_transcribe_segments(self, 
                                audio_path: Union[str, Path],
                                segments: List[Tuple[float, float]],
                                sample_rate: int = 16000) -> List[SpeechResult]:
        """
        Transcribe multiple audio segments using optimized batch processing
        
        This method transcribes the entire audio file once and then maps
        the results to the provided segments, significantly improving performance.
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time) tuples
            sample_rate: Target sample rate
            
        Returns:
            List of SpeechResults mapped to input segments
        """
        self.logger.info("batch_speech_recognition_optimized", 
                        file=str(audio_path),
                        segments_count=len(segments))
        
        try:
            # Step 1: Load entire audio file for analysis
            audio_data, sr = librosa.load(str(audio_path), sr=sample_rate)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Step 2: Transcribe entire audio file once
            full_transcription = self._transcribe_full_audio(audio_path, sample_rate)
            
            # Step 3: Map transcription results to segments with acoustic analysis
            results = self._map_transcription_to_segments_with_acoustics(
                full_transcription, segments, audio_data, sample_rate
            )
            
            self.logger.info("batch_speech_recognition_completed",
                           segments_processed=len(results),
                           total_words=sum(len(r.text.split()) for r in results if r.text))
            
            return results
            
        except Exception as e:
            self.logger.error("batch_speech_recognition_failed", error=str(e))
            # Fallback to individual segment processing if batch fails
            return self._fallback_individual_processing(audio_path, segments, sample_rate)
    
    def _transcribe_full_audio(self, audio_path: Union[str, Path], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe entire audio file using WhisperX
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            
        Returns:
            Full WhisperX transcription result with word-level timestamps
        """
        self._load_models()
        
        try:
            # Load entire audio file
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            
            # VAD-based optimization: Use larger batch sizes for better throughput
            batch_size = 16 if self.device != "cpu" else 8  # Optimized for VAD-based processing
            
            # Transcribe with WhisperX optimized parameters
            result = self.whisper_model.transcribe(
                y,
                batch_size=batch_size
            )
            
            # Detect language once for entire file
            detected_language = result.get("language", "en")
            self.logger.info("language_detected", language=detected_language)
            
            # Load alignment model once
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)
            
            # Align for word-level timestamps
            if self.alignment_model is not None:
                try:
                    result = whisperx.align(
                        result["segments"], 
                        self.alignment_model, 
                        self.alignment_metadata,
                        y, 
                        self.device,
                        return_char_alignments=False
                    )
                    result["language"] = detected_language
                except Exception as e:
                    self.logger.warning("alignment_failed_full_audio", error=str(e))
                    result["language"] = detected_language
            
            return result
            
        except Exception as e:
            self.logger.error("full_audio_transcription_failed", error=str(e))
            raise
    
    def _map_transcription_to_segments(self, 
                                     full_result: Dict[str, Any], 
                                     segments: List[Tuple[float, float]]) -> List[SpeechResult]:
        """
        Map full transcription results to speaker segments
        
        Args:
            full_result: WhisperX transcription result for entire audio
            segments: List of (start_time, end_time) tuples
            
        Returns:
            List of SpeechResult objects mapped to segments
        """
        try:
            # Extract all word-level segments from WhisperX result
            all_words = []
            for segment in full_result.get("segments", []):
                for word in segment.get("words", []):
                    all_words.append({
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "confidence": word.get("confidence", 0.0)
                    })
            
            # Map words to speaker segments
            results = []
            for start_time, end_time in segments:
                # Find words that overlap with this segment
                segment_words = []
                for word in all_words:
                    word_start = word["start"]
                    word_end = word["end"]
                    
                    # Check if word overlaps with segment (with small tolerance)
                    if (word_start < end_time and word_end > start_time):
                        segment_words.append(word)
                
                # Combine words into text
                text = " ".join([w["word"] for w in segment_words]).strip()
                
                # Calculate average confidence
                avg_confidence = np.mean([w["confidence"] for w in segment_words]) if segment_words else 0.0
                
                # Create SpeechResult
                result = SpeechResult(
                    text=text,
                    confidence=float(avg_confidence),
                    language=full_result.get("language", "en"),
                    word_segments=segment_words if segment_words else None
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error("transcription_mapping_failed", error=str(e))
            # Return empty results as fallback
            return [SpeechResult(text="", confidence=0.0) for _ in segments]
    
    def _map_transcription_to_segments_with_acoustics(self, 
                                                    full_result: Dict[str, Any], 
                                                    segments: List[Tuple[float, float]],
                                                    audio_data: np.ndarray,
                                                    sample_rate: int = 16000) -> List[SpeechResult]:
        """
        Map full transcription results to speaker segments with acoustic analysis
        
        Args:
            full_result: WhisperX transcription result for entire audio
            segments: List of (start_time, end_time) tuples
            audio_data: Full audio data for acoustic analysis
            sample_rate: Sample rate of audio
            
        Returns:
            List of SpeechResult objects with acoustic features
        """
        try:
            # Extract all word-level segments from WhisperX result
            all_words = []
            for segment in full_result.get("segments", []):
                for word in segment.get("words", []):
                    all_words.append({
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "confidence": word.get("confidence", 0.0)
                    })
            
            # Map words to speaker segments with acoustic analysis
            results = []
            for start_time, end_time in segments:
                # Find words that overlap with this segment
                segment_words = []
                for word in all_words:
                    word_start = word["start"]
                    word_end = word["end"]
                    
                    # Check if word overlaps with segment (with small tolerance)
                    if (word_start < end_time and word_end > start_time):
                        # Extract word audio and analyze acoustics
                        word_audio = self._extract_word_audio(
                            audio_data, word_start, word_end, sample_rate
                        )
                        acoustic_features = self._analyze_word_acoustics(word_audio, sample_rate)
                        
                        # Add acoustic features to word
                        enhanced_word = word.copy()
                        enhanced_word.update(acoustic_features)
                        
                        segment_words.append(enhanced_word)
                
                # Combine words into text
                text = " ".join([w["word"] for w in segment_words]).strip()
                
                # Calculate average confidence
                avg_confidence = np.mean([w["confidence"] for w in segment_words]) if segment_words else 0.0
                
                # Create SpeechResult with enhanced word segments
                result = SpeechResult(
                    text=text,
                    confidence=float(avg_confidence),
                    language=full_result.get("language", "en"),
                    word_segments=segment_words if segment_words else None
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error("transcription_acoustic_mapping_failed", error=str(e))
            # Fallback to basic mapping without acoustics
            return self._map_transcription_to_segments(full_result, segments)
    
    def _fallback_individual_processing(self, 
                                      audio_path: Union[str, Path],
                                      segments: List[Tuple[float, float]],
                                      sample_rate: int = 16000) -> List[SpeechResult]:
        """
        Fallback method: process segments individually (original approach)
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time) tuples
            sample_rate: Target sample rate
            
        Returns:
            List of SpeechResults
        """
        self.logger.warning("using_fallback_individual_processing")
        
        results = []
        for start_time, end_time in segments:
            duration = end_time - start_time
            result = self.transcribe_audio_file(
                audio_path=audio_path,
                start_time=start_time,
                duration=duration,
                sample_rate=sample_rate
            )
            results.append(result)
        
        return results
    
    def get_transcription_summary(self, results: List[SpeechResult]) -> Dict[str, Any]:
        """
        Get summary statistics for transcription results
        
        Args:
            results: List of SpeechResults
            
        Returns:
            Dictionary with transcription statistics
        """
        if not results:
            return {}
        
        # Calculate statistics
        total_segments = len(results)
        non_empty_results = [r for r in results if r.text.strip()]
        empty_segments = total_segments - len(non_empty_results)
        
        avg_confidence = np.mean([r.confidence for r in non_empty_results]) if non_empty_results else 0.0
        total_words = sum(len(r.text.split()) for r in non_empty_results)
        
        # Language detection
        languages = [r.language for r in results if r.language]
        dominant_language = max(set(languages), key=languages.count) if languages else None
        
        return {
            "total_segments": total_segments,
            "transcribed_segments": len(non_empty_results),
            "empty_segments": empty_segments,
            "success_rate": len(non_empty_results) / total_segments if total_segments > 0 else 0.0,
            "average_confidence": float(avg_confidence),
            "total_words": total_words,
            "dominant_language": dominant_language,
            "language_distribution": {
                lang: languages.count(lang) 
                for lang in set(languages)
            } if languages else {}
        }
    
    def _extract_word_audio(self, audio_data: np.ndarray, start_time: float, end_time: float, sample_rate: int = 16000) -> np.ndarray:
        """Extract audio segment for a specific word"""
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                return np.zeros(int(0.1 * sample_rate), dtype=np.float32)  # Return 0.1s of silence
                
            return audio_data[start_sample:end_sample].astype(np.float32)
        except Exception as e:
            self.logger.warning("word_audio_extraction_failed", error=str(e))
            return np.zeros(int(0.1 * sample_rate), dtype=np.float32)
    
    def _analyze_word_acoustics(self, word_audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """
        Analyze acoustic features of a word segment using GPU acceleration
        
        Args:
            word_audio: Audio data for the word
            sample_rate: Sample rate of audio
            
        Returns:
            Dict with volume_db, pitch_hz, harmonics_ratio, spectral_centroid
        """
        try:
            # Ensure minimum length
            if len(word_audio) < sample_rate * 0.05:  # At least 50ms
                return {
                    "volume_db": -60.0,
                    "pitch_hz": 0.0,
                    "harmonics_ratio": 0.0,
                    "spectral_centroid": 0.0
                }
            
            # Convert to torch tensor and move to device
            audio_tensor = torch.from_numpy(word_audio).to(self.device)
            
            # Calculate volume (RMS to dB)
            volume_db = self._calculate_rms_db(audio_tensor)
            
            # Calculate STFT for frequency analysis
            stft = torch.stft(
                audio_tensor, 
                n_fft=512, 
                hop_length=128,
                win_length=400,
                window=torch.hann_window(400).to(self.device),
                return_complex=True
            )
            magnitude = torch.abs(stft)
            
            # Calculate acoustic features
            pitch_hz = self._estimate_f0_gpu(magnitude, sample_rate)
            harmonics_ratio = self._calculate_harmonics_ratio(magnitude)
            spectral_centroid = self._calculate_spectral_centroid_gpu(magnitude, sample_rate)
            
            return {
                "volume_db": float(volume_db),
                "pitch_hz": float(pitch_hz),
                "harmonics_ratio": float(harmonics_ratio),
                "spectral_centroid": float(spectral_centroid)
            }
            
        except Exception as e:
            self.logger.warning("acoustic_analysis_failed", error=str(e))
            return {
                "volume_db": -60.0,
                "pitch_hz": 0.0,
                "harmonics_ratio": 0.0,
                "spectral_centroid": 0.0
            }
    
    def _analyze_words_acoustics(self, word_segments: List[Dict[str, Any]], 
                                audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """
        Analyze acoustic features for multiple word segments
        
        Args:
            word_segments: List of word segment dictionaries
            audio_data: Full audio data array
            sample_rate: Sample rate of audio
            
        Returns:
            List of word segments enhanced with acoustic data
        """
        enhanced_segments = []
        
        for word_data in word_segments:
            try:
                # Extract timing information
                start_time = word_data.get('start', 0.0)
                end_time = word_data.get('end', 0.0)
                
                # Extract audio segment for this word
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Ensure valid bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if start_sample < end_sample:
                    word_audio = audio_data[start_sample:end_sample]
                    acoustic_features = self._analyze_word_acoustics(word_audio, sample_rate)
                    
                    # Merge original data with acoustic features
                    enhanced_word = word_data.copy()
                    enhanced_word.update(acoustic_features)
                    enhanced_segments.append(enhanced_word)
                else:
                    # Invalid segment - add with default values
                    enhanced_word = word_data.copy()
                    enhanced_word.update({
                        "volume_db": -60.0,
                        "pitch_hz": 0.0,
                        "harmonics_ratio": 0.0,
                        "spectral_centroid": 0.0
                    })
                    enhanced_segments.append(enhanced_word)
                    
            except Exception as e:
                self.logger.warning("word_acoustic_analysis_failed", 
                                  word=word_data.get('word', ''), error=str(e))
                # Add word with default acoustic values
                enhanced_word = word_data.copy()
                enhanced_word.update({
                    "volume_db": -60.0,
                    "pitch_hz": 0.0,
                    "harmonics_ratio": 0.0,
                    "spectral_centroid": 0.0
                })
                enhanced_segments.append(enhanced_word)
        
        return enhanced_segments
    
    def _calculate_rms_db(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate RMS volume in decibels"""
        rms = torch.sqrt(torch.mean(audio_tensor ** 2))
        # Convert to dB with floor to prevent log(0)
        rms_db = 20 * torch.log10(torch.clamp(rms, min=1e-8))
        return rms_db
    
    def _estimate_f0_gpu(self, magnitude: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Estimate fundamental frequency using autocorrelation on GPU"""
        try:
            # Convert to power spectrum
            power_spectrum = magnitude ** 2
            
            # Sum across time frames
            avg_spectrum = torch.mean(power_spectrum, dim=1)
            
            # Find peak frequency (simplified F0 estimation)
            freqs = torch.linspace(0, sample_rate // 2, avg_spectrum.shape[0]).to(self.device)
            
            # Focus on human vocal range (80-400 Hz)
            vocal_range_mask = (freqs >= 80) & (freqs <= 400)
            if not vocal_range_mask.any():
                return torch.tensor(0.0)
            
            vocal_spectrum = avg_spectrum[vocal_range_mask]
            vocal_freqs = freqs[vocal_range_mask]
            
            # Find peak in vocal range
            peak_idx = torch.argmax(vocal_spectrum)
            f0 = vocal_freqs[peak_idx]
            
            return f0
            
        except Exception:
            return torch.tensor(0.0)
    
    def _calculate_harmonics_ratio(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Calculate harmonics-to-noise ratio"""
        try:
            # Convert to power spectrum
            power_spectrum = magnitude ** 2
            
            # Sum across time frames
            avg_spectrum = torch.mean(power_spectrum, dim=1)
            
            # Simple HNR approximation: ratio of peak to mean energy
            peak_energy = torch.max(avg_spectrum)
            mean_energy = torch.mean(avg_spectrum)
            
            if mean_energy > 0:
                hnr = peak_energy / mean_energy
                # Normalize to 0-1 range
                hnr_normalized = torch.clamp(hnr / 10.0, 0, 1)
                return hnr_normalized
            else:
                return torch.tensor(0.0)
                
        except Exception:
            return torch.tensor(0.0)
    
    def _calculate_spectral_centroid_gpu(self, magnitude: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Calculate spectral centroid on GPU"""
        try:
            # Power spectrum
            power_spectrum = magnitude ** 2
            
            # Sum across time frames
            avg_spectrum = torch.mean(power_spectrum, dim=1)
            
            # Frequency bins
            freqs = torch.linspace(0, sample_rate // 2, avg_spectrum.shape[0]).to(self.device)
            
            # Calculate weighted average frequency (spectral centroid)
            total_energy = torch.sum(avg_spectrum)
            if total_energy > 0:
                centroid = torch.sum(freqs * avg_spectrum) / total_energy
                return centroid
            else:
                return torch.tensor(0.0)
                
        except Exception:
            return torch.tensor(0.0)