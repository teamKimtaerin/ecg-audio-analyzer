"""
Audio Cleaner - Simple audio preprocessing for improved analysis quality
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from typing import Union, Optional, Tuple
from ..utils.logger import get_logger


class AudioCleaner:
    """
    Simple audio preprocessor focused on:
    - 16kHz mono conversion for WhisperX optimization
    - Audio normalization for consistent levels
    - Basic quality validation
    """
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio cleaner
        
        Args:
            target_sr: Target sample rate (16kHz optimized for WhisperX)
        """
        self.target_sr = target_sr
        self.logger = get_logger().bind_context(service="audio_cleaner")
    
    def clean_audio(self, 
                   input_path: Union[str, Path], 
                   output_path: Optional[Union[str, Path]] = None,
                   return_buffer: bool = False) -> Union[str, Tuple[np.ndarray, int]]:
        """
        Clean and optimize audio for analysis
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file (optional, creates temp file if None)
            return_buffer: If True, return (audio_array, sample_rate) instead of file path
            
        Returns:
            Path to cleaned audio file OR (audio_array, sample_rate) if return_buffer=True
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.logger.info("cleaning_audio", input_file=str(input_path))
        
        try:
            # Create output path if not provided
            if output_path is None:
                output_path = self._create_temp_output_path(input_path)
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load and process audio
            processed_audio, sr = self._process_audio_file(input_path)
            
            # Return buffer directly if requested (memory optimization)
            if return_buffer:
                duration = len(processed_audio) / sr
                self.logger.info("audio_cleaning_completed_memory", 
                               duration=duration,
                               sample_rate=sr)
                return processed_audio, sr
            
            # Save processed audio to file
            sf.write(str(output_path), processed_audio, sr, format='WAV')
            
            # Validate result
            file_info = self._validate_output(output_path)
            
            self.logger.info("audio_cleaning_completed", 
                           output_file=str(output_path),
                           duration=file_info["duration"],
                           sample_rate=file_info["sample_rate"])
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error("audio_cleaning_failed", error=str(e))
            raise
    
    def clean_audio_to_memory(self, input_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Clean audio and return processed audio data in memory (no temp file)
        Optimized for memory-efficient processing
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        return self.clean_audio(input_path, return_buffer=True)
    
    def _process_audio_file(self, input_path: Path) -> Tuple[np.ndarray, int]:
        """
        Process audio file with optimizations
        
        Args:
            input_path: Input audio file path
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        try:
            # Load audio with target sample rate and mono conversion
            audio, sr = librosa.load(
                str(input_path), 
                sr=self.target_sr, 
                mono=True  # Convert to mono automatically
            )
            
            original_duration = len(audio) / sr
            self.logger.info("audio_loaded", 
                           original_duration=original_duration,
                           sample_rate=sr)
            
            # Normalize audio levels (prevents clipping and ensures consistent volume)
            audio = librosa.util.normalize(audio)
            
            # Use gentler trimming to preserve content - only trim very quiet parts
            audio, _ = librosa.effects.trim(audio, top_db=10)  # Changed from 15 to 10 for better speaker separation
            
            trimmed_duration = len(audio) / sr
            self.logger.info("audio_processed", 
                           original_duration=original_duration,
                           trimmed_duration=trimmed_duration,
                           duration_loss=original_duration - trimmed_duration,
                           sample_rate=sr,
                           channels="mono")
            
            return audio, sr
            
        except Exception as e:
            self.logger.error("audio_processing_failed", 
                            file=str(input_path), 
                            error=str(e))
            raise
    
    def _create_temp_output_path(self, input_path: Path) -> Path:
        """Create temporary output file path"""
        temp_dir = Path(tempfile.gettempdir()) / "ecg-audio-cleaner"
        temp_dir.mkdir(exist_ok=True)
        
        # Create unique filename
        temp_filename = f"cleaned_{input_path.stem}_{os.getpid()}.wav"
        return temp_dir / temp_filename
    
    def _validate_output(self, output_path: Path) -> dict:
        """
        Validate processed audio file
        
        Args:
            output_path: Path to output file
            
        Returns:
            Dictionary with file information
        """
        try:
            # Load just for validation (quick check)
            info = sf.info(str(output_path))
            
            validation_result = {
                "valid": True,
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "file_size_mb": output_path.stat().st_size / (1024 * 1024)
            }
            
            # Check if meets requirements
            if info.samplerate != self.target_sr:
                self.logger.warning("sample_rate_mismatch", 
                                  expected=self.target_sr, 
                                  actual=info.samplerate)
            
            if info.channels != 1:
                self.logger.warning("not_mono", channels=info.channels)
            
            return validation_result
            
        except Exception as e:
            self.logger.error("validation_failed", 
                            file=str(output_path), 
                            error=str(e))
            return {"valid": False, "error": str(e)}
    
    def quick_analyze(self, audio_path: Union[str, Path]) -> dict:
        """
        Quick analysis of audio file quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio analysis
        """
        try:
            # Load small sample for analysis
            y, sr = librosa.load(str(audio_path), duration=10.0)  # First 10 seconds
            
            # Basic quality metrics
            rms_energy = librosa.feature.rms(y=y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            return {
                "sample_rate": sr,
                "duration_analyzed": len(y) / sr,
                "avg_energy": float(np.mean(rms_energy)),
                "avg_spectral_centroid": float(np.mean(spectral_centroid)),
                "avg_zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "dynamic_range": float(np.max(y) - np.min(y)),
                "needs_normalization": float(np.max(np.abs(y))) < 0.1,  # Very quiet audio
                "quality_score": self._calculate_quality_score(y, sr)
            }
            
        except Exception as e:
            return {"error": str(e), "quality_score": 0.0}
    
    def _calculate_quality_score(self, audio: np.ndarray, sr: int) -> float:
        """Calculate simple quality score (0-1)"""
        try:
            # Factors affecting quality
            dynamic_range = np.max(audio) - np.min(audio)
            energy_variance = np.var(librosa.feature.rms(y=audio))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            
            # Simple heuristic scoring
            range_score = min(1.0, dynamic_range / 0.5)  # Good if range > 0.5
            energy_score = min(1.0, energy_variance * 100)  # Some energy variation is good
            bandwidth_score = min(1.0, spectral_bandwidth / 2000)  # Good bandwidth
            
            return (range_score + energy_score + bandwidth_score) / 3.0
            
        except:
            return 0.5  # Default moderate score
    
    
    def clean_audio_from_memory(self, 
                               audio_data: np.ndarray, 
                               sample_rate: int,
                               output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Clean audio data already loaded in memory
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            output_path: Path for output file (optional, creates temp file if None)
            
        Returns:
            Path to cleaned audio file
        """
        self.logger.info("cleaning_audio_from_memory")
        
        try:
            # Process audio that's already in memory
            processed_audio = self._process_memory_audio(audio_data, sample_rate)
            
            # Create output path if not provided
            if output_path is None:
                output_path = self._create_temp_output_path(Path("memory_audio"))
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed audio
            sf.write(str(output_path), processed_audio, self.target_sr, format='WAV')
            
            # Validate result
            file_info = self._validate_output(output_path)
            
            self.logger.info("audio_cleaning_from_memory_completed", 
                           output_file=str(output_path),
                           duration=file_info["duration"])
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error("audio_cleaning_from_memory_failed", error=str(e))
            raise
    
    def _process_memory_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio data that's already in memory
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Current sample rate
            
        Returns:
            Processed audio data
        """
        try:
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
            
            # Resample if needed
            if sample_rate != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.target_sr)
            
            # Ensure correct data type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio levels
            audio_data = librosa.util.normalize(audio_data)
            
            # Remove silence at beginning/end - gentler trimming for speaker separation
            audio_data, _ = librosa.effects.trim(audio_data, top_db=10)
            
            self.logger.info("memory_audio_processed", 
                           duration=len(audio_data) / self.target_sr,
                           sample_rate=self.target_sr)
            
            return audio_data
            
        except Exception as e:
            self.logger.error("memory_audio_processing_failed", error=str(e))
            raise


def clean_audio_file(input_path: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convenience function for audio cleaning
    
    Args:
        input_path: Path to input audio file
        output_path: Optional output path
        
    Returns:
        Path to cleaned audio file
    """
    cleaner = AudioCleaner()
    return cleaner.clean_audio(input_path, output_path)


def analyze_audio_quality(audio_path: Union[str, Path]) -> dict:
    """
    Convenience function for audio quality analysis
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with quality metrics
    """
    cleaner = AudioCleaner()
    return cleaner.quick_analyze(audio_path)