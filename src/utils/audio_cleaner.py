"""
Audio Cleaner - Simplified audio preprocessing for improved analysis quality
"""

import tempfile
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf

from ..utils.logger import get_logger


class AudioCleaner:
    """
    Simplified audio preprocessor focused on:
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
    
    def process(self, 
                input_source: Union[str, Path, Tuple[np.ndarray, int]], 
                output_path: Optional[Union[str, Path]] = None) -> Union[str, Tuple[np.ndarray, int]]:
        """
        Clean and optimize audio from file or memory
        
        Args:
            input_source: Either file path or tuple of (audio_array, sample_rate)
            output_path: Path for output file (None = return array, "temp" = temp file)
            
        Returns:
            Either file path or (audio_array, sample_rate) based on output_path
        """
        try:
            # Load audio from appropriate source
            if isinstance(input_source, (str, Path)):
                audio, sr = self._load_from_file(input_source)
                self.logger.info("loaded_from_file", path=str(input_source))
            else:
                audio, sr = input_source
                self.logger.info("loaded_from_memory", sample_rate=sr)
            
            # Process audio
            processed = self._process_audio(audio, sr)
            
            # Return based on output preference
            if output_path is None:
                # Return as array
                return processed, self.target_sr
            
            # Save to file
            if output_path == "temp":
                output_path = self._get_temp_path()
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_path), processed, self.target_sr, format='WAV')
            
            self.logger.info("audio_saved", 
                           output_path=str(output_path),
                           duration=len(processed) / self.target_sr)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error("audio_processing_failed", error=str(e))
            raise
    
    def _load_from_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio from file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Load with target sample rate and mono conversion
        audio, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
        return audio, sr
    
    def _process_audio(self, audio: np.ndarray, current_sr: int) -> np.ndarray:
        """Process audio data"""
        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T if audio.shape[0] > audio.shape[1] else audio)
        
        # Resample if needed
        if current_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=self.target_sr)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=30)
        
        return audio.astype(np.float32)
    
    def _get_temp_path(self) -> Path:
        """Create temporary output path"""
        temp_dir = Path(tempfile.gettempdir()) / "audio-cleaner"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"cleaned_{tempfile.gettempprefix()}.wav"
    
    def analyze(self, file_path: Union[str, Path], duration: float = 10.0) -> dict:
        """
        Quick quality analysis of audio file
        
        Args:
            file_path: Path to audio file
            duration: Seconds to analyze (None = full file)
            
        Returns:
            Dictionary with audio metrics
        """
        try:
            # Load sample for analysis
            y, sr = librosa.load(str(file_path), duration=duration)
            
            # Calculate metrics
            rms = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Quality heuristics
            dynamic_range = float(np.ptp(y))  # peak-to-peak
            is_clipping = np.any(np.abs(y) > 0.99)
            is_too_quiet = np.max(np.abs(y)) < 0.1
            
            return {
                "sample_rate": sr,
                "duration_analyzed": len(y) / sr,
                "avg_energy": float(np.mean(rms)),
                "dynamic_range": dynamic_range,
                "avg_zcr": float(np.mean(zcr)),
                "is_clipping": is_clipping,
                "is_too_quiet": is_too_quiet,
                "quality_score": self._calculate_quality_score(
                    dynamic_range, is_clipping, is_too_quiet
                )
            }
        except Exception as e:
            return {"error": str(e), "quality_score": 0.0}
    
    def _calculate_quality_score(self, 
                                dynamic_range: float, 
                                is_clipping: bool, 
                                is_too_quiet: bool) -> float:
        """Simple quality score (0-1)"""
        score = 0.5  # Base score
        
        # Good dynamic range
        if 0.3 < dynamic_range < 1.5:
            score += 0.3
        
        # Penalties
        if is_clipping:
            score -= 0.3
        if is_too_quiet:
            score -= 0.2
            
        return max(0.0, min(1.0, score))


# Convenience functions
def clean_audio(input_path: Union[str, Path], 
               output_path: Optional[str] = "temp") -> str:
    """
    Quick audio cleaning with file output
    
    Args:
        input_path: Input audio file
        output_path: Output path ("temp" for temporary file)
        
    Returns:
        Path to cleaned audio file
    """
    cleaner = AudioCleaner()
    return cleaner.process(input_path, output_path)


def load_and_clean(input_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load and clean audio, returning array
    
    Args:
        input_path: Input audio file
        
    Returns:
        Tuple of (cleaned_audio, sample_rate)
    """
    cleaner = AudioCleaner()
    return cleaner.process(input_path, output_path=None)