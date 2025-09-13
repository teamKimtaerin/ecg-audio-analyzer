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
    def __init__(self, target_sr: int = 16000, top_db: float = 10.0):
        """
        Initialize audio cleaner

        Args:
            target_sr: Target sample rate (16kHz optimized for WhisperX)
            top_db: Silence trimming threshold
        """
        self.target_sr = target_sr
        self.top_db = top_db
        self.logger = get_logger().bind_context(service="audio_cleaner")

    def process(
        self,
        input_source: Union[str, Path, Tuple[np.ndarray, int]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Tuple[np.ndarray, int]]:
        """Clean and optimize audio from file or memory"""
        try:
            # Load audio from appropriate source
            if isinstance(input_source, (str, Path)):
                audio, sr = self._load_from_file(input_source)
            else:
                audio, sr = input_source

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

            sf.write(str(output_path), processed, self.target_sr, format="WAV")

            return str(output_path)

        except Exception as e:
            self.logger.error("audio_processing_failed", error=str(e))
            raise

    def _load_from_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio from file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        audio, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
        return audio, int(sr)

    def _process_audio(self, audio: np.ndarray, current_sr: int) -> np.ndarray:
        """Process audio data"""
        if audio.ndim > 1:
            audio = librosa.to_mono(
                audio.T if audio.shape[0] > audio.shape[1] else audio
            )

        if current_sr != self.target_sr:
            audio = librosa.resample(
                audio, orig_sr=current_sr, target_sr=self.target_sr
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio = librosa.util.normalize(audio)

        audio, _ = librosa.effects.trim(audio, top_db=self.top_db)

        return audio

    def _get_temp_path(self) -> Path:
        """Create temporary output path"""
        temp_dir = Path(tempfile.gettempdir()) / "audio-cleaner"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"cleaned_{tempfile.gettempprefix()}.wav"


# Convenience functions
def clean_audio(
    input_path: Union[str, Path], output_path: Optional[str] = "temp"
) -> str:
    """
    Quick audio cleaning with file output

    Args:
        input_path: Input audio file
        output_path: Output path ("temp" for temporary file)

    Returns:
        Path to cleaned audio file
    """
    cleaner = AudioCleaner()
    result = cleaner.process(input_path, output_path or "temp")
    # Ensure we always return a string for this convenience function
    if isinstance(result, tuple):
        # This shouldn't happen when output_path is provided, but handle it safely
        raise ValueError("Expected file path output, got audio array")
    return result


def load_and_clean(input_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load and clean audio, returning array

    Args:
        input_path: Input audio file

    Returns:
        Tuple of (cleaned_audio, sample_rate)
    """
    cleaner = AudioCleaner()
    result = cleaner.process(input_path, output_path=None)
    # Ensure we always return a tuple for this convenience function
    if isinstance(result, str):
        # This shouldn't happen when output_path=None, but handle it safely
        raise ValueError("Expected audio array output, got file path")
    return result
