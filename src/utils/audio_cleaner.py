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

    def process(
        self,
        input_source: Union[str, Path, Tuple[np.ndarray, int]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Tuple[np.ndarray, int]]:
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

            sf.write(str(output_path), processed, self.target_sr, format="WAV")

            self.logger.info(
                "audio_saved",
                output_path=str(output_path),
                duration=len(processed) / self.target_sr,
            )

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
        return audio, int(sr)

    def _process_audio(self, audio: np.ndarray, current_sr: int) -> np.ndarray:
        """Process audio data"""
        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = librosa.to_mono(
                audio.T if audio.shape[0] > audio.shape[1] else audio
            )

        # Resample if needed
        if current_sr != self.target_sr:
            audio = librosa.resample(
                audio, orig_sr=current_sr, target_sr=self.target_sr
            )

        # Normalize
        audio = librosa.util.normalize(audio)

        # Trim silence with gentler setting for better speaker separation
        audio, _ = librosa.effects.trim(
            audio, top_db=10
        )  # Changed from 30 to 10 for better speaker separation

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
                "is_clipping": bool(is_clipping),
                "is_too_quiet": bool(is_too_quiet),
                "quality_score": self._calculate_quality_score(
                    dynamic_range, bool(is_clipping), bool(is_too_quiet)
                ),
            }
        except Exception as e:
            return {"error": str(e), "quality_score": 0.0}

    def clean_audio_from_memory(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
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
                output_path = self._get_temp_path()
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save processed audio
            sf.write(str(output_path), processed_audio, self.target_sr, format="WAV")

            # Validate result
            file_info = self._validate_output(output_path)

            self.logger.info(
                "audio_cleaning_from_memory_completed",
                output_file=str(output_path),
                duration=file_info["duration"],
            )

            return str(output_path)

        except Exception as e:
            self.logger.error("audio_cleaning_from_memory_failed", error=str(e))
            raise

    def _process_memory_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
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
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=self.target_sr
                )

            # Ensure correct data type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio levels
            audio_data = librosa.util.normalize(audio_data)

            # Remove silence at beginning/end - gentler trimming for speaker separation
            audio_data, _ = librosa.effects.trim(audio_data, top_db=10)

            self.logger.info(
                "memory_audio_processed",
                duration=len(audio_data) / self.target_sr,
                sample_rate=self.target_sr,
            )

            return audio_data

        except Exception as e:
            self.logger.error("memory_audio_processing_failed", error=str(e))
            raise

    def _calculate_quality_score(
        self, dynamic_range: float, is_clipping: bool, is_too_quiet: bool
    ) -> float:
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

    def _validate_output(self, output_path: Union[str, Path]) -> dict:
        """
        Validate the output audio file and return basic info.

        Args:
            output_path: Path to the audio file

        Returns:
            Dictionary with duration and sample rate
        """
        try:
            y, sr = librosa.load(str(output_path), sr=None)
            duration = len(y) / sr if sr else 0
            return {"duration": duration, "sample_rate": sr}
        except Exception as e:
            self.logger.error("output_validation_failed", error=str(e))
            return {"duration": 0, "sample_rate": 0, "error": str(e)}


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
