"""
Audio Extraction Service - Simplified version
Convert MP4/URLs to analysis-ready audio format
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
import subprocess

import soundfile as sf
import yt_dlp
import ffmpeg

from ..utils.logger import get_logger


@dataclass
class AudioExtractionResult:
    """Result of audio extraction with enhanced duration validation"""

    success: bool
    output_path: Optional[Path] = None
    duration: float = 0.0
    original_duration: float = 0.0  # Original video duration
    sample_rate: int = 16000
    error: Optional[str] = None
    duration_validation_passed: bool = False
    duration_difference: float = 0.0
    duration_confidence: float = 0.0
    extraction_method: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "duration": self.duration,
            "original_duration": self.original_duration,
            "sample_rate": self.sample_rate,
            "error": self.error,
            "duration_validation_passed": self.duration_validation_passed,
            "duration_difference": self.duration_difference,
            "duration_confidence": self.duration_confidence,
            "extraction_method": self.extraction_method,
        }


class AudioExtractor:
    """
    Simple audio extraction service
    """

    def __init__(
        self,
        target_sr: int = 16000,
        temp_dir: Optional[Path] = None,
        duration_tolerance: float = 0.1,
    ):

        self.target_sr = target_sr
        self.duration_tolerance = duration_tolerance
        self.logger = get_logger().bind_context(service="audio_extractor")


        # Setup temp directory
        self.temp_dir = Path(temp_dir or tempfile.gettempdir()) / "audio_extract"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Check ffmpeg availability
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

        # Simple yt-dlp config
        self.ytdl_opts = {
            "format": "best",
            "outtmpl": str(self.temp_dir / "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        self.logger.info("audio_extractor_ready", temp_dir=str(self.temp_dir))

    def _probe_video_duration(self, video_path: Union[str, Path]) -> dict:
        """Simple video duration probing using ffprobe"""
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe['streams'][0]['duration'])
            return {
                'final_duration': duration,
                'confidence': 0.95,  # High confidence for direct probe
                'method_used': 'ffprobe',
                'warnings': []
            }
        except Exception as e:
            self.logger.warning("video_duration_probe_failed", error=str(e))
            return {
                'final_duration': 0.0,
                'confidence': 0.0,
                'method_used': 'failed',
                'warnings': [f"Duration probe failed: {str(e)}"]
            }

    def _validate_extracted_audio(self, audio_path: Union[str, Path], expected_duration: float) -> dict:
        """Simple audio duration validation using soundfile"""
        try:
            import soundfile as sf
            
            # Get audio duration
            info = sf.info(str(audio_path))
            actual_duration = info.duration
            difference = abs(actual_duration - expected_duration)
            
            # Consider valid if within tolerance
            is_valid = difference <= self.duration_tolerance
            
            return {
                'valid': is_valid,
                'actual_duration': actual_duration,
                'difference': difference,
                'tolerance': self.duration_tolerance
            }
        except Exception as e:
            self.logger.error("audio_validation_failed", error=str(e))
            return {
                'valid': False,
                'actual_duration': 0.0,
                'difference': float('inf'),
                'error': str(e)
            }

    def extract(
        self, source: Union[str, Path], output_path: Optional[Path] = None
    ) -> AudioExtractionResult:
        """
        Extract audio from file or URL with precise duration validation

        Args:
            source: File path or URL
            output_path: Optional output path

        Returns:
            AudioExtractionResult with duration validation
        """
        source_str = str(source)
        is_url = source_str.startswith(("http://", "https://"))

        try:
            # Handle URL
            if is_url:
                video_path = self._download_url(source_str)
                if not video_path:
                    return AudioExtractionResult(
                        success=False, error="Failed to download URL"
                    )
            else:
                video_path = Path(source)
                if not video_path.exists():
                    return AudioExtractionResult(
                        success=False, error=f"File not found: {source}"
                    )

            # Step 1: Probe original video duration with high precision
            self.logger.info(
                "probing_original_video_duration", video_path=str(video_path)
            )
            duration_probe = self._probe_video_duration(video_path)

            if not duration_probe['final_duration']:
                self.logger.warning(
                    "duration_probe_failed", warnings=duration_probe['warnings']
                )
                return AudioExtractionResult(
                    success=False, error="Failed to determine video duration"
                )

            original_duration = duration_probe['final_duration']
            self.logger.info(
                "original_duration_detected",
                duration=original_duration,
                confidence=duration_probe['confidence'],
                method=duration_probe['method_used'],
            )

            # Convert to WAV
            if not output_path:
                output_path = self.temp_dir / f"{video_path.stem}_audio.wav"

            # Step 2: Enhanced conversion with duration preservation
            success = self._convert_to_wav_with_validation(
                video_path, output_path, original_duration
            )

            if not success:
                return AudioExtractionResult(success=False, error="Conversion failed")

            # Step 3: Validate extracted audio duration
            validation_result = self._validate_extracted_audio(
                output_path, original_duration
            )

            # Get audio info
            info = sf.info(str(output_path))
            extracted_duration = info.duration

            # Clean up downloaded file if from URL
            if is_url and video_path.exists():
                try:
                    video_path.unlink()
                except:
                    pass

            return AudioExtractionResult(
                success=True,
                output_path=output_path,
                duration=extracted_duration,
                original_duration=original_duration,
                sample_rate=info.samplerate,
                duration_validation_passed=validation_result.get("valid", False),
                duration_difference=validation_result.get("difference", 0.0),
                duration_confidence=duration_probe['confidence'],
                extraction_method="enhanced_ffmpeg_with_validation",
            )

        except Exception as e:
            self.logger.error("extraction_failed", source=source_str, error=str(e))
            return AudioExtractionResult(success=False, error=str(e))

    def _download_url(self, url: str) -> Optional[Path]:
        """Download video from URL"""
        try:
            with yt_dlp.YoutubeDL(self.ytdl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None

                # Find downloaded file
                title = info.get("title", "download").replace("/", "_")
                for ext in [".mp4", ".webm", ".mkv", ".m4a", ".mp3"]:
                    file_path = self.temp_dir / f"{title}{ext}"
                    if file_path.exists():
                        return file_path

                # Fallback: find most recent file
                files = list(self.temp_dir.glob("*"))
                if files:
                    return max(files, key=lambda p: p.stat().st_mtime)

                return None

        except Exception as e:
            self.logger.error("download_failed", url=url, error=str(e))
            return None

    def _convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert video/audio to WAV (legacy method)"""
        return self._convert_to_wav_with_validation(input_path, output_path)

    def _convert_to_wav_with_validation(
        self,
        input_path: Path,
        output_path: Path,
        expected_duration: Optional[float] = None,
    ) -> bool:
        """
        Convert video/audio to WAV with enhanced duration preservation

        Args:
            input_path: Input video/audio file
            output_path: Output WAV file path
            expected_duration: Expected duration for validation

        Returns:
            True if conversion successful and duration matches expectations
        """
        try:
            self.logger.info(
                "starting_enhanced_conversion",
                input=str(input_path),
                output=str(output_path),
                expected_duration=expected_duration,
            )

            # Method 1: Use ffmpeg-python with enhanced settings for duration preservation
            try:
                stream = ffmpeg.input(str(input_path))
                stream = ffmpeg.output(
                    stream,
                    str(output_path),
                    acodec="pcm_s16le",  # 16-bit PCM
                    ac=1,  # Mono
                    ar=self.target_sr,  # Target sample rate
                    loglevel="warning",  # More verbose logging
                    # Enhanced settings for duration preservation
                    avoid_negative_ts="make_zero",  # Handle negative timestamps
                    fflags="+genpts",  # Generate presentation timestamps
                    # Ensure we don't truncate the stream
                    **(
                        {"t": expected_duration} if expected_duration else {}
                    ),  # Explicit duration if known
                )

                ffmpeg.run(stream, overwrite_output=True, quiet=False)

                if output_path.exists():
                    # Validate the conversion
                    if expected_duration:
                        validation = self._validate_extracted_audio(
                            output_path, expected_duration
                        )
                        if validation.get("valid", False):
                            self.logger.info(
                                "enhanced_conversion_successful",
                                method="ffmpeg_python_enhanced",
                                duration_validated=True,
                            )
                            return True
                        else:
                            self.logger.warning(
                                "duration_validation_failed_ffmpeg_python",
                                expected=expected_duration,
                                actual=validation.get("actual_duration"),
                            )
                            # Try fallback method
                            return self._convert_with_ffmpeg_direct(
                                input_path, output_path, expected_duration
                            )
                    else:
                        self.logger.info("conversion_successful_no_validation")
                        return True

            except ffmpeg.Error as e:
                self.logger.warning("ffmpeg_python_failed", error=str(e))
                # Try fallback method
                return self._convert_with_ffmpeg_direct(
                    input_path, output_path, expected_duration
                )

            return False

        except Exception as e:
            self.logger.error("enhanced_conversion_error", error=str(e))
            return False

    def _convert_with_ffmpeg_direct(
        self,
        input_path: Path,
        output_path: Path,
        expected_duration: Optional[float] = None,
    ) -> bool:
        """
        Fallback conversion using direct ffmpeg subprocess with maximum preservation settings
        """
        try:
            self.logger.info("trying_direct_ffmpeg_conversion")

            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i",
                str(input_path),
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # 16-bit PCM
                "-ac",
                "1",  # Mono
                "-ar",
                str(self.target_sr),  # Sample rate
                # Enhanced duration preservation settings
                "-avoid_negative_ts",
                "make_zero",
                "-fflags",
                "+genpts",
                "-max_muxing_queue_size",
                "1024",  # Handle large streams
                "-thread_queue_size",
                "512",
            ]

            # Add explicit duration if known to prevent truncation
            if expected_duration:
                cmd.extend(["-t", str(expected_duration)])

            cmd.append(str(output_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large files
            )

            if result.returncode != 0:
                self.logger.error("direct_ffmpeg_failed", stderr=result.stderr[:500])
                return False

            if output_path.exists():
                # Final validation
                if expected_duration:
                    validation = self._validate_extracted_audio(
                        output_path, expected_duration
                    )
                    self.logger.info(
                        "direct_ffmpeg_conversion_completed",
                        validation_passed=validation.get("valid", False),
                        duration_difference=validation.get("difference", 0),
                    )
                else:
                    self.logger.info("direct_ffmpeg_conversion_completed_no_validation")

                return True

            return False

        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg_conversion_timeout")
            return False
        except Exception as e:
            self.logger.error("direct_ffmpeg_error", error=str(e))
            return False

    def cleanup(self):
        """Clean up temp files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("cleanup_completed")
        except Exception as e:
            self.logger.warning("cleanup_failed", error=str(e))


# Convenience function
def extract_audio(
    source: Union[str, Path], output_path: Optional[Path] = None
) -> AudioExtractionResult:
    """
    Quick audio extraction

    Args:
        source: File path or URL
        output_path: Optional output path

    Returns:
        AudioExtractionResult
    """
    extractor = AudioExtractor()
    try:
        return extractor.extract(source, output_path)
    finally:
        extractor.cleanup()
