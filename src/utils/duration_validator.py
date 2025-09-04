"""
Duration Validator - Precise video/audio duration detection and validation
Ensures accurate duration measurement using multiple FFmpeg probing methods
"""

import subprocess
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from dataclasses import dataclass

import ffmpeg
import soundfile as sf
from ..utils.logger import get_logger


@dataclass
class DurationProbeResult:
    """Result of duration probing with multiple detection methods"""

    file_path: str
    ffprobe_duration: Optional[float] = None
    ffmpeg_duration: Optional[float] = None
    container_duration: Optional[float] = None
    stream_duration: Optional[float] = None
    audio_file_duration: Optional[float] = None
    final_duration: Optional[float] = None
    confidence: float = 0.0
    method_used: str = "unknown"
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DurationValidator:
    """
    Precise video/audio duration detection using multiple FFmpeg methods.

    Implements robust duration detection with fallback strategies to ensure
    accurate video length measurement before audio extraction and processing.
    """

    def __init__(self):
        self.logger = get_logger().bind_context(service="duration_validator")

        # Duration validation thresholds
        self.tolerance_seconds = 0.1  # Acceptable difference between methods
        self.max_difference_seconds = 2.0  # Maximum difference before warning

        self.logger.info("duration_validator_initialized")

    def probe_video_duration(self, file_path: Union[str, Path]) -> DurationProbeResult:
        """
        Probe video file duration using multiple detection methods.

        Args:
            file_path: Path to video file

        Returns:
            DurationProbeResult with comprehensive duration information
        """
        file_path = Path(file_path)
        result = DurationProbeResult(file_path=str(file_path))

        self.logger.info("probing_video_duration", file=str(file_path))

        try:
            # Method 1: FFprobe with format detection
            result.ffprobe_duration = self._probe_with_ffprobe(file_path)

            # Method 2: FFmpeg stream information
            result.ffmpeg_duration = self._probe_with_ffmpeg_python(file_path)

            # Method 3: Container-level duration
            result.container_duration = self._probe_container_duration(file_path)

            # Method 4: Stream-level duration (most accurate for some formats)
            result.stream_duration = self._probe_stream_duration(file_path)

            # Determine final duration with confidence scoring
            result.final_duration, result.confidence, result.method_used = (
                self._determine_final_duration(result)
            )

            # Validate consistency between methods
            self._validate_duration_consistency(result)

            self.logger.info(
                "duration_probe_completed",
                file=str(file_path),
                final_duration=result.final_duration,
                confidence=result.confidence,
                method=result.method_used,
            )

            return result

        except Exception as e:
            self.logger.error(
                "duration_probe_failed", file=str(file_path), error=str(e)
            )
            result.warnings.append(f"Duration probing failed: {str(e)}")
            return result

    def _probe_with_ffprobe(self, file_path: Path) -> Optional[float]:
        """Probe duration using ffprobe format information"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.logger.warning("ffprobe_format_failed", error=result.stderr)
                return None

            data = json.loads(result.stdout)
            duration_str = data.get("format", {}).get("duration")

            if duration_str:
                return float(duration_str)

        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
            self.logger.warning("ffprobe_format_error", error=str(e))

        return None

    def _probe_with_ffmpeg_python(self, file_path: Path) -> Optional[float]:
        """Probe duration using ffmpeg-python"""
        try:
            probe = ffmpeg.probe(str(file_path))
            format_info = probe.get("format", {})
            duration = format_info.get("duration")

            if duration:
                return float(duration)

        except Exception as e:
            self.logger.warning("ffmpeg_python_probe_failed", error=str(e))

        return None

    def _probe_container_duration(self, file_path: Path) -> Optional[float]:
        """Probe container-level duration using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

        except (subprocess.TimeoutExpired, ValueError) as e:
            self.logger.warning("container_duration_probe_failed", error=str(e))

        return None

    def _probe_stream_duration(self, file_path: Path) -> Optional[float]:
        """Probe stream-level duration (most accurate method)"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                duration_str = result.stdout.strip()
                if duration_str != "N/A":
                    return float(duration_str)

            # Fallback: try audio stream duration
            cmd[2] = "a:0"  # Select audio stream instead
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                duration_str = result.stdout.strip()
                if duration_str != "N/A":
                    return float(duration_str)

        except (subprocess.TimeoutExpired, ValueError) as e:
            self.logger.warning("stream_duration_probe_failed", error=str(e))

        return None

    def _determine_final_duration(
        self, result: DurationProbeResult
    ) -> tuple[Optional[float], float, str]:
        """
        Determine the most accurate duration from multiple probe results.

        Returns:
            Tuple of (final_duration, confidence, method_used)
        """
        durations = {
            "stream_duration": result.stream_duration,
            "ffprobe_duration": result.ffprobe_duration,
            "ffmpeg_duration": result.ffmpeg_duration,
            "container_duration": result.container_duration,
        }

        # Remove None values
        valid_durations = {k: v for k, v in durations.items() if v is not None}

        if not valid_durations:
            return None, 0.0, "no_valid_duration"

        # If only one method worked, use it with medium confidence
        if len(valid_durations) == 1:
            method, duration = next(iter(valid_durations.items()))
            return duration, 0.6, method

        # Check for consensus among multiple methods
        duration_values = list(valid_durations.values())

        # Find the most common duration (within tolerance)
        consensus_duration = None
        consensus_count = 0
        consensus_methods = []

        for method, duration in valid_durations.items():
            matching_count = sum(
                1
                for d in duration_values
                if abs(d - duration) <= self.tolerance_seconds
            )
            matching_methods = [
                m
                for m, d in valid_durations.items()
                if abs(d - duration) <= self.tolerance_seconds
            ]

            if matching_count > consensus_count:
                consensus_duration = duration
                consensus_count = matching_count
                consensus_methods = matching_methods

        # Calculate confidence based on consensus
        total_methods = len(valid_durations)
        confidence = (consensus_count / total_methods) * 0.9  # Max 0.9 confidence

        # Prefer stream_duration if it's part of consensus (most accurate)
        final_method = "consensus"
        if "stream_duration" in consensus_methods:
            final_method = "stream_duration"
        elif consensus_methods:
            final_method = consensus_methods[0]

        return consensus_duration, confidence, final_method

    def _validate_duration_consistency(self, result: DurationProbeResult):
        """Validate consistency between different duration detection methods"""
        if not result.final_duration:
            return

        durations = [
            result.ffprobe_duration,
            result.ffmpeg_duration,
            result.container_duration,
            result.stream_duration,
        ]

        valid_durations = [d for d in durations if d is not None]

        for duration in valid_durations:
            difference = abs(duration - result.final_duration)
            if difference > self.max_difference_seconds:
                warning = f"Large duration difference detected: {difference:.2f}s between methods"
                result.warnings.append(warning)
                self.logger.warning(
                    "duration_inconsistency",
                    final=result.final_duration,
                    other=duration,
                    difference=difference,
                )

    def validate_extracted_audio(
        self, audio_path: Union[str, Path], expected_duration: float
    ) -> Dict[str, Any]:
        """
        Validate that extracted audio matches expected duration.

        Args:
            audio_path: Path to extracted audio file
            expected_duration: Expected duration in seconds

        Returns:
            Dictionary with validation results
        """
        audio_path = Path(audio_path)

        try:
            # Get actual audio duration using soundfile
            info = sf.info(str(audio_path))
            actual_duration = info.duration

            difference = abs(actual_duration - expected_duration)
            percentage_diff = (
                (difference / expected_duration) * 100 if expected_duration > 0 else 0
            )

            is_valid = difference <= self.tolerance_seconds

            result = {
                "valid": is_valid,
                "expected_duration": expected_duration,
                "actual_duration": actual_duration,
                "difference": difference,
                "percentage_difference": percentage_diff,
                "tolerance": self.tolerance_seconds,
            }

            if not is_valid:
                self.logger.warning(
                    "audio_duration_mismatch",
                    expected=expected_duration,
                    actual=actual_duration,
                    difference=difference,
                )
            else:
                self.logger.info(
                    "audio_duration_validated",
                    duration=actual_duration,
                    difference=difference,
                )

            return result

        except Exception as e:
            self.logger.error(
                "audio_validation_failed", file=str(audio_path), error=str(e)
            )
            return {
                "valid": False,
                "error": str(e),
                "expected_duration": expected_duration,
                "actual_duration": None,
            }

    def detect_timeline_gaps(
        self, segments: List[Dict[str, Any]], total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in timeline coverage compared to total video duration.

        Args:
            segments: List of timeline segments with start_time and end_time
            total_duration: Total expected duration

        Returns:
            List of gap information dictionaries
        """
        if not segments:
            return [
                {
                    "start_time": 0.0,
                    "end_time": total_duration,
                    "gap_duration": total_duration,
                    "gap_type": "complete_missing",
                }
            ]

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.get("start_time", 0))

        gaps = []
        expected_start = 0.0

        for segment in sorted_segments:
            segment_start = segment.get("start_time", 0)
            segment_end = segment.get("end_time", segment_start)

            # Check for gap before this segment
            if segment_start > expected_start + self.tolerance_seconds:
                gaps.append(
                    {
                        "start_time": expected_start,
                        "end_time": segment_start,
                        "gap_duration": segment_start - expected_start,
                        "gap_type": "timeline_gap",
                    }
                )

            expected_start = max(expected_start, segment_end)

        # Check for gap at the end
        if expected_start < total_duration - self.tolerance_seconds:
            gaps.append(
                {
                    "start_time": expected_start,
                    "end_time": total_duration,
                    "gap_duration": total_duration - expected_start,
                    "gap_type": "ending_gap",
                }
            )

        if gaps:
            total_gap_duration = sum(gap["gap_duration"] for gap in gaps)
            self.logger.warning(
                "timeline_gaps_detected",
                gap_count=len(gaps),
                total_gap_duration=total_gap_duration,
                coverage_percentage=(1 - total_gap_duration / total_duration) * 100,
            )

        return gaps


# Convenience functions
def probe_video_duration(file_path: Union[str, Path]) -> DurationProbeResult:
    """Quick duration probing function"""
    validator = DurationValidator()
    return validator.probe_video_duration(file_path)


def validate_audio_duration(
    audio_path: Union[str, Path], expected_duration: float
) -> Dict[str, Any]:
    """Quick audio duration validation function"""
    validator = DurationValidator()
    return validator.validate_extracted_audio(audio_path, expected_duration)
