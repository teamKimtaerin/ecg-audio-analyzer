"""
Result Synthesizer Service - Refactored for Simplicity
Single Responsibility: Merge analysis outputs into unified JSON metadata
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..services.audio_extractor import AudioExtractionResult
from ..services.speaker_diarizer import DiarizationResult, SpeakerSegment
from ..models.output_models import (
    CompleteAnalysisResult,
    AnalysisMetadata,
    ModelVersions,
    SpeakerInfo,
    TimelineSegment,
    EmotionAnalysis,
    EmotionScores,
    AudioFeatures,
    VolumeCategory,
    PerformanceStats,
)
from ..utils.logger import get_logger


@dataclass
class SynthesisInput:
    """Input data for result synthesis"""

    filename: str
    duration: float
    audio_extraction_result: AudioExtractionResult
    diarization_result: DiarizationResult
    processing_start_time: float = 0.0
    gpu_acceleration_used: bool = False
    model_versions: Optional[ModelVersions] = None


class SimplifiedResultSynthesizer:
    """
    Simplified result synthesis service focusing on core functionality.

    Key improvements:
    - Removed mock data generation (should come from actual analyzers)
    - Simplified timeline generation
    - Removed unnecessary statistics calculations
    - Streamlined export process
    """

    def __init__(self):
        self.logger = get_logger().bind_context(service="result_synthesizer")
        self.logger.info("synthesizer_initialized")

    def _create_timeline_segment(
        self, speaker_segment: SpeakerSegment, segment_index: int
    ) -> TimelineSegment:
        """Create a timeline segment from speaker segment"""

        # Basic emotion data (neutral default until EmotionAnalyzer is implemented)
        emotion = EmotionAnalysis(
            primary="neutral",
            confidence=0.5,
            intensity=0.3,
            valence=0.0,
            arousal=0.3,
            all_scores=EmotionScores(
                joy=0.1,
                sadness=0.1,
                anger=0.1,
                fear=0.1,
                surprise=0.1,
                disgust=0.1,
                neutral=0.4,
            ),
        )

        # Basic audio features (defaults until AcousticAnalyzer is implemented)
        audio_features = AudioFeatures(
            rms_energy=0.05,
            rms_db=-26.0,
            pitch_mean=200.0,
            pitch_variance=20.0,
            speaking_rate=4.0,
            amplitude_max=0.7,
            silence_ratio=0.1,
            spectral_centroid=2000.0,
            zcr=0.05,
            mfcc=[12.0, -5.0, 3.0],
            volume_category=VolumeCategory.MEDIUM,
        )

        return TimelineSegment(
            segment_id=f"seg_{segment_index:04d}",
            start_time=speaker_segment.start_time,
            end_time=speaker_segment.end_time,
            duration=speaker_segment.duration,
            speaker_id=speaker_segment.speaker_id,
            speaker_confidence=speaker_segment.confidence,
            emotion=emotion,
            audio_features=audio_features,
        )

    def _create_speakers_dict(
        self, diarization_result: DiarizationResult
    ) -> Dict[str, SpeakerInfo]:
        """Create speaker information from diarization results"""

        speakers = {}

        for speaker_id, speaker_data in diarization_result.speakers.items():
            # Generate readable speaker name
            speaker_num = speaker_id.replace("SPEAKER_", "").replace("speaker_", "")
            try:
                speaker_num = int(speaker_num) + 1  # 1-indexed for user display
                speaker_name = f"Speaker {speaker_num}"
            except:
                speaker_name = f"Speaker {speaker_id}"

            speakers[speaker_id] = SpeakerInfo(
                name=speaker_name,
                total_duration=speaker_data.total_duration,
                avg_confidence=speaker_data.avg_confidence,
            )

        return speakers

    def _create_performance_stats(
        self, synthesis_input: SynthesisInput, synthesis_time: float
    ) -> PerformanceStats:
        """Create simplified performance statistics"""

        total_time = time.time() - synthesis_input.processing_start_time

        # Determine bottleneck
        extraction_time = (
            synthesis_input.audio_extraction_result.extraction_time_seconds
        )
        diarization_time = synthesis_input.diarization_result.processing_time

        times = {
            "audio_extraction": extraction_time,
            "speaker_diarization": diarization_time,
            "result_synthesis": synthesis_time,
        }
        bottleneck = max(times, key=times.get)

        # Calculate FPS
        fps = synthesis_input.duration / total_time if total_time > 0 else 0.0

        # Estimate memory usage (simplified)
        peak_memory = (
            synthesis_input.audio_extraction_result.file_size_mb * 2  # Audio processing
            + len(synthesis_input.diarization_result.segments) * 0.001  # Segments
        )

        return PerformanceStats(
            gpu_utilization=0.8 if synthesis_input.gpu_acceleration_used else 0.0,
            peak_memory_mb=int(peak_memory),
            avg_processing_fps=fps,
            bottleneck_stage=bottleneck,
        )

    def synthesize_results(
        self, synthesis_input: SynthesisInput
    ) -> CompleteAnalysisResult:
        """
        Synthesize all analysis results into comprehensive format.

        Args:
            synthesis_input: All analysis results and metadata

        Returns:
            CompleteAnalysisResult with analysis data
        """

        start_time = time.time()

        try:
            self.logger.info(
                "synthesis_started",
                filename=synthesis_input.filename,
                duration=synthesis_input.duration,
                speakers=synthesis_input.diarization_result.total_speakers,
            )

            # Generate timeline from diarization segments
            timeline = []
            for i, segment in enumerate(synthesis_input.diarization_result.segments):
                timeline_segment = self._create_timeline_segment(segment, i)
                timeline.append(timeline_segment)

            # Sort timeline by start time
            timeline.sort(key=lambda x: x.start_time)

            # Create speaker information
            speakers = self._create_speakers_dict(synthesis_input.diarization_result)

            # Create metadata
            processing_time = time.time() - synthesis_input.processing_start_time
            metadata = AnalysisMetadata(
                filename=synthesis_input.filename,
                duration=synthesis_input.duration,
                total_speakers=len(speakers),
                processing_time=processing_time,
                gpu_acceleration=synthesis_input.gpu_acceleration_used,
                model_versions=synthesis_input.model_versions or ModelVersions(),
            )

            # Create performance stats
            synthesis_time = time.time() - start_time
            performance = self._create_performance_stats(
                synthesis_input, synthesis_time
            )

            # Create final result
            result = CompleteAnalysisResult(
                metadata=metadata,
                speakers=speakers,
                timeline=timeline,
                performance_stats=performance,
            )

            self.logger.info(
                "synthesis_completed",
                segments=len(timeline),
                speakers=len(speakers),
                time=synthesis_time,
            )

            return result

        except Exception as e:
            self.logger.error("synthesis_failed", error=str(e))

            # Return minimal result on failure
            return CompleteAnalysisResult(
                metadata=AnalysisMetadata(
                    filename=synthesis_input.filename,
                    duration=synthesis_input.duration,
                    total_speakers=0,
                    processing_time=time.time() - synthesis_input.processing_start_time,
                ),
                performance_stats=PerformanceStats(
                    gpu_utilization=0.0,
                    peak_memory_mb=0,
                    avg_processing_fps=0.0,
                    bottleneck_stage="synthesis_error",
                ),
            )

    def export_json(
        self, result: CompleteAnalysisResult, output_path: Path, prettify: bool = True
    ) -> Path:
        """
        Export result to JSON file.

        Args:
            result: Analysis result to export
            output_path: Output file path
            prettify: Whether to format JSON

        Returns:
            Path to exported file
        """

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                if prettify:
                    json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
                else:
                    json.dump(result.model_dump(), f, ensure_ascii=False)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            self.logger.info(
                "json_exported", path=str(output_path), size_mb=f"{file_size_mb:.2f}"
            )

            return output_path

        except Exception as e:
            self.logger.error("export_failed", error=str(e))
            raise

    def synthesize_and_export(
        self, synthesis_input: SynthesisInput, output_path: Path, prettify: bool = True
    ) -> Tuple[CompleteAnalysisResult, Path]:
        """
        Synthesize and export in one operation.

        Args:
            synthesis_input: Analysis input data
            output_path: Output file path
            prettify: Whether to format JSON

        Returns:
            Tuple of (result, output_path)
        """

        # Synthesize
        result = self.synthesize_results(synthesis_input)

        # Export
        output_file = self.export_json(result, output_path, prettify)

        return result, output_file


# Backwards compatibility
ResultSynthesizer = SimplifiedResultSynthesizer
