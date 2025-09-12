"""
Data Models for JSON Output Structure
Comprehensive models for the high-performance audio analysis metadata output
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum


class VolumeCategory(str, Enum):
    """Volume categories for subtitle sizing"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMPHASIS = "emphasis"


# Audio Feature Models
class AudioFeatures(BaseModel):
    """Acoustic features for subtitle styling"""

    rms_energy: float = Field(
        ..., description="RMS energy for subtitle size determination"
    )
    rms_db: float = Field(..., description="Decibel value")
    pitch_mean: float = Field(..., ge=0.0, description="Average pitch in Hz")
    pitch_variance: float = Field(
        ..., ge=0.0, description="Pitch variation for emphasis detection"
    )
    speaking_rate: float = Field(..., ge=0.0, description="Words per second")
    amplitude_max: float = Field(..., ge=0.0, le=1.0, description="Peak amplitude")
    silence_ratio: float = Field(..., ge=0.0, le=1.0, description="Silence proportion")
    spectral_centroid: float = Field(..., ge=0.0, description="Tonal characteristics")
    zcr: float = Field(..., ge=0.0, description="Zero crossing rate")
    mfcc: List[float] = Field(..., description="First 3 MFCC coefficients")
    volume_category: VolumeCategory = Field(..., description="Categorized volume level")
    volume_peaks: List[float] = Field(
        default_factory=list, description="Volume peaks for waveform visualization"
    )

    @validator(
        "rms_energy",
        "rms_db",
        "pitch_mean",
        "pitch_variance",
        "speaking_rate",
        "amplitude_max",
        "silence_ratio",
        "spectral_centroid",
        "zcr",
        pre=True,
    )
    def round_float_values(cls, v):
        return round(float(v), 3) if v is not None else 0.0

    @validator("mfcc", pre=True)
    def round_mfcc_values(cls, v):
        if isinstance(v, list):
            return [round(float(x), 1) for x in v[:3]]  # Only first 3 coefficients
        return [0.0, 0.0, 0.0]

    @validator("volume_peaks", pre=True)
    def round_volume_peaks(cls, v):
        if isinstance(v, list):
            return [round(float(x), 3) for x in v]
        return []


# Timeline Segment Model
class TimelineSegment(BaseModel):
    """Individual timeline segment with comprehensive analysis"""

    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., gt=0.0, description="End time in seconds")
    speaker_id: str = Field(..., description="Speaker identifier")
    speaker_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Speaker identification confidence"
    )
    text_placeholder: str = Field(
        default="[TRANSCRIPTION_PENDING]", description="Placeholder for transcription"
    )
    audio_features: AudioFeatures = Field(..., description="Acoustic features")

    @validator("start_time", "end_time", "speaker_confidence", pre=True)
    def round_time_values(cls, v):
        return round(float(v), 1) if v is not None else 0.0

    @validator("end_time")
    def end_time_after_start(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be greater than start_time")
        return v

    @property
    def duration(self) -> float:
        """Calculate segment duration"""
        return round(self.end_time - self.start_time, 1)


# Performance Statistics Model
class PerformanceStats(BaseModel):
    """Performance monitoring statistics"""

    gpu_utilization: float = Field(
        ..., ge=0.0, le=1.0, description="GPU utilization ratio"
    )
    peak_memory_mb: int = Field(..., ge=0, description="Peak memory usage in MB")
    avg_processing_fps: float = Field(
        ..., ge=0.0, description="Average processing frames per second"
    )
    bottleneck_stage: str = Field(
        ..., description="Processing stage that was the bottleneck"
    )

    @validator("gpu_utilization", "avg_processing_fps", pre=True)
    def round_performance_values(cls, v):
        return round(float(v), 2) if v is not None else 0.0


# Main Metadata Model
class AnalysisMetadata(BaseModel):
    """Analysis metadata and processing information"""

    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., gt=0.0, description="Total duration in seconds")
    total_speakers: int = Field(..., ge=0, description="Number of speakers detected")
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO timestamp of analysis",
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    gpu_acceleration: bool = Field(
        default=False, description="Whether GPU acceleration was used"
    )

    @validator("duration", "processing_time", pre=True)
    def round_numeric_values(cls, v):
        return round(float(v), 1) if v is not None else 0.0


# Complete Analysis Result Model
class CompleteAnalysisResult(BaseModel):
    """Complete audio analysis result in JSON format"""

    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    timeline: List[TimelineSegment] = Field(
        default_factory=list, description="Timeline segments"
    )
    performance_stats: PerformanceStats = Field(
        ..., description="Performance statistics"
    )

    @model_validator(mode="before")
    def validate_timeline_consistency(cls, values):
        """Validate timeline segments are consistent"""
        timeline = values.get("timeline", [])
        metadata = values.get("metadata")

        if not timeline:
            return values

        # Validate total duration consistency
        if metadata and timeline:
            last_segment = max(timeline, key=lambda x: x.end_time)
            if last_segment.end_time > metadata.duration + 1.0:  # 1 second tolerance
                raise ValueError("Timeline extends beyond metadata duration")

        return values

    def model_dump_json_formatted(self) -> str:
        """Export as formatted JSON string"""
        return self.model_dump_json(indent=2, exclude_none=False)


# Factory Functions
def create_empty_analysis_result(
    filename: str, duration: float
) -> CompleteAnalysisResult:
    """Create empty analysis result with basic metadata"""
    return CompleteAnalysisResult(
        metadata=AnalysisMetadata(
            filename=filename,
            duration=max(duration, 0.1),  # Ensure duration > 0 for Pydantic validation
            total_speakers=0,
            processing_time=0.0,
        ),
        performance_stats=PerformanceStats(
            gpu_utilization=0.0,
            peak_memory_mb=0,
            avg_processing_fps=0.0,
            bottleneck_stage="initialization",
        ),
    )
