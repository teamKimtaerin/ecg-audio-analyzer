"""
Data Models for JSON Output Structure
Comprehensive models for the high-performance audio analysis metadata output
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum


class EmotionType(str, Enum):
    """Supported emotion types"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class VolumeCategory(str, Enum):
    """Volume categories for subtitle sizing"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMPHASIS = "emphasis"


class EmphasisLevel(str, Enum):
    """Emphasis levels for subtitle styling"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRONG = "strong"


class ColorHint(str, Enum):
    """Color temperature hints"""

    COOL = "cool"
    NEUTRAL = "neutral"
    WARM = "warm"


class AnimationType(str, Enum):
    """Animation types for subtitles"""

    NONE = "none"
    FADE_IN = "fade_in"
    SLIDE_UP = "slide_up"
    PULSE = "pulse"
    SHAKE = "shake"


# Emotion Analysis Models
class EmotionScores(BaseModel):
    """All emotion classification scores"""

    joy: float = Field(..., ge=0.0, le=1.0)
    sadness: float = Field(..., ge=0.0, le=1.0)
    anger: float = Field(..., ge=0.0, le=1.0)
    fear: float = Field(..., ge=0.0, le=1.0)
    surprise: float = Field(..., ge=0.0, le=1.0)
    disgust: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)

    @validator("*", pre=True)
    def round_scores(cls, v):
        return round(float(v), 3) if v is not None else 0.0


class EmotionAnalysis(BaseModel):
    """Complete emotion analysis results"""

    primary: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    intensity: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=-1.0, le=1.0)  # Positive/negative sentiment
    arousal: float = Field(..., ge=0.0, le=1.0)  # Activation level
    all_scores: EmotionScores

    @validator("confidence", "intensity", "arousal", pre=True)
    def round_confidence_values(cls, v):
        return round(float(v), 3) if v is not None else 0.0

    @validator("valence", pre=True)
    def round_valence(cls, v):
        return round(float(v), 3) if v is not None else 0.0


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
    mfcc: List[float] = Field(
        ..., description="First 3 MFCC coefficients", min_items=3, max_items=3
    )
    volume_category: VolumeCategory = Field(..., description="Categorized volume level")

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


# Subtitle Styling Models
class SubtitleStyling(BaseModel):
    """Subtitle styling hints based on audio analysis"""

    size_multiplier: float = Field(
        ..., ge=0.6, le=1.8, description="Size scaling factor"
    )
    emphasis_level: EmphasisLevel = Field(..., description="Emphasis level")
    duration_hint: float = Field(..., gt=0.0, description="Display duration in seconds")
    fade_in: bool = Field(default=False, description="Fade in effect")
    color_hint: ColorHint = Field(
        default=ColorHint.NEUTRAL, description="Color temperature"
    )
    animation_type: AnimationType = Field(
        default=AnimationType.NONE, description="Animation suggestion"
    )

    @validator("size_multiplier", "duration_hint", pre=True)
    def round_size_values(cls, v):
        return round(float(v), 1) if v is not None else 1.0


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
    emotion: EmotionAnalysis = Field(..., description="Emotion analysis results")
    audio_features: AudioFeatures = Field(..., description="Acoustic features")
    subtitle_styling: SubtitleStyling = Field(
        ..., description="Subtitle styling recommendations"
    )

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


# Speaker Information Model
class SpeakerInfo(BaseModel):
    """Speaker information and statistics"""

    name: str = Field(..., description="Speaker name/identifier")
    total_duration: float = Field(
        ..., ge=0.0, description="Total speaking time in seconds"
    )
    avg_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average speaker identification confidence"
    )

    @validator("total_duration", "avg_confidence", pre=True)
    def round_speaker_values(cls, v):
        return round(float(v), 1) if v is not None else 0.0


# Model Version Information
class ModelVersions(BaseModel):
    """Model versions used in analysis"""

    speaker_diarization: str = Field(
        default="pyannote-2.1", description="Speaker diarization model version"
    )
    emotion_analysis: str = Field(
        default="wav2vec2-emotion-v1.0", description="Emotion analysis model version"
    )
    acoustic_features: str = Field(
        default="opensmile-3.0", description="Acoustic feature extraction version"
    )


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
    processing_time: str = Field(..., description="Processing time in HH:MM:SS format")
    gpu_acceleration: bool = Field(
        default=False, description="Whether GPU acceleration was used"
    )
    model_versions: ModelVersions = Field(
        default_factory=ModelVersions, description="Model version information"
    )

    @validator("duration", pre=True)
    def round_duration(cls, v):
        return round(float(v), 1) if v is not None else 0.0

    @validator("processing_time", pre=True)
    def format_processing_time(cls, v):
        if isinstance(v, (int, float)):
            # Convert seconds to HH:MM:SS format
            hours = int(v // 3600)
            minutes = int((v % 3600) // 60)
            seconds = int(v % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return str(v)


# Complete Analysis Result Model
class CompleteAnalysisResult(BaseModel):
    """Complete audio analysis result in JSON format"""

    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    speakers: Dict[str, SpeakerInfo] = Field(
        default_factory=dict, description="Speaker information"
    )
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
        speakers = values.get("speakers", {})
        metadata = values.get("metadata")

        if not timeline:
            return values

        # Validate speaker IDs exist in speakers dict
        timeline_speakers = {seg.speaker_id for seg in timeline}
        speaker_keys = set(speakers.keys())

        if timeline_speakers and not timeline_speakers.issubset(speaker_keys):
            missing_speakers = timeline_speakers - speaker_keys
            raise ValueError(
                f"Timeline references speakers not in speakers dict: {missing_speakers}"
            )

        # Validate total duration consistency
        if metadata and timeline:
            last_segment = max(timeline, key=lambda x: x.end_time)
            if last_segment.end_time > metadata.duration + 1.0:  # 1 second tolerance
                raise ValueError("Timeline extends beyond metadata duration")

        # Validate segments don't overlap inappropriately
        sorted_timeline = sorted(timeline, key=lambda x: x.start_time)
        for i in range(len(sorted_timeline) - 1):
            current = sorted_timeline[i]
            next_seg = sorted_timeline[i + 1]

            # Allow small overlaps (up to 0.1 seconds) for natural speech
            if current.end_time > next_seg.start_time + 0.1:
                # Only raise error if same speaker overlaps significantly
                if (
                    current.speaker_id == next_seg.speaker_id
                    and current.end_time > next_seg.start_time + 0.5
                ):
                    raise ValueError(
                        f"Significant overlap in same speaker segments: {current.start_time}-{current.end_time} and {next_seg.start_time}-{next_seg.end_time}"
                    )

        return values

    def model_dump_json_formatted(self) -> str:
        """Export as formatted JSON string"""
        return self.model_dump_json(indent=2, exclude_none=False)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary statistics"""
        return {
            "total_duration": self.metadata.duration,
            "total_speakers": self.metadata.total_speakers,
            "total_segments": len(self.timeline),
            "processing_time": self.metadata.processing_time,
            "gpu_acceleration": self.metadata.gpu_acceleration,
            "avg_segment_duration": (
                round(
                    sum(seg.duration for seg in self.timeline) / len(self.timeline), 1
                )
                if self.timeline
                else 0.0
            ),
            "speaker_distribution": {
                k: v.total_duration for k, v in self.speakers.items()
            },
        }


# Factory Functions
def create_empty_analysis_result(
    filename: str, duration: float
) -> CompleteAnalysisResult:
    """Create empty analysis result with basic metadata"""
    return CompleteAnalysisResult(
        metadata=AnalysisMetadata(
            filename=filename,
            duration=duration,
            total_speakers=0,
            processing_time="00:00:00",
        ),
        performance_stats=PerformanceStats(
            gpu_utilization=0.0,
            peak_memory_mb=0,
            avg_processing_fps=0.0,
            bottleneck_stage="initialization",
        ),
    )


def create_timeline_segment(
    start_time: float,
    end_time: float,
    speaker_id: str,
    speaker_confidence: float,
    emotion_data: Dict[str, Any],
    audio_features_data: Dict[str, Any],
    styling_hints: Optional[Dict[str, Any]] = None,
) -> TimelineSegment:
    """Factory function to create timeline segment with validation"""

    # Create emotion analysis
    emotion = EmotionAnalysis(
        primary=EmotionType(emotion_data["primary"]),
        confidence=emotion_data["confidence"],
        intensity=emotion_data["intensity"],
        valence=emotion_data["valence"],
        arousal=emotion_data["arousal"],
        all_scores=EmotionScores(**emotion_data["all_scores"]),
    )

    # Create audio features
    features = AudioFeatures(**audio_features_data)

    # Create subtitle styling
    if styling_hints is None:
        # Generate basic styling based on features
        styling_hints = {
            "size_multiplier": min(1.8, max(0.6, 0.8 + features.rms_energy * 2)),
            "emphasis_level": EmphasisLevel.MEDIUM,
            "duration_hint": end_time - start_time,
            "fade_in": features.amplitude_max > 0.8,
            "color_hint": ColorHint.WARM if emotion.valence > 0.3 else ColorHint.COOL,
            "animation_type": (
                AnimationType.PULSE
                if features.volume_category == VolumeCategory.EMPHASIS
                else AnimationType.NONE
            ),
        }

    styling = SubtitleStyling(**styling_hints)

    return TimelineSegment(
        start_time=start_time,
        end_time=end_time,
        speaker_id=speaker_id,
        speaker_confidence=speaker_confidence,
        emotion=emotion,
        audio_features=features,
        subtitle_styling=styling,
    )
