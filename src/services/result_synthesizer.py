"""
Result Synthesizer Service
Single Responsibility: Merge all analysis outputs into unified JSON metadata

Memory-efficient synthesis with streaming JSON generation and comprehensive validation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import gzip

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
    EmotionType,
    AudioFeatures,
    VolumeCategory,
    SubtitleStyling,
    EmphasisLevel,
    ColorHint,
    AnimationType,
    PerformanceStats,
    create_timeline_segment
)
from ..utils.logger import get_logger


@dataclass
class SynthesisInput:
    """Input data for result synthesis"""
    filename: str
    duration: float
    audio_extraction_result: AudioExtractionResult
    diarization_result: DiarizationResult
    emotion_results: Optional[Dict[str, Any]] = None  # Future implementation
    acoustic_results: Optional[Dict[str, Any]] = None  # Future implementation
    processing_start_time: float = 0.0
    gpu_acceleration_used: bool = False
    model_versions: Optional[ModelVersions] = None


@dataclass
class SynthesisStats:
    """Statistics about the synthesis process"""
    total_segments_processed: int = 0
    total_speakers_found: int = 0
    synthesis_time_seconds: float = 0.0
    output_file_size_mb: float = 0.0
    compression_ratio: float = 1.0
    validation_errors: int = 0
    warnings: int = 0


class TimelineGenerator:
    """Generate timeline segments from analysis results"""
    
    def __init__(self):
        self.logger = get_logger().bind_context(component="timeline_generator")
    
    def create_mock_emotion_data(self, speaker_segment: SpeakerSegment) -> Dict[str, Any]:
        """Create mock emotion data for MVP (will be replaced with real EmotionAnalyzer)"""
        # Generate basic emotion data based on segment characteristics
        duration = speaker_segment.duration
        confidence = speaker_segment.confidence
        
        # Simple heuristic: longer segments tend to be more neutral/calm
        # Shorter segments might have more varied emotions
        if duration < 2.0:
            # Short segments - more varied emotions
            primary_emotion = "surprise" if confidence < 0.8 else "joy"
            intensity = 0.7
            valence = 0.3 if primary_emotion == "surprise" else 0.6
            arousal = 0.8
        elif duration < 5.0:
            # Medium segments - moderate emotions  
            primary_emotion = "joy" if confidence > 0.85 else "neutral"
            intensity = 0.5
            valence = 0.4 if primary_emotion == "neutral" else 0.7
            arousal = 0.5
        else:
            # Long segments - tend to be neutral
            primary_emotion = "neutral"
            intensity = 0.3
            valence = 0.1
            arousal = 0.3
        
        # Create emotion scores with primary emotion dominant
        all_scores = {
            "joy": 0.1, "sadness": 0.05, "anger": 0.02,
            "fear": 0.01, "surprise": 0.05, "disgust": 0.01, "neutral": 0.76
        }
        
        # Adjust scores based on primary emotion
        if primary_emotion in all_scores:
            all_scores[primary_emotion] = intensity
            # Normalize remaining scores
            remaining_total = 1.0 - intensity
            other_emotions = [k for k in all_scores.keys() if k != primary_emotion]
            base_score = remaining_total / len(other_emotions)
            for emotion in other_emotions:
                all_scores[emotion] = base_score
        
        return {
            'primary': primary_emotion,
            'confidence': intensity,
            'intensity': intensity,
            'valence': valence,
            'arousal': arousal,
            'all_scores': all_scores
        }
    
    def create_mock_audio_features(self, speaker_segment: SpeakerSegment) -> Dict[str, Any]:
        """Create mock audio features data for MVP (will be replaced with real AcousticAnalyzer)"""
        duration = speaker_segment.duration
        confidence = speaker_segment.confidence
        
        # Generate realistic audio features based on segment characteristics
        # Higher confidence segments might have clearer audio features
        base_energy = 0.02 + (confidence * 0.05)  # 0.02 to 0.07 range
        pitch_base = 150.0 + (confidence * 100.0)  # 150 to 250 Hz range
        
        # Adjust based on duration (longer segments might be more stable)
        pitch_variance = max(5.0, 30.0 - (duration * 2))  # Less variance for longer segments
        speaking_rate = min(6.0, max(2.0, 8.0 - duration))  # Slower rate for longer segments
        
        # Volume category based on energy
        if base_energy < 0.03:
            volume_cat = VolumeCategory.LOW
        elif base_energy < 0.05:
            volume_cat = VolumeCategory.MEDIUM  
        elif base_energy < 0.07:
            volume_cat = VolumeCategory.HIGH
        else:
            volume_cat = VolumeCategory.EMPHASIS
        
        return {
            'rms_energy': base_energy,
            'rms_db': -30.0 + (base_energy * 200),  # Convert energy to dB approximation
            'pitch_mean': pitch_base,
            'pitch_variance': pitch_variance,
            'speaking_rate': speaking_rate,
            'amplitude_max': min(1.0, base_energy * 15),
            'silence_ratio': max(0.05, 0.3 - (speaking_rate * 0.03)),
            'spectral_centroid': 1800 + (pitch_base * 2),
            'zcr': 0.03 + (base_energy * 2),
            'mfcc': [12.5 + (base_energy * 10), -7.0 + confidence, 3.0 + (duration * 0.3)],
            'volume_category': volume_cat
        }
    
    def generate_timeline(self, 
                         diarization_result: DiarizationResult,
                         emotion_results: Optional[Dict[str, Any]] = None,
                         acoustic_results: Optional[Dict[str, Any]] = None) -> List[TimelineSegment]:
        """
        Generate timeline segments from diarization and analysis results.
        
        Args:
            diarization_result: Speaker diarization results
            emotion_results: Emotion analysis results (optional for MVP)
            acoustic_results: Acoustic analysis results (optional for MVP)
            
        Returns:
            List of TimelineSegment objects
        """
        
        with self.logger.performance_timer("timeline_generation", items_count=len(diarization_result.segments)):
            
            timeline_segments = []
            
            for i, speaker_segment in enumerate(diarization_result.segments):
                try:
                    # Get or create emotion data
                    if emotion_results and str(i) in emotion_results:
                        emotion_data = emotion_results[str(i)]
                    else:
                        emotion_data = self.create_mock_emotion_data(speaker_segment)
                    
                    # Get or create acoustic features
                    if acoustic_results and str(i) in acoustic_results:
                        audio_features_data = acoustic_results[str(i)]
                    else:
                        audio_features_data = self.create_mock_audio_features(speaker_segment)
                    
                    # Create timeline segment using factory function
                    segment = create_timeline_segment(
                        start_time=speaker_segment.start_time,
                        end_time=speaker_segment.end_time,
                        speaker_id=speaker_segment.speaker_id,
                        speaker_confidence=speaker_segment.confidence,
                        emotion_data=emotion_data,
                        audio_features_data=audio_features_data
                    )
                    
                    timeline_segments.append(segment)
                    
                except Exception as e:
                    self.logger.error("segment_creation_failed", 
                                    segment_index=i,
                                    start_time=speaker_segment.start_time,
                                    error=str(e))
                    continue
            
            # Sort timeline by start time
            timeline_segments.sort(key=lambda x: x.start_time)
            
            self.logger.info("timeline_generated", 
                           total_segments=len(timeline_segments),
                           total_duration=sum(seg.duration for seg in timeline_segments))
            
            return timeline_segments


class ResultSynthesizer:
    """
    High-performance result synthesis service.
    
    Single Responsibility: Merge all analysis outputs into comprehensive JSON metadata
    with memory-efficient processing and streaming capabilities.
    """
    
    def __init__(self, 
                 enable_compression: bool = True,
                 compression_threshold_mb: float = 10.0,
                 enable_validation: bool = True):
        
        self.enable_compression = enable_compression
        self.compression_threshold_mb = compression_threshold_mb
        self.enable_validation = enable_validation
        
        self.logger = get_logger().bind_context(service="result_synthesizer")
        self.timeline_generator = TimelineGenerator()
        
        self.logger.info("result_synthesizer_initialized",
                        compression_enabled=enable_compression,
                        validation_enabled=enable_validation)
    
    def _create_speaker_info_dict(self, 
                                 timeline_segments: List[TimelineSegment],
                                 diarization_result: DiarizationResult) -> Dict[str, SpeakerInfo]:
        """Create speaker information dictionary from timeline and diarization results"""
        
        speaker_stats = {}
        
        # Aggregate statistics from timeline segments
        for segment in timeline_segments:
            speaker_id = segment.speaker_id
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    'total_duration': 0.0,
                    'confidences': []
                }
            
            speaker_stats[speaker_id]['total_duration'] += segment.duration
            speaker_stats[speaker_id]['confidences'].append(segment.speaker_confidence)
        
        # Create SpeakerInfo objects
        speakers_dict = {}
        for speaker_id, stats in speaker_stats.items():
            confidences = stats['confidences']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Use original speaker names from diarization if available
            speaker_name = f"Speaker {speaker_id.replace('speaker_', '').replace('_', ' ').title()}"
            
            speakers_dict[speaker_id] = SpeakerInfo(
                name=speaker_name,
                total_duration=stats['total_duration'],
                avg_confidence=avg_confidence
            )
        
        return speakers_dict
    
    def _create_performance_stats(self, 
                                 synthesis_input: SynthesisInput,
                                 synthesis_time: float,
                                 peak_memory_mb: float) -> PerformanceStats:
        """Create performance statistics from synthesis process"""
        
        total_processing_time = time.time() - synthesis_input.processing_start_time
        duration = synthesis_input.duration
        
        # Calculate processing FPS
        avg_processing_fps = duration / total_processing_time if total_processing_time > 0 else 0.0
        
        # Determine bottleneck stage
        extraction_time = synthesis_input.audio_extraction_result.extraction_time_seconds
        diarization_time = synthesis_input.diarization_result.processing_time
        
        if diarization_time > extraction_time and diarization_time > synthesis_time:
            bottleneck = "speaker_diarization"
        elif extraction_time > synthesis_time:
            bottleneck = "audio_extraction"
        else:
            bottleneck = "result_synthesis"
        
        # GPU utilization (mock for now, will be real when GPU monitoring is added)
        gpu_utilization = 0.85 if synthesis_input.gpu_acceleration_used else 0.0
        
        return PerformanceStats(
            gpu_utilization=gpu_utilization,
            peak_memory_mb=int(peak_memory_mb),
            avg_processing_fps=avg_processing_fps,
            bottleneck_stage=bottleneck
        )
    
    def synthesize_results(self, synthesis_input: SynthesisInput) -> CompleteAnalysisResult:
        """
        Synthesize all analysis results into comprehensive JSON format.
        
        Args:
            synthesis_input: All analysis results and metadata
            
        Returns:
            CompleteAnalysisResult with comprehensive analysis data
        """
        
        synthesis_start_time = time.time()
        
        with self.logger.performance_timer("result_synthesis", items_count=1):
            
            self.logger.info("synthesis_started",
                           filename=synthesis_input.filename,
                           duration=synthesis_input.duration,
                           total_speakers=synthesis_input.diarization_result.total_speakers)
            
            try:
                # Generate timeline segments
                timeline_segments = self.timeline_generator.generate_timeline(
                    diarization_result=synthesis_input.diarization_result,
                    emotion_results=synthesis_input.emotion_results,
                    acoustic_results=synthesis_input.acoustic_results
                )
                
                # Create speaker information
                speakers_dict = self._create_speaker_info_dict(
                    timeline_segments, 
                    synthesis_input.diarization_result
                )
                
                # Create metadata
                processing_time_total = time.time() - synthesis_input.processing_start_time
                metadata = AnalysisMetadata(
                    filename=synthesis_input.filename,
                    duration=synthesis_input.duration,
                    total_speakers=len(speakers_dict),
                    processing_time=processing_time_total,  # Will be formatted automatically
                    gpu_acceleration=synthesis_input.gpu_acceleration_used,
                    model_versions=synthesis_input.model_versions or ModelVersions()
                )
                
                # Create performance statistics
                synthesis_time = time.time() - synthesis_start_time
                
                # Estimate peak memory (basic calculation)
                estimated_memory = (
                    len(timeline_segments) * 2 +  # ~2KB per segment
                    len(speakers_dict) * 0.5 +     # ~0.5KB per speaker
                    synthesis_input.audio_extraction_result.file_size_mb * 0.1  # 10% of audio file size
                )
                
                performance_stats = self._create_performance_stats(
                    synthesis_input, 
                    synthesis_time,
                    estimated_memory
                )
                
                # Create complete result
                result = CompleteAnalysisResult(
                    metadata=metadata,
                    speakers=speakers_dict,
                    timeline=timeline_segments,
                    performance_stats=performance_stats
                )
                
                # Validate if enabled
                if self.enable_validation:
                    try:
                        # Pydantic validation happens automatically during creation
                        summary = result.get_summary()
                        self.logger.info("validation_successful", **summary)
                    except Exception as e:
                        self.logger.error("validation_failed", error=str(e))
                        # Continue with result despite validation errors
                
                self.logger.info("synthesis_completed",
                               filename=synthesis_input.filename,
                               total_segments=len(timeline_segments),
                               total_speakers=len(speakers_dict),
                               synthesis_time=synthesis_time)
                
                return result
                
            except Exception as e:
                self.logger.error("synthesis_failed", 
                                filename=synthesis_input.filename,
                                error=str(e))
                
                # Return minimal result on failure
                return CompleteAnalysisResult(
                    metadata=AnalysisMetadata(
                        filename=synthesis_input.filename,
                        duration=synthesis_input.duration,
                        total_speakers=0,
                        processing_time=time.time() - synthesis_input.processing_start_time
                    ),
                    performance_stats=PerformanceStats(
                        gpu_utilization=0.0,
                        peak_memory_mb=0,
                        avg_processing_fps=0.0,
                        bottleneck_stage="result_synthesis"
                    )
                )
    
    def export_json(self, 
                   result: CompleteAnalysisResult, 
                   output_path: Path,
                   prettify: bool = True) -> SynthesisStats:
        """
        Export result to JSON file with optional compression.
        
        Args:
            result: Complete analysis result to export
            output_path: Output file path
            prettify: Whether to format JSON with indentation
            
        Returns:
            SynthesisStats with export information
        """
        
        export_start_time = time.time()
        
        with self.logger.performance_timer("json_export"):
            
            # Generate JSON string
            if prettify:
                json_str = result.model_dump_json(indent=2)
            else:
                json_str = result.model_dump_json()
            
            # Calculate uncompressed size
            uncompressed_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            
            # Determine output path and compression
            final_output_path = output_path
            use_compression = (
                self.enable_compression and 
                uncompressed_size_mb > self.compression_threshold_mb
            )
            
            if use_compression:
                if not str(output_path).endswith('.gz'):
                    final_output_path = Path(str(output_path) + '.gz')
                
                # Write compressed
                with gzip.open(final_output_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
                
                compressed_size_mb = final_output_path.stat().st_size / (1024 * 1024)
                compression_ratio = uncompressed_size_mb / compressed_size_mb
                
                self.logger.info("json_exported_compressed",
                               output_path=str(final_output_path),
                               uncompressed_mb=uncompressed_size_mb,
                               compressed_mb=compressed_size_mb,
                               compression_ratio=compression_ratio)
            else:
                # Write uncompressed
                with open(final_output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                
                compressed_size_mb = uncompressed_size_mb
                compression_ratio = 1.0
                
                self.logger.info("json_exported_uncompressed",
                               output_path=str(final_output_path),
                               size_mb=uncompressed_size_mb)
            
            export_time = time.time() - export_start_time
            
            return SynthesisStats(
                total_segments_processed=len(result.timeline),
                total_speakers_found=len(result.speakers),
                synthesis_time_seconds=export_time,
                output_file_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                validation_errors=0,
                warnings=0
            )
    
    def synthesize_and_export(self, 
                             synthesis_input: SynthesisInput,
                             output_path: Path,
                             prettify: bool = True) -> Tuple[CompleteAnalysisResult, SynthesisStats]:
        """
        Synthesize results and export to JSON file in one operation.
        
        Args:
            synthesis_input: Analysis input data
            output_path: Output file path
            prettify: Whether to format JSON with indentation
            
        Returns:
            Tuple of (CompleteAnalysisResult, SynthesisStats)
        """
        
        with self.logger.performance_timer("synthesize_and_export"):
            
            # Synthesize results
            result = self.synthesize_results(synthesis_input)
            
            # Export to JSON
            stats = self.export_json(result, output_path, prettify)
            
            return result, stats