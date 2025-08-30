"""
Speaker Diarizer - Real speaker diarization using pyannote.audio
"""

import torch
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from ..utils.logger import get_logger
from .model_manager import get_model_manager

@dataclass
class SpeakerSegment:
    """Speaker segment information"""
    start: float
    end: float
    speaker: str
    confidence: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerDiarizer:
    """Real speaker diarization using pyannote.audio"""
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 min_speakers: int = 2,
                 max_speakers: int = 4,
                 device: Optional[str] = None):
        """
        Initialize speaker diarizer
        
        Args:
            model_name: Pyannote model name
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers  
            device: Device to use (None for auto-detection)
        """
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Get model manager
        self.model_manager = get_model_manager(device=device)
        self.device = self.model_manager.get_device()
        
        self.logger = get_logger().bind_context(
            service="speaker_diarizer",
            model=model_name,
            device=self.device
        )
        
        # Model will be loaded on first use
        self.pipeline = None
        
    def _load_model(self):
        """Load pyannote pipeline"""
        if self.pipeline is None:
            self.logger.info("loading_speaker_diarization_model")
            self.pipeline = self.model_manager.load_speaker_model(self.model_name)
            
            # Configure number of speakers
            if hasattr(self.pipeline, '_segmentation') and hasattr(self.pipeline._segmentation, 'model'):
                # Configure clustering
                try:
                    self.pipeline.instantiate({
                        "clustering": {
                            "min_cluster_size": self.min_speakers,
                            "max_num_speakers": self.max_speakers,
                        }
                    })
                except Exception as e:
                    self.logger.warning("failed_to_configure_clustering", error=str(e))
    
    def diarize_audio(self, 
                     audio_path: Union[str, Path],
                     sample_rate: int = 16000) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            
        Returns:
            List of speaker segments
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        self.logger.info("starting_diarization", 
                        file=str(audio_path),
                        sample_rate=sample_rate)
        
        try:
            # Apply diarization pipeline with speaker constraints
            diarization = self.pipeline(str(audio_path), 
                                      min_speakers=self.min_speakers,
                                      max_speakers=self.max_speakers)
            
            # Convert to our format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # pyannote doesn't provide confidence directly
                )
                segments.append(segment)
            
            # Sort by start time
            segments.sort(key=lambda x: x.start)
            
            # Apply automatic segment optimization
            segments = self.filter_short_segments(segments, min_duration=2.0)
            segments = self.merge_adjacent_segments(segments, max_gap=1.0)
            
            self.logger.info("diarization_completed",
                           segments_count=len(segments),
                           unique_speakers=len(set(s.speaker for s in segments)))
            
            return segments
            
        except Exception as e:
            self.logger.error("diarization_failed", error=str(e))
            # Fallback to librosa-based approach
            return self._fallback_diarization(audio_path, sample_rate)
    
    def _fallback_diarization(self, 
                            audio_path: Union[str, Path],
                            sample_rate: int,
                            segment_length: float = 3.0) -> List[SpeakerSegment]:
        """
        Fallback speaker diarization using acoustic features
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            segment_length: Length of each segment in seconds
            
        Returns:
            List of speaker segments
        """
        self.logger.warning("using_fallback_diarization")
        
        try:
            # Load audio with librosa
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            duration = len(y) / sr
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Create segments
            num_segments = int(duration // segment_length) + (1 if duration % segment_length > 0 else 0)
            segments = []
            
            # Calculate feature statistics for clustering
            features_per_segment = []
            segment_times = []
            
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = min((i + 1) * segment_length, duration)
                
                if end_time - start_time < 0.5:  # Skip very short segments
                    continue
                
                # Calculate frame indices
                start_frame = int(start_time * len(spectral_centroid) / duration)
                end_frame = int(end_time * len(spectral_centroid) / duration)
                
                if start_frame >= end_frame or end_frame > len(spectral_centroid):
                    continue
                
                # Extract features for this segment
                segment_features = np.array([
                    np.mean(spectral_centroid[start_frame:end_frame]),
                    np.std(spectral_centroid[start_frame:end_frame]),
                    np.mean(spectral_rolloff[start_frame:end_frame]),
                    np.std(spectral_rolloff[start_frame:end_frame]),
                    np.mean(zero_crossing_rate[start_frame:end_frame]),
                    np.std(zero_crossing_rate[start_frame:end_frame]),
                    np.mean(mfcc[:, start_frame:end_frame]),
                    np.std(mfcc[:, start_frame:end_frame])
                ])
                
                features_per_segment.append(segment_features)
                segment_times.append((start_time, end_time))
            
            # Simple clustering based on spectral centroid
            if len(features_per_segment) > 0:
                features_array = np.array(features_per_segment)
                
                # Use spectral centroid as primary feature for speaker separation
                centroids = features_array[:, 0]  # spectral centroid mean
                median_centroid = np.median(centroids)
                
                # Assign speakers based on spectral centroid
                for i, (start_time, end_time) in enumerate(segment_times):
                    if i < len(centroids):
                        speaker_id = "speaker_01" if centroids[i] <= median_centroid else "speaker_02"
                        confidence = 0.6 + 0.3 * abs(centroids[i] - median_centroid) / (np.max(centroids) - np.min(centroids))
                        confidence = min(0.95, max(0.3, confidence))
                        
                        segment = SpeakerSegment(
                            start=start_time,
                            end=end_time,
                            speaker=speaker_id,
                            confidence=confidence
                        )
                        segments.append(segment)
            
            self.logger.info("fallback_diarization_completed",
                           segments_count=len(segments))
            
            return segments
            
        except Exception as e:
            self.logger.error("fallback_diarization_failed", error=str(e))
            raise
    
    def get_speaker_statistics(self, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """
        Calculate speaker statistics
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary with speaker statistics
        """
        if not segments:
            return {}
        
        stats = {}
        speakers = set(s.speaker for s in segments)
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s.speaker == speaker]
            total_duration = sum(s.duration for s in speaker_segments)
            avg_confidence = sum(s.confidence for s in speaker_segments) / len(speaker_segments)
            
            stats[speaker] = {
                "total_duration": total_duration,
                "segment_count": len(speaker_segments),
                "avg_confidence": avg_confidence,
                "speaking_ratio": total_duration / sum(s.duration for s in segments)
            }
        
        return stats
    
    def filter_short_segments(self,
                            segments: List[SpeakerSegment],
                            min_duration: float = 2.0) -> List[SpeakerSegment]:
        """
        Filter out or merge segments that are too short
        
        Args:
            segments: List of speaker segments
            min_duration: Minimum duration for segments (seconds)
            
        Returns:
            List of filtered segments
        """
        if not segments:
            return segments
        
        filtered = []
        for i, segment in enumerate(segments):
            if segment.duration >= min_duration:
                filtered.append(segment)
            else:
                # Try to merge with previous segment of same speaker
                if filtered and filtered[-1].speaker == segment.speaker:
                    last = filtered[-1]
                    filtered[-1] = SpeakerSegment(
                        start=last.start,
                        end=segment.end,
                        speaker=last.speaker,
                        confidence=max(last.confidence, segment.confidence)
                    )
                # Try to merge with next segment of same speaker
                elif (i + 1 < len(segments) and 
                      segments[i + 1].speaker == segment.speaker):
                    # Will be merged when processing next segment
                    next_segment = segments[i + 1]
                    segments[i + 1] = SpeakerSegment(
                        start=segment.start,
                        end=next_segment.end,
                        speaker=segment.speaker,
                        confidence=max(segment.confidence, next_segment.confidence)
                    )
                else:
                    # Keep isolated short segments (better than losing speech)
                    filtered.append(segment)
        
        return filtered

    def merge_adjacent_segments(self, 
                              segments: List[SpeakerSegment],
                              max_gap: float = 1.0) -> List[SpeakerSegment]:
        """
        Merge adjacent segments from the same speaker
        
        Args:
            segments: List of speaker segments
            max_gap: Maximum gap to merge (seconds)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return segments
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x.start)
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # Check if same speaker and close enough
            if (current.speaker == last.speaker and 
                current.start - last.end <= max_gap):
                # Merge segments
                merged[-1] = SpeakerSegment(
                    start=last.start,
                    end=current.end,
                    speaker=last.speaker,
                    confidence=max(last.confidence, current.confidence)
                )
            else:
                merged.append(current)
        
        return merged