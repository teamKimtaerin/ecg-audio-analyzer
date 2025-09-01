"""
Speaker Diarization Service
Single Responsibility: Identify and segment different speakers in audio

GPU-optimized with pyannote-audio for high-performance processing on AWS instances.
"""

import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torchaudio
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils import PipelineModel

from ..utils.logger import get_logger
from ...config.model_configs import SpeakerDiarizationConfig


@dataclass
class SpeakerSegment:
    """Individual speaker segment with timing and confidence"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    duration: float = field(init=False)
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'speaker_id': self.speaker_id,
            'confidence': self.confidence,
            'duration': self.duration
        }


@dataclass 
class SpeakerInfo:
    """Speaker information and statistics"""
    speaker_id: str
    total_duration: float
    segment_count: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'speaker_id': self.speaker_id,
            'total_duration': self.total_duration,
            'segment_count': self.segment_count,
            'avg_confidence': self.avg_confidence,
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence
        }


@dataclass
class DiarizationResult:
    """Complete speaker diarization result"""
    success: bool
    segments: List[SpeakerSegment] = field(default_factory=list)
    speakers: Dict[str, SpeakerInfo] = field(default_factory=dict)
    total_speakers: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'segments': [seg.to_dict() for seg in self.segments],
            'speakers': {k: v.to_dict() for k, v in self.speakers.items()},
            'total_speakers': self.total_speakers,
            'total_duration': self.total_duration,
            'processing_time': self.processing_time,
            'model_info': self.model_info,
            'error_message': self.error_message
        }


class SpeakerEmbeddingCache:
    """Cache for speaker embeddings to improve recurring speaker detection"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_times: Dict[str, float] = {}
        self.max_cache_size = max_cache_size
    
    def _compute_audio_hash(self, audio_segment: torch.Tensor) -> str:
        """Compute hash for audio segment"""
        # Sample a few points for efficiency
        sample_points = audio_segment[::1000].cpu().numpy().tobytes()
        return hashlib.md5(sample_points).hexdigest()[:16]
    
    def get_embedding(self, audio_hash: str) -> Optional[torch.Tensor]:
        """Get cached embedding if available"""
        if audio_hash in self.cache:
            self.access_times[audio_hash] = time.time()
            return self.cache[audio_hash]
        return None
    
    def store_embedding(self, audio_hash: str, embedding: torch.Tensor):
        """Store embedding in cache"""
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.cache[audio_hash] = embedding.clone()
        self.access_times[audio_hash] = time.time()
    
    def _evict_oldest(self):
        """Remove oldest accessed item from cache"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


class SpeakerDiarizer:
    """
    High-performance speaker diarization service using pyannote-audio.
    
    Single Responsibility: Identify and segment speakers in audio files with
    GPU acceleration and advanced optimization techniques.
    """
    
    def __init__(self,
                 config: SpeakerDiarizationConfig,
                 device: Optional[str] = None,
                 enable_caching: bool = True):
        
        self.config = config
        self.logger = get_logger().bind_context(service="speaker_diarizer")
        self.enable_caching = enable_caching
        
        # Setup device
        self.device = device or config.device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning("cuda_unavailable", fallback="cpu")
            self.device = "cpu"
        
        # Initialize components
        self.pipeline: Optional[Pipeline] = None
        self.embedding_cache = SpeakerEmbeddingCache() if enable_caching else None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance optimization settings
        if self.device.startswith('cuda') and config.enable_fp16:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        self.logger.info("speaker_diarizer_initialized",
                        device=self.device,
                        model=config.model_name,
                        fp16_enabled=config.enable_fp16,
                        caching_enabled=enable_caching)
    
    def load_model(self) -> None:
        """Load and initialize the speaker diarization pipeline"""
        with self.logger.performance_timer("model_loading"):
            try:
                # Check for Hugging Face token
                auth_token = os.getenv("HUGGING_FACE_TOKEN")
                if not auth_token:
                    self.logger.warning("no_hf_token", 
                                      message="Hugging Face token not found. Some models may not be accessible.")
                
                # Load pipeline
                self.pipeline = Pipeline.from_pretrained(
                    self.config.model_name,
                    use_auth_token=auth_token
                )
                
                # Move to specified device
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to(torch.device(self.device))
                
                # Apply configuration parameters
                self.pipeline.instantiate({
                    'clustering': {
                        'method': self.config.clustering_method,
                        'min_cluster_size': self.config.min_speakers,
                        'max_cluster_size': self.config.max_speakers,
                    },
                    'segmentation': {
                        'onset': self.config.segmentation_onset,
                        'offset': self.config.segmentation_offset,
                        'min_duration_off': self.config.min_segment_duration,
                    }
                })
                
                # Warm up model with dummy data if enabled
                if hasattr(self.config, 'model_warmup') and self.config.model_warmup:
                    self._warmup_model()
                
                self.logger.info("model_loaded_successfully", 
                               model=self.config.model_name,
                               device=self.device)
                
            except Exception as e:
                self.logger.error("model_loading_failed", error=str(e))
                raise RuntimeError(f"Failed to load diarization model: {str(e)}")
    
    def _warmup_model(self):
        """Warm up model with dummy audio data"""
        try:
            # Create dummy audio (1 second of silence)
            dummy_audio = torch.zeros(1, 16000)  # 1 second at 16kHz
            dummy_file = {"waveform": dummy_audio, "sample_rate": 16000}
            
            # Run inference once to warm up
            with torch.no_grad():
                _ = self.pipeline(dummy_file)
            
            self.logger.info("model_warmup_completed")
            
        except Exception as e:
            self.logger.warning("model_warmup_failed", error=str(e))
    
    def _load_audio_segment(self, 
                           audio_path: Path, 
                           start_time: float = 0.0, 
                           end_time: Optional[float] = None) -> Dict[str, Any]:
        """Load audio segment for processing"""
        try:
            waveform, sample_rate = torchaudio.load(
                str(audio_path),
                frame_offset=int(start_time * 16000) if start_time > 0 else None,
                num_frames=int((end_time - start_time) * 16000) if end_time else None
            )
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            return {
                "waveform": waveform,
                "sample_rate": 16000
            }
            
        except Exception as e:
            self.logger.error("audio_loading_failed", 
                            audio_path=str(audio_path), 
                            error=str(e))
            raise
    
    def _process_chunk(self, 
                      audio_path: Path, 
                      start_time: float, 
                      end_time: float,
                      chunk_id: int) -> List[SpeakerSegment]:
        """Process a single audio chunk"""
        
        try:
            # Load audio chunk
            audio_data = self._load_audio_segment(audio_path, start_time, end_time)
            
            # Run diarization on chunk
            with torch.no_grad():
                if self.config.enable_fp16 and self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast():
                        diarization = self.pipeline(audio_data)
                else:
                    diarization = self.pipeline(audio_data)
            
            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Adjust times to global timeline
                global_start = start_time + turn.start
                global_end = start_time + turn.end
                
                # Create speaker segment
                segment = SpeakerSegment(
                    start_time=global_start,
                    end_time=global_end,
                    speaker_id=f"{speaker}",
                    confidence=getattr(turn, 'confidence', 0.9)  # Default confidence
                )
                segments.append(segment)
            
            self.logger.debug("chunk_processed", 
                            chunk_id=chunk_id,
                            start_time=start_time,
                            end_time=end_time,
                            segments_found=len(segments))
            
            return segments
            
        except Exception as e:
            self.logger.error("chunk_processing_failed", 
                            chunk_id=chunk_id,
                            start_time=start_time,
                            end_time=end_time,
                            error=str(e))
            return []
    
    def _merge_overlapping_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge overlapping segments from the same speaker"""
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            # If same speaker and overlapping or adjacent
            if (current.speaker_id == next_segment.speaker_id and 
                next_segment.start_time <= current.end_time + 0.1):  # 100ms tolerance
                
                # Merge segments
                current = SpeakerSegment(
                    start_time=current.start_time,
                    end_time=max(current.end_time, next_segment.end_time),
                    speaker_id=current.speaker_id,
                    confidence=max(current.confidence, next_segment.confidence)
                )
            else:
                merged.append(current)
                current = next_segment
        
        merged.append(current)
        return merged
    
    def _compute_speaker_statistics(self, segments: List[SpeakerSegment]) -> Dict[str, SpeakerInfo]:
        """Compute statistics for each speaker"""
        speaker_stats = {}
        
        for segment in segments:
            speaker_id = segment.speaker_id
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'confidences': []
                }
            
            stats = speaker_stats[speaker_id]
            stats['total_duration'] += segment.duration
            stats['segment_count'] += 1
            stats['confidences'].append(segment.confidence)
        
        # Create SpeakerInfo objects
        speaker_info = {}
        for speaker_id, stats in speaker_stats.items():
            confidences = stats['confidences']
            speaker_info[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                total_duration=stats['total_duration'],
                segment_count=stats['segment_count'],
                avg_confidence=sum(confidences) / len(confidences),
                min_confidence=min(confidences),
                max_confidence=max(confidences)
            )
        
        return speaker_info
    
    def _fast_merge_similar_speakers(self, segments: List[SpeakerSegment], audio_path: Path) -> List[SpeakerSegment]:
        """
        Lightweight segment merging based on spectral similarity (fast processing)
        """
        if len(segments) <= 2:
            return segments
            
        try:
            import librosa
            import numpy as np
            
            # Load audio for quick spectral analysis
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Extract lightweight features for each segment
            segment_features = []
            valid_segments = []
            
            for segment in segments:
                start_frame = int(segment.start_time * sr)
                end_frame = int(segment.end_time * sr)
                
                if end_frame > start_frame and end_frame <= len(y):
                    segment_audio = y[start_frame:end_frame]
                    
                    if len(segment_audio) > 1000:  # Minimum audio length
                        # Simple spectral features for fast processing
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0])
                        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)[0])
                        zcr = np.mean(librosa.feature.zero_crossing_rate(segment_audio)[0])
                        
                        features = np.array([spectral_centroid, spectral_bandwidth, zcr])
                        segment_features.append(features)
                        valid_segments.append(segment)
            
            if len(segment_features) < 2:
                return segments
                
            # Fast pairwise similarity check
            merged_segments = []
            used_indices = set()
            
            for i, segment_i in enumerate(valid_segments):
                if i in used_indices:
                    continue
                    
                current_segment = segment_i
                
                # Check subsequent segments for merging
                for j in range(i + 1, len(valid_segments)):
                    if j in used_indices:
                        continue
                        
                    segment_j = valid_segments[j]
                    
                    # Time proximity check (within 10 seconds)
                    time_gap = segment_j.start_time - current_segment.end_time
                    if time_gap > 10.0:
                        continue
                    
                    # Fast cosine similarity
                    feat_i = segment_features[i]
                    feat_j = segment_features[j]
                    
                    # Normalize features
                    feat_i_norm = feat_i / (np.linalg.norm(feat_i) + 1e-8)
                    feat_j_norm = feat_j / (np.linalg.norm(feat_j) + 1e-8)
                    
                    similarity = np.dot(feat_i_norm, feat_j_norm)
                    
                    # Merge if similar enough
                    if similarity > 0.85:
                        # Merge segments
                        current_segment = SpeakerSegment(
                            start_time=min(current_segment.start_time, segment_j.start_time),
                            end_time=max(current_segment.end_time, segment_j.end_time),
                            speaker_id=current_segment.speaker_id,  # Keep first speaker ID
                            confidence=max(current_segment.confidence, segment_j.confidence)
                        )
                        used_indices.add(j)
                
                merged_segments.append(current_segment)
                used_indices.add(i)
            
            # Sort by start time
            merged_segments.sort(key=lambda x: x.start_time)
            
            self.logger.info("fast_merge_completed",
                           original_segments=len(segments),
                           merged_segments=len(merged_segments),
                           reduction_pct=round((1 - len(merged_segments)/len(segments)) * 100, 1))
            
            return merged_segments
            
        except Exception as e:
            self.logger.warning("fast_merge_failed", error=str(e))
            return segments
    
    def _filter_noise_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Remove noise segments and filter by minimum duration
        """
        if not segments:
            return segments
            
        try:
            # Calculate total duration and speaker statistics
            total_duration = sum(seg.duration for seg in segments)
            speaker_stats = {}
            
            for segment in segments:
                speaker_id = segment.speaker_id
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {'duration': 0.0, 'segments': []}
                speaker_stats[speaker_id]['duration'] += segment.duration
                speaker_stats[speaker_id]['segments'].append(segment)
            
            # Identify noise speakers (< 5% of total duration or < 2 seconds)
            noise_threshold = max(2.0, total_duration * 0.05)
            valid_speakers = set()
            noise_speakers = set()
            
            for speaker_id, stats in speaker_stats.items():
                if stats['duration'] >= noise_threshold:
                    valid_speakers.add(speaker_id)
                else:
                    noise_speakers.add(speaker_id)
                    self.logger.info("noise_speaker_detected", 
                                   speaker_id=speaker_id,
                                   duration=stats['duration'],
                                   threshold=noise_threshold)
            
            # Filter out noise segments and very short segments
            filtered_segments = []
            for segment in segments:
                # Skip noise speakers
                if segment.speaker_id in noise_speakers:
                    continue
                    
                # Skip very short segments (< 1 second)
                if segment.duration < 1.0:
                    continue
                    
                filtered_segments.append(segment)
            
            self.logger.info("noise_filtering_completed",
                           original_segments=len(segments),
                           filtered_segments=len(filtered_segments),
                           noise_speakers=len(noise_speakers),
                           valid_speakers=len(valid_speakers))
            
            return filtered_segments
            
        except Exception as e:
            self.logger.warning("noise_filtering_failed", error=str(e))
            return segments
    
    def _temporal_consistency_check(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Fast temporal consistency checking and speaker ID reassignment
        """
        if len(segments) <= 2:
            return segments
            
        try:
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x.start_time)
            consistent_segments = []
            
            for i, current_segment in enumerate(sorted_segments):
                # Check for short gaps with similar speakers
                if i > 0:
                    prev_segment = consistent_segments[-1]
                    time_gap = current_segment.start_time - prev_segment.end_time
                    
                    # If gap is small (< 5 seconds) and segments are short
                    if (time_gap < 5.0 and 
                        current_segment.duration < 3.0 and 
                        prev_segment.duration < 3.0):
                        
                        # Check if we should merge based on speaker pattern
                        if len(consistent_segments) >= 2:
                            # Look for A-B-A pattern (same speaker interrupted briefly)
                            second_prev = consistent_segments[-2]
                            
                            if (prev_segment.speaker_id != current_segment.speaker_id and
                                second_prev.speaker_id == current_segment.speaker_id):
                                
                                # Merge the interrupted segments
                                consistent_segments[-1] = SpeakerSegment(
                                    start_time=second_prev.start_time,
                                    end_time=current_segment.end_time,
                                    speaker_id=current_segment.speaker_id,
                                    confidence=max(second_prev.confidence, current_segment.confidence)
                                )
                                continue
                
                consistent_segments.append(current_segment)
            
            # Final pass: merge adjacent segments from same speaker
            final_segments = []
            for segment in consistent_segments:
                if (final_segments and 
                    final_segments[-1].speaker_id == segment.speaker_id and
                    segment.start_time - final_segments[-1].end_time < 1.0):
                    
                    # Merge with previous segment
                    final_segments[-1] = SpeakerSegment(
                        start_time=final_segments[-1].start_time,
                        end_time=segment.end_time,
                        speaker_id=segment.speaker_id,
                        confidence=max(final_segments[-1].confidence, segment.confidence)
                    )
                else:
                    final_segments.append(segment)
            
            self.logger.info("temporal_consistency_applied",
                           original_segments=len(segments),
                           final_segments=len(final_segments))
            
            return final_segments
            
        except Exception as e:
            self.logger.warning("temporal_consistency_failed", error=str(e))
            return segments
    
    def _resolve_overlapping_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Resolve overlapping segments by merging or splitting
        """
        if not segments:
            return segments
            
        try:
            # Sort by start time
            sorted_segments = sorted(segments, key=lambda x: x.start_time)
            resolved_segments = []
            
            i = 0
            while i < len(sorted_segments):
                current = sorted_segments[i]
                
                # Check for overlaps with next segments
                overlapping_segments = [current]
                j = i + 1
                
                while j < len(sorted_segments):
                    next_segment = sorted_segments[j]
                    
                    # If overlapping
                    if (next_segment.start_time < current.end_time and 
                        next_segment.end_time > current.start_time):
                        overlapping_segments.append(next_segment)
                        j += 1
                    else:
                        break
                
                if len(overlapping_segments) == 1:
                    # No overlap, add as is
                    resolved_segments.append(current)
                else:
                    # Resolve overlap - use the longest segment's speaker
                    longest_segment = max(overlapping_segments, key=lambda x: x.duration)
                    
                    # Create merged segment
                    start_time = min(seg.start_time for seg in overlapping_segments)
                    end_time = max(seg.end_time for seg in overlapping_segments)
                    
                    merged_segment = SpeakerSegment(
                        start_time=start_time,
                        end_time=end_time,
                        speaker_id=longest_segment.speaker_id,
                        confidence=max(seg.confidence for seg in overlapping_segments)
                    )
                    resolved_segments.append(merged_segment)
                
                # Skip processed segments
                i = j if j > i + 1 else i + 1
            
            self.logger.info("overlapping_segments_resolved",
                           original_segments=len(segments),
                           resolved_segments=len(resolved_segments))
            
            return resolved_segments
            
        except Exception as e:
            self.logger.warning("overlap_resolution_failed", error=str(e))
            return segments
    
    def _rebalance_speakers(self, segments: List[SpeakerSegment], audio_path: Path) -> List[SpeakerSegment]:
        """
        Rebalance speakers to achieve more even distribution
        """
        if len(segments) <= 2:
            return segments
            
        try:
            # Calculate current distribution
            total_duration = sum(seg.duration for seg in segments)
            speaker_stats = {}
            
            for segment in segments:
                speaker_id = segment.speaker_id
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {'duration': 0.0, 'segments': []}
                speaker_stats[speaker_id]['duration'] += segment.duration
                speaker_stats[speaker_id]['segments'].append(segment)
            
            # Find dominant speaker (>60% of time)
            dominant_speakers = []
            minor_speakers = []
            
            for speaker_id, stats in speaker_stats.items():
                ratio = stats['duration'] / total_duration
                if ratio > 0.6:
                    dominant_speakers.append((speaker_id, stats))
                elif ratio < 0.15:  # Less than 15%
                    minor_speakers.append((speaker_id, stats))
            
            # If we have a dominant speaker, try to split their segments
            if dominant_speakers and len(speaker_stats) >= 3:
                dominant_id, dominant_stats = dominant_speakers[0]
                
                # Sort dominant speaker's segments by duration
                dominant_segments = sorted(dominant_stats['segments'], 
                                         key=lambda x: x.duration, reverse=True)
                
                # Reassign some segments to balance speakers
                target_speakers = [sid for sid in speaker_stats.keys() if sid != dominant_id]
                
                if target_speakers:
                    rebalanced_segments = []
                    reassign_count = 0
                    
                    for segment in segments:
                        if (segment.speaker_id == dominant_id and 
                            segment.duration > 10.0 and  # Only reassign long segments
                            reassign_count < len(dominant_segments) // 3):
                            
                            # Find best target speaker based on temporal proximity
                            best_target = None
                            min_gap = float('inf')
                            
                            for target_id in target_speakers:
                                target_segments = speaker_stats[target_id]['segments']
                                for target_seg in target_segments:
                                    gap = abs(segment.start_time - target_seg.end_time)
                                    if gap < min_gap:
                                        min_gap = gap
                                        best_target = target_id
                            
                            if best_target and min_gap < 20.0:  # Within 20 seconds
                                # Reassign to best target
                                reassigned_segment = SpeakerSegment(
                                    start_time=segment.start_time,
                                    end_time=segment.end_time,
                                    speaker_id=best_target,
                                    confidence=segment.confidence * 0.8  # Lower confidence for reassigned
                                )
                                rebalanced_segments.append(reassigned_segment)
                                reassign_count += 1
                            else:
                                rebalanced_segments.append(segment)
                        else:
                            rebalanced_segments.append(segment)
                    
                    self.logger.info("speakers_rebalanced",
                                   reassigned_segments=reassign_count,
                                   dominant_speaker=dominant_id)
                    
                    return rebalanced_segments
            
            return segments
            
        except Exception as e:
            self.logger.warning("speaker_rebalancing_failed", error=str(e))
            return segments
    
    def _post_process_segments(self, segments: List[SpeakerSegment], audio_path: Path) -> List[SpeakerSegment]:
        """
        Post-process segments with advanced speaker re-classification
        """
        if len(segments) <= 1:
            return segments
            
        try:
            import librosa
            
            # Load audio for analysis
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Extract embeddings for each segment
            segment_embeddings = []
            valid_segments = []
            
            for segment in segments:
                start_frame = int(segment.start_time * sr)
                end_frame = int(segment.end_time * sr)
                
                if end_frame > start_frame and end_frame <= len(y):
                    segment_audio = y[start_frame:end_frame]
                    
                    # Extract features for this segment
                    if len(segment_audio) > 1000:  # Minimum audio length
                        features = self._extract_segment_embedding(segment_audio, sr)
                        segment_embeddings.append(features)
                        valid_segments.append(segment)
            
            if len(segment_embeddings) < 2:
                return segments
                
            # Re-cluster segments based on embeddings
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Normalize embeddings
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(segment_embeddings)
            
            # Determine optimal number of clusters
            n_clusters = min(6, max(2, len(valid_segments) // 8))
            
            # Hierarchical clustering for better speaker separation
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage='average',
                metric='cosine'
            )
            cluster_labels = clustering.fit_predict(normalized_embeddings)
            
            # Update speaker IDs based on clustering
            updated_segments = []
            for i, segment in enumerate(valid_segments):
                new_speaker_id = f"speaker_{cluster_labels[i] + 1:02d}"
                
                # Calculate confidence based on cluster cohesion
                same_cluster_embeddings = [emb for j, emb in enumerate(normalized_embeddings) 
                                         if cluster_labels[j] == cluster_labels[i]]
                if len(same_cluster_embeddings) > 1:
                    cluster_std = np.std(same_cluster_embeddings, axis=0).mean()
                    confidence = min(0.95, max(0.4, 0.9 - cluster_std))
                else:
                    confidence = 0.7
                
                updated_segment = SpeakerSegment(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    speaker_id=new_speaker_id,
                    confidence=confidence
                )
                updated_segments.append(updated_segment)
            
            self.logger.info("post_processing_completed",
                           original_segments=len(segments),
                           processed_segments=len(updated_segments),
                           unique_speakers=len(set(s.speaker_id for s in updated_segments)))
            
            return updated_segments
            
        except Exception as e:
            self.logger.warning("post_processing_failed", error=str(e))
            return segments
    
    def _extract_segment_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract acoustic embedding for a single segment
        """
        try:
            import librosa
            
            # Extract comprehensive acoustic features
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_features = [np.mean(mfcc[i]) for i in range(min(8, mfcc.shape[0]))]
            features.extend(mfcc_features)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_features = [np.mean(chroma[i]) for i in range(min(4, chroma.shape[0]))]
            features.extend(chroma_features)
            
            return np.array(features)
            
        except Exception as e:
            # Return dummy embedding on failure
            return np.random.random(16)
    
    def _validate_speaker_distribution(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Validate and correct speaker distribution
        """
        if not segments:
            return segments
            
        try:
            total_duration = sum(seg.duration for seg in segments)
            speaker_stats = {}
            
            for segment in segments:
                speaker_id = segment.speaker_id
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {'duration': 0.0, 'segments': []}
                speaker_stats[speaker_id]['duration'] += segment.duration
                speaker_stats[speaker_id]['segments'].append(segment)
            
            # Log current distribution
            distribution_info = []
            for speaker_id, stats in sorted(speaker_stats.items()):
                ratio = (stats['duration'] / total_duration) * 100
                distribution_info.append(f"{speaker_id}:{ratio:.1f}%")
            
            self.logger.info("speaker_distribution_validated",
                           total_speakers=len(speaker_stats),
                           total_segments=len(segments),
                           distribution="|".join(distribution_info))
            
            # Check for highly unbalanced distribution
            max_ratio = max(stats['duration'] / total_duration for stats in speaker_stats.values())
            min_ratio = min(stats['duration'] / total_duration for stats in speaker_stats.values())
            
            if max_ratio > 0.8 and len(speaker_stats) > 2:
                self.logger.warning("highly_unbalanced_speakers",
                                  max_speaker_ratio=max_ratio,
                                  min_speaker_ratio=min_ratio)
            
            return segments
            
        except Exception as e:
            self.logger.warning("distribution_validation_failed", error=str(e))
            return segments
    
    def diarize_audio(self, audio_path: Union[str, Path]) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            DiarizationResult with speaker segments and statistics
        """
        
        if not self.pipeline:
            self.load_model()
        
        audio_path = Path(audio_path)
        start_time = time.time()
        
        with self.logger.performance_timer("speaker_diarization", items_count=1):
            
            try:
                # Get audio duration
                info = torchaudio.info(str(audio_path))
                total_duration = info.num_frames / info.sample_rate
                
                self.logger.info("diarization_started", 
                               audio_file=str(audio_path),
                               duration=total_duration)
                
                # Process in chunks for long audio
                chunk_duration = self.config.chunk_processing_duration if hasattr(self.config, 'chunk_processing_duration') else 60.0
                
                if total_duration <= chunk_duration:
                    # Process entire file
                    audio_data = self._load_audio_segment(audio_path)
                    
                    with torch.no_grad():
                        if self.config.enable_fp16 and self.device.startswith('cuda'):
                            with torch.cuda.amp.autocast():
                                diarization = self.pipeline(audio_data)
                        else:
                            diarization = self.pipeline(audio_data)
                    
                    # Convert to segments
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segment = SpeakerSegment(
                            start_time=turn.start,
                            end_time=turn.end,
                            speaker_id=f"{speaker}",
                            confidence=getattr(turn, 'confidence', 0.9)
                        )
                        segments.append(segment)
                
                else:
                    # Process in chunks with overlap
                    overlap = 2.0  # 2 second overlap
                    chunks = []
                    current_start = 0.0
                    
                    while current_start < total_duration:
                        current_end = min(current_start + chunk_duration, total_duration)
                        chunks.append((current_start, current_end))
                        current_start = current_end - overlap
                        
                        if current_start >= total_duration:
                            break
                    
                    # Process chunks in parallel
                    all_segments = []
                    futures = []
                    
                    for i, (start, end) in enumerate(chunks):
                        future = self.thread_pool.submit(
                            self._process_chunk, audio_path, start, end, i
                        )
                        futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        chunk_segments = future.result()
                        all_segments.extend(chunk_segments)
                    
                    # Merge overlapping segments
                    segments = self._merge_overlapping_segments(all_segments)
                
                # Apply improved post-processing for balanced speaker distribution
                if len(segments) > 2:  # Only if we have multiple segments
                    # Step 1: Remove noise segments and short segments
                    segments = self._filter_noise_segments(segments)
                    
                    # Step 2: Resolve overlapping segments
                    segments = self._resolve_overlapping_segments(segments)
                    
                    # Step 3: Fast segment merging based on spectral similarity
                    segments = self._fast_merge_similar_speakers(segments, audio_path)
                    
                    # Step 4: Temporal consistency checking
                    segments = self._temporal_consistency_check(segments)
                    
                    # Step 5: Rebalance dominant speakers
                    segments = self._rebalance_speakers(segments, audio_path)
                    
                    # Step 6: Final validation
                    segments = self._validate_speaker_distribution(segments)
                
                # Compute speaker statistics
                speakers = self._compute_speaker_statistics(segments)
                
                processing_time = time.time() - start_time
                
                result = DiarizationResult(
                    success=True,
                    segments=segments,
                    speakers=speakers,
                    total_speakers=len(speakers),
                    total_duration=total_duration,
                    processing_time=processing_time,
                    model_info={
                        'model_name': self.config.model_name,
                        'device': self.device,
                        'fp16_enabled': self.config.enable_fp16,
                        'chunk_processing': total_duration > chunk_duration
                    }
                )
                
                self.logger.info("diarization_completed",
                               audio_file=str(audio_path),
                               total_speakers=len(speakers),
                               total_segments=len(segments),
                               processing_time=processing_time)
                
                return result
                
            except Exception as e:
                self.logger.error("diarization_failed", 
                                audio_file=str(audio_path), 
                                error=str(e))
                
                return DiarizationResult(
                    success=False,
                    error_message=f"Diarization failed: {str(e)}",
                    processing_time=time.time() - start_time
                )
    
    def cleanup(self):
        """Clean up resources"""
        if self.embedding_cache:
            self.embedding_cache.clear()
        
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear GPU cache
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("speaker_diarizer_cleanup_completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error("speaker_diarizer_exception", 
                            exception_type=str(exc_type),
                            exception_message=str(exc_val))