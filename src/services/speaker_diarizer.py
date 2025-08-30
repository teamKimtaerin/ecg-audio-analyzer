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
                        'min_duration_on': self.config.min_segment_duration,
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