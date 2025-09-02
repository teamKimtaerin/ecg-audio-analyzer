"""
Unified WhisperX Pipeline - Speech Recognition with Speaker Diarization
"""

import whisperx
import torch
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from ..utils.logger import get_logger
from .model_manager import get_model_manager

@dataclass
class SpeechResult:
    """Speech recognition result"""
    text: str
    confidence: float
    language: Optional[str] = None
    word_segments: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_whisperx_result(cls, result: Dict[str, Any]) -> 'SpeechResult':
        """Create SpeechResult from WhisperX output"""
        # WhisperX returns segments with word-level timestamps
        segments = result.get("segments", [])
        
        # Combine all text from segments
        full_text = " ".join([seg.get("text", "").strip() for seg in segments]).strip()
        
        # Calculate average confidence if available
        confidences = []
        word_segments = []
        
        for seg in segments:
            # Segment-level confidence
            if "confidence" in seg:
                confidences.append(seg["confidence"])
            
            # Word-level information
            words = seg.get("words", [])
            for word in words:
                word_segments.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                    "confidence": word.get("confidence", 0.0)
                })
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        language = result.get("language", None)
        
        return cls(
            text=full_text,
            confidence=float(avg_confidence),
            language=language,
            word_segments=word_segments if word_segments else None
        )


class WhisperXPipeline:
    """Unified WhisperX pipeline with integrated speaker diarization"""
    
    def __init__(self, 
                 model_size: str = "base",
                 device: Optional[str] = None,
                 compute_type: str = "float16",
                 language: Optional[str] = None,
                 hf_auth_token: Optional[str] = None):
        """
        Initialize unified WhisperX pipeline
        
        Args:
            model_size: WhisperX model size (tiny, base, small, medium, large-v2)
            device: Device to use (None for auto-detection)
            compute_type: Compute type (float16, float32, int8)
            language: Language code (None for auto-detection)
            hf_auth_token: Hugging Face token for diarization model access
        """
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self.hf_auth_token = hf_auth_token
        
        # Get model manager
        self.model_manager = get_model_manager(device=device)
        self.device = self.model_manager.get_device()
        
        # Use CPU compute type if on CPU
        if self.device == "cpu":
            self.compute_type = "float32"
        
        self.logger = get_logger().bind_context(
            service="whisperx_pipeline",
            model=f"whisperx-{model_size}",
            device=self.device
        )
        
        # Models will be loaded on first use
        self.whisper_model = None
        self.alignment_model = None
        self.alignment_metadata = None
        self.diarization_pipeline = None
    
    def _load_models(self):
        """Load WhisperX models"""
        if self.whisper_model is None:
            self.logger.info("loading_whisperx_model")
            
            try:
                # Try to load with GPU/MPS first, fallback to CPU if needed
                device_to_use = self.device
                compute_type_to_use = self.compute_type
                
                # WhisperX only supports CUDA and CPU, fallback to CPU if CUDA not available
                if self.device == "cuda" and not torch.cuda.is_available():
                    self.logger.info("cuda_not_available_fallback_to_cpu")
                    device_to_use = "cpu"
                    compute_type_to_use = "float32"
                
                # Load WhisperX model
                self.whisper_model = whisperx.load_model(
                    self.model_size,
                    device=device_to_use,
                    compute_type=compute_type_to_use,
                    language=self.language
                )
                
                # Update device info based on what actually worked
                self.device = device_to_use
                self.compute_type = compute_type_to_use
                
                self.logger.info("whisperx_model_loaded", 
                               model_size=self.model_size,
                               device=self.device,
                               compute_type=self.compute_type)
                
            except Exception as e:
                if "CUDA" in str(e) or "CTranslate2" in str(e):
                    self.logger.warning("gpu_failed_fallback_to_cpu", error=str(e))
                    # Fallback to CPU
                    device_to_use = "cpu"
                    compute_type_to_use = "float32"
                    
                    self.whisper_model = whisperx.load_model(
                        self.model_size,
                        device=device_to_use,
                        compute_type=compute_type_to_use,
                        language=self.language
                    )
                    
                    # Update device info
                    self.device = device_to_use
                    self.compute_type = compute_type_to_use
                    
                    self.logger.info("whisperx_model_loaded_cpu_fallback", 
                                   model_size=self.model_size,
                                   device=self.device,
                                   compute_type=self.compute_type)
                else:
                    self.logger.error("failed_to_load_whisperx_model", error=str(e))
                    raise
    
    def _load_alignment_model(self, language_code: str):
        """Load alignment model for better word-level timestamps"""
        try:
            if self.alignment_model is None or getattr(self, '_last_language', None) != language_code:
                self.logger.info("loading_alignment_model", language=language_code)
                
                self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self._last_language = language_code
                
                self.logger.info("alignment_model_loaded", language=language_code)
                
        except Exception as e:
            self.logger.warning("failed_to_load_alignment_model", 
                              language=language_code,
                              error=str(e))
            self.alignment_model = None
            self.alignment_metadata = None
    
    def _load_diarization_model(self):
        """Load diarization pipeline"""
        if self.diarization_pipeline is None:
            self.logger.info("loading_diarization_model")
            
            try:
                # Use pyannote directly for diarization
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_auth_token
                )
                
                # Move to device if CUDA
                if self.device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                
                self.logger.info("diarization_model_loaded", 
                               model="pyannote/speaker-diarization-3.1",
                               device=self.device)
                
            except Exception as e:
                self.logger.error("failed_to_load_diarization_model", error=str(e))
                raise
    
    def process_audio_with_diarization(self, 
                                     audio_path: Union[str, Path],
                                     min_speakers: int = 2,
                                     max_speakers: int = 4,
                                     sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Complete WhisperX pipeline with integrated speaker diarization
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            sample_rate: Target sample rate
            
        Returns:
            Dictionary with transcription, alignment, and speaker diarization
        """
        self.logger.info("starting_whisperx_pipeline", 
                        file=str(audio_path),
                        min_speakers=min_speakers,
                        max_speakers=max_speakers)
        
        try:
            # Step 1: ASR (transcribe)
            self._load_models()
            
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            
            audio_duration = len(y) / sr
            self.logger.info("audio_loaded_for_processing", 
                           duration=round(audio_duration, 2),
                           sample_rate=sr)
            
            batch_size = 16 if self.device != "cpu" else 8
            
            self.logger.info("transcribing_audio")
            asr_result = self.whisper_model.transcribe(
                y,
                batch_size=batch_size
            )
            
            detected_language = asr_result.get("language", self.language or "en")
            self.logger.info("language_detected", language=detected_language)
            
            # Step 2: Alignment
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)
            
            if self.alignment_model is not None:
                self.logger.info("aligning_transcription")
                try:
                    aligned_result = whisperx.align(
                        asr_result["segments"], 
                        self.alignment_model, 
                        self.alignment_metadata,
                        y, 
                        self.device,
                        return_char_alignments=False
                    )
                    aligned_result["language"] = detected_language
                except Exception as e:
                    self.logger.warning("alignment_failed", error=str(e))
                    aligned_result = asr_result
                    aligned_result["language"] = detected_language
            else:
                aligned_result = asr_result
                aligned_result["language"] = detected_language
            
            # Step 3: Speaker Diarization
            self._load_diarization_model()
            
            self.logger.info("performing_speaker_diarization", 
                           min_speakers=min_speakers, 
                           max_speakers=max_speakers)
            
            # Use pyannote diarization with speaker constraints
            diarization = self.diarization_pipeline(
                str(audio_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Step 4: Assign speakers to words
            self.logger.info("assigning_speakers_to_words")
            try:
                final_result = whisperx.assign_word_speakers(
                    diarization, aligned_result
                )
                self.logger.info("whisperx_speaker_assignment_successful")
            except Exception as e:
                self.logger.warning("speaker_assignment_failed", 
                                  error=str(e),
                                  error_type=type(e).__name__,
                                  diarization_type=type(diarization).__name__,
                                  segments_count=len(aligned_result.get("segments", [])))
                # Fallback: use actual diarization results to assign speakers based on timing overlap
                final_result = aligned_result
                final_result = self._assign_speakers_from_diarization(final_result, diarization)
            
            # Remove duplicate segments (same as reference code)
            final_result["segments"] = self._remove_duplicate_speaker_segments(
                final_result["segments"]
            )
            
            # Validation and detailed logging
            segments_count = len(final_result["segments"])
            unique_speakers = set(
                seg.get("speaker", "UNKNOWN") 
                for seg in final_result["segments"] 
                if seg.get("speaker")
            )
            speakers_count = len(unique_speakers)
            
            # Calculate total duration from segments
            total_segment_duration = sum(
                seg.get("end", 0) - seg.get("start", 0) 
                for seg in final_result["segments"]
                if seg.get("start") is not None and seg.get("end") is not None
            )
            
            self.logger.info("whisperx_pipeline_completed",
                           total_segments=segments_count,
                           unique_speakers=speakers_count,
                           speaker_list=list(unique_speakers),
                           total_segment_duration=round(total_segment_duration, 2),
                           audio_file_duration=round(audio_duration, 2))
            
            return final_result
            
        except Exception as e:
            self.logger.error("whisperx_pipeline_failed", error=str(e))
            raise
    
    def _remove_duplicate_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove duplicate speaker segments (same time range, same text)
        Based on reference implementation
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Filtered list of segments
        """
        seen = set()
        filtered = []
        
        for segment in segments:
            # Create unique key based on timing and text
            start = round(segment.get("start", 0), 2)
            end = round(segment.get("end", 0), 2)
            text = segment.get("text", "").strip()
            
            key = (start, end, text)
            
            if key not in seen:
                filtered.append(segment)
                seen.add(key)
        
        return filtered
    
    def _assign_speakers_from_diarization(self, transcription_result: Dict[str, Any], diarization) -> Dict[str, Any]:
        """
        Improved fallback method to assign speakers using actual diarization timeline
        
        Args:
            transcription_result: WhisperX transcription result with segments
            diarization: PyAnnote diarization result
            
        Returns:
            Updated transcription result with speaker assignments
        """
        try:
            # Convert diarization to list of (start, end, speaker) tuples
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            # Sort by start time for efficient processing
            speaker_timeline.sort(key=lambda x: x['start'])
            
            self.logger.info("diarization_timeline_extracted", 
                           timeline_entries=len(speaker_timeline))
            
            # Assign speakers to segments with improved algorithm
            for segment in transcription_result["segments"]:
                seg_start = segment.get("start", 0.0)
                seg_end = segment.get("end", 0.0)
                seg_duration = seg_end - seg_start
                
                # Find all overlapping speaker turns
                overlapping_turns = []
                for entry in speaker_timeline:
                    # Check for any overlap
                    if entry['end'] > seg_start and entry['start'] < seg_end:
                        # Calculate overlap ratio
                        overlap_start = max(seg_start, entry['start'])
                        overlap_end = min(seg_end, entry['end'])
                        overlap_duration = overlap_end - overlap_start
                        
                        # Calculate overlap percentage relative to segment
                        overlap_ratio = overlap_duration / seg_duration if seg_duration > 0 else 0
                        
                        overlapping_turns.append({
                            'speaker': entry['speaker'],
                            'overlap_duration': overlap_duration,
                            'overlap_ratio': overlap_ratio,
                            'speaker_confidence': min(1.0, entry['duration'] / 2.0)  # Longer turns = higher confidence
                        })
                
                # Assign speaker based on best overlap
                if overlapping_turns:
                    # Sort by overlap ratio first, then by overlap duration
                    overlapping_turns.sort(key=lambda x: (x['overlap_ratio'], x['overlap_duration']), reverse=True)
                    best_turn = overlapping_turns[0]
                    
                    # Only assign if overlap is significant (>10% of segment)
                    if best_turn['overlap_ratio'] > 0.1:
                        segment["speaker"] = best_turn['speaker']
                        segment["speaker_confidence"] = best_turn['speaker_confidence']
                    else:
                        # Find closest speaker by time if no good overlap
                        segment["speaker"] = self._find_closest_speaker(seg_start, seg_end, speaker_timeline)
                        segment["speaker_confidence"] = 0.3  # Low confidence
                else:
                    # No overlapping turns - find closest
                    segment["speaker"] = self._find_closest_speaker(seg_start, seg_end, speaker_timeline)
                    segment["speaker_confidence"] = 0.2  # Very low confidence
            
            # Post-process: merge consecutive segments from same speaker if confidence is low
            self._merge_consecutive_same_speaker_segments(transcription_result["segments"])
            
            unique_speakers = set(
                seg.get("speaker", "UNKNOWN") 
                for seg in transcription_result["segments"]
                if seg.get("speaker")
            )
            
            self.logger.info("speakers_assigned_from_diarization",
                           unique_speakers=len(unique_speakers),
                           speaker_list=list(unique_speakers))
            
            return transcription_result
            
        except Exception as e:
            self.logger.error("fallback_speaker_assignment_failed", error=str(e))
            # Ultimate fallback: assign speakers based on timing patterns
            return self._assign_speakers_by_timing_pattern(transcription_result)
    
    def _find_closest_speaker(self, seg_start: float, seg_end: float, speaker_timeline: List[Dict]) -> str:
        """Find the closest speaker turn in time"""
        seg_mid = (seg_start + seg_end) / 2
        
        min_distance = float('inf')
        closest_speaker = "SPEAKER_00"
        
        for entry in speaker_timeline:
            turn_mid = (entry['start'] + entry['end']) / 2
            distance = abs(seg_mid - turn_mid)
            
            if distance < min_distance:
                min_distance = distance
                closest_speaker = entry['speaker']
        
        return closest_speaker
    
    def _merge_consecutive_same_speaker_segments(self, segments: List[Dict]) -> None:
        """Merge consecutive segments from the same speaker with improved logic"""
        if len(segments) < 2:
            return
        
        i = 0
        while i < len(segments) - 1:
            current = segments[i]
            next_seg = segments[i + 1]
            
            # More aggressive merging conditions for better speaker consistency
            should_merge = False
            
            # Case 1: Same speaker and very close in time (< 0.3 seconds gap)
            time_gap = next_seg.get("start", 0) - current.get("end", 0)
            if (current.get("speaker") == next_seg.get("speaker") and time_gap < 0.3):
                should_merge = True
            
            # Case 2: Same speaker with low confidence and reasonable gap (< 1.0 seconds)
            elif (current.get("speaker") == next_seg.get("speaker") and
                  (current.get("speaker_confidence", 1.0) < 0.6 or 
                   next_seg.get("speaker_confidence", 1.0) < 0.6) and
                  time_gap < 1.0):
                should_merge = True
            
            # Case 3: Very short segments (< 1 second) from same speaker
            elif (current.get("speaker") == next_seg.get("speaker") and
                  (current.get("end", 0) - current.get("start", 0) < 1.0 or
                   next_seg.get("end", 0) - next_seg.get("start", 0) < 1.0) and
                  time_gap < 2.0):
                should_merge = True
            
            if should_merge:
                # Merge segments with improved text handling
                current_text = current.get("text", "").strip()
                next_text = next_seg.get("text", "").strip()
                
                # Add proper spacing
                if current_text and next_text:
                    current["text"] = current_text + " " + next_text
                elif next_text:
                    current["text"] = next_text
                
                current["end"] = next_seg.get("end", current.get("end"))
                
                # Update confidence - take weighted average based on duration
                current_duration = current.get("end", 0) - current.get("start", 0)
                next_duration = next_seg.get("end", 0) - next_seg.get("start", 0)
                total_duration = current_duration + next_duration
                
                if total_duration > 0:
                    current_conf = current.get("speaker_confidence", 0.5)
                    next_conf = next_seg.get("speaker_confidence", 0.5)
                    weighted_conf = (current_conf * current_duration + next_conf * next_duration) / total_duration
                    current["speaker_confidence"] = min(0.9, weighted_conf + 0.1)  # Boost confidence after merge
                
                # Merge word-level information if available
                if "words" in current and "words" in next_seg:
                    current["words"].extend(next_seg["words"])
                elif "words" in next_seg:
                    current["words"] = next_seg["words"]
                
                # Remove next segment
                segments.pop(i + 1)
                continue
            
            i += 1
    
    def _assign_speakers_by_timing_pattern(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ultimate fallback: assign speakers based on timing patterns"""
        segments = transcription_result["segments"]
        
        if not segments:
            return transcription_result
        
        # Simple alternating pattern with gaps detection
        current_speaker = 0
        last_end_time = 0
        
        for segment in segments:
            seg_start = segment.get("start", 0.0)
            
            # If there's a significant gap (>2 seconds), potentially switch speaker
            if seg_start - last_end_time > 2.0:
                current_speaker = (current_speaker + 1) % min(3, len(segments))  # Max 3 speakers
            
            segment["speaker"] = f"SPEAKER_{current_speaker:02d}"
            segment["speaker_confidence"] = 0.1  # Very low confidence
            last_end_time = segment.get("end", seg_start)
        
        return transcription_result
    
    def transcribe_audio_segment(self, 
                               audio_data: np.ndarray, 
                               sample_rate: int = 16000) -> SpeechResult:
        """
        Transcribe audio segment to text (backward compatibility)
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            SpeechResult with transcription
        """
        self._load_models()
        
        try:
            # Ensure audio is float32 and correct sample rate
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # WhisperX expects 16kHz audio
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe with WhisperX
            batch_size = 32 if self.device != "cpu" else 16
            result = self.whisper_model.transcribe(
                audio_data,
                batch_size=batch_size
            )
            
            # Detect language if not specified
            detected_language = result.get("language", "en")
            
            # Load alignment model for better word-level timestamps
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)
            
            # Align whisper output for better word-level timestamps
            if self.alignment_model is not None:
                try:
                    result = whisperx.align(
                        result["segments"], 
                        self.alignment_model, 
                        self.alignment_metadata,
                        audio_data, 
                        self.device,
                        return_char_alignments=False
                    )
                except Exception as e:
                    self.logger.warning("alignment_failed", error=str(e))
                    # Continue with unaligned result
            
            # Convert to SpeechResult
            speech_result = SpeechResult.from_whisperx_result(result)
            speech_result.language = detected_language
            
            return speech_result
            
        except Exception as e:
            self.logger.error("speech_recognition_failed", error=str(e))
            raise
    
    def batch_transcribe_segments(self, 
                                audio_path: Union[str, Path],
                                segments: List[Tuple[float, float]],
                                sample_rate: int = 16000) -> List[SpeechResult]:
        """
        Legacy method for backward compatibility
        Now uses the unified pipeline and extracts segment-based results
        """
        try:
            # Use the unified pipeline
            full_result = self.process_audio_with_diarization(
                audio_path, 
                min_speakers=2, 
                max_speakers=10,  # Allow more flexibility for legacy compatibility
                sample_rate=sample_rate
            )
            
            # Map results to requested segments
            results = []
            for start_time, end_time in segments:
                # Find segments that overlap with requested timeframe
                matching_segments = [
                    seg for seg in full_result["segments"]
                    if (seg.get("start", 0) < end_time and seg.get("end", 0) > start_time)
                ]
                
                # Combine text from matching segments
                combined_text = " ".join([
                    seg.get("text", "").strip() 
                    for seg in matching_segments
                ]).strip()
                
                # Calculate average confidence
                confidences = [
                    seg.get("confidence", 0.0) 
                    for seg in matching_segments 
                    if "confidence" in seg
                ]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                result = SpeechResult(
                    text=combined_text,
                    confidence=float(avg_confidence),
                    language=full_result.get("language", "en")
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error("batch_transcribe_segments_failed", error=str(e))
            raise
    
    def get_pipeline_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics for pipeline results
        
        Args:
            result: WhisperX pipeline result
            
        Returns:
            Dictionary with pipeline statistics
        """
        if not result or "segments" not in result:
            return {}
        
        segments = result["segments"]
        
        # Calculate statistics
        total_segments = len(segments)
        segments_with_text = [s for s in segments if s.get("text", "").strip()]
        segments_with_speakers = [s for s in segments if s.get("speaker")]
        
        unique_speakers = len(set(
            seg.get("speaker", "UNKNOWN") 
            for seg in segments_with_speakers
        ))
        
        total_words = sum(
            len(seg.get("text", "").split()) 
            for seg in segments_with_text
        )
        
        return {
            "total_segments": total_segments,
            "transcribed_segments": len(segments_with_text),
            "segments_with_speakers": len(segments_with_speakers),
            "unique_speakers": unique_speakers,
            "total_words": total_words,
            "language": result.get("language", "unknown"),
            "success_rate": len(segments_with_text) / total_segments if total_segments > 0 else 0.0
        }


# Keep SpeechRecognizer as alias for backward compatibility
SpeechRecognizer = WhisperXPipeline