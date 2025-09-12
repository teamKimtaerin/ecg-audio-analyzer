"""
Unified WhisperX Pipeline - Speech Recognition with Speaker Diarization
"""

import whisperx
import torch
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union
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
    def from_whisperx_result(cls, result: Dict[str, Any]) -> "SpeechResult":
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
                word_segments.append(
                    {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "confidence": word.get("confidence", 0.0),
                    }
                )

        avg_confidence = np.mean(confidences) if confidences else 0.0
        language = result.get("language", None)

        return cls(
            text=full_text,
            confidence=float(avg_confidence),
            language=language,
            word_segments=word_segments if word_segments else None,
        )


class WhisperXPipeline:
    """Unified WhisperX pipeline with integrated speaker diarization"""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        language: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
    ):
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

        # Assume GPU environment (AWS instances)

        self.logger = get_logger().bind_context(
            service="whisperx_pipeline",
            model=f"whisperx-{model_size}",
            device=self.device,
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

            # Load WhisperX model (GPU-first approach)
            self.whisper_model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

            self.logger.info(
                "whisperx_model_loaded",
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def _load_alignment_model(self, language_code: str):
        """Load alignment model for better word-level timestamps"""
        try:
            if (
                self.alignment_model is None
                or getattr(self, "_last_language", None) != language_code
            ):
                self.logger.info("loading_alignment_model", language=language_code)

                self.alignment_model, self.alignment_metadata = (
                    whisperx.load_align_model(
                        language_code=language_code, device=self.device
                    )
                )
                self._last_language = language_code

                self.logger.info("alignment_model_loaded", language=language_code)

        except Exception as e:
            self.logger.warning(
                "failed_to_load_alignment_model", language=language_code, error=str(e)
            )
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
                    use_auth_token=self.hf_auth_token,
                )

                # Move to device if CUDA
                if self.device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(
                        torch.device("cuda")
                    )

                self.logger.info(
                    "diarization_model_loaded",
                    model="pyannote/speaker-diarization-3.1",
                    device=self.device,
                )

            except Exception as e:
                self.logger.error("failed_to_load_diarization_model", error=str(e))
                raise

    def process_audio_with_diarization(
        self,
        audio_path: Union[str, Path],
        min_speakers: int = 2,
        max_speakers: int = 4,
        sample_rate: int = 16000,
        expected_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Complete WhisperX pipeline with integrated speaker diarization

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            sample_rate: Target sample rate
            expected_duration: Expected duration for validation

        Returns:
            Dictionary with transcription, alignment, and speaker diarization
        """
        self.logger.info(
            "starting_whisperx_pipeline",
            file=str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            expected_duration=expected_duration,
        )

        try:

            # Step 1: ASR (transcribe) with enhanced audio loading
            self._load_models()

            # Load and process audio
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            
            audio_duration = len(y) / sr
            self.logger.info("audio_loaded", duration=round(audio_duration, 2))

            batch_size = 16  # GPU optimized batch size

            # Ensure models are loaded
            if self.whisper_model is None:
                self._load_models()
            assert self.whisper_model is not None, "Failed to load WhisperX model"

            self.logger.info("transcribing_audio")
            asr_result = self.whisper_model.transcribe(y, batch_size=batch_size)

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
                        return_char_alignments=False,
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

            self.logger.info(
                "performing_speaker_diarization",
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Ensure diarization pipeline is loaded
            if self.diarization_pipeline is None:
                self._load_diarization_model()
            assert self.diarization_pipeline is not None, "Failed to load diarization pipeline"

            # Use pyannote diarization with speaker constraints
            diarization = self.diarization_pipeline(
                str(audio_path), min_speakers=min_speakers, max_speakers=max_speakers
            )

            # Step 4: Assign speakers to words
            self.logger.info("assigning_speakers_to_words")
            try:
                final_result = whisperx.assign_word_speakers(
                    diarization, aligned_result
                )
                self.logger.info("whisperx_speaker_assignment_successful")
            except Exception as e:
                self.logger.warning(
                    "speaker_assignment_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    diarization_type=type(diarization).__name__,
                    segments_count=len(aligned_result.get("segments", [])),
                )
                # Fallback: use actual diarization results to assign speakers based on timing overlap
                final_result = aligned_result
                final_result = self._assign_speakers_from_diarization(
                    final_result, diarization
                )

            # Remove duplicate segments (same as reference code)
            final_result["segments"] = self._remove_duplicate_speaker_segments(
                final_result["segments"]
            )

            # Basic result validation
            segments_count = len(final_result["segments"])
            unique_speakers = set(
                seg.get("speaker", "UNKNOWN")
                for seg in final_result["segments"]
                if seg.get("speaker")
            )
            speakers_count = len(unique_speakers)

            # Add basic metadata
            final_result["duration_metadata"] = {
                "audio_file_duration": round(audio_duration, 2),
                "total_segments": segments_count,
            }

            self.logger.info(
                "whisperx_pipeline_completed",
                total_segments=segments_count,
                unique_speakers=speakers_count,
                speaker_list=list(unique_speakers),
            )

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

    def _assign_speakers_from_diarization(
        self, transcription_result: Dict[str, Any], diarization
    ) -> Dict[str, Any]:
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
                speaker_timeline.append(
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                        "duration": turn.end - turn.start,
                    }
                )

            # Sort by start time for efficient processing
            speaker_timeline.sort(key=lambda x: x["start"])

            self.logger.info(
                "diarization_timeline_extracted", timeline_entries=len(speaker_timeline)
            )

            # Assign speakers to segments with improved algorithm
            for segment in transcription_result["segments"]:
                seg_start = segment.get("start", 0.0)
                seg_end = segment.get("end", 0.0)
                seg_duration = seg_end - seg_start

                # Find all overlapping speaker turns
                overlapping_turns = []
                for entry in speaker_timeline:
                    # Check for any overlap
                    if entry["end"] > seg_start and entry["start"] < seg_end:
                        # Calculate overlap ratio
                        overlap_start = max(seg_start, entry["start"])
                        overlap_end = min(seg_end, entry["end"])
                        overlap_duration = overlap_end - overlap_start

                        # Calculate overlap percentage relative to segment
                        overlap_ratio = (
                            overlap_duration / seg_duration if seg_duration > 0 else 0
                        )

                        overlapping_turns.append(
                            {
                                "speaker": entry["speaker"],
                                "overlap_duration": overlap_duration,
                                "overlap_ratio": overlap_ratio,
                                "speaker_confidence": min(
                                    1.0, entry["duration"] / 2.0
                                ),  # Longer turns = higher confidence
                            }
                        )

                # Assign speaker based on best overlap
                if overlapping_turns:
                    # Sort by overlap ratio first, then by overlap duration
                    overlapping_turns.sort(
                        key=lambda x: (x["overlap_ratio"], x["overlap_duration"]),
                        reverse=True,
                    )
                    best_turn = overlapping_turns[0]

                    # Only assign if overlap is significant (>10% of segment)
                    if best_turn["overlap_ratio"] > 0.1:
                        segment["speaker"] = best_turn["speaker"]
                        segment["speaker_confidence"] = best_turn["speaker_confidence"]
                    else:
                        # Find closest speaker by time if no good overlap
                        segment["speaker"] = self._find_closest_speaker(
                            seg_start, seg_end, speaker_timeline
                        )
                        segment["speaker_confidence"] = 0.3  # Low confidence
                else:
                    # No overlapping turns - find closest
                    segment["speaker"] = self._find_closest_speaker(
                        seg_start, seg_end, speaker_timeline
                    )
                    segment["speaker_confidence"] = 0.2  # Very low confidence

            # Simple post-processing: basic segment cleanup
            self._basic_segment_cleanup(transcription_result["segments"])

            unique_speakers = set(
                seg.get("speaker", "UNKNOWN")
                for seg in transcription_result["segments"]
                if seg.get("speaker")
            )

            self.logger.info(
                "speakers_assigned_from_diarization",
                unique_speakers=len(unique_speakers),
                speaker_list=list(unique_speakers),
            )

            return transcription_result

        except Exception as e:
            self.logger.error("fallback_speaker_assignment_failed", error=str(e))
            # Simple fallback: assign basic speaker IDs
            return self._simple_speaker_fallback(transcription_result)

    def _find_closest_speaker(
        self, seg_start: float, seg_end: float, speaker_timeline: List[Dict]
    ) -> str:
        """Find the closest speaker turn in time"""
        seg_mid = (seg_start + seg_end) / 2

        min_distance = float("inf")
        closest_speaker = "SPEAKER_00"

        for entry in speaker_timeline:
            turn_mid = (entry["start"] + entry["end"]) / 2
            distance = abs(seg_mid - turn_mid)

            if distance < min_distance:
                min_distance = distance
                closest_speaker = entry["speaker"]

        return closest_speaker

    def _basic_segment_cleanup(self, segments: List[Dict]) -> None:
        """Basic segment cleanup and merging"""
        if len(segments) < 2:
            return

        i = 0
        while i < len(segments) - 1:
            current = segments[i]
            next_seg = segments[i + 1]

            # Simple merging: same speaker and close in time (< 0.5 seconds gap)
            time_gap = next_seg.get("start", 0) - current.get("end", 0)
            if current.get("speaker") == next_seg.get("speaker") and time_gap < 0.5:
                # Merge segments
                current_text = current.get("text", "").strip()
                next_text = next_seg.get("text", "").strip()
                
                if current_text and next_text:
                    current["text"] = current_text + " " + next_text
                elif next_text:
                    current["text"] = next_text
                
                current["end"] = next_seg.get("end", current.get("end"))
                
                # Merge word-level information if available
                if "words" in current and "words" in next_seg:
                    current["words"].extend(next_seg["words"])
                elif "words" in next_seg:
                    current["words"] = next_seg["words"]
                
                segments.pop(i + 1)
                continue
            
            i += 1

    def _simple_speaker_fallback(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback: assign basic speaker IDs"""
        segments = transcription_result["segments"]
        
        for i, segment in enumerate(segments):
            # Simple alternating pattern
            speaker_id = f"SPEAKER_{(i % 2):02d}"
            segment["speaker"] = speaker_id
            segment["speaker_confidence"] = 0.3
        
        return transcription_result



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

        unique_speakers = len(
            set(seg.get("speaker", "UNKNOWN") for seg in segments_with_speakers)
        )

        total_words = sum(
            len(seg.get("text", "").split()) for seg in segments_with_text
        )

        return {
            "total_segments": total_segments,
            "transcribed_segments": len(segments_with_text),
            "segments_with_speakers": len(segments_with_speakers),
            "unique_speakers": unique_speakers,
            "total_words": total_words,
            "language": result.get("language", "unknown"),
            "success_rate": (
                len(segments_with_text) / total_segments if total_segments > 0 else 0.0
            ),
        }


# Keep SpeechRecognizer as alias for backward compatibility
SpeechRecognizer = WhisperXPipeline
