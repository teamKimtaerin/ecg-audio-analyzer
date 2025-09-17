"""
Unified WhisperX Pipeline - Speech Recognition with Speaker Diarization
"""

import whisperx
import torch
import librosa
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class WhisperXPipeline:
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        language: Optional[str] = "en",
        hf_auth_token: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self.hf_auth_token = hf_auth_token or os.getenv("HF_TOKEN")

        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Adjust compute type for CPU
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "float32"

        # Models will be loaded on first use
        self.whisper_model = None
        self.alignment_model = None
        self.alignment_metadata = None
        self.diarization_pipeline = None

    def _load_models(self):
        """Load WhisperX models"""
        if self.whisper_model is None:
            # Load WhisperX model (GPU-first approach)
            self.whisper_model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

    def _load_alignment_model(self, language_code: str):
        """Load alignment model for better word-level timestamps"""
        try:
            if (
                self.alignment_model is None
                or getattr(self, "_last_language", None) != language_code
            ):
                self.alignment_model, self.alignment_metadata = (
                    whisperx.load_align_model(
                        language_code=language_code, device=self.device
                    )
                )
                self._last_language = language_code

        except Exception:
            self.alignment_model = None
            self.alignment_metadata = None

    def _load_diarization_model(self):
        """Load diarization pipeline"""
        if self.diarization_pipeline is None:
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

            except Exception:
                raise

    def process_audio_with_diarization(
        self,
        audio_path: Union[str, Path],
        min_speakers: int = 2,
        max_speakers: int = 8,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        try:
            # Step 1: ASR (transcribe) with enhanced audio loading
            self._load_models()

            # Load and process audio
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            if y.dtype != np.float32:
                y = y.astype(np.float32)

            audio_duration = len(y) / sr
            batch_size = 16

            assert self.whisper_model is not None, "Failed to load WhisperX model"

            asr_result = self.whisper_model.transcribe(y, batch_size=batch_size)
            detected_language = asr_result.get("language", self.language or "en")

            # Step 2: Alignment
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)

            if self.alignment_model is not None:
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
                    
                    # Check if word alignment was successful
                    words_found = any(
                        "words" in seg and len(seg["words"]) > 0 
                        for seg in aligned_result.get("segments", [])
                    )
                    
                    if not words_found:
                        print("⚠️ Word alignment failed, adding fallback word timing")
                        aligned_result = self._add_fallback_word_timing(aligned_result, y)
                    else:
                        print(f"✅ Word alignment successful for {len(aligned_result.get('segments', []))} segments")
                        
                except Exception as e:
                    print(f"❌ Alignment failed: {e}, using fallback")
                    aligned_result = self._add_fallback_word_timing(asr_result, y)
                    aligned_result["language"] = detected_language
            else:
                aligned_result = asr_result
                aligned_result["language"] = detected_language

            # Step 3: Speaker Diarization
            self._load_diarization_model()

            assert (
                self.diarization_pipeline is not None
            ), "Failed to load diarization pipeline"

            # Use pyannote diarization with speaker constraints
            diarization = self.diarization_pipeline(
                str(audio_path), min_speakers=min_speakers, max_speakers=max_speakers
            )

            # Step 4: Assign speakers to words
            try:
                final_result = whisperx.assign_word_speakers(
                    diarization, aligned_result
                )
            except Exception:
                # Fallback: use actual diarization results to assign speakers based on timing overlap
                final_result = aligned_result
                final_result = self._assign_speakers_from_diarization(
                    final_result, diarization
                )

            # Remove duplicate segments (same as reference code)
            final_result["segments"] = self._remove_duplicate_speaker_segments(
                final_result["segments"]
            )

            final_result["duration_metadata"] = {
                "audio_file_duration": round(audio_duration, 2),
                "total_segments": len(final_result["segments"]),
            }

            return final_result

        except Exception:
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

            # Track unique speakers for debugging
            _ = set(
                seg.get("speaker", "UNKNOWN")
                for seg in transcription_result["segments"]
                if seg.get("speaker")
            )

            return transcription_result

        except Exception:
            # Return transcription result without speaker assignments
            return transcription_result

    def _find_closest_speaker(
        self, seg_start: float, seg_end: float, speaker_timeline: List[Dict]
    ) -> str:
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

    def _add_fallback_word_timing(self, result: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Add word-level timing when WhisperX alignment fails"""
        import librosa
        
        sr = 16000
        segments = result.get("segments", [])
        
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                seg["words"] = []
                continue
                
            words = text.split()
            if not words:
                seg["words"] = []
                continue
                
            start_time = seg.get("start", 0.0)
            end_time = seg.get("end", 0.0)
            duration = end_time - start_time
            
            if duration <= 0:
                seg["words"] = []
                continue
            
            word_duration = duration / len(words)
            word_list = []
            
            for i, word in enumerate(words):
                word_start = start_time + (i * word_duration)
                word_end = word_start + word_duration
                
                # Extract word-level audio for feature analysis
                try:
                    start_sample = max(0, int(word_start * sr))
                    end_sample = min(len(audio), int(word_end * sr))
                    
                    if end_sample > start_sample:
                        word_audio = audio[start_sample:end_sample]
                        
                        # Calculate basic acoustic features
                        if len(word_audio) > 0:
                            rms_energy = np.sqrt(np.mean(word_audio**2))
                            volume_db = float(20 * np.log10(rms_energy + 1e-8))
                        else:
                            volume_db = -30.0
                    else:
                        volume_db = -30.0
                        
                except Exception:
                    volume_db = -30.0
                
                word_data = {
                    "word": word,
                    "start": round(word_start, 2),
                    "end": round(word_end, 2),
                    "score": 0.8,  # Default confidence
                    "volume_db": round(volume_db, 1)
                }
                word_list.append(word_data)
            
            seg["words"] = word_list
        
        return result


# Keep SpeechRecognizer as alias for backward compatibility
SpeechRecognizer = WhisperXPipeline
