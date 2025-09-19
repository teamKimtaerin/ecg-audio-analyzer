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
        """Load WhisperX models with language optimization"""
        if self.whisper_model is None:
            # 언어별 최적화된 모델 설정
            optimized_config = self._get_optimized_model_config(self.language)

            # Load WhisperX model (GPU-first approach with optimization)
            self.whisper_model = whisperx.load_model(
                optimized_config["model_size"],
                device=self.device,
                compute_type=optimized_config["compute_type"],
                language=self.language if self.language != "auto" else None
            )

    def _get_optimized_model_config(self, language: Optional[str]) -> Dict[str, Any]:
        """언어별 최적화된 모델 설정 반환"""
        # 언어별 최적화 설정
        language_configs = {
            "ko": {
                "model_size": "large-v2",
                "compute_type": "float16",
                "optimization_type": "Korean-optimized",
            },
            "en": {
                "model_size": "large-v2",
                "compute_type": "float16",
                "optimization_type": "English-optimized",
            },
            "ja": {
                "model_size": "medium",
                "compute_type": "float16",
                "optimization_type": "Japanese-optimized",
            },
            "zh": {
                "model_size": "large-v2",
                "compute_type": "float16",
                "optimization_type": "Chinese-optimized",
            },
        }

        # CPU 환경에서는 compute_type 조정
        base_config = {
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "optimization_type": "default",
        }

        if language and language != "auto" and language in language_configs:
            config = language_configs[language].copy()
            if self.device == "cpu":
                config["compute_type"] = "float32"
            config["optimization_type"] += " (targeted)"
            return config
        else:
            # auto 모드 또는 지원하지 않는 언어의 경우 기본 설정
            base_config["optimization_type"] = "universal (auto-detect)"
            if self.device == "cpu":
                base_config["compute_type"] = "float32"
            return base_config

    def _load_alignment_model(self, language_code: str):
        """Load alignment model for better word-level timestamps"""
        try:
            if (
                self.alignment_model is None
                or getattr(self, "_last_language", None) != language_code
            ):
                print(f"📥 Loading alignment model for language: {language_code}")
                (
                    self.alignment_model,
                    self.alignment_metadata,
                ) = whisperx.load_align_model(
                    language_code=language_code, device=self.device
                )
                self._last_language = language_code
                print(f"✅ Alignment model loaded successfully for {language_code}")

        except Exception as e:
            print(f"⚠️ Alignment model loading failed for {language_code}: {e}")
            print(f"   This is likely the HTTP 301 issue. Continuing without alignment...")
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

            # 언어별 최적화된 transcribe 실행
            if self.language and self.language != "auto":
                # 언어 지정 시 최적화
                asr_result = self.whisper_model.transcribe(
                    y,
                    batch_size=batch_size,
                    language=self.language,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    prompt_reset_on_temperature=0.5
                )
                detected_language = self.language  # 지정된 언어 사용
            else:
                # 자동 감지 모드 - 기존 방식
                asr_result = self.whisper_model.transcribe(
                    y,
                    batch_size=batch_size,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    prompt_reset_on_temperature=0.5
                )
                detected_language = asr_result.get("language", "en")

            # Step 2: Alignment
            if self.alignment_model is None:
                self._load_alignment_model(detected_language)

            if self.alignment_model is not None:
                try:
                    print("🔄 Performing word-level alignment...")
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
                        aligned_result = self._add_fallback_word_timing(
                            aligned_result, y
                        )
                    else:
                        print(
                            f"✅ Word alignment successful for {len(aligned_result.get('segments', []))} segments"
                        )

                except Exception as e:
                    print(f"❌ Alignment failed: {e}, using fallback")
                    aligned_result = self._add_fallback_word_timing(asr_result, y)
                    aligned_result["language"] = detected_language
            else:
                print("⚠️ No alignment model available, using fallback timing")
                aligned_result = self._add_fallback_word_timing(asr_result, y)
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

            # Ensure all segments have word-level timing before gap filling
            for seg in final_result["segments"]:
                words = seg.get("words", [])
                speech_words = [
                    w
                    for w in words
                    if w.get("word", "").strip() and w.get("word") != " "
                ]

                # If segment has no speech words but has text, create fallback word timing
                if not speech_words and seg.get("text", "").strip():
                    print(
                        f"⚠️ Creating fallback words for segment: {seg.get('text', '')[:50]}..."
                    )
                    seg_with_words = self._add_fallback_word_timing(
                        {"segments": [seg]}, y
                    )
                    seg["words"] = seg_with_words["segments"][0].get("words", [])

            # Fill gaps by extending nearest speaker segments with silence words
            final_result["segments"] = self._fill_gaps_into_speaker_segments(
                final_result["segments"], audio_duration, y
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

    def transcribe(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Wrapper method for backward compatibility and simpler interface.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            Dictionary containing transcription results with speaker diarization
        """
        return self.process_audio_with_diarization(audio_path)
    def _add_fallback_word_timing(
        self, result: Dict[str, Any], audio: np.ndarray
    ) -> Dict[str, Any]:
        """Add word-level timing when WhisperX alignment fails"""
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
                    "start_time": round(word_start, 2),
                    "end_time": round(word_end, 2),
                    "duration": round(word_end - word_start, 2),
                    "acoustic_features": {
                        "volume_db": round(volume_db, 1),
                        "pitch_hz": 150.0,
                        "spectral_centroid": 1500.0,
                    },
                }
                word_list.append(word_data)

            seg["words"] = word_list

        return result

    def _fill_gaps_into_speaker_segments(
        self, segments: List[Dict], total_duration: float, audio: np.ndarray
    ) -> List[Dict]:
        """
        Fill gaps by extending nearest speaker segments with silence words instead of creating separate segments

        Args:
            segments: List of existing speech segments
            total_duration: Total audio duration in seconds
            audio: Audio data for acoustic feature extraction

        Returns:
            Updated segments list with gaps integrated as silence words
        """
        if not segments:
            return segments

        sr = 16000
        # Sort segments by start time
        segments.sort(key=lambda x: x.get("start", 0))

        # Identify gaps that need to be filled
        gaps = []

        # Gap at the beginning
        if segments[0].get("start", 0) > 0.5:
            gaps.append(
                {"start": 0.0, "end": segments[0].get("start", 0), "type": "beginning"}
            )

        # Gaps between segments
        for i in range(len(segments) - 1):
            current_end = segments[i].get("end", 0)
            next_start = segments[i + 1].get("start", 0)

            if next_start - current_end > 0.5:
                gaps.append(
                    {
                        "start": current_end,
                        "end": next_start,
                        "type": "between",
                        "prev_segment_idx": i,
                        "next_segment_idx": i + 1,
                    }
                )

        # Gap at the end
        last_end = segments[-1].get("end", 0)
        if total_duration - last_end > 0.5:
            gaps.append({"start": last_end, "end": total_duration, "type": "ending"})

        # Extend segments to include gaps as silence words
        for gap in gaps:
            gap_duration = gap["end"] - gap["start"]
            silence_words = self._create_silence_words(
                gap["start"], gap["end"], gap_duration, audio, sr
            )

            if gap["type"] == "beginning":
                # Extend first segment backwards
                target_segment = segments[0]
                target_segment["start"] = gap["start"]
                target_segment["text"] = "[SILENCE] " + target_segment.get("text", "")
                # Merge silence words with existing words, maintaining chronological order
                existing_words = target_segment.get("words", [])
                all_words = silence_words + existing_words
                target_segment["words"] = self._sort_words_by_time(all_words)

            elif gap["type"] == "ending":
                # Extend last segment forwards
                target_segment = segments[-1]
                target_segment["end"] = gap["end"]
                target_segment["text"] = target_segment.get("text", "") + " [SILENCE]"
                # Merge silence words with existing words, maintaining chronological order
                existing_words = target_segment.get("words", [])
                all_words = existing_words + silence_words
                target_segment["words"] = self._sort_words_by_time(all_words)

            elif gap["type"] == "between":
                # Assign gap to closer segment
                prev_segment = segments[gap["prev_segment_idx"]]
                next_segment = segments[gap["next_segment_idx"]]

                gap_mid = (gap["start"] + gap["end"]) / 2
                prev_end = prev_segment.get("end", 0)
                next_start = next_segment.get("start", 0)

                # Determine which segment is closer to gap midpoint
                dist_to_prev = gap_mid - prev_end
                dist_to_next = next_start - gap_mid

                if dist_to_prev <= dist_to_next:
                    # Extend previous segment forward
                    prev_segment["end"] = gap["end"]
                    prev_segment["text"] = prev_segment.get("text", "") + " [SILENCE]"
                    # Merge silence words with existing words, maintaining chronological order
                    existing_words = prev_segment.get("words", [])
                    all_words = existing_words + silence_words
                    prev_segment["words"] = self._sort_words_by_time(all_words)
                else:
                    # Extend next segment backward
                    next_segment["start"] = gap["start"]
                    next_segment["text"] = "[SILENCE] " + next_segment.get("text", "")
                    # Merge silence words with existing words, maintaining chronological order
                    existing_words = next_segment.get("words", [])
                    all_words = silence_words + existing_words
                    next_segment["words"] = self._sort_words_by_time(all_words)

        return segments

    def _create_silence_words(
        self,
        start_time: float,
        end_time: float,
        duration: float,
        audio: np.ndarray,
        sr: int,
    ) -> List[Dict]:
        """
        Create silence words for gap periods to be included in speaker segments

        Args:
            start_time: Gap start time
            end_time: Gap end time
            duration: Gap duration
            audio: Audio data for feature extraction
            sr: Sample rate

        Returns:
            List of silence word dictionaries
        """
        # Extract audio features for the silence period
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(audio), int(end_time * sr))

        if end_sample > start_sample:
            silence_audio = audio[start_sample:end_sample]

            # Calculate basic acoustic features for silence
            if len(silence_audio) > 0:
                rms_energy = np.sqrt(np.mean(silence_audio**2))
                volume_db = float(20 * np.log10(rms_energy + 1e-8))
            else:
                volume_db = -60.0  # Very quiet
        else:
            volume_db = -60.0

        # Create single silence word for entire gap duration (using consistent field names)
        words = [
            {
                "word": " ",  # Single space character for entire silence period
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(end_time - start_time, 2),
                "acoustic_features": {
                    "volume_db": round(volume_db, 1),
                    "pitch_hz": 0.0,  # No pitch in silence
                    "spectral_centroid": 0.0,
                },
            }
        ]

        return words

    def _sort_words_by_time(self, words: List[Dict]) -> List[Dict]:
        """
        Sort words by their start time and ensure proper field names

        Args:
            words: List of word dictionaries

        Returns:
            Sorted list of words
        """

        def get_start_time(word):
            # Handle both 'start' and 'start_time' field names
            return word.get("start_time", word.get("start", 0))

        return sorted(words, key=get_start_time)

    def _create_silence_segment(
        self,
        start_time: float,
        end_time: float,
        duration: float,
        audio: np.ndarray,
        sr: int,
    ) -> Dict:
        """
        Create a silence segment with word-level timing for gap periods

        Args:
            start_time: Gap start time
            end_time: Gap end time
            duration: Gap duration
            audio: Audio data for feature extraction
            sr: Sample rate

        Returns:
            Silence segment dictionary
        """
        # Extract audio features for the silence period
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(audio), int(end_time * sr))

        if end_sample > start_sample:
            silence_audio = audio[start_sample:end_sample]

            # Calculate basic acoustic features for silence
            if len(silence_audio) > 0:
                rms_energy = np.sqrt(np.mean(silence_audio**2))
                volume_db = float(20 * np.log10(rms_energy + 1e-8))

                # Calculate spectral features (librosa already imported at top)
                spectral_centroid = float(
                    np.mean(librosa.feature.spectral_centroid(y=silence_audio, sr=sr))
                )
                zero_crossing_rate = float(
                    np.mean(librosa.feature.zero_crossing_rate(silence_audio))
                )
            else:
                volume_db = -60.0  # Very quiet
                spectral_centroid = 0.0
                zero_crossing_rate = 0.0
        else:
            volume_db = -60.0
            spectral_centroid = 0.0
            zero_crossing_rate = 0.0

        # Create word-level timing for silence periods
        # Split longer silences into chunks for better granularity
        words = []
        if duration > 2.0:
            # Split into 1-second chunks for longer silences
            num_chunks = max(1, int(duration))
            chunk_duration = duration / num_chunks

            for i in range(num_chunks):
                word_start = start_time + (i * chunk_duration)
                word_end = word_start + chunk_duration

                words.append(
                    {
                        "word": "[SILENCE]",
                        "start": round(word_start, 2),
                        "end": round(word_end, 2),
                        "score": 0.0,  # No confidence for silence
                        "volume_db": round(volume_db, 1),
                    }
                )
        else:
            # Single silence marker for short gaps
            words.append(
                {
                    "word": "[SILENCE]",
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "score": 0.0,
                    "volume_db": round(volume_db, 1),
                }
            )

        return {
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "text": "[SILENCE]",
            "speaker": "SILENCE",
            "words": words,
            "acoustic_features": {
                "volume_db": round(volume_db, 1),
                "pitch_hz": 0.0,  # No pitch in silence
                "spectral_centroid": round(spectral_centroid, 1),
                "zero_crossing_rate": round(zero_crossing_rate, 3),
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "mfcc_mean": [0.0, 0.0, 0.0],  # Minimal MFCC for silence
            },
        }


# Keep SpeechRecognizer as alias for backward compatibility
SpeechRecognizer = WhisperXPipeline
