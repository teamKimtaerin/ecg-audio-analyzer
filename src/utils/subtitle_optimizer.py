"""
Subtitle Optimizer - Smart text segmentation and timing for subtitle generation
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..utils.logger import get_logger
from .duration_validator import DurationValidator


@dataclass
class OptimizedSubtitle:
    """Optimized subtitle segment for display"""

    start_time: float
    end_time: float
    text: str
    speaker_id: str
    confidence: float
    word_count: int
    characters_per_second: float
    subtitle_ready: bool = True


class SubtitleOptimizer:
    """
    Optimizes audio segments for subtitle display by:
    - Splitting long segments into readable chunks
    - Ensuring appropriate timing for reading speed
    - Respecting sentence boundaries and natural pauses
    """

    def __init__(
        self,
        max_duration: float = 4.0,  # Maximum subtitle duration
        min_duration: float = 1.2,  # Minimum subtitle duration
        max_cps: float = 200.0,  # Maximum characters per second
        target_cps: float = 150.0,  # Target characters per second
        max_words_per_line: int = 7,
    ):  # Maximum words per subtitle line
        """
        Initialize subtitle optimizer

        Args:
            max_duration: Maximum duration for a subtitle (seconds)
            min_duration: Minimum duration for a subtitle (seconds)
            max_cps: Maximum characters per second reading speed
            target_cps: Target characters per second for optimal reading
            max_words_per_line: Maximum words per subtitle line
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_cps = max_cps
        self.target_cps = target_cps
        self.max_words_per_line = max_words_per_line

        self.logger = get_logger().bind_context(service="subtitle_optimizer")

        # Initialize duration validator for gap detection
        self.duration_validator = DurationValidator()

        # Sentence boundary markers
        self.sentence_endings = r"[.!?]+\s*"
        self.clause_breaks = r"[,;:]\s*"

        # Common break points for natural pausing
        self.break_words = {
            "and",
            "but",
            "or",
            "so",
            "then",
            "now",
            "well",
            "oh",
            "um",
            "uh",
            "because",
            "since",
            "while",
            "when",
            "where",
            "how",
            "what",
            "that",
        }

    def optimize_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[OptimizedSubtitle]:
        """
        Optimize audio segments for subtitle display

        Args:
            segments: List of audio segments with text, timing, and speaker info

        Returns:
            List of OptimizedSubtitle objects ready for display
        """
        self.logger.info(
            "optimizing_segments_for_subtitles", total_segments=len(segments)
        )

        optimized_segments = []

        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            start_time = segment.get("start_time", 0.0)
            end_time = segment.get("end_time", 0.0)
            speaker_id = segment.get("speaker", {}).get("speaker_id", "unknown")
            confidence = segment.get("speaker", {}).get("confidence", 0.0)

            # Check if segment needs optimization
            duration = end_time - start_time
            cps = len(text) / duration if duration > 0 else 0

            if self._needs_splitting(text, duration, cps):
                # Split into multiple subtitles
                split_subtitles = self._split_segment(
                    text, start_time, end_time, speaker_id, confidence
                )
                optimized_segments.extend(split_subtitles)
            else:
                # Use as-is but adjust timing if needed
                optimized_subtitle = self._adjust_timing(
                    text, start_time, end_time, speaker_id, confidence
                )
                optimized_segments.append(optimized_subtitle)

        # Post-process: merge very short adjacent segments from same speaker
        optimized_segments = self._merge_short_segments(optimized_segments)

        self.logger.info(
            "subtitle_optimization_completed",
            input_segments=len(segments),
            output_subtitles=len(optimized_segments),
        )

        return optimized_segments

    def optimize_segments_with_gap_filling(
        self, segments: List[Dict[str, Any]], total_duration: Optional[float] = None
    ) -> List[OptimizedSubtitle]:
        """
        Optimize audio segments for subtitle display with gap detection and filling

        Args:
            segments: List of audio segments with text, timing, and speaker info
            total_duration: Total expected duration for gap detection

        Returns:
            List of OptimizedSubtitle objects with gaps filled
        """
        self.logger.info(
            "optimizing_segments_with_gap_filling",
            total_segments=len(segments),
            total_duration=total_duration,
        )

        # First, do standard optimization
        optimized_segments = self.optimize_segments(segments)

        # Then detect and fill gaps if total duration is provided
        if total_duration and optimized_segments:
            optimized_segments = self._detect_and_fill_gaps(
                optimized_segments, total_duration
            )

        return optimized_segments

    def _detect_and_fill_gaps(
        self, optimized_segments: List[OptimizedSubtitle], total_duration: float
    ) -> List[OptimizedSubtitle]:
        """
        Detect gaps in subtitle timeline and fill them with placeholder subtitles

        Args:
            optimized_segments: List of optimized subtitle segments
            total_duration: Total expected duration

        Returns:
            List with gaps filled
        """
        if not optimized_segments:
            return optimized_segments

        # Sort segments by start time
        sorted_segments = sorted(optimized_segments, key=lambda x: x.start_time)

        # Detect gaps using duration validator
        segment_data = [
            {"start_time": seg.start_time, "end_time": seg.end_time}
            for seg in sorted_segments
        ]

        gaps = self.duration_validator.detect_timeline_gaps(
            segment_data, total_duration
        )

        if not gaps:
            self.logger.info("no_timeline_gaps_detected")
            return sorted_segments

        self.logger.info(
            "filling_timeline_gaps",
            gap_count=len(gaps),
            total_gap_duration=sum(g["gap_duration"] for g in gaps),
        )

        # Fill gaps with placeholder subtitles
        filled_segments = []
        current_segments = sorted_segments.copy()

        for gap in gaps:
            gap_start = gap["start_time"]
            gap_end = gap["end_time"]
            gap_duration = gap["gap_duration"]
            gap_type = gap["gap_type"]

            # Add original segments that come before this gap
            while current_segments and current_segments[0].end_time <= gap_start:
                filled_segments.append(current_segments.pop(0))

            # Create gap-filling subtitle
            if gap_duration >= self.min_duration:  # Only fill significant gaps
                gap_filler = self._create_gap_filler_subtitle(
                    gap_start, gap_end, gap_duration, gap_type
                )
                filled_segments.append(gap_filler)

                self.logger.debug(
                    "gap_filled",
                    gap_start=gap_start,
                    gap_end=gap_end,
                    gap_duration=gap_duration,
                    gap_type=gap_type,
                )

        # Add remaining segments
        filled_segments.extend(current_segments)

        # Final sort by start time
        filled_segments.sort(key=lambda x: x.start_time)

        self.logger.info(
            "gap_filling_completed",
            original_segments=len(sorted_segments),
            final_segments=len(filled_segments),
            gaps_filled=len(filled_segments) - len(sorted_segments),
        )

        return filled_segments

    def _create_gap_filler_subtitle(
        self, start_time: float, end_time: float, duration: float, gap_type: str
    ) -> OptimizedSubtitle:
        """
        Create a placeholder subtitle for timeline gaps

        Args:
            start_time: Gap start time
            end_time: Gap end time
            duration: Gap duration
            gap_type: Type of gap (timeline_gap, ending_gap, etc.)

        Returns:
            OptimizedSubtitle placeholder
        """
        # Create contextual placeholder text based on gap type and duration
        if gap_type == "ending_gap":
            text = "[No dialogue]"  # For gaps at the end
        elif duration > 5.0:
            text = "[Silence]"  # For long gaps
        elif duration > 2.0:
            text = "[Pause]"  # For medium gaps
        else:
            text = "[...]"  # For short gaps

        # Determine likely speaker (use previous segment's speaker if possible)
        speaker_id = "unknown"

        return OptimizedSubtitle(
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker_id=speaker_id,
            confidence=0.1,  # Low confidence for gap fillers
            word_count=len(text.split()),
            characters_per_second=len(text) / duration if duration > 0 else 0,
            subtitle_ready=True,
        )

    def validate_subtitle_timeline_coverage(
        self, subtitles: List[OptimizedSubtitle], expected_duration: float
    ) -> Dict[str, Any]:
        """
        Validate that subtitles cover the expected timeline duration

        Args:
            subtitles: List of optimized subtitles
            expected_duration: Expected total duration

        Returns:
            Dictionary with coverage validation results
        """
        if not subtitles:
            return {
                "valid": False,
                "coverage_percentage": 0.0,
                "last_subtitle_time": 0.0,
                "gaps_detected": True,
                "total_gap_duration": expected_duration,
            }

        # Sort by start time
        sorted_subtitles = sorted(subtitles, key=lambda x: x.start_time)

        # Find last subtitle end time
        last_subtitle_time = max(sub.end_time for sub in sorted_subtitles)

        # Calculate total subtitle duration
        total_subtitle_duration = sum(
            sub.end_time - sub.start_time for sub in sorted_subtitles
        )

        # Detect gaps
        segment_data = [
            {"start_time": sub.start_time, "end_time": sub.end_time}
            for sub in sorted_subtitles
        ]
        gaps = self.duration_validator.detect_timeline_gaps(
            segment_data, expected_duration
        )

        # Calculate coverage metrics
        coverage_percentage = (
            (last_subtitle_time / expected_duration) * 100
            if expected_duration > 0
            else 0
        )
        time_coverage_percentage = (
            (total_subtitle_duration / expected_duration) * 100
            if expected_duration > 0
            else 0
        )

        total_gap_duration = sum(gap["gap_duration"] for gap in gaps) if gaps else 0

        is_valid = (
            coverage_percentage >= 95.0  # Last subtitle covers at least 95% of duration
            and len(gaps) == 0  # No significant gaps
        )

        result = {
            "valid": is_valid,
            "coverage_percentage": round(coverage_percentage, 2),
            "time_coverage_percentage": round(time_coverage_percentage, 2),
            "last_subtitle_time": round(last_subtitle_time, 2),
            "expected_duration": expected_duration,
            "gaps_detected": len(gaps) > 0,
            "gap_count": len(gaps),
            "total_gap_duration": round(total_gap_duration, 2),
            "subtitle_count": len(subtitles),
            "validation_summary": f"Coverage: {coverage_percentage:.1f}%, Gaps: {len(gaps)}, Duration: {last_subtitle_time:.1f}s/{expected_duration:.1f}s",
        }

        log_level = "info" if is_valid else "warning"
        getattr(self.logger, log_level)(
            "subtitle_timeline_validation",
            **{k: v for k, v in result.items() if k not in ["validation_summary"]},
        )

        return result

    def _needs_splitting(self, text: str, duration: float, cps: float) -> bool:
        """Check if segment needs to be split for better subtitle display"""
        word_count = len(text.split())

        return (
            duration > self.max_duration  # Too long duration
            or cps > self.max_cps  # Reading speed too fast
            or word_count > self.max_words_per_line * 2  # Too many words
        )

    def _split_segment(
        self,
        text: str,
        start_time: float,
        end_time: float,
        speaker_id: str,
        confidence: float,
    ) -> List[OptimizedSubtitle]:
        """Split a long segment into multiple subtitle-friendly parts"""

        words = text.split()
        if len(words) <= 1:
            return [
                OptimizedSubtitle(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    speaker_id=speaker_id,
                    confidence=confidence,
                    word_count=len(words),
                    characters_per_second=(
                        len(text) / (end_time - start_time)
                        if end_time > start_time
                        else 0
                    ),
                )
            ]

        # Find natural break points
        break_points = self._find_break_points(text)
        if not break_points:
            # Fallback: split by word count
            break_points = self._split_by_word_count(words)

        # Create subtitle segments
        subtitles = []
        duration = end_time - start_time

        for i, (chunk_start, chunk_end) in enumerate(break_points):
            chunk_text = " ".join(words[chunk_start:chunk_end])

            # Calculate timing proportionally
            chunk_start_time = start_time + (chunk_start / len(words)) * duration
            chunk_end_time = start_time + (chunk_end / len(words)) * duration

            # Ensure minimum duration
            chunk_duration = chunk_end_time - chunk_start_time
            if chunk_duration < self.min_duration:
                chunk_end_time = chunk_start_time + self.min_duration

            # Ensure maximum duration
            if chunk_duration > self.max_duration:
                chunk_end_time = chunk_start_time + self.max_duration

            word_count = len(chunk_text.split())
            cps = len(chunk_text) / chunk_duration if chunk_duration > 0 else 0

            subtitles.append(
                OptimizedSubtitle(
                    start_time=chunk_start_time,
                    end_time=chunk_end_time,
                    text=chunk_text,
                    speaker_id=speaker_id,
                    confidence=confidence,
                    word_count=word_count,
                    characters_per_second=cps,
                )
            )

        return subtitles

    def _find_break_points(self, text: str) -> List[Tuple[int, int]]:
        """Find natural break points in text for subtitle splitting"""
        words = text.split()
        break_points = []
        current_start = 0

        # Look for sentence boundaries first
        sentences = re.split(self.sentence_endings, text)
        if len(sentences) > 1:
            word_pos = 0
            for sentence in sentences[:-1]:  # Exclude last empty element
                sentence_words = len(sentence.split())
                if sentence_words > 0:
                    break_points.append((word_pos, word_pos + sentence_words))
                    word_pos += sentence_words

            # Add remaining words if any
            if word_pos < len(words):
                break_points.append((word_pos, len(words)))

            return break_points

        # Fallback: look for clause breaks
        current_pos = 0
        current_chunk = []

        for i, word in enumerate(words):
            current_chunk.append(word)

            # Check for natural break after this word
            if (
                word.lower() in self.break_words
                or word.endswith(",")
                or len(current_chunk) >= self.max_words_per_line
            ):

                if len(current_chunk) > 1:  # Don't create single-word chunks
                    break_points.append((current_pos, current_pos + len(current_chunk)))
                    current_pos += len(current_chunk)
                    current_chunk = []

        # Add remaining words
        if current_chunk:
            break_points.append((current_pos, len(words)))

        return break_points if break_points else [(0, len(words))]

    def _split_by_word_count(self, words: List[str]) -> List[Tuple[int, int]]:
        """Fallback: split by maximum word count per subtitle"""
        break_points = []

        for i in range(0, len(words), self.max_words_per_line):
            end_idx = min(i + self.max_words_per_line, len(words))
            break_points.append((i, end_idx))

        return break_points

    def _adjust_timing(
        self,
        text: str,
        start_time: float,
        end_time: float,
        speaker_id: str,
        confidence: float,
    ) -> OptimizedSubtitle:
        """Adjust timing for optimal reading speed"""

        # Calculate optimal duration based on text length
        char_count = len(text)
        optimal_duration = char_count / self.target_cps
        current_duration = end_time - start_time

        # Adjust if current duration is too short for comfortable reading
        if current_duration < optimal_duration:
            if current_duration < self.min_duration:
                end_time = start_time + self.min_duration

        word_count = len(text.split())
        final_duration = end_time - start_time
        cps = char_count / final_duration if final_duration > 0 else 0

        return OptimizedSubtitle(
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker_id=speaker_id,
            confidence=confidence,
            word_count=word_count,
            characters_per_second=cps,
        )

    def _merge_short_segments(
        self, segments: List[OptimizedSubtitle]
    ) -> List[OptimizedSubtitle]:
        """Merge very short adjacent segments from the same speaker"""
        if not segments:
            return segments

        merged = []
        current = segments[0]

        for next_segment in segments[1:]:
            # Check if we should merge with current
            current_duration = current.end_time - current.start_time
            next_duration = next_segment.end_time - next_segment.start_time

            should_merge = (
                current.speaker_id == next_segment.speaker_id  # Same speaker
                and current_duration < self.min_duration  # Current is too short
                and next_duration < self.min_duration  # Next is too short
                and (next_segment.start_time - current.end_time) < 0.5  # Close timing
            )

            if should_merge:
                # Merge segments
                merged_text = f"{current.text} {next_segment.text}"
                merged_duration = next_segment.end_time - current.start_time
                merged_word_count = current.word_count + next_segment.word_count
                merged_cps = (
                    len(merged_text) / merged_duration if merged_duration > 0 else 0
                )

                current = OptimizedSubtitle(
                    start_time=current.start_time,
                    end_time=next_segment.end_time,
                    text=merged_text,
                    speaker_id=current.speaker_id,
                    confidence=(current.confidence + next_segment.confidence) / 2,
                    word_count=merged_word_count,
                    characters_per_second=merged_cps,
                )
            else:
                # Add current to results and move to next
                merged.append(current)
                current = next_segment

        # Add the last segment
        merged.append(current)

        return merged

    def generate_subtitle_json(
        self, optimized_segments: List[OptimizedSubtitle]
    ) -> Dict[str, Any]:
        """Generate JSON output with subtitle-optimized segments"""

        segments_data = []
        for segment in optimized_segments:
            segments_data.append(
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.end_time - segment.start_time,
                    "text": segment.text,
                    "speaker_id": segment.speaker_id,
                    "confidence": segment.confidence,
                    "word_count": segment.word_count,
                    "characters_per_second": round(segment.characters_per_second, 2),
                    "subtitle_ready": segment.subtitle_ready,
                    "reading_time_optimal": segment.characters_per_second
                    <= self.max_cps,
                }
            )

        return {
            "subtitle_optimized_segments": segments_data,
            "optimization_stats": {
                "total_subtitles": len(optimized_segments),
                "avg_duration": (
                    sum(s.end_time - s.start_time for s in optimized_segments)
                    / len(optimized_segments)
                    if optimized_segments
                    else 0
                ),
                "avg_words_per_subtitle": (
                    sum(s.word_count for s in optimized_segments)
                    / len(optimized_segments)
                    if optimized_segments
                    else 0
                ),
                "avg_characters_per_second": (
                    sum(s.characters_per_second for s in optimized_segments)
                    / len(optimized_segments)
                    if optimized_segments
                    else 0
                ),
                "optimization_settings": {
                    "max_duration": self.max_duration,
                    "min_duration": self.min_duration,
                    "max_cps": self.max_cps,
                    "target_cps": self.target_cps,
                    "max_words_per_line": self.max_words_per_line,
                },
            },
        }
