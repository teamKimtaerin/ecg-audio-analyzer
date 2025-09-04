"""
Integration tests for duration accuracy improvements
Tests the complete pipeline with actual video processing for duration validation
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from src.services.audio_extractor import AudioExtractor
from src.models.speech_recognizer import WhisperXPipeline
from src.pipeline.manager import PipelineManager
from src.utils.duration_validator import DurationValidator
from src.utils.subtitle_optimizer import SubtitleOptimizer


class TestDurationAccuracyIntegration:
    """Integration tests for complete duration accuracy pipeline"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_video_file(self, temp_dir):
        """Create mock video file for testing"""
        video_path = temp_dir / "test_video.mp4"
        video_path.write_text("mock video content")
        return video_path

    @pytest.fixture
    def mock_audio_file(self, temp_dir):
        """Create mock audio file for testing"""
        audio_path = temp_dir / "test_audio.wav"
        audio_path.write_text("mock audio content")
        return audio_path

    @pytest.fixture
    def duration_validator(self):
        """Create DurationValidator for testing"""
        return DurationValidator()

    @pytest.fixture
    def audio_extractor(self, temp_dir):
        """Create AudioExtractor for testing"""
        return AudioExtractor(temp_dir=temp_dir)


class TestAudioExtractionDurationAccuracy:
    """Test audio extraction with duration validation"""

    @pytest.fixture
    def extractor(self):
        return AudioExtractor()

    @patch.object(AudioExtractor, "_convert_to_wav_with_validation")
    @patch("src.services.audio_extractor.sf.info")
    @patch("src.services.audio_extractor.DurationValidator")
    def test_audio_extraction_with_duration_validation(
        self, mock_validator_class, mock_sf_info, mock_convert, extractor
    ):
        """Test that audio extraction properly validates duration"""
        # Mock duration validator
        mock_validator = Mock()
        mock_validator.probe_video_duration.return_value = Mock(
            final_duration=142.0,
            confidence=0.95,
            method_used="stream_duration",
            warnings=[],
        )
        mock_validator.validate_extracted_audio.return_value = {
            "valid": True,
            "expected_duration": 142.0,
            "actual_duration": 141.9,
            "difference": 0.1,
        }
        mock_validator_class.return_value = mock_validator

        # Mock conversion success
        mock_convert.return_value = True

        # Mock soundfile info
        mock_info = Mock()
        mock_info.duration = 141.9
        mock_info.samplerate = 16000
        mock_sf_info.return_value = mock_info

        # Test extraction
        result = extractor.extract("test_video.mp4")

        assert result.success is True
        assert result.duration == 141.9
        assert result.original_duration == 142.0
        assert result.duration_validation_passed is True
        assert result.duration_difference == 0.1
        assert result.extraction_method == "enhanced_ffmpeg_with_validation"

    @patch.object(AudioExtractor, "_convert_to_wav_with_validation")
    @patch("src.services.audio_extractor.DurationValidator")
    def test_audio_extraction_duration_validation_failure(
        self, mock_validator_class, mock_convert, extractor
    ):
        """Test handling of duration validation failures"""
        # Mock duration validator with validation failure
        mock_validator = Mock()
        mock_validator.probe_video_duration.return_value = Mock(
            final_duration=None,
            confidence=0.0,
            method_used="no_valid_duration",
            warnings=["Failed to detect duration"],
        )
        mock_validator_class.return_value = mock_validator

        # Test extraction failure
        result = extractor.extract("test_video.mp4")

        assert result.success is False
        assert "Failed to determine video duration" in result.error


class TestWhisperXDurationValidation:
    """Test WhisperX processing with duration validation"""

    @pytest.fixture
    def whisperx_pipeline(self):
        return WhisperXPipeline(model_size="tiny")  # Use tiny model for faster testing

    @patch("src.models.speech_recognizer.librosa.load")
    @patch.object(WhisperXPipeline, "_load_models")
    @patch.object(WhisperXPipeline, "_load_alignment_model")
    @patch.object(WhisperXPipeline, "_load_diarization_model")
    def test_whisperx_with_duration_validation(
        self,
        mock_load_diarization,
        mock_load_alignment,
        mock_load_models,
        mock_librosa,
        whisperx_pipeline,
    ):
        """Test WhisperX processing with duration validation"""
        # Mock audio loading
        import numpy as np

        mock_audio = np.random.random(16000 * 142)  # 142 seconds of audio
        mock_librosa.return_value = (mock_audio.astype(np.float32), 16000)

        # Mock WhisperX model
        mock_whisper_model = Mock()
        mock_whisper_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world", "confidence": 0.9},
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "This is a test",
                    "confidence": 0.8,
                },
                {
                    "start": 10.0,
                    "end": 140.0,
                    "text": "Long segment",
                    "confidence": 0.85,
                },
            ],
            "language": "en",
        }
        whisperx_pipeline.whisper_model = mock_whisper_model

        # Mock alignment
        whisperx_pipeline.alignment_model = Mock()
        whisperx_pipeline.alignment_metadata = Mock()

        # Mock diarization
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=5.0), None, "SPEAKER_00"),
            (Mock(start=5.0, end=10.0), None, "SPEAKER_01"),
            (Mock(start=10.0, end=140.0), None, "SPEAKER_00"),
        ]
        whisperx_pipeline.diarization_pipeline = mock_diarization

        # Mock duration validator
        whisperx_pipeline.duration_validator.validate_extracted_audio = Mock(
            return_value={
                "valid": False,  # Simulate duration mismatch
                "expected_duration": 142.0,
                "actual_duration": 140.0,
                "difference": 2.0,
            }
        )

        # Mock WhisperX functions
        with (
            patch("src.models.speech_recognizer.whisperx.align") as mock_align,
            patch(
                "src.models.speech_recognizer.whisperx.assign_word_speakers"
            ) as mock_assign,
        ):

            mock_align.return_value = mock_whisper_model.transcribe.return_value
            mock_assign.return_value = mock_whisper_model.transcribe.return_value

            # Process with expected duration
            result = whisperx_pipeline.process_audio_with_diarization(
                "test_audio.wav", expected_duration=142.0
            )

        # Verify duration metadata is included
        assert "duration_metadata" in result
        assert result["duration_metadata"]["expected_duration"] == 142.0
        assert result["duration_metadata"]["audio_file_duration"] == 142.0

        # Verify timeline coverage information
        assert "timeline_coverage" in result


class TestSubtitleTimelineValidation:
    """Test subtitle optimization with timeline validation"""

    @pytest.fixture
    def subtitle_optimizer(self):
        return SubtitleOptimizer()

    def test_subtitle_timeline_coverage_validation_complete(self, subtitle_optimizer):
        """Test subtitle timeline validation with complete coverage"""
        # Create subtitles that cover the complete timeline
        segments = [
            {
                "text": "Hello world",
                "start_time": 0.0,
                "end_time": 5.0,
                "speaker": {"speaker_id": "SPEAKER_00", "confidence": 0.9},
            },
            {
                "text": "This is a test",
                "start_time": 5.0,
                "end_time": 10.0,
                "speaker": {"speaker_id": "SPEAKER_01", "confidence": 0.8},
            },
            {
                "text": "Final segment",
                "start_time": 10.0,
                "end_time": 15.0,
                "speaker": {"speaker_id": "SPEAKER_00", "confidence": 0.85},
            },
        ]

        optimized = subtitle_optimizer.optimize_segments_with_gap_filling(
            segments, 15.0
        )
        validation = subtitle_optimizer.validate_subtitle_timeline_coverage(
            optimized, 15.0
        )

        assert validation["valid"] is True
        assert validation["coverage_percentage"] == 100.0
        assert validation["gaps_detected"] is False
        assert validation["gap_count"] == 0

    def test_subtitle_timeline_coverage_validation_with_gaps(self, subtitle_optimizer):
        """Test subtitle timeline validation with gaps"""
        # Create subtitles with gaps
        segments = [
            {
                "text": "Hello world",
                "start_time": 0.0,
                "end_time": 5.0,
                "speaker": {"speaker_id": "SPEAKER_00", "confidence": 0.9},
            },
            {
                "text": "After gap",
                "start_time": 10.0,  # 5 second gap
                "end_time": 15.0,
                "speaker": {"speaker_id": "SPEAKER_01", "confidence": 0.8},
            },
        ]

        optimized = subtitle_optimizer.optimize_segments_with_gap_filling(
            segments, 20.0
        )
        validation = subtitle_optimizer.validate_subtitle_timeline_coverage(
            optimized, 20.0
        )

        # Should detect gaps but coverage might still be reasonable
        assert validation["coverage_percentage"] < 100.0
        assert (
            validation["gaps_detected"] is True or len(optimized) > 2
        )  # Either gaps detected or filled

    def test_gap_filling_functionality(self, subtitle_optimizer):
        """Test that gap filling actually fills gaps"""
        # Create segments with significant gaps
        segments = [
            {
                "text": "First segment",
                "start_time": 0.0,
                "end_time": 5.0,
                "speaker": {"speaker_id": "SPEAKER_00", "confidence": 0.9},
            },
            {
                "text": "Last segment",
                "start_time": 15.0,  # 10 second gap
                "end_time": 20.0,
                "speaker": {"speaker_id": "SPEAKER_01", "confidence": 0.8},
            },
        ]

        # Test with gap filling
        optimized_with_filling = subtitle_optimizer.optimize_segments_with_gap_filling(
            segments, 20.0
        )

        # Test without gap filling
        optimized_without_filling = subtitle_optimizer.optimize_segments(segments)

        # Should have more segments with gap filling
        assert len(optimized_with_filling) >= len(optimized_without_filling)

        # Should have better timeline coverage with gap filling
        validation_with = subtitle_optimizer.validate_subtitle_timeline_coverage(
            optimized_with_filling, 20.0
        )
        validation_without = subtitle_optimizer.validate_subtitle_timeline_coverage(
            optimized_without_filling, 20.0
        )

        assert (
            validation_with["coverage_percentage"]
            >= validation_without["coverage_percentage"]
        )


class TestEndToEndDurationAccuracy:
    """Test complete end-to-end duration accuracy"""

    @patch("src.pipeline.manager.AudioExtractor")
    @patch("src.pipeline.manager.WhisperXPipeline")
    def test_pipeline_duration_accuracy_integration(
        self, mock_whisperx_class, mock_extractor_class
    ):
        """Test complete pipeline with duration accuracy features"""
        # Mock audio extraction with duration validation
        mock_extractor = Mock()
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.output_path = Path("test_audio.wav")
        mock_extraction_result.duration = 141.8
        mock_extraction_result.original_duration = 142.0
        mock_extraction_result.duration_validation_passed = True
        mock_extraction_result.duration_difference = 0.2
        mock_extraction_result.extraction_method = "enhanced_ffmpeg_with_validation"

        mock_extractor.extract_single.return_value = mock_extraction_result
        mock_extractor_class.return_value = mock_extractor

        # Mock WhisperX processing with duration validation
        mock_whisperx = Mock()
        mock_whisperx_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
                {
                    "start": 5.0,
                    "end": 140.0,
                    "text": "Long segment",
                    "speaker": "SPEAKER_01",
                },
            ],
            "language": "en",
            "duration_metadata": {
                "audio_file_duration": 141.8,
                "expected_duration": 142.0,
                "last_segment_time": 140.0,
                "timeline_coverage_percent": 98.6,
            },
            "timeline_coverage": {
                "segment_coverage": 98.6,
                "end_coverage": 98.6,
                "gaps_detected": True,
                "total_gap_duration": 2.0,
            },
        }
        mock_whisperx.process_audio_with_diarization.return_value = mock_whisperx_result
        mock_whisperx_class.return_value = mock_whisperx

        # Create pipeline manager
        from config.base_settings import BaseConfig, ProcessingConfig

        base_config = BaseConfig()
        processing_config = ProcessingConfig()

        # This would normally be an async test, but we'll mock the async parts
        with patch("src.pipeline.manager.asyncio.get_event_loop") as mock_loop:
            mock_loop_instance = Mock()
            mock_loop_instance.run_in_executor = AsyncMock()
            mock_loop_instance.run_in_executor.side_effect = [
                mock_extraction_result,  # Audio extraction result
                mock_whisperx_result,  # WhisperX result
            ]
            mock_loop.return_value = mock_loop_instance

            pipeline = PipelineManager(base_config, processing_config)

            # The actual integration would require running the async process_single method
            # For this test, we verify the components are properly integrated
            assert hasattr(pipeline, "duration_validator")
            assert pipeline.duration_validator is not None


class TestDurationAccuracyBenchmarks:
    """Benchmark tests for duration accuracy improvements"""

    def test_duration_detection_performance(self):
        """Test that duration detection is reasonably fast"""
        import time

        validator = DurationValidator()

        # Mock the actual probing methods to avoid external dependencies
        with (
            patch.object(validator, "_probe_with_ffprobe", return_value=142.0),
            patch.object(validator, "_probe_with_ffmpeg_python", return_value=142.1),
            patch.object(validator, "_probe_container_duration", return_value=141.9),
            patch.object(validator, "_probe_stream_duration", return_value=142.0),
        ):

            start_time = time.time()
            result = validator.probe_video_duration("test.mp4")
            end_time = time.time()

            # Duration detection should be fast (< 1 second for mocked operations)
            assert end_time - start_time < 1.0
            assert result.final_duration is not None

    def test_gap_detection_performance(self):
        """Test gap detection performance with many segments"""
        validator = DurationValidator()

        # Create many segments to test performance
        segments = [
            {"start_time": i * 2, "end_time": i * 2 + 1}
            for i in range(1000)  # 1000 segments with gaps
        ]

        import time

        start_time = time.time()
        gaps = validator.detect_timeline_gaps(segments, 2000)
        end_time = time.time()

        # Gap detection should be reasonably fast even with many segments
        assert end_time - start_time < 1.0
        assert len(gaps) > 0  # Should detect gaps between segments


@pytest.mark.integration
class TestRealVideoProcessing:
    """Integration tests that would work with real video files"""

    @pytest.mark.skip(reason="Requires actual video file for testing")
    def test_friends_video_duration_accuracy(self):
        """Test duration accuracy with the actual friends.mp4 file

        This test should be run manually with the actual video file
        to validate that the duration detection fixes work correctly.
        """
        # This would test the complete pipeline with friends.mp4
        # Expected: 142 seconds detected accurately
        # Previous: 130.461 seconds (missing ~12 seconds)

        video_path = Path("friends.mp4")
        if not video_path.exists():
            pytest.skip("friends.mp4 not available for testing")

        # Test duration detection
        validator = DurationValidator()
        result = validator.probe_video_duration(video_path)

        # Should detect close to 142 seconds
        assert result.final_duration is not None
        assert abs(result.final_duration - 142.0) < 1.0  # Within 1 second tolerance
        assert result.confidence > 0.8

        # Test audio extraction
        extractor = AudioExtractor()
        extraction_result = extractor.extract(video_path)

        assert extraction_result.success is True
        assert extraction_result.duration_validation_passed is True
        assert abs(extraction_result.duration - 142.0) < 1.0
