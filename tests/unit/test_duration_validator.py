"""
Unit tests for duration validator functionality
Tests precise video duration detection and validation
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json

from src.utils.duration_validator import DurationValidator, DurationProbeResult, probe_video_duration


class TestDurationValidator:
    """Test duration validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a DurationValidator instance for testing"""
        return DurationValidator()
    
    def test_duration_validator_initialization(self, validator):
        """Test that DurationValidator initializes correctly"""
        assert validator.tolerance_seconds == 0.1
        assert validator.max_difference_seconds == 2.0
    
    @patch('subprocess.run')
    @patch('json.loads')
    def test_probe_with_ffprobe_success(self, mock_json, mock_subprocess, validator):
        """Test successful FFprobe duration detection"""
        # Mock successful ffprobe response
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = '{"format": {"duration": "142.5"}}'
        mock_json.return_value = {"format": {"duration": "142.5"}}
        
        duration = validator._probe_with_ffprobe(Path("test.mp4"))
        
        assert duration == 142.5
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_probe_with_ffprobe_failure(self, mock_subprocess, validator):
        """Test FFprobe failure handling"""
        # Mock failed ffprobe response
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "File not found"
        
        duration = validator._probe_with_ffprobe(Path("nonexistent.mp4"))
        
        assert duration is None
    
    @patch('ffmpeg.probe')
    def test_probe_with_ffmpeg_python_success(self, mock_probe, validator):
        """Test successful ffmpeg-python duration detection"""
        mock_probe.return_value = {
            "format": {"duration": "142.3"}
        }
        
        duration = validator._probe_with_ffmpeg_python(Path("test.mp4"))
        
        assert duration == 142.3
    
    @patch('ffmpeg.probe')
    def test_probe_with_ffmpeg_python_failure(self, mock_probe, validator):
        """Test ffmpeg-python failure handling"""
        mock_probe.side_effect = Exception("Probe failed")
        
        duration = validator._probe_with_ffmpeg_python(Path("test.mp4"))
        
        assert duration is None
    
    def test_determine_final_duration_single_method(self, validator):
        """Test final duration determination with single valid method"""
        result = DurationProbeResult(file_path="test.mp4")
        result.stream_duration = 142.0
        result.ffprobe_duration = None
        result.ffmpeg_duration = None
        result.container_duration = None
        
        final, confidence, method = validator._determine_final_duration(result)
        
        assert final == 142.0
        assert confidence == 0.6
        assert method == "stream_duration"
    
    def test_determine_final_duration_consensus(self, validator):
        """Test final duration determination with consensus"""
        result = DurationProbeResult(file_path="test.mp4")
        result.stream_duration = 142.0
        result.ffprobe_duration = 142.1  # Within tolerance
        result.ffmpeg_duration = 142.0
        result.container_duration = 141.9  # Within tolerance
        
        final, confidence, method = validator._determine_final_duration(result)
        
        assert abs(final - 142.0) < 0.1
        assert confidence > 0.8
        assert method == "stream_duration"  # Preferred method
    
    def test_determine_final_duration_no_valid(self, validator):
        """Test final duration determination with no valid methods"""
        result = DurationProbeResult(file_path="test.mp4")
        result.stream_duration = None
        result.ffprobe_duration = None
        result.ffmpeg_duration = None
        result.container_duration = None
        
        final, confidence, method = validator._determine_final_duration(result)
        
        assert final is None
        assert confidence == 0.0
        assert method == "no_valid_duration"
    
    @patch('soundfile.info')
    def test_validate_extracted_audio_success(self, mock_sf_info, validator):
        """Test successful audio duration validation"""
        # Mock soundfile info
        mock_info = Mock()
        mock_info.duration = 142.1
        mock_sf_info.return_value = mock_info
        
        result = validator.validate_extracted_audio(Path("test.wav"), 142.0)
        
        assert result["valid"] is True
        assert result["expected_duration"] == 142.0
        assert result["actual_duration"] == 142.1
        assert result["difference"] == 0.1
    
    @patch('soundfile.info')
    def test_validate_extracted_audio_failure(self, mock_sf_info, validator):
        """Test audio duration validation failure"""
        # Mock soundfile info with large difference
        mock_info = Mock()
        mock_info.duration = 130.0  # 12 second difference
        mock_sf_info.return_value = mock_info
        
        result = validator.validate_extracted_audio(Path("test.wav"), 142.0)
        
        assert result["valid"] is False
        assert result["expected_duration"] == 142.0
        assert result["actual_duration"] == 130.0
        assert result["difference"] == 12.0
    
    def test_detect_timeline_gaps_no_segments(self, validator):
        """Test gap detection with no segments"""
        gaps = validator.detect_timeline_gaps([], 142.0)
        
        assert len(gaps) == 1
        assert gaps[0]["start_time"] == 0.0
        assert gaps[0]["end_time"] == 142.0
        assert gaps[0]["gap_duration"] == 142.0
        assert gaps[0]["gap_type"] == "complete_missing"
    
    def test_detect_timeline_gaps_with_gaps(self, validator):
        """Test gap detection with actual gaps"""
        segments = [
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 15.0, "end_time": 20.0},
            {"start_time": 25.0, "end_time": 30.0}
        ]
        
        gaps = validator.detect_timeline_gaps(segments, 35.0)
        
        # Should detect: beginning gap (0-5), middle gap (10-15), middle gap (20-25), end gap (30-35)
        assert len(gaps) == 4
        
        # Check first gap (beginning)
        assert gaps[0]["start_time"] == 0.0
        assert gaps[0]["end_time"] == 5.0
        assert gaps[0]["gap_duration"] == 5.0
        assert gaps[0]["gap_type"] == "timeline_gap"
        
        # Check end gap
        last_gap = [g for g in gaps if g["gap_type"] == "ending_gap"][0]
        assert last_gap["start_time"] == 30.0
        assert last_gap["end_time"] == 35.0
        assert last_gap["gap_duration"] == 5.0
    
    def test_detect_timeline_gaps_no_gaps(self, validator):
        """Test gap detection with complete coverage"""
        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0},
            {"start_time": 20.0, "end_time": 30.0}
        ]
        
        gaps = validator.detect_timeline_gaps(segments, 30.0)
        
        assert len(gaps) == 0
    
    @patch.object(DurationValidator, '_probe_with_ffprobe')
    @patch.object(DurationValidator, '_probe_with_ffmpeg_python')
    @patch.object(DurationValidator, '_probe_container_duration')
    @patch.object(DurationValidator, '_probe_stream_duration')
    def test_probe_video_duration_integration(self, mock_stream, mock_container, 
                                            mock_ffmpeg, mock_ffprobe, validator):
        """Test full video duration probing integration"""
        # Mock all probe methods with consistent results
        mock_ffprobe.return_value = 142.0
        mock_ffmpeg.return_value = 142.1
        mock_container.return_value = 141.9
        mock_stream.return_value = 142.0
        
        result = validator.probe_video_duration(Path("test.mp4"))
        
        assert result.file_path == str(Path("test.mp4"))
        assert result.final_duration is not None
        assert abs(result.final_duration - 142.0) < 0.2
        assert result.confidence > 0.8
        assert result.method_used in ["stream_duration", "consensus"]
    
    def test_validation_consistency_check(self, validator):
        """Test duration consistency validation"""
        result = DurationProbeResult(file_path="test.mp4")
        result.ffprobe_duration = 142.0
        result.ffmpeg_duration = 145.0  # 3 second difference - should trigger warning
        result.container_duration = 142.1
        result.stream_duration = 142.0
        result.final_duration = 142.0
        
        validator._validate_duration_consistency(result)
        
        # Should have warning about large difference
        assert len(result.warnings) > 0
        assert "Large duration difference" in result.warnings[0]


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch.object(DurationValidator, 'probe_video_duration')
    def test_probe_video_duration_convenience(self, mock_probe):
        """Test convenience function for video duration probing"""
        mock_result = DurationProbeResult(
            file_path="test.mp4",
            final_duration=142.0,
            confidence=0.9,
            method_used="stream_duration"
        )
        mock_probe.return_value = mock_result
        
        result = probe_video_duration("test.mp4")
        
        assert result.final_duration == 142.0
        assert result.confidence == 0.9
        assert result.method_used == "stream_duration"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def validator(self):
        return DurationValidator()
    
    def test_zero_duration_handling(self, validator):
        """Test handling of zero duration files"""
        result = DurationProbeResult(file_path="test.mp4")
        result.stream_duration = 0.0
        result.final_duration = 0.0
        
        gaps = validator.detect_timeline_gaps([], 0.0)
        assert len(gaps) == 0  # No gaps expected for zero duration
    
    def test_negative_duration_handling(self, validator):
        """Test handling of invalid negative durations"""
        segments = [{"start_time": -1.0, "end_time": 5.0}]
        gaps = validator.detect_timeline_gaps(segments, 10.0)
        
        # Should still detect gaps correctly
        assert len(gaps) > 0
    
    @patch('soundfile.info')
    def test_audio_validation_error_handling(self, mock_sf_info, validator):
        """Test error handling in audio validation"""
        mock_sf_info.side_effect = Exception("File read error")
        
        result = validator.validate_extracted_audio(Path("test.wav"), 142.0)
        
        assert result["valid"] is False
        assert "error" in result
        assert result["expected_duration"] == 142.0
        assert result["actual_duration"] is None