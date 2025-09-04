"""
Unit tests for SpeakerDiarizer service
"""

import torch
from pathlib import Path
from unittest.mock import patch

from services.speaker_diarizer import (
    SpeakerDiarizer,
    SpeakerSegment,
    SpeakerInfo,
    DiarizationResult,
    SpeakerEmbeddingCache,
)


class TestSpeakerSegment:
    """Test cases for SpeakerSegment data class"""

    def test_speaker_segment_creation(self):
        """Test SpeakerSegment creation and duration calculation"""
        segment = SpeakerSegment(
            start_time=1.5, end_time=4.2, speaker_id="speaker_0", confidence=0.85
        )

        assert segment.start_time == 1.5
        assert segment.end_time == 4.2
        assert segment.duration == 2.7
        assert segment.speaker_id == "speaker_0"
        assert segment.confidence == 0.85

    def test_speaker_segment_to_dict(self):
        """Test SpeakerSegment dictionary conversion"""
        segment = SpeakerSegment(
            start_time=0.0, end_time=2.0, speaker_id="speaker_1", confidence=0.9
        )

        result_dict = segment.to_dict()

        assert result_dict["start_time"] == 0.0
        assert result_dict["end_time"] == 2.0
        assert result_dict["duration"] == 2.0
        assert result_dict["speaker_id"] == "speaker_1"
        assert result_dict["confidence"] == 0.9


class TestSpeakerInfo:
    """Test cases for SpeakerInfo data class"""

    def test_speaker_info_creation(self):
        """Test SpeakerInfo creation"""
        speaker_info = SpeakerInfo(
            speaker_id="speaker_0",
            total_duration=45.3,
            segment_count=12,
            avg_confidence=0.87,
            min_confidence=0.65,
            max_confidence=0.95,
        )

        assert speaker_info.speaker_id == "speaker_0"
        assert speaker_info.total_duration == 45.3
        assert speaker_info.segment_count == 12
        assert speaker_info.avg_confidence == 0.87

    def test_speaker_info_to_dict(self):
        """Test SpeakerInfo dictionary conversion"""
        speaker_info = SpeakerInfo(
            speaker_id="speaker_1",
            total_duration=30.0,
            segment_count=8,
            avg_confidence=0.8,
            min_confidence=0.7,
            max_confidence=0.9,
        )

        result_dict = speaker_info.to_dict()

        assert result_dict["speaker_id"] == "speaker_1"
        assert result_dict["total_duration"] == 30.0
        assert result_dict["segment_count"] == 8


class TestSpeakerEmbeddingCache:
    """Test cases for SpeakerEmbeddingCache"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = SpeakerEmbeddingCache(max_cache_size=100)
        assert len(cache.cache) == 0
        assert cache.max_cache_size == 100

    def test_store_and_retrieve_embedding(self):
        """Test storing and retrieving embeddings"""
        cache = SpeakerEmbeddingCache(max_cache_size=10)

        # Create mock embedding
        embedding = torch.randn(128)
        audio_hash = "test_hash_123"

        # Store embedding
        cache.store_embedding(audio_hash, embedding)

        # Retrieve embedding
        retrieved = cache.get_embedding(audio_hash)

        assert retrieved is not None
        assert torch.equal(retrieved, embedding)

    def test_cache_eviction(self):
        """Test cache eviction when max size exceeded"""
        cache = SpeakerEmbeddingCache(max_cache_size=2)

        # Store 3 embeddings (should trigger eviction)
        for i in range(3):
            embedding = torch.randn(64)
            cache.store_embedding(f"hash_{i}", embedding)

        # Cache should only have 2 items
        assert len(cache.cache) == 2

        # First item should be evicted
        assert cache.get_embedding("hash_0") is None
        assert cache.get_embedding("hash_1") is not None
        assert cache.get_embedding("hash_2") is not None

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = SpeakerEmbeddingCache()

        # Store some embeddings
        cache.store_embedding("hash_1", torch.randn(64))
        cache.store_embedding("hash_2", torch.randn(64))

        assert len(cache.cache) == 2

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0


class TestSpeakerDiarizer:
    """Test cases for SpeakerDiarizer service"""

    def test_initialization_cpu(self, speaker_config):
        """Test SpeakerDiarizer initialization with CPU"""
        speaker_config.device = "cpu"

        with patch("torch.cuda.is_available", return_value=False):
            diarizer = SpeakerDiarizer(
                config=speaker_config, device="cpu", enable_caching=True
            )

            assert diarizer.config == speaker_config
            assert diarizer.device == "cpu"
            assert diarizer.enable_caching is True
            assert diarizer.pipeline is None  # Not loaded yet
            assert diarizer.embedding_cache is not None

    def test_initialization_gpu_fallback(self, speaker_config):
        """Test GPU fallback to CPU when CUDA unavailable"""
        speaker_config.device = "cuda:0"

        with patch("torch.cuda.is_available", return_value=False):
            diarizer = SpeakerDiarizer(config=speaker_config, device="cuda:0")

            assert diarizer.device == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_initialization_gpu_available(self, mock_cuda, speaker_config):
        """Test initialization when GPU is available"""
        speaker_config.device = "cuda:0"

        diarizer = SpeakerDiarizer(config=speaker_config, device="cuda:0")

        assert diarizer.device == "cuda:0"

    def test_load_model_mock(self, speaker_config, mock_pyannote):
        """Test model loading with mocked pyannote"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Load model (mocked)
        diarizer.load_model()

        assert diarizer.pipeline is not None

    def test_audio_segment_loading(
        self, speaker_config, sample_audio_file, mock_pyannote
    ):
        """Test loading audio segment"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Test loading audio segment
        audio_data = diarizer._load_audio_segment(sample_audio_file, 1.0, 3.0)

        assert "waveform" in audio_data
        assert "sample_rate" in audio_data
        assert audio_data["sample_rate"] == 16000

        # Check waveform shape (should be mono)
        waveform = audio_data["waveform"]
        assert waveform.shape[0] == 1  # Mono
        # Approximate duration check (2 seconds ± tolerance)
        expected_samples = 2.0 * 16000
        assert abs(waveform.shape[1] - expected_samples) < 1000

    def test_diarize_audio_success(
        self, speaker_config, sample_audio_file, mock_pyannote
    ):
        """Test successful audio diarization"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        result = diarizer.diarize_audio(sample_audio_file)

        assert isinstance(result, DiarizationResult)
        assert result.success is True
        assert result.total_speakers >= 1
        assert len(result.segments) >= 1
        assert len(result.speakers) >= 1
        assert result.processing_time > 0

        # Check segment properties
        for segment in result.segments:
            assert isinstance(segment, SpeakerSegment)
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert 0 <= segment.confidence <= 1

    def test_diarize_audio_file_not_found(self, speaker_config, mock_pyannote):
        """Test diarization with non-existent file"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        result = diarizer.diarize_audio(Path("non_existent_file.wav"))

        assert isinstance(result, DiarizationResult)
        assert result.success is False
        assert result.error_message is not None

    def test_merge_overlapping_segments(self, speaker_config, mock_pyannote):
        """Test merging overlapping segments from same speaker"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Create overlapping segments from same speaker
        segments = [
            SpeakerSegment(0.0, 2.0, "speaker_0", 0.9),
            SpeakerSegment(1.8, 4.0, "speaker_0", 0.8),  # Overlaps with first
            SpeakerSegment(5.0, 7.0, "speaker_1", 0.85),  # Different speaker
        ]

        merged = diarizer._merge_overlapping_segments(segments)

        # Should have 2 segments (merged + separate speaker)
        assert len(merged) == 2

        # First segment should be merged (0.0 to 4.0)
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 4.0
        assert merged[0].speaker_id == "speaker_0"

        # Second segment should be unchanged
        assert merged[1].start_time == 5.0
        assert merged[1].end_time == 7.0
        assert merged[1].speaker_id == "speaker_1"

    def test_compute_speaker_statistics(self, speaker_config, mock_pyannote):
        """Test speaker statistics computation"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        segments = [
            SpeakerSegment(0.0, 2.0, "speaker_0", 0.9),
            SpeakerSegment(2.0, 4.0, "speaker_0", 0.8),
            SpeakerSegment(4.0, 6.0, "speaker_1", 0.85),
        ]

        stats = diarizer._compute_speaker_statistics(segments)

        assert len(stats) == 2
        assert "speaker_0" in stats
        assert "speaker_1" in stats

        # Check speaker_0 stats
        speaker_0_stats = stats["speaker_0"]
        assert speaker_0_stats.total_duration == 4.0  # 2 + 2 seconds
        assert speaker_0_stats.segment_count == 2
        assert speaker_0_stats.avg_confidence == 0.85  # (0.9 + 0.8) / 2

        # Check speaker_1 stats
        speaker_1_stats = stats["speaker_1"]
        assert speaker_1_stats.total_duration == 2.0
        assert speaker_1_stats.segment_count == 1
        assert speaker_1_stats.avg_confidence == 0.85

    def test_chunk_processing(
        self, speaker_config, sample_long_audio_file, mock_pyannote
    ):
        """Test processing with chunking for long audio"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Mock chunk processing duration to force chunking
        with patch.object(diarizer.config, "chunk_processing_duration", 10.0):
            result = diarizer.diarize_audio(sample_long_audio_file)

            assert isinstance(result, DiarizationResult)
            assert result.success is True
            # Should still process the file, even if chunked
            assert result.total_duration > 0

    def test_context_manager(self, speaker_config, mock_pyannote):
        """Test SpeakerDiarizer as context manager"""
        with SpeakerDiarizer(config=speaker_config, device="cpu") as diarizer:
            assert diarizer is not None
            assert diarizer.embedding_cache is not None

    def test_cleanup(self, speaker_config, mock_pyannote):
        """Test resource cleanup"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Add some data to cache
        if diarizer.embedding_cache:
            diarizer.embedding_cache.store_embedding("test", torch.randn(64))
            assert len(diarizer.embedding_cache.cache) == 1

        # Cleanup
        diarizer.cleanup()

        # Cache should be cleared
        if diarizer.embedding_cache:
            assert len(diarizer.embedding_cache.cache) == 0

    def test_error_handling_during_processing(
        self, speaker_config, sample_audio_file, mock_pyannote
    ):
        """Test error handling during audio processing"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Mock torchaudio.info to raise an exception
        with patch("torchaudio.info", side_effect=Exception("Audio loading error")):
            result = diarizer.diarize_audio(sample_audio_file)

            assert isinstance(result, DiarizationResult)
            assert result.success is False
            assert "Diarization failed" in result.error_message

    def test_warmup_model(self, speaker_config, mock_pyannote):
        """Test model warmup functionality"""
        diarizer = SpeakerDiarizer(config=speaker_config, device="cpu")

        # Load model first
        diarizer.load_model()

        # Test warmup (should not raise exception)
        diarizer._warmup_model()

    def test_diarization_result_to_dict(self):
        """Test DiarizationResult dictionary conversion"""
        segments = [
            SpeakerSegment(0.0, 2.0, "speaker_0", 0.9),
            SpeakerSegment(2.0, 4.0, "speaker_1", 0.8),
        ]

        speakers = {
            "speaker_0": SpeakerInfo("speaker_0", 2.0, 1, 0.9, 0.9, 0.9),
            "speaker_1": SpeakerInfo("speaker_1", 2.0, 1, 0.8, 0.8, 0.8),
        }

        result = DiarizationResult(
            success=True,
            segments=segments,
            speakers=speakers,
            total_speakers=2,
            total_duration=4.0,
            processing_time=1.5,
            model_info={"model": "test"},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["total_speakers"] == 2
        assert result_dict["total_duration"] == 4.0
        assert result_dict["processing_time"] == 1.5
        assert len(result_dict["segments"]) == 2
        assert len(result_dict["speakers"]) == 2
