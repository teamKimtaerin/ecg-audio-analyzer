"""
Pytest configuration and fixtures for ECG Audio Analysis tests
"""

import pytest
import tempfile
import numpy as np
import torch
import torchaudio
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.base_settings import BaseConfig, ProcessingConfig, ValidationConfig
from config.aws_settings import AWSConfig
from config.model_configs import (
    SpeakerDiarizationConfig,
    EmotionAnalysisConfig,
    AcousticAnalysisConfig,
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_audio_file(test_data_dir):
    """Create a sample audio file for testing"""
    # Generate 5 seconds of test audio (sine wave)
    sample_rate = 16000
    duration = 5.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = np.sin(2 * np.pi * frequency * t) * 0.3

    # Add some variation to make it more realistic
    noise = np.random.normal(0, 0.05, len(waveform))
    waveform = waveform + noise

    # Convert to torch tensor
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

    # Save to file
    audio_file = test_data_dir / "test_audio.wav"
    torchaudio.save(str(audio_file), waveform_tensor, sample_rate)

    return audio_file


@pytest.fixture
def sample_long_audio_file(test_data_dir):
    """Create a longer sample audio file for testing"""
    sample_rate = 16000
    duration = 30.0  # 30 seconds

    # Generate more complex audio with multiple frequencies
    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = (
        np.sin(2 * np.pi * 220 * t) * 0.2  # Low frequency
        + np.sin(2 * np.pi * 440 * t) * 0.2  # Mid frequency
        + np.sin(2 * np.pi * 880 * t) * 0.1  # High frequency
    )

    # Add speech-like modulation
    modulation = np.sin(2 * np.pi * 5 * t) * 0.3 + 1.0
    waveform = waveform * modulation

    # Add noise
    noise = np.random.normal(0, 0.02, len(waveform))
    waveform = waveform + noise

    # Normalize
    waveform = waveform / np.max(np.abs(waveform)) * 0.8

    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

    audio_file = test_data_dir / "test_long_audio.wav"
    torchaudio.save(str(audio_file), waveform_tensor, sample_rate)

    return audio_file


@pytest.fixture
def base_config(test_data_dir):
    """Base configuration for testing"""
    return BaseConfig(
        temp_dir=test_data_dir / "temp",
        output_dir=test_data_dir / "output",
        max_workers=2,  # Reduced for testing
        enable_cleanup=True,
        log_level="DEBUG",
    )


@pytest.fixture
def processing_config():
    """Processing configuration for testing"""
    return ProcessingConfig(
        enable_multiprocessing=False,  # Disabled for testing
        enable_async_io=True,
        enable_memory_monitoring=False,
        max_workers=2,
    )


@pytest.fixture
def validation_config():
    """Validation configuration for testing"""
    return ValidationConfig(
        min_duration_seconds=0.5,  # Shorter for testing
        max_duration_seconds=60.0,
        enable_quality_checks=True,
    )


@pytest.fixture
def aws_config():
    """AWS configuration for testing"""
    return AWSConfig(
        instance_type="test_instance",
        cuda_device="cpu",  # Use CPU for testing
        enable_gpu_monitoring=False,
        concurrent_workers=2,
        s3_bucket="test-bucket",
    )


@pytest.fixture
def speaker_config():
    """Speaker diarization configuration for testing"""
    return SpeakerDiarizationConfig(
        device="cpu",  # Use CPU for testing
        enable_fp16=False,
        batch_size=2,
        min_speakers=1,
        max_speakers=4,
    )


@pytest.fixture
def emotion_config():
    """Emotion analysis configuration for testing"""
    return EmotionAnalysisConfig(
        device="cpu", enable_fp16=False, batch_size=2, confidence_threshold=0.5
    )


@pytest.fixture
def acoustic_config():
    """Acoustic analysis configuration for testing"""
    return AcousticAnalysisConfig(
        sample_rate=16000, enable_gpu_acceleration=False, parallel_processing=False
    )


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailable for CPU-only testing"""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_huggingface_models():
    """Mock Hugging Face models to avoid downloading in tests"""

    class MockProcessor:
        def __call__(self, audio, sampling_rate, return_tensors="pt", padding=True):
            return {"input_values": torch.randn(1, len(audio))}

    class MockModel:
        def __init__(self):
            self.training = False

        def eval(self):
            return self

        def to(self, device):
            return self

        def half(self):
            return self

        def __call__(self, **inputs):
            # Return mock model outputs
            batch_size = inputs["input_values"].shape[0]
            num_labels = 7  # Number of emotions
            logits = torch.randn(batch_size, num_labels)
            return Mock(logits=logits)

        def parameters(self):
            return [torch.randn(100, 100)]

    class MockPipeline:
        def __call__(self, text, return_all_scores=True):
            # Return mock emotion scores
            emotions = [
                "joy",
                "sadness",
                "anger",
                "fear",
                "surprise",
                "disgust",
                "neutral",
            ]
            scores = np.random.dirichlet([1] * len(emotions))

            return [
                {"label": emotion, "score": float(score)}
                for emotion, score in zip(emotions, scores)
            ]

    with (
        patch(
            "transformers.Wav2Vec2Processor.from_pretrained",
            return_value=MockProcessor(),
        ),
        patch(
            "transformers.Wav2Vec2ForSequenceClassification.from_pretrained",
            return_value=MockModel(),
        ),
        patch("transformers.pipeline", return_value=MockPipeline()),
    ):
        yield


@pytest.fixture
def mock_pyannote():
    """Mock pyannote-audio components"""

    class MockAnnotation:
        def __init__(self, segments):
            self.segments = segments

        def itertracks(self, yield_label=False):
            for i, (start, end) in enumerate(self.segments):
                segment = Mock()
                segment.start = start
                segment.end = end
                if yield_label:
                    yield segment, None, f"speaker_{i % 2}"
                else:
                    yield segment

    class MockPipeline:
        def __init__(self):
            self.device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def instantiate(self, params):
            pass

        def __call__(self, audio_data):
            # Return mock diarization with 2 speakers
            duration = audio_data["waveform"].shape[1] / audio_data["sample_rate"]
            segments = [
                (0.0, duration * 0.6),  # Speaker 0
                (duration * 0.4, duration),  # Speaker 1 (with overlap)
            ]
            return MockAnnotation(segments)

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=MockPipeline()):
        yield


@pytest.fixture
def mock_opensmile():
    """Mock OpenSMILE for acoustic analysis"""

    def mock_subprocess_run(*args, **kwargs):
        # Mock successful OpenSMILE execution
        result = Mock()
        result.returncode = 0
        return result

    with (
        patch("subprocess.run", side_effect=mock_subprocess_run),
        patch("shutil.which", return_value="/usr/bin/SMILExtract"),
    ):
        yield


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce log noise in tests
    yield
    # Cleanup after tests
    logging.getLogger().handlers.clear()


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration optimized for performance benchmarking"""
    return {
        "iterations": 10,
        "timeout_seconds": 30,
        "memory_limit_mb": 1000,
        "acceptable_variance": 0.2,  # 20% variance acceptable
    }


# Integration test fixtures
@pytest.fixture
def integration_test_files(test_data_dir):
    """Create multiple test files for integration testing"""
    files = []

    # Create 3 different audio files
    for i in range(3):
        sample_rate = 16000
        duration = 3.0 + i  # Different durations
        frequency = 200 + (i * 100)  # Different frequencies

        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * frequency * t) * 0.3

        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

        audio_file = test_data_dir / f"integration_test_{i}.wav"
        torchaudio.save(str(audio_file), waveform_tensor, sample_rate)

        files.append(audio_file)

    return files


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    test_env = {
        "HUGGING_FACE_TOKEN": "test_token",
        "ECG_S3_BUCKET": "test-bucket",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    with patch.dict(os.environ, test_env):
        yield test_env


# Error testing fixtures
@pytest.fixture
def corrupt_audio_file(test_data_dir):
    """Create a corrupt audio file for error testing"""
    corrupt_file = test_data_dir / "corrupt_audio.wav"
    with open(corrupt_file, "w") as f:
        f.write("This is not an audio file")
    return corrupt_file


@pytest.fixture
def empty_audio_file(test_data_dir):
    """Create an empty audio file for error testing"""
    empty_file = test_data_dir / "empty_audio.wav"
    empty_file.touch()
    return empty_file
