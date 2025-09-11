"""
Unit tests for AudioExtractor service
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.services.audio_extractor import AudioExtractor, AudioExtractionResult


class TestAudioExtractor:
    """Test cases for AudioExtractor service"""

    def test_initialization(self, base_config, processing_config, validation_config):
        """Test AudioExtractor initialization"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            assert extractor.config == base_config
            assert extractor.processing_config == processing_config
            assert extractor.validation_config == validation_config
            assert extractor.temp_dir.exists()

    def test_dependency_validation_success(
        self, base_config, processing_config, validation_config
    ):
        """Test successful dependency validation"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )
            # Should not raise exception
            assert extractor is not None

    def test_dependency_validation_failure(
        self, base_config, processing_config, validation_config
    ):
        """Test dependency validation failure"""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Missing required tools"):
                AudioExtractor(
                    config=base_config,
                    processing_config=processing_config,
                    validation_config=validation_config,
                )

    def test_audio_validation_success(
        self, base_config, processing_config, validation_config, sample_audio_file
    ):
        """Test successful audio file validation"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            is_valid = extractor._validate_audio_file(sample_audio_file)
            assert is_valid is True

    def test_audio_validation_too_short(
        self, base_config, processing_config, validation_config, test_data_dir
    ):
        """Test audio validation with file too short"""
        import torch
        import torchaudio

        # Create very short audio file (0.1 seconds)
        short_audio = torch.randn(1, 1600)  # 0.1 seconds at 16kHz
        short_file = test_data_dir / "short_audio.wav"
        torchaudio.save(str(short_file), short_audio, 16000)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            is_valid = extractor._validate_audio_file(short_file)
            assert is_valid is False

    def test_gpu_acceleration_detection(
        self, base_config, processing_config, validation_config
    ):
        """Test GPU acceleration detection"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("subprocess.run") as mock_run,
        ):

            # Mock successful nvidia-smi call
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
                enable_gpu_acceleration=True,
            )

            gpu_opts = extractor._get_ffmpeg_gpu_options()
            assert "hwaccel" in gpu_opts or len(gpu_opts) == 0  # Fallback to empty dict

    def test_convert_to_wav_success(
        self,
        base_config,
        processing_config,
        validation_config,
        sample_audio_file,
        test_data_dir,
    ):
        """Test successful WAV conversion"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("ffmpeg.run") as mock_ffmpeg,
            patch.object(AudioExtractor, "_validate_audio_file", return_value=True),
        ):

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            output_path = test_data_dir / "converted.wav"
            result = extractor._convert_to_wav(sample_audio_file, output_path)

            assert isinstance(result, AudioExtractionResult)
            # In test environment, ffmpeg.run is mocked, so we check the structure
            mock_ffmpeg.assert_called_once()

    def test_extract_single_file_success(
        self, base_config, processing_config, validation_config, sample_audio_file
    ):
        """Test successful single file extraction"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(AudioExtractor, "_convert_to_wav") as mock_convert,
        ):

            # Mock successful conversion
            mock_result = AudioExtractionResult(
                success=True,
                output_path=Path("test_output.wav"),
                duration_seconds=5.0,
                sample_rate=16000,
                channels=1,
                file_size_mb=0.5,
            )
            mock_convert.return_value = mock_result

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            result = extractor.extract_single(sample_audio_file)

            assert result.success is True
            assert result.duration_seconds == 5.0
            assert result.sample_rate == 16000
            mock_convert.assert_called_once()

    def test_extract_single_file_not_found(
        self, base_config, processing_config, validation_config
    ):
        """Test extraction with non-existent file"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            result = extractor.extract_single(Path("non_existent_file.mp4"))

            assert result.success is False
            assert "File not found" in result.error_message

    def test_extract_single_file_too_large(
        self, base_config, processing_config, validation_config, sample_audio_file
    ):
        """Test extraction with file too large"""
        # Set very small file size limit
        base_config.max_file_size_gb = 0.000001  # 1 byte

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            result = extractor.extract_single(sample_audio_file)

            assert result.success is False
            assert "File too large" in result.error_message

    @pytest.mark.asyncio
    async def test_extract_batch_success(
        self, base_config, processing_config, validation_config, integration_test_files
    ):
        """Test successful batch extraction"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(AudioExtractor, "extract_single") as mock_extract,
        ):

            # Mock successful extraction for each file
            mock_result = AudioExtractionResult(
                success=True,
                output_path=Path("test_output.wav"),
                duration_seconds=3.0,
                sample_rate=16000,
                channels=1,
                file_size_mb=0.3,
            )
            mock_extract.return_value = mock_result

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            results = await extractor.extract_batch(integration_test_files)

            assert len(results) == len(integration_test_files)
            assert all(r.success for r in results)
            assert mock_extract.call_count == len(integration_test_files)

    @pytest.mark.asyncio
    async def test_extract_batch_mixed_results(
        self, base_config, processing_config, validation_config, integration_test_files
    ):
        """Test batch extraction with mixed success/failure"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(AudioExtractor, "extract_single") as mock_extract,
        ):

            # Mock mixed results
            def side_effect(source):
                if "0" in str(source):  # First file succeeds
                    return AudioExtractionResult(
                        success=True,
                        output_path=Path("test_output.wav"),
                        duration_seconds=3.0,
                        sample_rate=16000,
                        channels=1,
                        file_size_mb=0.3,
                    )
                else:  # Other files fail
                    return AudioExtractionResult(
                        success=False, error_message="Mock extraction failure"
                    )

            mock_extract.side_effect = side_effect

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            results = await extractor.extract_batch(integration_test_files)

            assert len(results) == len(integration_test_files)
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            assert len(successful) == 1
            assert len(failed) == 2

    def test_url_extraction_mock(
        self, base_config, processing_config, validation_config
    ):
        """Test URL extraction (mocked)"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("yt_dlp.YoutubeDL") as mock_ytdl_class,
        ):

            # Mock yt-dlp
            mock_ytdl = Mock()
            mock_ytdl.extract_info.return_value = {
                "title": "Test Video",
                "duration": 120,
            }
            mock_ytdl.download.return_value = None
            mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl

            # Mock converted file creation
            with patch.object(AudioExtractor, "_convert_to_wav") as mock_convert:
                mock_result = AudioExtractionResult(
                    success=True,
                    output_path=Path("test_output.wav"),
                    duration_seconds=120.0,
                    sample_rate=16000,
                    channels=1,
                    file_size_mb=5.0,
                )
                mock_convert.return_value = mock_result

                extractor = AudioExtractor(
                    config=base_config,
                    processing_config=processing_config,
                    validation_config=validation_config,
                )

                result = extractor._extract_from_url("https://youtube.com/watch?v=test")

                assert result.success is True
                assert result.duration_seconds == 120.0
                mock_ytdl.extract_info.assert_called_once()
                mock_ytdl.download.assert_called_once()

    def test_cleanup_temp_files(
        self, base_config, processing_config, validation_config, test_data_dir
    ):
        """Test temporary file cleanup"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
                temp_dir=test_data_dir / "extractor_temp",
            )

            # Create some test files in temp dir
            temp_file = extractor.temp_dir / "test_file.txt"
            temp_file.write_text("test content")

            assert temp_file.exists()

            extractor.cleanup_temp_files()

            # Files should be cleaned up, but directory should still exist
            assert not temp_file.exists()
            assert extractor.temp_dir.exists()

    def test_context_manager(self, base_config, processing_config, validation_config):
        """Test AudioExtractor as context manager"""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            ) as extractor:
                assert extractor is not None
                assert extractor.temp_dir.exists()

    def test_error_handling_in_conversion(
        self, base_config, processing_config, validation_config, sample_audio_file
    ):
        """Test error handling during conversion"""
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("ffmpeg.run", side_effect=Exception("FFmpeg error")),
        ):

            extractor = AudioExtractor(
                config=base_config,
                processing_config=processing_config,
                validation_config=validation_config,
            )

            result = extractor._convert_to_wav(sample_audio_file)

            assert result.success is False
            assert "Conversion failed" in result.error_message
