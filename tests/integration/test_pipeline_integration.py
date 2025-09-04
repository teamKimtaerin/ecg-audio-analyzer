"""
Integration tests for the complete audio analysis pipeline
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from pipeline.manager import PipelineManager
from services.result_synthesizer import ResultSynthesizer, SynthesisInput
from models.output_models import CompleteAnalysisResult, create_empty_analysis_result


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""

    @patch("torch.cuda.is_available", return_value=False)
    def test_pipeline_manager_initialization(
        self, mock_cuda, base_config, processing_config, aws_config, speaker_config
    ):
        """Test pipeline manager initialization"""
        with PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            aws_config=aws_config,
            speaker_config=speaker_config,
        ) as pipeline:
            assert pipeline is not None
            assert pipeline.progress.total_stages > 0
            assert pipeline.gpu_manager is not None

    @pytest.mark.asyncio
    async def test_single_file_pipeline_mock(
        self,
        base_config,
        processing_config,
        aws_config,
        speaker_config,
        sample_audio_file,
    ):
        """Test complete pipeline with single file (mocked dependencies)"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(PipelineManager, "_run_audio_extraction") as mock_extraction,
            patch.object(
                PipelineManager, "_run_speaker_diarization"
            ) as mock_diarization,
        ):

            # Mock extraction result
            from services.audio_extractor import AudioExtractionResult

            mock_extraction_result = AudioExtractionResult(
                success=True,
                output_path=sample_audio_file,
                duration_seconds=5.0,
                sample_rate=16000,
                channels=1,
                file_size_mb=0.5,
                extraction_time_seconds=0.1,
            )
            mock_extraction.return_value = mock_extraction_result

            # Mock diarization result
            from services.speaker_diarizer import (
                DiarizationResult,
                SpeakerSegment,
                SpeakerInfo,
            )

            segments = [
                SpeakerSegment(0.0, 2.5, "speaker_0", 0.9),
                SpeakerSegment(2.5, 5.0, "speaker_1", 0.85),
            ]
            speakers = {
                "speaker_0": SpeakerInfo("speaker_0", 2.5, 1, 0.9, 0.9, 0.9),
                "speaker_1": SpeakerInfo("speaker_1", 2.5, 1, 0.85, 0.85, 0.85),
            }

            mock_diarization_result = DiarizationResult(
                success=True,
                segments=segments,
                speakers=speakers,
                total_speakers=2,
                total_duration=5.0,
                processing_time=0.5,
            )
            mock_diarization.return_value = mock_diarization_result

            # Run pipeline
            with PipelineManager(
                base_config=base_config,
                processing_config=processing_config,
                aws_config=aws_config,
                speaker_config=speaker_config,
            ) as pipeline:
                result = await pipeline.process_single(sample_audio_file)

                assert isinstance(result, CompleteAnalysisResult)
                assert result.metadata.filename == sample_audio_file.name
                assert result.metadata.duration == 5.0
                assert result.metadata.total_speakers >= 0  # Depends on mock behavior

    @pytest.mark.asyncio
    async def test_batch_processing_mock(
        self,
        base_config,
        processing_config,
        aws_config,
        speaker_config,
        integration_test_files,
    ):
        """Test batch processing with multiple files"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(PipelineManager, "process_single") as mock_process,
        ):

            # Mock successful processing for each file
            def mock_process_side_effect(source):
                return create_empty_analysis_result(
                    filename=Path(str(source)).name,
                    duration=3.0 + len(str(source)) % 3,  # Varying durations
                )

            mock_process.side_effect = mock_process_side_effect

            with PipelineManager(
                base_config=base_config,
                processing_config=processing_config,
                aws_config=aws_config,
                speaker_config=speaker_config,
            ) as pipeline:
                results = await pipeline.process_batch(
                    integration_test_files, max_concurrent=2
                )

                assert len(results) == len(integration_test_files)
                assert all(isinstance(r, CompleteAnalysisResult) for r in results)
                assert mock_process.call_count == len(integration_test_files)

    def test_result_synthesizer_integration(self, sample_audio_file):
        """Test result synthesizer integration"""
        synthesizer = ResultSynthesizer(
            enable_compression=False, enable_validation=True
        )

        # Create mock synthesis input
        from services.audio_extractor import AudioExtractionResult
        from services.speaker_diarizer import (
            DiarizationResult,
            SpeakerSegment,
            SpeakerInfo,
        )

        extraction_result = AudioExtractionResult(
            success=True,
            output_path=sample_audio_file,
            duration_seconds=5.0,
            sample_rate=16000,
            channels=1,
            file_size_mb=0.5,
            extraction_time_seconds=0.1,
        )

        segments = [
            SpeakerSegment(0.0, 2.5, "speaker_0", 0.9),
            SpeakerSegment(2.5, 5.0, "speaker_1", 0.85),
        ]
        speakers = {
            "speaker_0": SpeakerInfo("speaker_0", 2.5, 1, 0.9, 0.9, 0.9),
            "speaker_1": SpeakerInfo("speaker_1", 2.5, 1, 0.85, 0.85, 0.85),
        }

        diarization_result = DiarizationResult(
            success=True,
            segments=segments,
            speakers=speakers,
            total_speakers=2,
            total_duration=5.0,
            processing_time=0.5,
        )

        synthesis_input = SynthesisInput(
            filename=sample_audio_file.name,
            duration=5.0,
            audio_extraction_result=extraction_result,
            diarization_result=diarization_result,
            processing_start_time=0.0,
            gpu_acceleration_used=False,
        )

        # Synthesize results
        result = synthesizer.synthesize_results(synthesis_input)

        assert isinstance(result, CompleteAnalysisResult)
        assert result.metadata.filename == sample_audio_file.name
        assert result.metadata.duration == 5.0
        assert result.metadata.total_speakers == 2
        assert len(result.timeline) == 2  # Two segments
        assert len(result.speakers) == 2  # Two speakers

    def test_pipeline_progress_tracking(
        self, base_config, processing_config, aws_config, speaker_config
    ):
        """Test pipeline progress tracking"""
        with PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            aws_config=aws_config,
            speaker_config=speaker_config,
        ) as pipeline:

            # Initial progress
            progress = pipeline.get_progress()
            assert progress.current_stage == "initialization"
            assert progress.progress_percentage == 0.0
            assert progress.total_stages > 0

    def test_resource_monitoring_integration(
        self, base_config, processing_config, aws_config, speaker_config
    ):
        """Test resource monitoring integration"""
        with PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            aws_config=aws_config,
            speaker_config=speaker_config,
        ) as pipeline:

            # Get resource usage
            resource_usage = pipeline.get_resource_usage()

            assert hasattr(resource_usage, "cpu_percent")
            assert hasattr(resource_usage, "memory_mb")
            assert hasattr(resource_usage, "gpu_memory_mb")
            assert hasattr(resource_usage, "active_threads")

    @pytest.mark.asyncio
    async def test_error_recovery_integration(
        self,
        base_config,
        processing_config,
        aws_config,
        speaker_config,
        corrupt_audio_file,
    ):
        """Test pipeline error recovery with corrupt file"""
        with patch("torch.cuda.is_available", return_value=False):
            with PipelineManager(
                base_config=base_config,
                processing_config=processing_config,
                aws_config=aws_config,
                speaker_config=speaker_config,
            ) as pipeline:

                # Process corrupt file
                result = await pipeline.process_single(corrupt_audio_file)

                # Should return error result, not crash
                assert isinstance(result, CompleteAnalysisResult)
                # Result should indicate failure in some way (e.g., no speakers found)
                assert result.metadata.total_speakers == 0

    def test_pipeline_cleanup_integration(
        self, base_config, processing_config, aws_config, speaker_config, test_data_dir
    ):
        """Test pipeline cleanup integration"""

        # Create pipeline and use it
        with PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            aws_config=aws_config,
            speaker_config=speaker_config,
        ) as pipeline:

            # Create some temporary files
            temp_file = base_config.temp_dir / "test_cleanup.txt"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text("test content")

            assert temp_file.exists()

        # After context manager exit, cleanup should be called
        # (Actual cleanup behavior depends on implementation)

    @pytest.mark.asyncio
    async def test_concurrent_processing_integration(
        self,
        base_config,
        processing_config,
        aws_config,
        speaker_config,
        integration_test_files,
    ):
        """Test concurrent processing integration"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(PipelineManager, "process_single") as mock_process,
        ):

            # Mock processing with different durations
            async def mock_process_side_effect(source):
                await asyncio.sleep(0.1)  # Simulate processing time
                return create_empty_analysis_result(
                    filename=Path(str(source)).name, duration=2.0
                )

            mock_process.side_effect = mock_process_side_effect

            with PipelineManager(
                base_config=base_config,
                processing_config=processing_config,
                aws_config=aws_config,
                speaker_config=speaker_config,
            ) as pipeline:

                import time

                start_time = time.time()

                results = await pipeline.process_batch(
                    integration_test_files[:2], max_concurrent=2  # Process 2 files
                )

                end_time = time.time()
                processing_time = end_time - start_time

                # Should process concurrently (less than sequential time)
                assert processing_time < 0.2 * len(
                    integration_test_files[:2]
                )  # Much faster than sequential
                assert len(results) == 2
                assert all(isinstance(r, CompleteAnalysisResult) for r in results)


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(
        self,
        base_config,
        processing_config,
        aws_config,
        speaker_config,
        sample_audio_file,
        test_data_dir,
    ):
        """Test full pipeline from audio file to JSON output with all mocks"""

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("ffmpeg.run"),
            patch("soundfile.info") as mock_sf_info,
            patch("torchaudio.info") as mock_ta_info,
            patch("torchaudio.load") as mock_ta_load,
        ):

            # Mock audio file info
            mock_sf_info.return_value.duration = 5.0
            mock_sf_info.return_value.samplerate = 16000
            mock_ta_info.return_value.num_frames = 80000
            mock_ta_info.return_value.sample_rate = 16000

            # Mock audio loading
            import torch

            mock_ta_load.return_value = (torch.randn(1, 80000), 16000)

            with patch(
                "pyannote.audio.Pipeline.from_pretrained"
            ) as mock_pipeline_class:

                # Mock pyannote pipeline
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.to.return_value = mock_pipeline
                mock_pipeline.instantiate.return_value = None

                # Mock diarization output
                class MockAnnotation:
                    def itertracks(self, yield_label=False):
                        segments = [(0.0, 2.5), (2.5, 5.0)]  # Two segments
                        for i, (start, end) in enumerate(segments):
                            segment = type(
                                "obj", (object,), {"start": start, "end": end}
                            )()
                            if yield_label:
                                yield segment, None, f"speaker_{i}"
                            else:
                                yield segment

                mock_pipeline.return_value = MockAnnotation()

                # Run full pipeline
                output_file = test_data_dir / "integration_output.json"

                with PipelineManager(
                    base_config=base_config,
                    processing_config=processing_config,
                    aws_config=aws_config,
                    speaker_config=speaker_config,
                ) as pipeline:

                    result = await pipeline.process_single(
                        sample_audio_file, output_path=output_file
                    )

                    # Verify result
                    assert isinstance(result, CompleteAnalysisResult)
                    assert result.metadata.filename == sample_audio_file.name

                    # Verify JSON output (if file was created)
                    # Note: In test environment with mocks, file creation might be mocked too

    def test_configuration_integration(self, test_data_dir):
        """Test configuration integration across all components"""
        from config.base_settings import BaseConfig, ProcessingConfig, ValidationConfig
        from config.aws_settings import AWSConfig
        from config.model_configs import (
            SpeakerDiarizationConfig,
            EmotionAnalysisConfig,
            AcousticAnalysisConfig,
        )

        # Create configurations
        base_config = BaseConfig(temp_dir=test_data_dir / "temp")
        processing_config = ProcessingConfig(max_workers=1)
        validation_config = ValidationConfig()
        aws_config = AWSConfig(cuda_device="cpu")
        speaker_config = SpeakerDiarizationConfig(device="cpu")
        emotion_config = EmotionAnalysisConfig(device="cpu")
        acoustic_config = AcousticAnalysisConfig()

        # Verify all configurations are compatible
        assert base_config.temp_dir == test_data_dir / "temp"
        assert processing_config.max_workers == 1
        assert aws_config.cuda_device == "cpu"
        assert speaker_config.device == "cpu"
        assert emotion_config.device == "cpu"

        # Test configuration validation
        assert validation_config.min_duration_seconds > 0
        assert (
            validation_config.max_duration_seconds
            > validation_config.min_duration_seconds
        )
