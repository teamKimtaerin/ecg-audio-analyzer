"""
Pipeline Manager - Streamlined Service Orchestration
Single Responsibility: Coordinate analysis workflow efficiently
"""

import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .resource_manager import ResourceManager
from .progress_tracker import ProgressTracker
from .stage_executor import StageExecutor
from ..models.output_models import (
    CompleteAnalysisResult,
    AnalysisMetadata,
    PerformanceStats,
    create_empty_analysis_result,
)
from ..utils.logger import get_logger
from config.base_settings import BaseConfig, ProcessingConfig
from config.aws_settings import AWSConfig
from config.model_configs import SpeakerDiarizationConfig


class PipelineManager:
    """
    Streamlined pipeline orchestration manager.
    Single Responsibility: Coordinate analysis services efficiently.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        processing_config: ProcessingConfig,
        aws_config: Optional[AWSConfig] = None,
        speaker_config: Optional[SpeakerDiarizationConfig] = None,
        language: str = "en",
    ):
        self.base_config = base_config
        self.processing_config = processing_config
        self.aws_config = aws_config
        self.speaker_config = speaker_config or SpeakerDiarizationConfig()
        self.language = language

        self.logger = get_logger().bind_context(component="pipeline_manager")

        # Initialize components
        max_gpu_workers = aws_config.concurrent_workers if aws_config else 2
        self.resource_manager = ResourceManager(
            max_concurrent_gpu_tasks=max_gpu_workers
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=base_config.max_workers)

        # Pipeline stage configuration
        self.stage_config = self._get_stage_config()

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(self.stage_config)

        # Initialize stage executor
        self.stage_executor = StageExecutor(
            base_config, processing_config, aws_config, language, self.thread_pool
        )

        self.logger.info(
            "pipeline_manager_initialized",
            total_stages=self.progress_tracker.progress.total_stages,
            gpu_workers=max_gpu_workers,
            thread_workers=base_config.max_workers,
        )

        # Store results for API access
        self._last_whisperx_result = None
        self._last_audio_path = None

    def _get_stage_config(self) -> Dict[str, Dict]:
        """Get simplified stage configuration for essential stages only"""
        return {
            "audio_extraction": {"required": True},
            "whisperx_pipeline": {"required": True},
            "result_synthesis": {"required": True},
        }

    async def preload_models(self, enable_warmup: bool = True) -> Dict[str, Any]:
        """Preload models to reduce first-time processing latency"""
        try:
            from ..models.model_manager import get_model_manager

            model_manager = get_model_manager()

            # Run preloading in thread pool
            loop = asyncio.get_event_loop()

            preload_results = await loop.run_in_executor(
                self.thread_pool,
                model_manager.preload_models,
                True,  # load_speaker
                True,  # load_whisperx
                False,  # load_emotion (optional)
            )

            return {"status": "success", "models": preload_results}

        except Exception as e:
            self.logger.error("model_preloading_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    def _create_basic_result(
        self,
        source: Union[str, Path],
        extraction_result,
        whisperx_result: Dict[str, Any],
    ) -> CompleteAnalysisResult:
        """Create basic analysis result from available data"""
        filename = Path(source).name if isinstance(source, (str, Path)) else "unknown"
        duration = extraction_result.duration or 0.0

        # Extract speaker information from WhisperX result
        segments = whisperx_result.get("segments", [])
        unique_speakers = len(
            set(seg.get("speaker", "UNKNOWN") for seg in segments if seg.get("speaker"))
        )

        # Create basic metadata
        metadata = AnalysisMetadata(
            filename=filename,
            duration=duration,
            total_speakers=unique_speakers,
            processing_time=0.0,
            gpu_acceleration=(
                self.aws_config.cuda_device.startswith("cuda")
                if self.aws_config and self.aws_config.cuda_device
                else False
            ),
        )

        # Create performance stats
        resource_usage = self.resource_manager.get_resource_usage()
        performance_stats = PerformanceStats(
            gpu_utilization=0.0,  # Will be updated by resource manager
            peak_memory_mb=int(resource_usage.peak_memory_mb),
            avg_processing_fps=0.0,
            bottleneck_stage="whisperx_pipeline",
        )

        # Create basic result with available data
        result = CompleteAnalysisResult(
            metadata=metadata,
            timeline=[],  # Will be populated by ResultSynthesizer
            performance_stats=performance_stats,
        )

        return result

    async def process_single(
        self, source: Union[str, Path], output_path: Optional[Path] = None
    ) -> CompleteAnalysisResult:
        """
        Process a single audio/video file through the complete pipeline.
        """
        start_time = time.time()
        source_str = str(source)

        with self.logger.timer("complete_pipeline"):
            self.logger.info("pipeline_started", source=source_str)

            try:
                # Stage 1: Audio Extraction
                self.progress_tracker.update_stage("audio_extraction")
                extraction_result = await self.stage_executor.execute_audio_extraction(
                    source
                )

                # Validate extraction result
                if extraction_result.error or not extraction_result.output_path:
                    error_msg = f"Audio extraction failed: {extraction_result.error or 'No output path'}"
                    self.logger.error("audio_extraction_failed", error=error_msg)
                    raise ValueError(error_msg)

                self.progress_tracker.mark_stage_completed("audio_extraction")

                # Stage 2: WhisperX Pipeline
                self.progress_tracker.update_stage("whisperx_pipeline")

                # Get expected duration for validation
                expected_duration = getattr(
                    extraction_result, "original_duration", extraction_result.duration
                )

                if extraction_result.output_path:
                    whisperx_result = (
                        await self.stage_executor.execute_whisperx_pipeline(
                            extraction_result.output_path, self.resource_manager
                        )
                    )
                else:
                    whisperx_result = {}

                # Validate WhisperX result
                if not whisperx_result or not whisperx_result.get("segments"):
                    error_msg = "Speech recognition failed: No audio segments found"
                    self.logger.error("whisperx_failed", error=error_msg)
                    raise ValueError(error_msg)

                self.progress_tracker.mark_stage_completed("whisperx_pipeline")

                # Stage 3: Timeline validation
                # Check for timeline coverage gaps
                if whisperx_result.get("segments"):
                    segments = whisperx_result["segments"]
                    last_end = segments[-1].get("end", 0) if segments else 0
                    coverage_ratio = last_end / expected_duration if expected_duration > 0 else 0
                    
                    if coverage_ratio < 0.8:  # Less than 80% coverage
                        warning = f"Low timeline coverage: {coverage_ratio:.1%} of expected duration"
                        self.logger.warning("timeline_validation_warning", warning=warning)
                    self.progress_tracker.add_warning()

                # Store results for API access
                self._last_whisperx_result = whisperx_result
                self._last_audio_path = (
                    str(extraction_result.output_path)
                    if extraction_result.output_path
                    else None
                )

                # Stage 4: Result synthesis
                self.progress_tracker.update_stage("result_synthesis")
                result = self._create_basic_result(
                    source, extraction_result, whisperx_result
                )
                self.progress_tracker.mark_stage_completed("result_synthesis")

                # Save result if output path specified
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        f.write(result.model_dump_json_formatted())
                    self.logger.info("results_saved", output_path=str(output_path))

                total_time = time.time() - start_time
                result.metadata.processing_time = total_time

                self.logger.info(
                    "pipeline_completed",
                    source=source_str,
                    total_time=total_time,
                    total_speakers=result.metadata.total_speakers,
                    success=True,
                )

                return result

            except Exception as e:
                self.progress_tracker.add_error()
                error_msg = str(e)
                self.logger.error(
                    "pipeline_failed",
                    source=source_str,
                    error=error_msg,
                    traceback=traceback.format_exc(),
                )

                # Return error result
                result = create_empty_analysis_result(
                    filename=Path(source_str).name, duration=0.0
                )
                result.metadata.processing_time = time.time() - start_time

                return result

    def get_progress(self):
        """Get current pipeline progress"""
        return self.progress_tracker.get_progress()

    def get_resource_usage(self):
        """Get current resource usage"""
        return self.resource_manager.get_resource_usage()

    def cleanup(self):
        """Clean up pipeline resources"""
        try:
            # Clean up stage executor
            self.stage_executor.cleanup()

            # Clean up GPU memory
            self.resource_manager.cleanup_gpu_memory()

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            self.logger.info("pipeline_cleanup_completed")

        except Exception as e:
            self.logger.error("pipeline_cleanup_failed", error=str(e))

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()

        if exc_type is not None:
            self.logger.error(
                "pipeline_manager_exception",
                exception_type=str(exc_type),
                exception_message=str(exc_val),
            )
