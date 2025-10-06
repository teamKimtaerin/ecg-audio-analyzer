"""
Pipeline Manager - Simplified Service Orchestration
Single Responsibility: Coordinate analysis workflow efficiently
"""

import asyncio
import time
import traceback
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable

from ..services.audio_extractor import AudioExtractor
from ..models.speech_recognizer import WhisperXPipeline
from ..models.output_models import (
    CompleteAnalysisResult,
    AnalysisMetadata,
    PerformanceStats,
    create_empty_analysis_result,
)
from ..utils.logger import get_logger
from config.base_settings import BaseConfig, ProcessingConfig
from config.aws_settings import AWSConfig
from config.model_configs import SpeakerDiarizationConfig, WhisperXConfig


class PipelineManager:
    """
    Simplified pipeline orchestration manager.
    Single Responsibility: Coordinate analysis services efficiently.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        processing_config: ProcessingConfig,
        aws_config: Optional[AWSConfig] = None,
        speaker_config: Optional[SpeakerDiarizationConfig] = None,
        whisperx_config: Optional[WhisperXConfig] = None,
        language: str = "en",
        progress_callback: Optional[Callable] = None,
    ):
        self.base_config = base_config
        self.processing_config = processing_config
        self.aws_config = aws_config
        self.speaker_config = speaker_config or SpeakerDiarizationConfig()
        self.whisperx_config = whisperx_config or WhisperXConfig()
        self.language = language
        self.progress_callback = progress_callback

        self.logger = get_logger().bind_context(component="pipeline_manager")

        # Simple progress tracking
        self.current_stage = "initialization"
        self.completed_stages = []
        self.error_count = 0

        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=base_config.max_workers)

        # GPU availability check
        self.gpu_available = torch.cuda.is_available()

        # Service instances (lazy initialization)
        self._audio_extractor: Optional[AudioExtractor] = None
        self._whisperx_pipeline: Optional[WhisperXPipeline] = None

        # Store results for API access
        self._last_whisperx_result = None
        self._last_audio_path = None

        self.logger.info(
            "pipeline_manager_initialized",
            gpu_available=self.gpu_available,
            thread_workers=base_config.max_workers,
        )

    def _get_audio_extractor(self) -> AudioExtractor:
        """Get or create audio extractor instance"""
        if self._audio_extractor is None:
            self._audio_extractor = AudioExtractor(
                target_sr=self.base_config.sample_rate,
                temp_dir=self.base_config.temp_dir,
            )
        return self._audio_extractor

    def _get_whisperx_pipeline(self) -> WhisperXPipeline:
        """Get or create WhisperX pipeline instance"""
        if self._whisperx_pipeline is None:
            device = self.aws_config.cuda_device if self.aws_config else None
            hf_token = self.base_config.hugging_face_token

            self._whisperx_pipeline = WhisperXPipeline(
                model_size=self.whisperx_config.model_size,
                device=device or self.whisperx_config.device,
                compute_type=(
                    self.whisperx_config.compute_type
                    if device and device.startswith("cuda")
                    else "float32"
                ),
                language=self.whisperx_config.language or self.language,
                hf_auth_token=hf_token,
            )
        return self._whisperx_pipeline

    async def _update_progress(
        self,
        stage: str,
        progress_percent: Optional[int] = None,
        message: Optional[str] = None,
    ):
        """Update current stage"""
        self.current_stage = stage
        self.logger.info("progress_updated", stage=stage)

        # 콜백이 있으면 호출
        if self.progress_callback and progress_percent is not None:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_percent, message or stage)
                else:
                    self.progress_callback(progress_percent, message or stage)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def _mark_stage_completed(self, stage: str):
        """Mark stage as completed"""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
            self.logger.info("stage_completed", stage=stage)

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024

            gpu_memory_mb = 0.0
            if self.gpu_available:
                try:
                    device = torch.cuda.current_device()
                    gpu_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
                except Exception:
                    pass

            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "gpu_memory_mb": gpu_memory_mb,
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "gpu_memory_mb": 0.0}

    async def preload_models(self, enable_warmup: bool = True) -> Dict[str, Any]:
        """Preload WhisperX model by creating pipeline instance"""
        try:
            # Simply create WhisperX pipeline to preload models
            pipeline = self._get_whisperx_pipeline()
            if pipeline:
                self.logger.info("whisperx_model_preloaded")
                return {"status": "success", "models": {"whisperx_model": True}}
            else:
                return {
                    "status": "failed",
                    "error": "Failed to create WhisperX pipeline",
                }

        except Exception as e:
            self.logger.error("model_preloading_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    async def _execute_audio_extraction(self, source: Union[str, Path]):
        """Execute audio extraction stage"""
        self.logger.info("executing_audio_extraction", source=str(source))

        try:
            extractor = self._get_audio_extractor()

            # Run in thread pool for I/O intensive operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool, extractor.extract, source
            )

            if result.success:
                self.logger.info("audio_extraction_completed", duration=result.duration)
            else:
                self.logger.error("audio_extraction_failed", error=result.error)

            return result

        except Exception as e:
            self.logger.error("audio_extraction_stage_failed", error=str(e))
            from ..services.audio_extractor import AudioExtractionResult

            return AudioExtractionResult(
                success=False, error=str(e), output_path=None, duration=0.0
            )

    async def _execute_whisperx_pipeline(self, audio_path: Path) -> Dict[str, Any]:
        """Execute WhisperX integrated pipeline with language optimization"""
        # 언어 최적화 정보 로깅
        processing_mode = "auto-detect" if self.language == "auto" else "targeted"
        self.logger.info(
            "executing_whisperx_pipeline",
            audio_path=str(audio_path),
            language=self.language,
            processing_mode=processing_mode,
        )

        try:
            pipeline = self._get_whisperx_pipeline()

            # Create wrapper function with enhanced logging
            def whisperx_wrapper():
                return pipeline.transcribe(audio_path=audio_path)

            # Run WhisperX pipeline in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_pool, whisperx_wrapper)

            if result and "segments" in result:
                segments_count = len(result["segments"])
                speakers = {
                    seg.get("speaker")
                    for seg in result["segments"]
                    if seg.get("speaker")
                }

                self.logger.info(
                    "whisperx_pipeline_completed",
                    speakers=len(speakers),
                    segments=segments_count,
                    language_requested=self.language,
                )
            else:
                self.logger.error("whisperx_pipeline_no_result")
                result = {"segments": [], "language": self.language}

            return result

        except Exception as e:
            self.logger.error("whisperx_pipeline_failed", error=str(e))
            return {"segments": [], "language": self.language, "error": str(e)}

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
        resource_usage = self._get_resource_usage()
        performance_stats = PerformanceStats(
            gpu_utilization=0.0,
            peak_memory_mb=int(resource_usage["memory_mb"]),
            avg_processing_fps=0.0,
            bottleneck_stage="whisperx_pipeline",
        )

        # Create basic result with available data
        result = CompleteAnalysisResult(
            metadata=metadata,
            timeline=[],
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
                await self._update_progress("audio_extraction", 35, "오디오 추출 시작")
                extraction_result = await self._execute_audio_extraction(source)

                # Validate extraction result
                if extraction_result.error or not extraction_result.output_path:
                    error_msg = f"Audio extraction failed: {extraction_result.error or 'No output path'}"
                    self.logger.error("audio_extraction_failed", error=error_msg)
                    raise ValueError(error_msg)

                self._mark_stage_completed("audio_extraction")

                # Stage 2: WhisperX Pipeline
                await self._update_progress("whisperx_pipeline", 45, "음성 인식 시작")

                if extraction_result.output_path:
                    whisperx_result = await self._execute_whisperx_pipeline(
                        extraction_result.output_path
                    )
                else:
                    whisperx_result = {}

                # Validate WhisperX result
                if not whisperx_result or not whisperx_result.get("segments"):
                    error_msg = "Speech recognition failed: No audio segments found"
                    self.logger.error("whisperx_failed", error=error_msg)
                    raise ValueError(error_msg)

                self._mark_stage_completed("whisperx_pipeline")

                # Store results for API access
                self._last_whisperx_result = whisperx_result
                self._last_audio_path = (
                    str(extraction_result.output_path)
                    if extraction_result.output_path
                    else None
                )

                # Stage 3: Result synthesis
                await self._update_progress("result_synthesis", 65, "결과 합성 중")
                result = self._create_basic_result(
                    source, extraction_result, whisperx_result
                )
                self._mark_stage_completed("result_synthesis")

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
                self.error_count += 1
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
        total_stages = 3  # audio_extraction, whisperx_pipeline, result_synthesis
        progress_percentage = (len(self.completed_stages) / total_stages) * 100

        return {
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "progress_percentage": progress_percentage,
            "error_count": self.error_count,
        }

    def get_resource_usage(self):
        """Get current resource usage"""
        return self._get_resource_usage()

    def cleanup(self):
        """Clean up pipeline resources"""
        try:
            # Clean up GPU memory
            if self.gpu_available:
                torch.cuda.empty_cache()
                self.logger.debug("gpu_memory_cleared")

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
