"""
Stage Executor - Individual pipeline stage execution
Single Responsibility: Execute individual pipeline stages with proper error handling
"""

import asyncio
from pathlib import Path
from typing import Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from ..services.audio_extractor import AudioExtractor, AudioExtractionResult
from ..models.speech_recognizer import WhisperXPipeline
from ..utils.logger import get_logger


class StageExecutor:
    """Execute individual pipeline stages"""

    def __init__(
        self,
        base_config,
        processing_config,
        aws_config=None,
        language: str = "en",
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        self.base_config = base_config
        self.processing_config = processing_config
        self.aws_config = aws_config
        self.language = language
        self.logger = get_logger().bind_context(component="stage_executor")

        # Use provided thread pool or create new one
        self.thread_pool = thread_pool or ThreadPoolExecutor(
            max_workers=base_config.max_workers
        )
        self._own_thread_pool = thread_pool is None

        # Service instances (lazy initialization)
        self._audio_extractor: Optional[AudioExtractor] = None
        self._whisperx_pipeline: Optional[WhisperXPipeline] = None

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
            hf_token = getattr(self.base_config, "hugging_face_token", None)

            self._whisperx_pipeline = WhisperXPipeline(
                model_size="base",
                device=device,
                compute_type=(
                    "float16" if device and device.startswith("cuda") else "float32"
                ),
                language=self.language,
                hf_auth_token=hf_token,
            )
        return self._whisperx_pipeline

    async def execute_audio_extraction(
        self, source: Union[str, Path]
    ) -> AudioExtractionResult:
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
                self.logger.info(
                    "audio_extraction_completed",
                    duration=result.duration,
                )
            else:
                self.logger.error("audio_extraction_failed", error=result.error)

            return result

        except Exception as e:
            self.logger.error("audio_extraction_stage_failed", error=str(e))
            # Return failed result
            return AudioExtractionResult(
                success=False, error=str(e), output_path=None, duration=0.0
            )

    async def execute_whisperx_pipeline(
        self, audio_path: Path, resource_manager=None
    ) -> Dict[str, Any]:
        """Execute WhisperX integrated pipeline (transcription + speaker diarization)"""
        self.logger.info("executing_whisperx_pipeline", audio_path=str(audio_path))

        # Use GPU resource manager if provided
        if resource_manager:
            async with resource_manager.acquire_gpu_resource():
                return await self._run_whisperx_inference(audio_path)
        else:
            return await self._run_whisperx_inference(audio_path)

    async def _run_whisperx_inference(self, audio_path: Path) -> Dict[str, Any]:
        """Run WhisperX inference"""
        try:
            pipeline = self._get_whisperx_pipeline()

            # Create wrapper function with correct signature
            def whisperx_wrapper():
                return pipeline.process_audio_with_diarization(
                    audio_path=audio_path,
                    min_speakers=2,
                    max_speakers=4,
                    sample_rate=16000,
                )

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
                    language=result.get("language", "unknown"),
                )
            else:
                self.logger.error("whisperx_pipeline_no_result")
                result = {"segments": [], "language": self.language}

            return result

        except Exception as e:
            self.logger.error("whisperx_pipeline_failed", error=str(e))
            return {"segments": [], "language": self.language, "error": str(e)}

    def cleanup(self):
        """Clean up stage executor resources"""
        try:
            # Shutdown thread pool only if we own it
            if self._own_thread_pool and self.thread_pool:
                self.thread_pool.shutdown(wait=True)

            self.logger.info("stage_executor_cleanup_completed")

        except Exception as e:
            self.logger.error("stage_executor_cleanup_failed", error=str(e))
