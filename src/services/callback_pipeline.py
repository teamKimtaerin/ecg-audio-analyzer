"""
콜백 통합 파이프라인 래퍼

기존 PipelineManager를 확장하여 FastAPI 백엔드로 실시간 진행 상황과 결과를 전송
"""

import asyncio
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path

from ..pipeline.manager import PipelineManager
from ..models.output_models import CompleteAnalysisResult
from .callback_client import FastAPIClient, ProcessingStatus, get_fastapi_client
from ..utils.logger import get_logger
from config.base_settings import BaseConfig, ProcessingConfig


class CallbackPipelineManager:
    """FastAPI API 호출이 통합된 파이프라인 매니저"""

    def __init__(self, config: Any, fastapi_client: Optional[FastAPIClient] = None):
        """
        Args:
            config: 파이프라인 설정 (AnalysisConfig)
            fastapi_client: FastAPI 클라이언트 (None이면 전역 인스턴스 사용)
        """
        # BaseConfig와 ProcessingConfig 객체 생성
        base_config = BaseConfig()
        processing_config = ProcessingConfig()

        # AnalysisConfig의 설정들을 BaseConfig에 적용
        if hasattr(config, "max_workers"):
            base_config.max_workers = config.max_workers
        if hasattr(config, "sample_rate"):
            base_config.sample_rate = config.sample_rate
        if hasattr(config, "language"):
            self.language = config.language
        else:
            self.language = "en"  # 기본값

        # PipelineManager 초기화 (필수 파라미터 전달)
        self.pipeline_manager = PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            language=self.language,
        )
        self.fastapi_client = fastapi_client or get_fastapi_client()
        self.logger = get_logger(__name__)

    async def process_with_callbacks(
        self,
        input_source: Union[str, Path],
        job_id: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> CompleteAnalysisResult:
        """
        콜백과 함께 오디오 분석 파이프라인 실행

        Args:
            input_source: 분석할 비디오 파일 경로 또는 URL
            job_id: 작업 ID (콜백 식별용)
            output_path: 결과 저장 경로

        Returns:
            CompleteAnalysisResult: 분석 결과
        """

        try:
            # 시작 콜백 전송
            if self.fastapi_client:
                from .callback_client import CallbackPayload

                await self.fastapi_client.send_ml_results(
                    CallbackPayload(
                        job_id=job_id, status=ProcessingStatus.STARTED, progress=0.0
                    )
                )

            # 진행 상황 콜백 함수 정의
            async def progress_callback(
                stage: str, progress: float, message: Optional[str] = None
            ):
                """파이프라인 진행 상황을 FastAPI로 전송"""
                if self.fastapi_client:
                    await self.fastapi_client.send_progress_update(
                        job_id=job_id,
                        progress=progress,
                        message=f"[{stage}] {message or ''}",
                    )

                self.logger.info(
                    f"Job {job_id} - Stage: {stage}, Progress: {progress:.1%}"
                )

            # 파이프라인 실행 (진행 상황 콜백과 함께)
            result = await self._run_pipeline_with_progress(
                input_source=input_source,
                progress_callback=progress_callback,
                output_path=output_path,
            )

            # 완료 콜백 전송
            if self.fastapi_client:
                # 결과를 직렬화 가능한 형태로 변환
                serializable_result = self._serialize_result(result)

                await self.fastapi_client.send_completion(
                    job_id=job_id, results=serializable_result
                )

            self.logger.info(
                f"Job {job_id} 완료 - 총 처리 시간: {result.metadata.processing_time:.2f}초"
            )
            return result

        except Exception as e:
            error_message = f"파이프라인 실행 중 오류: {str(e)}"
            self.logger.error(f"Job {job_id} 실패 - {error_message}")

            # 오류 콜백 전송
            if self.fastapi_client:
                await self.fastapi_client.send_error(
                    job_id=job_id, error_message=error_message
                )

            raise

    async def _run_pipeline_with_progress(
        self,
        input_source: Union[str, Path],
        progress_callback: Callable,
        output_path: Optional[Union[str, Path]] = None,
    ) -> CompleteAnalysisResult:
        """진행 상황 콜백과 함께 파이프라인 실행"""

        # 1. 오디오 추출 단계 (0% ~ 10%)
        await progress_callback(
            "audio_extraction", 0.05, "비디오에서 오디오 추출 중..."
        )

        # 기존 파이프라인 매니저 사용
        # Convert output_path to Path if it's a string
        output_path_obj = (
            Path(output_path) if isinstance(output_path, str) else output_path
        )

        result = await self.pipeline_manager.process_single(
            source=input_source, output_path=output_path_obj
        )

        # 단계별 진행 상황 시뮬레이션 (실제로는 PipelineManager 내부에서 콜백을 받아야 함)
        stages = [
            ("audio_extraction", 0.10, "오디오 추출 완료"),
            ("speaker_diarization", 0.30, "화자 분리 진행 중"),
            ("speech_recognition", 0.60, "음성 인식 진행 중"),
            ("emotion_analysis", 0.80, "감정 분석 진행 중"),
            ("acoustic_analysis", 0.95, "음향 특성 분석 중"),
            ("finalization", 1.0, "분석 완료"),
        ]

        # 각 단계마다 진행 상황 전송 (실제로는 PipelineManager에서 호출되어야 함)
        for stage, progress, message in stages[1:]:  # 첫 번째는 이미 전송함
            await progress_callback(stage, progress, message)
            await asyncio.sleep(0.1)  # 짧은 딜레이

        return result

    def _serialize_result(self, result: CompleteAnalysisResult) -> Dict[str, Any]:
        """분석 결과를 JSON 직렬화 가능한 형태로 변환"""
        try:
            # CompleteAnalysisResult를 딕셔너리로 변환 (올바른 속성 사용)
            serialized = {
                "metadata": {
                    "filename": result.metadata.filename,
                    "processing_time": result.metadata.processing_time,
                    "analysis_timestamp": result.metadata.analysis_timestamp,  # 이미 문자열
                    "total_speakers": result.metadata.total_speakers,
                    "duration": result.metadata.duration,  # video_duration 대신 duration 사용
                    "gpu_acceleration": result.metadata.gpu_acceleration,
                },
                "transcript_segments": [],
                "speaker_segments": [],
                "emotion_segments": [],
                "acoustic_features": [],
                "performance_stats": (
                    {
                        "gpu_utilization": result.performance_stats.gpu_utilization,
                        "peak_memory_mb": result.performance_stats.peak_memory_mb,
                        "avg_processing_fps": result.performance_stats.avg_processing_fps,
                        "bottleneck_stage": result.performance_stats.bottleneck_stage,
                    }
                    if result.performance_stats
                    else None
                ),
            }

            # timeline에서 세그먼트 데이터 변환
            if result.timeline:
                for segment in result.timeline:
                    # 전사 세그먼트 추가
                    serialized["transcript_segments"].append(
                        {
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "text": segment.text_placeholder,
                            "confidence": segment.speaker_confidence,
                            "speaker_id": segment.speaker_id,
                        }
                    )

                    # 화자 세그먼트 추가
                    serialized["speaker_segments"].append(
                        {
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "speaker_id": segment.speaker_id,
                            "confidence": segment.speaker_confidence,
                        }
                    )

                    # 감정 세그먼트 추가
                    if segment.emotion:
                        serialized["emotion_segments"].append(
                            {
                                "start_time": segment.start_time,
                                "end_time": segment.end_time,
                                "emotion": segment.emotion.primary,
                                "confidence": segment.emotion.confidence,
                                "valence": segment.emotion.valence,
                                "arousal": segment.emotion.arousal,
                            }
                        )

                    # 음향 특성 추가
                    if segment.audio_features:
                        serialized["acoustic_features"].append(
                            {
                                "start_time": segment.start_time,
                                "end_time": segment.end_time,
                                "rms_energy": segment.audio_features.rms_energy,
                                "rms_db": segment.audio_features.rms_db,
                                "pitch_mean": segment.audio_features.pitch_mean,
                                "spectral_centroid": segment.audio_features.spectral_centroid,
                            }
                        )

            return serialized

        except Exception as e:
            self.logger.error(f"결과 직렬화 중 오류: {str(e)}")
            return {"error": "결과 직렬화 실패", "message": str(e)}


# 편의 함수들
async def process_video_with_fastapi(
    input_source: Union[str, Path],
    job_id: str,
    config: Any,
    fastapi_base_url: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> CompleteAnalysisResult:
    """
    비디오 분석을 FastAPI API 호출과 함께 실행하는 편의 함수

    Args:
        input_source: 비디오 파일 경로 또는 URL
        job_id: 작업 ID
        config: 파이프라인 설정
        fastapi_base_url: FastAPI 기본 URL (None이면 전역 클라이언트 사용)
        output_path: 결과 저장 경로

    Returns:
        CompleteAnalysisResult: 분석 결과
    """
    fastapi_client = None
    if fastapi_base_url:
        fastapi_client = FastAPIClient(fastapi_base_url)

    pipeline = CallbackPipelineManager(config, fastapi_client)
    return await pipeline.process_with_callbacks(input_source, job_id, output_path)
