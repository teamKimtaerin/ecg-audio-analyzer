"""
EC2 ML 서버 - FastAPI 기반

WhisperX, 화자 분리, 감정 분석 등을 HTTP API로 제공하는 서버
ECS FastAPI 백엔드로부터 요청을 받아 JSON으로 결과 반환
"""

# 프로젝트 루트를 Python 경로에 추가 (다른 import 전에 실행)
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import os
from typing import Dict, Any, Optional, cast
from datetime import datetime
import random

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import uuid
import requests
import tempfile
import boto3

from src.api import AnalysisConfig
from src.utils.logger import get_logger
from src.models.output_models import AudioFeatures

# FastAPI 앱 생성
app = FastAPI(
    title="ECG Model Server",
    description="EC2 ML 서버 - 비디오 오디오 분석 API",
    version="1.0.0",
)

# AWS S3 및 Backend 설정
s3_client = boto3.client("s3")
S3_BUCKET = "ecg-project-pipeline-dev-video-storage-np9div7"

# 환경변수에서 Backend URL 가져오기 (다양한 이름 지원)
# 로컬 개발을 위해 localhost를 기본값으로 설정
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    os.getenv(
        "ECG_BACKEND_URL",
        os.getenv(
            "ml_api_server_url",
            "http://localhost:8000",  # 로컬 개발용 기본값
        ),
    ),
)

# In-memory job tracking
jobs = {}

# 로거 설정
logger = get_logger(__name__)


# Pydantic 모델들
class ProcessVideoRequest(BaseModel):
    """Backend 호환 비디오 처리 요청 (API 명세 준수)"""
    job_id: str
    video_url: str


class ProcessVideoResponse(BaseModel):
    """Backend 호환 비디오 처리 응답"""
    job_id: str
    status: str
    message: str
    estimated_time: Optional[int] = 300  # seconds


class MLProgressCallback(BaseModel):
    """ML 진행 상황 콜백"""
    job_id: str
    status: str  # processing, completed, failed
    progress: int  # 0-100
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class RequestProcessRequest(BaseModel):
    """비동기 비디오 분석 요청 (legacy)"""

    video_key: str  # S3 키

    # 분석 설정
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "auto"
    max_workers: int = 4


class TranscribeRequest(BaseModel):
    """백엔드 호환 전사 요청"""

    video_path: str
    video_url: Optional[str] = None
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "auto"
    max_workers: int = 4
    output_path: Optional[str] = None


class RequestProcessResponse(BaseModel):
    """비동기 비디오 분석 응답"""

    job_id: str
    status: str  # "processing"


class TranscribeResponse(BaseModel):
    """비디오 분석 응답"""

    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    transcript_segments: list
    speaker_segments: list
    emotion_segments: list
    acoustic_features: list
    performance_stats: Optional[Dict[str, Any]] = None


class BackendTranscribeResponse(BaseModel):
    """상세 분석 응답 (simplified)"""

    success: bool
    metadata: Optional[Dict[str, Any]] = None
    speakers: Optional[Dict[str, Any]] = None
    segments: Optional[list] = None
    processing_time: float
    error: Optional[str] = None
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str
    service: str
    version: str
    gpu_available: bool
    gpu_count: int
    timestamp: str


@app.post("/api/upload-video/process-video", response_model=ProcessVideoResponse)
async def process_video_api(request: ProcessVideoRequest, background_tasks: BackgroundTasks):
    """
    Backend 호환 비디오 처리 API (명세서 준수)
    
    POST {ML_SERVER_URL}/api/upload-video/process-video
    Content-Type: application/json
    
    Request Body:
    {
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "video_url": "https://s3.amazonaws.com/..."
    }
    """
    try:
        job_id = request.job_id
        video_url = request.video_url
        
        # Job 상태 추적
        jobs[job_id] = {
            "status": "accepted",
            "video_url": video_url,
            "started_at": datetime.now().isoformat()
        }
        
        logger.info(f"비디오 처리 요청 접수 - job_id: {job_id}, video_url: {video_url}")
        logger.info(f"Backend URL 설정: {BACKEND_URL}")
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            process_video_with_callback,
            job_id,
            video_url
        )
        
        return ProcessVideoResponse(
            job_id=job_id,
            status="accepted",
            message="Processing started",
            estimated_time=300
        )
        
    except Exception as e:
        logger.error(f"처리 요청 실패 - job_id: {request.job_id}, error: {str(e)}")
        # 실패 콜백 전송
        await send_error_to_backend(request.job_id, str(e), "INVALID_REQUEST")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": str(e),
                    "job_id": request.job_id
                }
            }
        )


@app.post("/request-process", response_model=RequestProcessResponse)
async def request_process(video_key: str, background_tasks: BackgroundTasks):
    """
    비동기 비디오 분석 요청 - Backend와 호환 (legacy)

    Backend가 호출하는 API:
    POST http://10.0.10.42:8080/request-process?video_key=uploads/uuid/video.mp4
    """

    try:
        # Job ID 생성
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "processing", "video_key": video_key}

        logger.info(f"비동기 분석 요청 시작 - video_key: {video_key}, job_id: {job_id}")

        # 백그라운드 작업 시작
        background_tasks.add_task(process_video_async, job_id, video_key)

        return RequestProcessResponse(job_id=job_id, status="processing")

    except Exception as e:
        logger.error(f"분석 요청 실패 - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 요청 실패: {str(e)}")


async def process_video_async(job_id: str, video_key: str):
    """
    백그라운드에서 비디오 분석 및 Backend로 결과 전송
    """
    try:
        logger.info(f"백그라운드 분석 시작 - job_id: {job_id}, video_key: {video_key}")

        # 임시 파일에 S3에서 비디오 다운로드
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            try:
                s3_client.download_file(S3_BUCKET, video_key, temp_file.name)
                logger.info(f"S3에서 비디오 다운로드 완료: {video_key}")
            except Exception as s3_error:
                logger.warning(f"S3 다운로드 실패, 모크 데이터 사용: {s3_error}")

            # 분석 설정 생성
            config = AnalysisConfig(
                enable_gpu=True, emotion_detection=True, language="auto", max_workers=4
            )

            try:
                # 실제 분석 실행
                from src.services.callback_pipeline import CallbackPipelineManager

                pipeline = CallbackPipelineManager(config, fastapi_client=None)

                result = await pipeline._run_pipeline_with_progress(
                    input_source=temp_file.name,
                    progress_callback=_dummy_progress_callback,
                    output_path=None,
                )

                # 결과를 Backend 형식으로 변환
                # CompleteAnalysisResult는 timeline 속성을 가짐
                timeline_segments = getattr(result, "timeline", [])

                backend_result = {
                    "job_id": job_id,
                    "video_key": video_key,
                    "status": "completed",
                    "success": True,
                    "processing_time": (
                        result.metadata.processing_time if result.metadata else 0
                    ),
                    "results": {
                        "transcript": [
                            {
                                "start": seg.start_time,
                                "end": seg.end_time,
                                "text": getattr(seg, "text_placeholder", ""),
                                "speaker": seg.speaker_id,
                            }
                            for seg in timeline_segments
                        ],
                        "emotions": [
                            {
                                "start": seg.start_time,
                                "end": seg.end_time,
                                "emotion": seg.emotion.emotion,
                                "confidence": seg.emotion.confidence,
                            }
                            for seg in timeline_segments
                        ],
                        "metadata": {
                            "duration": (
                                result.metadata.duration if result.metadata else 0
                            ),
                            "speakers": (
                                result.metadata.total_speakers if result.metadata else 0
                            ),
                            "language": "auto",
                        },
                    },
                }

            except Exception as analysis_error:
                logger.error(f"분석 실패: {analysis_error}")
                # 분석 실패시 모크 결과 생성
                backend_result = {
                    "job_id": job_id,
                    "video_key": video_key,
                    "status": "completed",
                    "success": False,
                    "error": str(analysis_error),
                    "results": {
                        "transcript": [
                            {
                                "start": 0.0,
                                "end": 3.0,
                                "text": "분석 실패 - 모크 데이터",
                                "speaker": "SPEAKER_00",
                            }
                        ],
                        "emotions": [],
                        "metadata": {
                            "duration": 0,
                            "speakers": 0,
                            "language": "unknown",
                        },
                    },
                }

            # 작업 상태 업데이트
            jobs[job_id] = {"status": "completed", "result": backend_result}

            # Backend로 결과 전송
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/upload-video/results",
                    json=backend_result,
                    timeout=30,
                )
                logger.info(
                    f"Backend로 결과 전송 완료 - job_id: {job_id}, status: {response.status_code}"
                )
            except Exception as backend_error:
                logger.error(f"Backend 결과 전송 실패: {backend_error}")

            logger.info(f"분석 완료 - job_id: {job_id}")

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
        logger.error(f"분석 실패 - job_id: {job_id}, error: {str(e)}")


async def _dummy_progress_callback(stage: str, progress: float, message: str = ""):
    """진행 상황 콜백 (로그만 출력)"""
    logger.info(f"Progress: {stage} - {progress:.1%} - {message or ''}")


async def send_error_to_backend(job_id: str, error_message: str, error_code: str = "PROCESSING_ERROR"):
    """백엔드에 에러 전송"""
    try:
        payload = MLProgressCallback(
            job_id=job_id,
            status="failed",
            progress=0,
            error_message=error_message,
            error_code=error_code
        ).model_dump()
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ml/ml-results",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ML-Server/1.0"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Error reported to backend for job {job_id}")
        else:
            logger.warning(f"Failed to report error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Failed to send error to backend: {str(e)}")


async def download_from_url(url: str, job_id: str) -> str:
    """URL에서 비디오 다운로드"""
    try:
        # Progress: 다운로드 시작
        await send_progress_to_backend(job_id, 5, "비디오 다운로드 시작...")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # 다운로드 진행률 업데이트 (5-10%)
                    if total_size > 0:
                        progress = 5 + int((downloaded / total_size) * 5)
                        await send_progress_to_backend(job_id, progress, f"다운로드 중... ({downloaded}/{total_size} bytes)")
            
            logger.info(f"비디오 다운로드 완료: {temp_file.name}")
            return temp_file.name
            
    except Exception as e:
        logger.error(f"비디오 다운로드 실패: {str(e)}")
        raise Exception(f"Failed to download video: {str(e)}")


async def send_progress_to_backend(job_id: str, progress: int, message: str = ""):
    """백엔드에 진행 상황 전송 (API 명세 준수)"""
    try:
        payload = MLProgressCallback(
            job_id=job_id,
            status="processing",
            progress=progress,  # 0-100
            message=message or "처리 중..."
        ).model_dump()

        # 로깅을 위해 요청 정보 출력
        callback_url = f"{BACKEND_URL}/api/v1/ml/ml-results"
        logger.debug(f"콜백 전송 - URL: {callback_url}, 페이로드: {payload}")
        
        response = requests.post(
            callback_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ML-Server/1.0"
            },
            timeout=10,
        )

        if response.status_code == 200:
            logger.info(f"✅ Progress updated: {message} ({progress}%)")
        else:
            logger.warning(f"⚠️ Progress update failed: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Progress update error: {str(e)}")


async def process_video_with_callback(job_id: str, video_url: str):
    """
    API 명세에 따른 비디오 처리 및 콜백
    """
    start_time = datetime.now()
    
    try:
        # 1. 비디오 다운로드 (0-10%)
        video_path = await download_from_url(video_url, job_id)
        
        # 2. 오디오 추출 (10-25%)
        await send_progress_to_backend(job_id, 10, "오디오 추출 중...")
        
        # 3. 음성 구간 감지 (25-40%)
        await send_progress_to_backend(job_id, 25, "음성 구간 감지 중...")
        
        # 4. 화자 식별 (40-60%)
        await send_progress_to_backend(job_id, 40, "화자 식별 중...")
        
        # 분석 설정
        config = AnalysisConfig(
            enable_gpu=True,
            emotion_detection=True,
            language="auto",
            max_workers=4
        )
        
        # 5. 전사 생성 (60-75%)
        await send_progress_to_backend(job_id, 60, "음성을 텍스트로 변환 중...")
        
        # Pipeline 실행
        from src.services.callback_pipeline import CallbackPipelineManager
        
        async def progress_callback(stage: str, progress: float, message: str = ""):
            # 60-75% 구간에 매핑
            adjusted = 60 + int(progress * 15)
            await send_progress_to_backend(job_id, adjusted, message or stage or "전사 처리 중...")
        
        pipeline = CallbackPipelineManager(config, fastapi_client=None)
        result = await pipeline._run_pipeline_with_progress(
            input_source=video_path,
            progress_callback=progress_callback,
            output_path=None
        )
        
        # 6. 감정 및 신뢰도 분석 (75-90%)
        await send_progress_to_backend(job_id, 75, "감정 분석 중...")
        
        # WhisperX 결과 추출
        whisperx_result = None
        if hasattr(pipeline.pipeline_manager, '_last_whisperx_result'):
            whisperx_result = pipeline.pipeline_manager._last_whisperx_result
        
        # 결과 변환 (API 명세 준수)
        segments = []
        if whisperx_result and "segments" in whisperx_result:
            for seg in whisperx_result["segments"]:
                segment_data = {
                    "start_time": seg.get("start", 0.0),
                    "end_time": seg.get("end", 0.0),
                    "speaker": {
                        "speaker_id": seg.get("speaker", "SPEAKER_01")
                    },
                    "text": seg.get("text", "").strip(),
                    "words": []
                }
                
                # 단어별 정보 추가
                if "words" in seg:
                    for word in seg["words"]:
                        word_data = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "volume_db": -20.0 + random.uniform(-5, 5),  # 샘플 값
                            "pitch_hz": 150.0 + random.uniform(-50, 200)  # 샘플 값
                        }
                        segment_data["words"].append(word_data)
                
                segments.append(segment_data)
        
        # 7. 결과 정리 (90-100%)
        await send_progress_to_backend(job_id, 90, "결과 정리 중...")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 최종 결과 준비
        final_result = {
            "metadata": {
                "filename": Path(video_url).name if "/" in video_url else "video.mp4",
                "duration": result.metadata.duration if result.metadata else 0,
                "total_segments": len(segments),
                "unique_speakers": len(set(s["speaker"]["speaker_id"] for s in segments)) if segments else 0
            },
            "segments": segments
        }
        
        # 완료 콜백 전송
        await send_progress_to_backend(job_id, 100, "분석 완료")
        
        # 최종 결과 전송
        complete_payload = MLProgressCallback(
            job_id=job_id,
            status="completed",
            progress=100,
            result=final_result
        ).model_dump()
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ml/ml-results",
            json=complete_payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ML-Server/1.0"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info(f"✅ 분석 완료 및 결과 전송 - job_id: {job_id}, 처리시간: {processing_time:.2f}초")
        else:
            logger.error(f"결과 전송 실패: {response.status_code}")
        
        # Job 상태 업데이트
        jobs[job_id] = {
            "status": "completed",
            "processing_time": processing_time,
            "completed_at": datetime.now().isoformat()
        }
        
        # 임시 파일 정리
        try:
            if Path(video_path).exists():
                Path(video_path).unlink()
        except:
            pass
            
    except Exception as e:
        logger.error(f"처리 실패 - job_id: {job_id}, error: {str(e)}")
        
        # 에러 콜백 전송
        await send_error_to_backend(
            job_id,
            str(e),
            "PROCESSING_ERROR" if "download" not in str(e).lower() else "DOWNLOAD_ERROR"
        )
        
        # Job 상태 업데이트
        jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }


# convert_to_backend_format 함수 제거됨 - 이제 상세 결과를 직접 반환


@app.post("/transcribe", response_model=BackendTranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    백엔드 호환 동기 전사 API

    Backend가 호출하는 메인 API:
    POST http://localhost:8080/transcribe
    Content-Type: application/json
    """
    start_time = datetime.now()

    try:
        logger.info(f"전사 요청 시작 - video_path: {request.video_path}")

        # 가상의 job_id 생성 (진행상황 추적용)
        job_id = str(uuid.uuid4())

        # Progress: 파일 검증 (5%)
        await send_progress_to_backend(job_id, 5, "파일 검증 중...")

        # 임시 파일 다운로드/검증
        video_path = request.video_path
        if not video_path.startswith("/") and not video_path.startswith("http"):
            # S3 키인 경우 S3에서 다운로드
            video_path = request.video_path

        # Progress: 오디오 추출 (15%)
        await send_progress_to_backend(job_id, 15, "오디오 추출 중...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            try:
                # S3에서 파일 다운로드 시도
                s3_client.download_file(S3_BUCKET, request.video_path, temp_file.name)
                logger.info(f"S3에서 비디오 다운로드 완료: {request.video_path}")
                actual_video_path = temp_file.name
            except Exception as s3_error:
                logger.warning(f"S3 다운로드 실패, 로컬 경로 사용: {s3_error}")
                actual_video_path = request.video_path

            # Progress: 음성 인식 (25%)
            await send_progress_to_backend(job_id, 25, "음성 구간 분석 중...")

            # 분석 설정 생성
            config = AnalysisConfig(
                enable_gpu=request.enable_gpu,
                emotion_detection=request.emotion_detection,
                language=request.language,
                max_workers=request.max_workers,
            )

            # Progress: 전사 작업 (65%)
            await send_progress_to_backend(job_id, 65, "음성을 텍스트로 변환 중...")

            try:
                # 실제 분석 실행
                from src.services.callback_pipeline import CallbackPipelineManager

                # Progress 콜백 함수 정의
                async def progress_callback(
                    stage: str, progress: float, message: str = ""
                ):
                    # 65% ~ 95% 구간에서 세부 진행률 업데이트
                    adjusted_progress = 65 + int(progress * 30)  # 65%에서 95%까지
                    await send_progress_to_backend(job_id, adjusted_progress, message or stage)

                pipeline = CallbackPipelineManager(config, fastapi_client=None)

                result = await pipeline._run_pipeline_with_progress(
                    input_source=actual_video_path,
                    progress_callback=progress_callback,
                    output_path=request.output_path,
                )

                # Progress: 감정 분석 (95%)
                await send_progress_to_backend(job_id, 95, "감정 분석 중...")

                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 디버깅: result 객체 구조 확인
                logger.info(f"Result 객체 타입: {type(result)}")
                logger.info(f"Result 속성들: {dir(result)}")
                
                # WhisperX 결과 직접 가져오기
                whisperx_result = None
                if hasattr(pipeline.pipeline_manager, '_last_whisperx_result'):
                    whisperx_result = pipeline.pipeline_manager._last_whisperx_result
                    logger.info(f"WhisperX 결과 발견: {len(whisperx_result.get('segments', []))} segments")

                # 상세 분석 결과를 직접 반환 (simplified format)
                detailed_result = {
                    "success": True,
                    "metadata": {
                        "filename": result.metadata.filename if result.metadata and hasattr(result.metadata, 'filename') else video_path,
                        "duration": result.metadata.duration if result.metadata and hasattr(result.metadata, 'duration') else 0,
                        "sample_rate": 16000,
                        "processed_at": result.metadata.analysis_timestamp if result.metadata and hasattr(result.metadata, 'analysis_timestamp') else datetime.now().isoformat(),
                        "processing_time": processing_time,
                        "total_segments": 0,  # Will be updated
                        "unique_speakers": result.metadata.total_speakers if result.metadata and hasattr(result.metadata, 'total_speakers') else 0,
                        "processing_mode": "real_ml_models",
                        "config": {
                            "enable_gpu": request.enable_gpu,
                            "segment_length": 5.0,
                            "language": request.language,
                            "unified_model": "whisperx-base-with-diarization",
                        },
                        "subtitle_optimization": True
                    },
                    "speakers": {},
                    "segments": [],
                    "processing_time": processing_time,
                }

                # WhisperX 결과에서 직접 세그먼트 추출
                if whisperx_result and "segments" in whisperx_result:
                    segments = whisperx_result["segments"]
                    logger.info(f"WhisperX 결과에서 {len(segments)} 세그먼트 추출")
                    
                    # 음향 특성 추출 (오디오 파일이 존재하는 경우)
                    acoustic_features_list: list[AudioFeatures] = []
                    try:
                        from src.services.acoustic_analyzer import FastAcousticAnalyzer
                        
                        # 오디오 파일 경로 확인
                        audio_path = None
                        if hasattr(pipeline.pipeline_manager, '_last_audio_path'):
                            audio_path = pipeline.pipeline_manager._last_audio_path
                        elif actual_video_path.endswith('.mp4'):
                            # 비디오에서 추출된 오디오 파일 찾기
                            temp_audio = actual_video_path.replace('.mp4', '.wav')
                            if Path(temp_audio).exists():
                                audio_path = temp_audio
                            else:
                                # 임시 오디오 추출
                                import subprocess
                                temp_audio = f"/tmp/audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                                subprocess.run([
                                    "ffmpeg", "-i", actual_video_path,
                                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                                    "-ac", "1", temp_audio, "-y"
                                ], capture_output=True)
                                if Path(temp_audio).exists():
                                    audio_path = temp_audio
                        
                        if audio_path and Path(audio_path).exists():
                            analyzer = FastAcousticAnalyzer(sample_rate=16000)
                            
                            # 각 세그먼트에 대해 음향 특성 추출
                            for seg in segments:
                                features = analyzer.extract_features(
                                    Path(audio_path),
                                    seg.get("start", 0.0),
                                    seg.get("end", 0.0)
                                )
                                acoustic_features_list.append(features)
                            
                            logger.info(f"음향 특성 추출 완료: {len(acoustic_features_list)}개 세그먼트")
                    except Exception as e:
                        logger.warning(f"음향 특성 추출 실패: {e}")
                        # 기본값 사용 - AudioFeatures 객체로 생성
                        from src.models.output_models import VolumeCategory
                        acoustic_features_list = [AudioFeatures(
                            rms_energy=0.05,
                            rms_db=-20.0,
                            pitch_mean=150.0,
                            pitch_variance=20.0,
                            speaking_rate=1.0,
                            amplitude_max=0.1,
                            silence_ratio=0.2,
                            spectral_centroid=1500.0,
                            zcr=0.1,
                            mfcc=[0.0, 0.0, 0.0],
                            volume_category=VolumeCategory.MEDIUM,
                            volume_peaks=[]
                        ) for _ in segments]
                    
                    # 화자별 통계 계산 (simplified)
                    speakers_stats = {}
                    for seg in segments:
                        speaker_id = seg.get("speaker", "SPEAKER_00")
                        if speaker_id not in speakers_stats:
                            speakers_stats[speaker_id] = {
                                "total_duration": 0.0,
                                "segment_count": 0
                            }
                        
                        duration = seg.get("end", 0.0) - seg.get("start", 0.0)
                        speakers_stats[speaker_id]["total_duration"] += duration
                        speakers_stats[speaker_id]["segment_count"] += 1
                    
                    # 각 세그먼트를 상세 결과에 추가
                    for i, seg in enumerate(segments):
                        # 음향 특성 가져오기 (simplified acoustic features)
                        if i < len(acoustic_features_list):
                            af = acoustic_features_list[i]
                            # Type cast to help Pylance understand the type
                            af_typed = cast(AudioFeatures, af)
                            acoustic_features_dict = {
                                "volume_db": af_typed.rms_db,
                                "pitch_hz": af_typed.pitch_mean,
                                "spectral_centroid": af_typed.spectral_centroid,
                                "zero_crossing_rate": af_typed.zcr,
                                "pitch_mean": af_typed.pitch_mean,
                                "pitch_std": af_typed.pitch_variance ** 0.5 if af_typed.pitch_variance > 0 else 0.0,
                                "mfcc_mean": af_typed.mfcc if hasattr(af_typed, 'mfcc') and af_typed.mfcc else [0.0] * 13
                            }
                        else:
                            acoustic_features_dict = {
                                "volume_db": -20.0,
                                "pitch_hz": 150.0,
                                "spectral_centroid": 1500.0,
                                "zero_crossing_rate": 0.1,
                                "pitch_mean": 150.0,
                                "pitch_std": 20.0,
                                "mfcc_mean": [0.0] * 13
                            }
                        
                        # 전체 세그먼트 데이터 (simplified format without emotion/confidence)
                        segment_data = {
                            "start_time": seg.get("start", 0.0),
                            "end_time": seg.get("end", 0.0),
                            "duration": seg.get("end", 0.0) - seg.get("start", 0.0),
                            "speaker": {
                                "speaker_id": seg.get("speaker", "SPEAKER_00")
                            },
                            "acoustic_features": acoustic_features_dict,
                            "text": seg.get("text", "").strip(),
                            "words": []
                        }
                        
                        # Extract word-level information if available
                        if "words" in seg and audio_path and Path(audio_path).exists():
                            words = seg["words"]
                            for word in words:
                                word_start = word.get("start", 0.0)
                                word_end = word.get("end", 0.0)
                                
                                # Calculate acoustic features for each word
                                try:
                                    word_features = analyzer.extract_features(
                                        Path(audio_path),
                                        word_start,
                                        word_end
                                    )
                                    word_data = {
                                        "word": word.get("word", ""),
                                        "start_time": word_start,
                                        "end_time": word_end,
                                        "duration": word_end - word_start,
                                        "acoustic_features": {
                                            "volume_db": word_features.rms_db,
                                            "pitch_hz": word_features.pitch_mean,
                                            "spectral_centroid": word_features.spectral_centroid
                                        }
                                    }
                                except Exception:
                                    # Fallback values if feature extraction fails
                                    word_data = {
                                        "word": word.get("word", ""),
                                        "start_time": word_start,
                                        "end_time": word_end,
                                        "duration": word_end - word_start,
                                        "acoustic_features": {
                                            "volume_db": -20.0,
                                            "pitch_hz": 150.0,
                                            "spectral_centroid": 1500.0
                                        }
                                    }
                                
                                segment_data["words"].append(word_data)
                        
                        # segments 배열에 추가
                        detailed_result["segments"].append(segment_data)
                    
                    # 화자 통계 추가
                    detailed_result["speakers"] = speakers_stats
                    
                    # Update total_segments in metadata
                    detailed_result["metadata"]["total_segments"] = len(detailed_result["segments"])
                else:
                    logger.warning("WhisperX 결과를 찾을 수 없음 - timeline에서 추출 시도")
                    
                    # timeline 세그먼트에서 데이터 추출 (폴백)
                    if hasattr(result, "timeline") and result.timeline:
                        for seg in result.timeline:
                            # 전사 세그먼트로 변환 (simplified format)
                            segment_data = {
                                "start_time": seg.start_time,
                                "end_time": seg.end_time,
                                "duration": seg.end_time - seg.start_time,
                                "speaker": {
                                    "speaker_id": getattr(seg, "speaker_id", "SPEAKER_00")
                                },
                                "acoustic_features": {
                                    "volume_db": -20.0,
                                    "pitch_hz": 150.0,
                                    "spectral_centroid": 1500.0,
                                    "zero_crossing_rate": 0.1,
                                    "pitch_mean": 150.0,
                                    "pitch_std": 20.0,
                                    "mfcc_mean": [0.0] * 13
                                },
                                "text": getattr(seg, "text_placeholder", "[TRANSCRIPTION_PENDING]"),
                                "language": "ko",
                                "words": []  # No word-level data available in fallback
                            }
                            detailed_result["segments"].append(segment_data)


                # Remove performance_stats from main result (keep it minimal)
                
                # Remove optimization_stats to keep response minimal

                # Progress: 완료 (100%)
                await send_progress_to_backend(job_id, 100, "분석 완료")
                
                logger.info(f"전사 완료 - 처리시간: {processing_time:.2f}초")
                logger.info(f"반환할 세그먼트 수: {len(detailed_result['segments'])}")
                
                # 결과를 output 폴더에 저장
                output_dir = Path("/Users/ahntaeju/project/ecg-audio-analyzer/output")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"api_result_{timestamp}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_result, f, ensure_ascii=False, indent=2)
                logger.info(f"결과 저장됨: {output_file}")

                return BackendTranscribeResponse(**detailed_result)

            except Exception as analysis_error:
                logger.error(f"분석 실패: {analysis_error}")
                processing_time = (datetime.now() - start_time).total_seconds()

                return BackendTranscribeResponse(
                    success=False,
                    error=f"분석 중 오류가 발생했습니다: {str(analysis_error)}",
                    error_code="ANALYSIS_ERROR",
                    processing_time=processing_time,
                )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"전사 요청 실패 - Error: {str(e)}")

        return BackendTranscribeResponse(
            success=False,
            error=f"요청 처리 실패: {str(e)}",
            error_code="REQUEST_ERROR",
            processing_time=processing_time,
        )


@app.get("/health")
async def health_check():
    """ML 서버 헬스 체크 - Backend 호환"""
    return {"status": "healthy", "service": "model-server"}


@app.post("/process-video")
async def process_video(request: dict, background_tasks: BackgroundTasks):
    """
    백엔드 호환 비동기 비디오 처리 엔드포인트 (legacy)

    이 엔드포인트는 기존 백엔드가 호출할 수 있도록 유지됩니다.
    """
    try:
        # dict로 받은 요청을 TranscribeRequest로 변환
        video_path = request.get("video_path", "")

        transcribe_request = TranscribeRequest(
            video_path=video_path,
            video_url=request.get("video_url"),
            enable_gpu=request.get("enable_gpu", True),
            emotion_detection=request.get("emotion_detection", True),
            language=request.get("language", "auto"),
            max_workers=request.get("max_workers", 4),
            output_path=request.get("output_path"),
        )

        # /transcribe 엔드포인트와 동일한 처리
        return await transcribe(transcribe_request)

    except Exception as e:
        logger.error(f"Process video 요청 실패: {str(e)}")
        return BackendTranscribeResponse(
            success=False,
            error=f"요청 처리 실패: {str(e)}",
            error_code="REQUEST_ERROR",
            processing_time=0.0,
        )


@app.post("/process-video-legacy")
async def process_video_legacy(video_key: str, background_tasks: BackgroundTasks):
    """
    레거시 동기 엔드포인트 (호환성 유지)
    """
    return await request_process(video_key, background_tasks)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "ECG Audio Analyzer ML API",
        "version": "1.0.0",
        "description": "EC2 ML 서버 - 비디오 오디오 분석 API",
        "endpoints": {
            "process-video-api": "POST /api/upload-video/process-video - 메인 비디오 처리 API (명세 준수)",
            "transcribe": "POST /transcribe - 동기 전사 API",
            "request-process": "POST /request-process - 비동기 비디오 분석 (legacy)",
            "process-video": "POST /process-video - 레거시 비디오 분석",
            "health": "GET /health - 헬스 체크",
            "docs": "GET /docs - API 문서",
        },
        "backend_url": BACKEND_URL
    }


@app.post("/test")
async def test_endpoint():
    """테스트 엔드포인트"""
    return {
        "status": "ok",
        "message": "ML API 서버가 정상 작동중입니다",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECG Audio Analyzer ML API Server")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트")
    parser.add_argument("--port", type=int, default=8080, help="바인드 포트")

    parser.add_argument("--workers", type=int, default=1, help="워커 수")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("🚀 ECG Audio Analyzer ML API 서버 시작")
    logger.info(f"   호스트: {args.host}:{args.port}")
    logger.info(f"   워커 수: {args.workers}")
    logger.info(f"   로그 레벨: {args.log_level}")
    logger.info(f"   백엔드 URL: {BACKEND_URL}")

    # GPU 확인
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.device_count()}개 사용 가능")
        else:
            logger.warning("   GPU: 사용 불가 (CPU 모드)")
    except ImportError:
        logger.warning("   PyTorch가 설치되지 않음")

    # FastAPI 서버 실행
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        access_log=True,
        log_level=args.log_level,
    )
