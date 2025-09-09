"""
EC2 ML 서버 - FastAPI 기반

WhisperX, 화자 분리, 감정 분석 등을 HTTP API로 제공하는 서버
ECS FastAPI 백엔드로부터 요청을 받아 JSON으로 결과 반환
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import uuid
import requests
import tempfile
import boto3

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.callback_pipeline import process_video_with_fastapi
from src.api import AnalysisConfig
from src.utils.logger import get_logger

# FastAPI 앱 생성
app = FastAPI(
    title="ECG Model Server",
    description="EC2 ML 서버 - 비디오 오디오 분석 API",
    version="1.0.0"
)

# AWS S3 및 Backend 설정
s3_client = boto3.client('s3')
S3_BUCKET = "ecg-project-pipeline-dev-video-storage-np9digv7"
BACKEND_URL = "http://ecg-project-pipeline-dev-alb-1703405864.us-east-1.elb.amazonaws.com"

# In-memory job tracking
jobs = {}

# 로거 설정
logger = get_logger(__name__)


# Pydantic 모델들
class RequestProcessRequest(BaseModel):
    """비동기 비디오 분석 요청"""
    video_key: str  # S3 키
    
    # 분석 설정
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "auto"
    max_workers: int = 4

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


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    version: str
    gpu_available: bool
    gpu_count: int
    timestamp: str


@app.post("/request-process", response_model=RequestProcessResponse)
async def request_process(video_key: str, background_tasks: BackgroundTasks):
    """
    비동기 비디오 분석 요청 - Backend와 호환
    
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
        
        return RequestProcessResponse(
            job_id=job_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"분석 요청 실패 - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 요청 실패: {str(e)}"
        )


async def process_video_async(job_id: str, video_key: str):
    """
    백그라운드에서 비디오 분석 및 Backend로 결과 전송
    """
    try:
        logger.info(f"백그라운드 분석 시작 - job_id: {job_id}, video_key: {video_key}")
        
        # 임시 파일에 S3에서 비디오 다운로드
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            try:
                s3_client.download_file(S3_BUCKET, video_key, temp_file.name)
                logger.info(f"S3에서 비디오 다운로드 완료: {video_key}")
            except Exception as s3_error:
                logger.warning(f"S3 다운로드 실패, 모크 데이터 사용: {s3_error}")
            
            # 분석 설정 생성
            config = AnalysisConfig(
                enable_gpu=True,
                emotion_detection=True,
                language="auto",
                max_workers=4
            )
            
            try:
                # 실제 분석 실행
                from src.services.callback_pipeline import CallbackPipelineManager
                
                pipeline = CallbackPipelineManager(config, fastapi_client=None)
                
                result = await pipeline._run_pipeline_with_progress(
                    input_source=temp_file.name,
                    progress_callback=_dummy_progress_callback,
                    output_path=None
                )
                
                # 결과를 Backend 형식으로 변환
                backend_result = {
                    "job_id": job_id,
                    "video_key": video_key,
                    "status": "completed",
                    "success": True,
                    "processing_time": result.metadata.processing_time if result.metadata else 0,
                    "results": {
                        "transcript": [
                            {
                                "start": seg.start_time,
                                "end": seg.end_time,
                                "text": seg.text,
                                "speaker": seg.speaker_id
                            }
                            for seg in (result.transcript_segments or [])
                        ],
                        "emotions": [
                            {
                                "start": seg.start_time,
                                "end": seg.end_time,
                                "emotion": seg.emotion,
                                "confidence": seg.confidence
                            }
                            for seg in (result.emotion_segments or [])
                        ],
                        "metadata": {
                            "duration": result.metadata.video_duration if result.metadata else 0,
                            "speakers": result.metadata.total_speakers if result.metadata else 0,
                            "language": result.metadata.language_detected if result.metadata else "unknown"
                        }
                    }
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
                            {"start": 0.0, "end": 3.0, "text": "분석 실패 - 모크 데이터", "speaker": "SPEAKER_00"}
                        ],
                        "emotions": [],
                        "metadata": {"duration": 0, "speakers": 0, "language": "unknown"}
                    }
                }
            
            # 작업 상태 업데이트
            jobs[job_id] = {"status": "completed", "result": backend_result}
            
            # Backend로 결과 전송
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/upload-video/results",
                    json=backend_result,
                    timeout=30
                )
                logger.info(f"Backend로 결과 전송 완료 - job_id: {job_id}, status: {response.status_code}")
            except Exception as backend_error:
                logger.error(f"Backend 결과 전송 실패: {backend_error}")
            
            logger.info(f"분석 완료 - job_id: {job_id}")
            
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
        logger.error(f"분석 실패 - job_id: {job_id}, error: {str(e)}")


async def _dummy_progress_callback(stage: str, progress: float, message: str = None):
    """진행 상황 콜백 (로그만 출력)"""
    logger.info(f"Progress: {stage} - {progress:.1%} - {message or ''}")


@app.get("/health")
async def health_check():
    """ML 서버 헬스 체크 - Backend 호환"""
    return {"status": "healthy", "service": "model-server"}


@app.post("/process-video")
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
            "request-process": "POST /request-process - 비동기 비디오 분석",
            "process-video": "POST /process-video - 레거시 비디오 분석", 
            "health": "GET /health - 헬스 체크",
            "docs": "GET /docs - API 문서"
        }
    }


@app.post("/test")
async def test_endpoint():
    """테스트 엔드포인트"""
    return {
        "status": "ok",
        "message": "ML API 서버가 정상 작동중입니다",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG Audio Analyzer ML API Server")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트")
    parser.add_argument("--port", type=int, default=8080, help="바인드 포트") 
    parser.add_argument("--workers", type=int, default=1, help="워커 수")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"🚀 ECG Audio Analyzer ML API 서버 시작")
    logger.info(f"   호스트: {args.host}:{args.port}")
    logger.info(f"   워커 수: {args.workers}")
    logger.info(f"   로그 레벨: {args.log_level}")
    
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
        log_level=args.log_level
    )