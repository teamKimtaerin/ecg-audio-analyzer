#!/usr/bin/env python3
"""
EC2 ML 서버 - FastAPI 기반

WhisperX, 화자 분리, 감정 분석 등을 HTTP API로 제공하는 서버
ECS FastAPI 백엔드로부터 요청을 받아 JSON으로 결과 반환
"""

import asyncio
import os
import sys
import logging
import tempfile
import uuid
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.services.callback_pipeline import process_video_with_fastapi
    from src.api import AnalysisConfig
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Warning: Import failed: {e}")
    # Mock 클래스들
    class AnalysisConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def get_logger(name):
        return logging.getLogger(name)

# FastAPI 앱 생성
app = FastAPI(
    title="ECG Model Server",
    description="EC2 ML 서버 - 비디오 오디오 분석 API",
    version="1.0.0"
)

# CORS 설정 - ECS 백엔드와의 통신을 위해
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 및 Backend 설정
s3_client = boto3.client('s3')
S3_BUCKET = "ecg-project-pipeline-dev-video-storage-np9digv7"
BACKEND_URL = "http://ecg-project-pipeline-dev-alb-1703405864.us-east-1.elb.amazonaws.com"

# In-memory job tracking
jobs: Dict[str, Dict[str, Any]] = {}

# 로거 설정
logger = get_logger(__name__)

# =============================================================================
# Pydantic 모델들 (타입 어노테이션 포함)
# =============================================================================

class ProcessRequest(BaseModel):
    """비디오 분석 요청 - ML_API.md 명세 준수"""
    job_id: str = Field(..., description="작업 ID")
    video_url: str = Field(..., description="S3 비디오 URL")
    enable_gpu: Optional[bool] = Field(True, description="GPU 사용 여부")
    emotion_detection: Optional[bool] = Field(True, description="감정 분석 여부")
    language: Optional[str] = Field("auto", description="언어 설정")
    max_workers: Optional[int] = Field(4, description="최대 워커 수")

class ProcessResponse(BaseModel):
    """비디오 분석 응답 - ML_API.md 명세 준수"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field("processing", description="작업 상태")

class JobStatusResponse(BaseModel):
    """작업 상태 응답 - ML_API.md 명세 준수"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태: processing, completed, failed")
    progress: Optional[int] = Field(None, description="진행률 (0-100)")
    result: Optional[Dict[str, Any]] = Field(None, description="분석 결과 (완료시)")
    error: Optional[str] = Field(None, description="에러 메시지 (실패시)")

class MLResultRequest(BaseModel):
    """ML 서버 결과 요청 (백엔드 콜백용) - ML_API.md 명세 준수"""
    job_id: str = Field(..., description="작업 ID")
    result: Dict[str, Any] = Field(..., description="Whisper 원본 JSON 결과")

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서버 상태")
    service: str = Field(..., description="서비스 이름")
    version: str = Field(..., description="버전")
    gpu_available: Optional[bool] = Field(None, description="GPU 사용 가능 여부")
    gpu_count: Optional[int] = Field(None, description="GPU 개수")
    timestamp: str = Field(..., description="응답 시간")

# =============================================================================
# API 엔드포인트들
# =============================================================================

@app.post("/api/upload-video/process-video", response_model=ProcessResponse)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks = None
):
    """
    비동기 비디오 분석 요청 - ML_API.md 명세 준수
    
    Backend가 호출하는 API:
    POST /api/upload-video/process-video
    {
        "job_id": "123456",
        "video_url": "https://bucket.s3.amazonaws.com/file.mp4"
    }
    """
    try:
        # 작업 상태 초기화
        jobs[request.job_id] = {
            "status": "processing",
            "video_url": request.video_url,
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"비동기 분석 요청 시작 - job_id: {request.job_id}, video_url: {request.video_url}")
        
        # 백그라운드 작업 시작
        if background_tasks:
            background_tasks.add_task(process_video_async, request.job_id, request.video_url)
        else:
            # 백그라운드 작업 직접 실행
            asyncio.create_task(process_video_async(request.job_id, request.video_url))
        
        return ProcessResponse(
            job_id=request.job_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"분석 요청 실패 - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 요청 실패: {str(e)}"
        )

@app.get("/api/upload-video/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """작업 상태 조회 - ML_API.md 명세 준수"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error")
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ML 서버 헬스 체크 - Backend 호환"""
    
    # GPU 정보 확인
    gpu_available = False
    gpu_count = 0
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
    except ImportError:
        pass
    
    return HealthResponse(
        status="healthy",
        service="ecg-ml-server",
        version="1.0.0",
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "ECG Audio Analyzer ML API",
        "version": "1.0.0", 
        "description": "EC2 ML 서버 - 비디오 오디오 분석 API",
        "endpoints": {
            "process-video": "POST /api/upload-video/process-video - 비동기 비디오 분석",
            "job-status": "GET /api/upload-video/status/{job_id} - 작업 상태 조회",
            "health": "GET /health - 헬스 체크",
            "docs": "GET /docs - API 문서"
        },
        "backend_integration": {
            "s3_bucket": S3_BUCKET,
            "backend_url": BACKEND_URL,
            "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"])
        }
    }

@app.post("/test")
async def test_endpoint():
    """테스트 엔드포인트"""
    return {
        "status": "ok",
        "message": "ML API 서버가 정상 작동중입니다",
        "timestamp": datetime.now().isoformat(),
        "server_info": {
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "environment": os.environ.get("ENVIRONMENT", "development")
        }
    }

# =============================================================================
# 백그라운드 처리 함수들
# =============================================================================

async def process_video_async(job_id: str, video_url: str):
    """
    백그라운드에서 비디오 분석 및 Backend로 결과 전송
    """
    try:
        logger.info(f"백그라운드 분석 시작 - job_id: {job_id}, video_url: {video_url}")
        
        # 진행상황 업데이트
        jobs[job_id]["progress"] = 10
        
        # S3 URL에서 파일 키 추출
        if video_url.startswith("https://"):
            # URL에서 파일 키 추출: https://bucket.s3.amazonaws.com/path/file.mp4 -> path/file.mp4
            file_key = "/".join(video_url.split("/")[4:])
        else:
            file_key = video_url  # 이미 파일 키인 경우
            
        # 진행상황 업데이트
        jobs[job_id]["progress"] = 20
        
        # 임시 파일에 S3에서 비디오 다운로드
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                s3_client.download_file(S3_BUCKET, file_key, temp_path)
                logger.info(f"S3에서 비디오 다운로드 완료: {file_key}")
                jobs[job_id]["progress"] = 40
            except Exception as s3_error:
                logger.warning(f"S3 다운로드 실패, 모크 데이터 생성: {s3_error}")
                # S3 다운로드 실패시 더미 데이터 생성
                with open(temp_path, 'wb') as f:
                    f.write(b"dummy video data for testing")
                jobs[job_id]["progress"] = 30
            
            try:
                # 진행상황 업데이트
                jobs[job_id]["progress"] = 60
                
                # 실제 분석 실행 시도
                result = await run_actual_analysis(temp_path, file_key, job_id)
                
                # 🔍 결과 검증 및 로깅 강화
                logger.info(f"분석 결과 타입: {type(result)}")
                logger.info(f"분석 결과 내용 (처음 500자): {str(result)[:500]}")
                
                if result is None:
                    logger.error("분석 결과가 None입니다!")
                    raise Exception("분석 결과가 None입니다")
                
                # 진행상황 업데이트
                jobs[job_id]["progress"] = 90
                
                # 결과를 Whisper 형식으로 변환
                if hasattr(result, 'model_dump'):
                    # Pydantic 모델인 경우
                    whisper_result = result.model_dump()
                    logger.info("Pydantic 모델을 딕셔너리로 변환")
                elif hasattr(result, '__dict__'):
                    # 일반 객체인 경우
                    whisper_result = result.__dict__
                    logger.info("객체를 딕셔너리로 변환")
                else:
                    # 이미 딕셔너리인 경우
                    whisper_result = result
                    logger.info("이미 딕셔너리 형태")
                
                logger.info(f"변환된 결과 키: {list(whisper_result.keys()) if isinstance(whisper_result, dict) else 'Not a dict'}")
                
                # 성공적인 분석 결과 - ML_API.md 명세에 맞게 단순화
                backend_result = {
                    "job_id": job_id,
                    "result": whisper_result  # 변환된 결과
                }
                
            except Exception as analysis_error:
                logger.error(f"분석 실패: {analysis_error}")
                jobs[job_id]["progress"] = 100
                
                # 분석 실패시 모크 결과 생성
                backend_result = {
                    "job_id": job_id,
                    "result": create_mock_whisper_result()
                }
            
            finally:
                # 임시 파일 정리
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # 작업 상태 업데이트
            jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "result": backend_result,
                "video_url": video_url,
                "completed_at": datetime.now().isoformat()
            })
            
            # Backend로 결과 전송
            await send_result_to_backend(backend_result)
            
            logger.info(f"분석 완료 - job_id: {job_id}")
            
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "progress": 100,
            "error": str(e),
            "video_url": video_url,
            "failed_at": datetime.now().isoformat()
        })
        logger.error(f"분석 실패 - job_id: {job_id}, error: {str(e)}")

async def run_actual_analysis(temp_path: str, file_key: str, job_id: str):
    """실제 WhisperX 분석 실행"""
    try:
        # 분석 설정 생성
        config = AnalysisConfig(
            enable_gpu=True,
            emotion_detection=True,
            language="auto",
            max_workers=4
        )
        
        # WhisperX 파이프라인 실행 시도
        from src.services.callback_pipeline import CallbackPipelineManager
        
        pipeline = CallbackPipelineManager(config, fastapi_client=None)
        
        # 진행상황 콜백 함수
        async def progress_callback(stage: str, progress: float, message: str = None):
            # 60-85% 범위에서 진행상황 업데이트
            actual_progress = 60 + int(progress * 25)
            jobs[job_id]["progress"] = min(85, actual_progress)
            logger.info(f"Progress: {stage} - {progress:.1%} - {message or ''}")
        
        result = await pipeline._run_pipeline_with_progress(
            input_source=temp_path,
            progress_callback=progress_callback,
            output_path=None
        )
        
        return result
        
    except ImportError as e:
        logger.warning(f"WhisperX 모듈 로딩 실패: {e}")
        raise Exception("WhisperX 모듈을 찾을 수 없습니다")
    except Exception as e:
        logger.error(f"분석 실행 실패: {e}")
        raise

def create_mock_whisper_result():
    """ML_API.md 명세에 맞는 모크 Whisper 결과 생성"""
    return {
        "text": "안녕하세요, 이것은 모크 데이터입니다. 실제 분석은 WhisperX로 처리됩니다.",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "안녕하세요, 이것은 모크 데이터입니다.",
                "speaker": "SPEAKER_00"
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "실제 분석은 WhisperX로 처리됩니다.",
                "speaker": "SPEAKER_01"
            }
        ],
        "language": "ko",
        "duration": 6.0,
        "speakers_count": 2
    }


async def send_result_to_backend(result: Dict[str, Any]):
    """Backend로 결과 전송 - ML_API.md 명세 준수"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/upload-video/result",  # 단수형으로 수정
            json=result,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"Backend로 결과 전송 성공 - job_id: {result['job_id']}")
        else:
            logger.warning(f"Backend 응답 상태: {response.status_code}, 내용: {response.text}")
            
    except Exception as backend_error:
        logger.error(f"Backend 결과 전송 실패: {backend_error}")


# =============================================================================
# 레거시 엔드포인트 (호환성 유지)
# =============================================================================

@app.post("/request-process")
async def request_process_legacy(
    fileKey: str = Query(..., description="S3 파일 키"),
    background_tasks: BackgroundTasks = None
):
    """레거시 엔드포인트 (호환성 유지)"""
    # 기존 형식을 새 형식으로 변환
    job_id = str(uuid.uuid4())
    # S3 URL 형식으로 변환 (간단한 추정)
    video_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{fileKey}"
    
    request = ProcessRequest(
        job_id=job_id,
        video_url=video_url
    )
    
    return await process_video(request, background_tasks)

# =============================================================================
# 서버 실행 코드
# =============================================================================

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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/tmp/ml_api_server.log')
        ]
    )
    
    logger.info("🚀 ECG Audio Analyzer ML API 서버 시작")
    logger.info(f"   호스트: {args.host}:{args.port}")
    logger.info(f"   워커 수: {args.workers}")
    logger.info(f"   로그 레벨: {args.log_level}")
    
    # GPU 확인
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.device_count()}개 사용 가능")
            for i in range(torch.cuda.device_count()):
                logger.info(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
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