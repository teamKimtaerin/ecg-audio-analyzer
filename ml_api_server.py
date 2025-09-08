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

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.callback_pipeline import process_video_with_fastapi
from src.api import AnalysisConfig
from src.utils.logger import get_logger

# FastAPI 앱 생성
app = FastAPI(
    title="ECG Audio Analyzer ML API",
    description="EC2 ML 서버 - 비디오 오디오 분석 API",
    version="1.0.0"
)

# 로거 설정
logger = get_logger(__name__)


# Pydantic 모델들
class TranscribeRequest(BaseModel):
    """비디오 분석 요청"""
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    
    # 분석 설정
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "auto"
    max_workers: int = 4
    
    # 출력 설정
    output_path: Optional[str] = None


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


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_video(request: TranscribeRequest):
    """
    비디오 오디오 분석 메인 엔드포인트
    
    ECS FastAPI 백엔드가 호출하는 API:
    POST http://ec2-ml-server:8001/transcribe
    {
        "video_path": "/path/to/video.mp4",
        "enable_gpu": true,
        "emotion_detection": true,
        "language": "auto"
    }
    """
    
    start_time = datetime.now()
    
    try:
        # 입력 검증
        if not request.video_path and not request.video_url:
            raise HTTPException(
                status_code=400,
                detail="video_path 또는 video_url 중 하나는 필수입니다"
            )
        
        input_source = request.video_path or request.video_url
        
        logger.info(f"비디오 분석 요청 수신 - Source: {input_source}")
        
        # 분석 설정 생성
        config = AnalysisConfig(
            enable_gpu=request.enable_gpu,
            emotion_detection=request.emotion_detection,
            language=request.language,
            max_workers=request.max_workers
        )
        
        # 분석 실행 (콜백 없이 직접 실행)
        from src.services.callback_pipeline import CallbackPipelineManager
        
        pipeline = CallbackPipelineManager(config, fastapi_client=None)  # 콜백 없음
        
        # 직접 분석 실행 (콜백 없는 버전)
        result = await pipeline._run_pipeline_with_progress(
            input_source=input_source,
            progress_callback=_dummy_progress_callback,
            output_path=request.output_path
        )
        
        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"비디오 분석 완료 - 처리 시간: {processing_time:.2f}초")
        
        # 응답 생성
        response = TranscribeResponse(
            success=True,
            processing_time=processing_time,
            metadata={
                "video_path": str(result.metadata.video_path) if result.metadata.video_path else None,
                "processing_time": result.metadata.processing_time,
                "analysis_timestamp": result.metadata.analysis_timestamp.isoformat() if result.metadata.analysis_timestamp else None,
                "total_speakers": result.metadata.total_speakers,
                "video_duration": result.metadata.video_duration,
                "audio_duration": result.metadata.audio_duration,
                "language_detected": result.metadata.language_detected
            },
            transcript_segments=[
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "speaker_id": seg.speaker_id
                }
                for seg in (result.transcript_segments or [])
            ],
            speaker_segments=[
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "speaker_id": seg.speaker_id,
                    "confidence": seg.confidence
                }
                for seg in (result.speaker_segments or [])
            ],
            emotion_segments=[
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "emotion": seg.emotion,
                    "confidence": seg.confidence,
                    "valence": seg.valence,
                    "arousal": seg.arousal
                }
                for seg in (result.emotion_segments or [])
            ],
            acoustic_features=[
                # 음향 특성 데이터 (필요시 추가)
            ],
            performance_stats={
                "cpu_time": result.performance_stats.cpu_time if result.performance_stats else None,
                "memory_peak_mb": result.performance_stats.memory_peak_mb if result.performance_stats else None,
                "gpu_time": result.performance_stats.gpu_time if result.performance_stats else None,
                "total_processing_time": result.performance_stats.total_processing_time if result.performance_stats else None
            } if result.performance_stats else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"비디오 분석 실패 - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"비디오 분석 실패: {str(e)}"
        )


async def _dummy_progress_callback(stage: str, progress: float, message: str = None):
    """진행 상황 콜백 (로그만 출력)"""
    logger.info(f"Progress: {stage} - {progress:.1%} - {message or ''}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ML 서버 헬스 체크"""
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        return HealthResponse(
            status="healthy",
            service="ECG Audio Analyzer ML API",
            version="1.0.0",
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"헬스 체크 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="헬스 체크 실패")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "ECG Audio Analyzer ML API",
        "version": "1.0.0", 
        "description": "EC2 ML 서버 - 비디오 오디오 분석 API",
        "endpoints": {
            "transcribe": "POST /transcribe - 비디오 오디오 분석",
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
    parser.add_argument("--port", type=int, default=8001, help="바인드 포트") 
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