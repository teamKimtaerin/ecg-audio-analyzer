import sys
from pathlib import Path
import logging
import os
import warnings
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import uuid
import re

from dotenv import load_dotenv
load_dotenv()
# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
import boto3

from src.utils.logger import get_logger
from src.pipeline.manager import PipelineManager
from config.base_settings import BaseConfig, ProcessingConfig

# FastAPI 앱 생성
app = FastAPI(
    title="ECG Model Server",
    description="EC2 ML 서버 - 비디오 오디오 분석 API",
    version="1.0.0",
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # 프론트엔드
        "http://localhost:8000",  # 백엔드
        "http://ecg-project-pipeline-dev-alb-1703405864.us-east-1.elb.amazonaws.com",  # Fargate 백엔드
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 및 Backend 설정
s3_client = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", os.getenv("S3_BUCKET", ""))
# 프로덕션 URL 우선, 없으면 개발 URL 사용
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "")  # https://ho-it.site
BACKEND_URL = os.getenv("BACKEND_URL", "")  # Fargate 개발 URL
DEFAULT_CALLBACK_URL = FASTAPI_BASE_URL or BACKEND_URL
ENABLE_CALLBACKS = bool(DEFAULT_CALLBACK_URL and DEFAULT_CALLBACK_URL.strip())

# In-memory job tracking
jobs = {}

# 중복 콜백 방지를 위한 완료 작업 추적
completed_jobs = set()
failed_jobs = set()

# 로거 설정
logger = get_logger(__name__)


# ========== Pydantic Models ==========


class ProcessVideoRequest(BaseModel):
    """Backend 호환 비디오 처리 요청"""

    job_id: str
    video_url: str
    # 백엔드에서 보내는 추가 필드들
    fastapi_base_url: Optional[str] = None  # 동적 콜백 URL
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "auto"
    max_workers: int = 4


class ProcessVideoResponse(BaseModel):
    """Backend 호환 비디오 처리 응답"""

    job_id: str
    status: str
    message: str
    status_url: Optional[str] = None  # 추가: 상태 조회 URL
    estimated_time: Optional[int] = 300


class MLProgressCallback(BaseModel):
    """ML 진행 상황 콜백"""

    job_id: str
    status: str  # processing, completed, failed
    progress: int  # 0-100
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class TranscribeRequest(BaseModel):
    """백엔드 호환 전사 요청"""

    video_path: str  # Deprecated: 하위 호환성을 위해 유지
    audio_path: Optional[str] = None  # 새로운 필드: S3 오디오 파일 경로
    video_url: Optional[str] = None
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "en"


class BackendTranscribeResponse(BaseModel):
    """상세 분석 응답"""

    success: bool
    metadata: Optional[Dict[str, Any]] = None
    speakers: Optional[Dict[str, Any]] = None
    segments: Optional[list] = None
    processing_time: float
    error: Optional[str] = None
    error_code: Optional[str] = None


# ========== Helper Functions ==========


def create_pipeline(language: str = "en") -> PipelineManager:
    return PipelineManager(
        base_config=BaseConfig(),
        processing_config=ProcessingConfig(),
        language=language,
    )


async def send_callback(
    job_id: str,
    status: str,
    progress: int,
    message: str = "",
    result: Optional[Dict] = None,
    error_message: Optional[str] = None,
    error_code: Optional[str] = None,
    callback_base_url: Optional[str] = None,
):
    """Send progress/error callback to backend"""
    # 중복 콜백 방지 체크
    if status == "completed" and job_id in completed_jobs:
        logger.debug(f"이미 완료된 작업 콜백 스킵: {job_id}")
        return
    elif status == "failed" and job_id in failed_jobs:
        logger.debug(f"이미 실패한 작업 콜백 스킵: {job_id}")
        return

    # 동적 콜백 URL 결정 (우선순위: 요청 제공 URL > FASTAPI_BASE_URL > BACKEND_URL)
    base_url = callback_base_url or DEFAULT_CALLBACK_URL
    if not base_url or not base_url.strip():
        logger.debug(f"No callback URL configured - {status}: {message} ({progress}%)")
        return

    try:
        payload = MLProgressCallback(
            job_id=job_id,
            status=status,
            progress=progress,
            message=message or "처리 중...",
            result=result,
            error_message=error_message,
            error_code=error_code,
        ).model_dump()

        # 백엔드 API 경로에 맞춰 수정 (기존: /api/v1/ml/ml-results)
        callback_endpoint = f"{base_url}/api/upload-video/result"

        response = requests.post(
            callback_endpoint,
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ML-Server/1.0"},
            timeout=30 if result else 10,
        )

        if response.status_code == 200:
            # 백엔드 응답 처리
            try:
                result_data = response.json()
                if result_data.get("status") == "ignored":
                    logger.info(
                        f"콜백이 무시됨 - Job ID: {job_id}, "
                        f"이유: {result_data.get('reason', 'unknown')}"
                    )
                    # 이미 완료/실패한 작업일 수 있으므로 상태 추적에 추가
                    if status == "completed":
                        completed_jobs.add(job_id)
                    elif status == "failed":
                        failed_jobs.add(job_id)
                else:
                    logger.info(
                        f"Callback sent to {callback_endpoint}: {status} - {message} ({progress}%)"
                    )
                    # 성공적으로 처리된 완료/실패 상태 추적
                    if status == "completed":
                        completed_jobs.add(job_id)
                    elif status == "failed":
                        failed_jobs.add(job_id)
            except:
                # JSON 파싱 실패 시 기본 로깅
                logger.info(
                    f"Callback sent to {callback_endpoint}: {status} - {message} ({progress}%)"
                )
        elif response.status_code == 422:
            # 백엔드 검증 실패 처리
            try:
                error_detail = response.json()
                if "Frontend API Misuse" in str(error_detail.get("detail", {})):
                    logger.warning("백엔드가 프론트엔드 오용을 감지함 - ML 서버는 정상")
                else:
                    logger.error(f"콜백 데이터 검증 실패 - Job ID: {job_id}, Error: {error_detail}")
            except:
                logger.error(f"422 에러 - Job ID: {job_id}, Response: {response.text}")
        else:
            logger.warning(
                f"Callback failed to {callback_endpoint}: {response.status_code} - {response.text}"
            )

    except Exception as e:
        logger.error(f"Callback error to {base_url}: {str(e)}")


async def download_from_url(
    url: str, job_id: str, callback_base_url: Optional[str] = None
) -> str:
    """URL에서 비디오 다운로드 또는 로컬 파일 확인"""
    try:
        # 로컬 파일 경로인지 확인
        local_path = Path(url)
        if local_path.exists():
            logger.info(f"로컬 파일 사용: {url}")
            await send_callback(
                job_id,
                "processing",
                10,
                "로컬 파일 확인 완료",
                callback_base_url=callback_base_url,
            )
            return str(local_path.absolute())

        # URL 다운로드
        await send_callback(
            job_id,
            "processing",
            5,
            "비디오 다운로드 시작...",
            callback_base_url=callback_base_url,
        )

        # S3 URL인지 확인하고 boto3로 다운로드
        if url.startswith("https://") and ".s3." in url:
            # S3 URL 파싱: https://bucket.s3.region.amazonaws.com/key
            s3_pattern = r"https://([^.]+)\.s3\.([^.]+)\.amazonaws\.com/(.+)"
            match = re.match(s3_pattern, url)
            
            if match:
                bucket_name = match.group(1)
                key = match.group(3)
                
                logger.info(f"S3에서 다운로드 시작: s3://{bucket_name}/{key}")
                
                # 임시 파일 생성 (with 블록 밖에서)
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.close()  # 파일 핸들 닫기
                
                # boto3로 S3에서 다운로드
                s3_client.download_file(bucket_name, key, temp_file.name)
                
                await send_callback(
                    job_id,
                    "processing",
                    10,
                    "S3에서 다운로드 완료",
                    callback_base_url=callback_base_url,
                )
                
                logger.info(f"S3 다운로드 완료: {temp_file.name}")
                return temp_file.name

        # 일반 HTTP URL 처리
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = 5 + int((downloaded / total_size) * 5)
                        await send_callback(
                            job_id,
                            "processing",
                            progress,
                            f"다운로드 중... ({downloaded}/{total_size} bytes)",
                            callback_base_url=callback_base_url,
                        )

            logger.info(f"HTTP 다운로드 완료: {temp_file.name}")
            return temp_file.name

    except Exception as e:
        logger.error(f"비디오 다운로드/파일 확인 실패: {str(e)}")
        raise Exception(f"Failed to access video: {str(e)}")


def get_default_acoustic_features() -> Dict[str, Any]:
    """기본 음향 특성 반환"""
    return {
        "volume_db": -20.0,
        "pitch_hz": 150.0,
        "spectral_centroid": 1500.0,
        "zero_crossing_rate": 0.1,
        "pitch_mean": 150.0,
        "pitch_std": 20.0,
        "mfcc_mean": [0.0] * 13,
    }


def extract_acoustic_features(audio_path: Path, segments: list) -> list[Dict[str, Any]]:
    """Extract acoustic features for segments"""
    acoustic_features_list = []

    try:
        from src.services.acoustic_analyzer import FastAcousticAnalyzer

        analyzer = FastAcousticAnalyzer(sample_rate=16000)

        for seg in segments:
            features = analyzer.extract_features(
                audio_path,
                seg.get("start", 0.0),
                seg.get("end", 0.0),
            )

            acoustic_features_list.append(
                {
                    "volume_db": features.rms_db,
                    "pitch_hz": features.pitch_mean,
                    "spectral_centroid": features.spectral_centroid,
                    "zero_crossing_rate": features.zcr,
                    "pitch_mean": features.pitch_mean,
                    "pitch_std": (
                        features.pitch_variance**0.5
                        if features.pitch_variance > 0
                        else 0.0
                    ),
                    "mfcc_mean": (
                        features.mfcc
                        if hasattr(features, "mfcc") and features.mfcc
                        else [0.0] * 13
                    ),
                }
            )

        logger.info(f"음향 특성 추출 완료: {len(acoustic_features_list)}개 세그먼트")

    except Exception as e:
        logger.warning(f"음향 특성 추출 실패: {e}")
        # Return default features for all segments
        acoustic_features_list = [get_default_acoustic_features() for _ in segments]

    return acoustic_features_list


def process_whisperx_segments(
    whisperx_result: Optional[Dict], audio_path: Optional[Path] = None
) -> tuple[list, dict]:
    """WhisperX 결과를 통합 처리"""
    if not whisperx_result or "segments" not in whisperx_result:
        logger.warning("WhisperX 결과가 없거나 세그먼트를 찾을 수 없음")
        return [], {}

    segments = whisperx_result["segments"]

    # DEBUG: 첫 번째 세그먼트 구조 로깅
    if segments:
        first_segment = segments[0]
        logger.debug(f"🔍 DEBUG: 첫 번째 세그먼트 키들: {list(first_segment.keys())}")
        logger.debug(f"🔍 DEBUG: 첫 번째 세그먼트 내용: {first_segment}")
        if "words" in first_segment and first_segment["words"]:
            first_word = first_segment["words"][0]
            logger.debug(f"🔍 DEBUG: 첫 번째 단어 키들: {list(first_word.keys())}")
            logger.debug(f"🔍 DEBUG: 첫 번째 단어 내용: {first_word}")

    # 음향 특성 추출
    acoustic_features_list = []
    if audio_path and audio_path.exists():
        acoustic_features_list = extract_acoustic_features(audio_path, segments)
    else:
        acoustic_features_list = [get_default_acoustic_features() for _ in segments]

    # 세그먼트 처리
    processed_segments = []
    speakers_stats = {}

    for i, seg in enumerate(segments):
        speaker_id = seg.get("speaker", "SPEAKER_00")

        # Update speaker stats
        if speaker_id not in speakers_stats:
            speakers_stats[speaker_id] = {
                "total_duration": 0.0,
                "segment_count": 0,
            }

        duration = seg.get("end", 0.0) - seg.get("start", 0.0)
        speakers_stats[speaker_id]["total_duration"] += duration
        speakers_stats[speaker_id]["segment_count"] += 1

        # Extract timing information from segment
        start_time = seg.get("start", 0.0) if "start" in seg else seg.get("start_time", 0.0)
        end_time = seg.get("end", 0.0) if "end" in seg else seg.get("end_time", 0.0)

        # 디버그 로깅: 타임스탬프 확인
        logger.debug(f"🔍 세그먼트 {i}: start={start_time}, end={end_time}, keys={list(seg.keys())}")

        # Build segment data
        segment_data = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time if end_time > start_time else 0.0,
            "speaker_id": speaker_id,
            "acoustic_features": (
                acoustic_features_list[i]
                if i < len(acoustic_features_list)
                else get_default_acoustic_features()
            ),
            "text": seg.get("text", "").strip(),
            "words": [],
        }

        # Process words if available
        if "words" in seg:
            for word_idx, word in enumerate(seg["words"]):
                # Extract timing information from word
                word_start = word.get("start", 0.0) if "start" in word else word.get("start_time", 0.0)
                word_end = word.get("end", 0.0) if "end" in word else word.get("end_time", 0.0)

                # 디버그 로깅: 단어 타임스탬프 확인
                logger.debug(f"🔍 단어 {word_idx}: start={word_start}, end={word_end}, keys={list(word.keys())}")

                word_data = {
                    "word": word.get("word", ""),
                    "start_time": word_start,
                    "end_time": word_end,
                    "duration": word_end - word_start if word_end > word_start else 0.0,
                    "acoustic_features": {
                        "volume_db": -20.0,
                        "pitch_hz": 150.0,
                        "spectral_centroid": 1500.0,
                    },
                }
                segment_data["words"].append(word_data)

        processed_segments.append(segment_data)

    # 타임스탬프 검증 및 통계
    validate_timestamps(processed_segments)

    return processed_segments, speakers_stats


def validate_timestamps(segments: list) -> None:
    """타임스탬프 검증 및 통계 출력"""
    total_segments = len(segments)
    zero_timestamp_segments = 0
    valid_timestamp_segments = 0

    for i, segment in enumerate(segments):
        start_time = segment.get("start_time", 0.0)
        end_time = segment.get("end_time", 0.0)

        if start_time == 0.0 and end_time == 0.0:
            zero_timestamp_segments += 1
        elif start_time < end_time:
            valid_timestamp_segments += 1

        # 첫 3개와 마지막 3개 세그먼트의 타임스탬프 로깅
        if i < 3 or i >= total_segments - 3:
            logger.info(f"✅ 세그먼트 {i}: {start_time:.2f}s - {end_time:.2f}s | '{segment.get('text', '')[:50]}...'")

    logger.info(f"📊 타임스탬프 통계: 전체={total_segments}, 유효={valid_timestamp_segments}, 0값={zero_timestamp_segments}")

    if zero_timestamp_segments > 0:
        logger.warning(f"⚠️ {zero_timestamp_segments}/{total_segments} 세그먼트에서 타임스탬프가 0입니다!")
    else:
        logger.info("✅ 모든 세그먼트의 타임스탬프가 유효합니다!")


async def process_audio_core(file_path: str, language: str = "en") -> Dict[str, Any]:
    import time
    start_time = time.time()

    # 언어 최적화 모드 결정
    processing_mode = "targeted" if language != "auto" else "auto-detect"
    logger.info(f"🎯 처리 모드: {processing_mode} (언어: {language})")

    # Pipeline 실행
    file_path_obj = Path(file_path)

    # 파일 타입에 따른 처리 방식 결정
    is_audio_file = file_path_obj.suffix.lower() in [
        ".wav",
        ".mp3",
        ".flac",
        ".aac",
        ".m4a",
    ]
    if is_audio_file:
        logger.info("📄 오디오 파일 직접 처리")
    else:
        logger.info("🎬 비디오 파일에서 오디오 추출 후 처리")

    pipeline = create_pipeline(language=language)

    result = await pipeline.process_single(
        source=file_path_obj,
        output_path=None,
    )
    # 처리 시간 측정
    processing_time = time.time() - start_time

    # WhisperX 결과 추출
    whisperx_result = None
    detected_language = language  # 기본값
    if hasattr(pipeline, "_last_whisperx_result"):
        whisperx_result = pipeline._last_whisperx_result

        # 실제 감지된 언어 정보 추출
        if whisperx_result and "language" in whisperx_result:
            detected_language = whisperx_result["language"]
    # 오디오 경로 가져오기
    audio_path = None
    if hasattr(pipeline, "_last_audio_path") and pipeline._last_audio_path:
        audio_path = Path(pipeline._last_audio_path)

    # 세그먼트 처리
    processed_segments, speakers_stats = process_whisperx_segments(
        whisperx_result, audio_path
    )
    logger.info(
        f"✅ 세그먼트 처리 완료: {len(processed_segments)}개 세그먼트, {len(speakers_stats)}명 화자"
    )

    # 메타데이터 생성 (언어 최적화 정보 포함)
    metadata = {
        "filename": Path(file_path).name,
        "duration": result.metadata.duration if result.metadata else 0,
        "total_segments": len(processed_segments),
        "unique_speakers": len(speakers_stats),
        "processing_time": processing_time,
        "language_requested": language,
        "language_detected": detected_language,
        "processing_mode": processing_mode,
    }

    return {
        "metadata": metadata,
        "segments": processed_segments,
        "speakers": speakers_stats,
        "whisperx_result": whisperx_result,
    }


# ========== API Endpoints ==========


@app.post("/api/upload-video/process-video", response_model=ProcessVideoResponse)
async def process_video_api(
    request: ProcessVideoRequest, background_tasks: BackgroundTasks
):
    """Backend 호환 비디오 처리 API"""
    try:
        job_id = request.job_id

        # Job 상태 추적
        jobs[job_id] = {
            "status": "accepted",
            "video_url": request.video_url,
            "started_at": datetime.now().isoformat(),
        }

        logger.info(f"비디오 처리 요청 접수 - job_id: {job_id}")

        # 백그라운드 작업 시작 (추가 파라미터와 함께)
        background_tasks.add_task(
            process_video_with_callback,
            job_id,
            request.video_url,
            request.fastapi_base_url,
            request.language,
        )

        return ProcessVideoResponse(
            job_id=job_id,
            status="processing",  # "accepted" → "processing"으로 변경
            message="비디오 처리가 시작되었습니다",
            status_url=f"/api/upload-video/status/{job_id}",  # 추가
            estimated_time=300,
        )

    except Exception as e:
        logger.error(f"처리 요청 실패 - job_id: {request.job_id}, error: {str(e)}")
        await send_callback(
            request.job_id,
            "failed",
            0,
            error_message=str(e),
            error_code="INVALID_REQUEST",
            callback_base_url=request.fastapi_base_url,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": str(e),
                    "job_id": request.job_id,
                }
            },
        )


async def process_video_with_callback(
    job_id: str,
    video_url: str,
    callback_base_url: Optional[str] = None,
    language: str = "en",
):
    """API 명세에 따른 비디오 처리 및 콜백"""
    start_time = datetime.now()
    video_path = None

    try:
        # Progress milestones
        milestones = [
            (10, "비디오 다운로드 완료"),
            (25, "음성 구간 감지 중..."),
            (40, "화자 식별 중..."),
            (60, "음성을 텍스트로 변환 중..."),
            (75, "감정 분석 중..."),
            (90, "결과 정리 중..."),
        ]

        # 1. 비디오 다운로드
        logger.info(f"작업 시작: {job_id}")
        video_path = await download_from_url(video_url, job_id, callback_base_url)
        await send_callback(
            job_id,
            "processing",
            milestones[0][0],
            milestones[0][1],
            callback_base_url=callback_base_url,
        )
        logger.info("비디오 준비 완료, ML 처리 시작")

        # 2-5. Pipeline 처리 (진행상황 업데이트)
        for progress, message in milestones[1:4]:
            await send_callback(
                job_id,
                "processing",
                progress,
                message,
                callback_base_url=callback_base_url,
            )

        # 비디오 처리 실행
        logger.info("ML 파이프라인 실행 중...")
        result = await process_audio_core(video_path, language=language)
        logger.info("ML 파이프라인 처리 완료")

        # 6. 감정 분석
        await send_callback(
            job_id,
            "processing",
            milestones[4][0],
            milestones[4][1],
            callback_base_url=callback_base_url,
        )

        # 7. 결과 정리
        await send_callback(
            job_id,
            "processing",
            milestones[5][0],
            milestones[5][1],
            callback_base_url=callback_base_url,
        )

        # API 응답 형식으로 변환 (백엔드가 기대하는 형식)
        segments_for_api = []
        word_segments = []  # 단어 단위 세그먼트 추가

        for seg in result["segments"]:
            segment_data = {
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "speaker": {"speaker_id": seg["speaker_id"]},
                "text": seg["text"],
                "words": [
                    {
                        "word": w["word"],
                        "start": w["start_time"],
                        "end": w["end_time"],
                        "acoustic_features": {  # 중첩 객체로 변경
                            "volume_db": w["acoustic_features"]["volume_db"],
                            "pitch_hz": w["acoustic_features"]["pitch_hz"],
                            "spectral_centroid": w["acoustic_features"].get("spectral_centroid", 1500.0)
                        }
                    }
                    for w in seg.get("words", [])
                ],
            }
            segments_for_api.append(segment_data)

            # word_segments 생성
            for word in seg.get("words", []):
                word_segments.append({
                    "word": word["word"],
                    "start_time": word["start_time"],
                    "end_time": word["end_time"],
                    "speaker_id": seg["speaker_id"],
                    "confidence": 0.95
                })

        processing_time = (datetime.now() - start_time).total_seconds()

        # 백엔드가 기대하는 올바른 결과 구조
        final_result = {
            "segments": segments_for_api,
            "word_segments": word_segments,  # 추가
            "speakers": result["speakers"],  # result 내부로 이동
            "text": " ".join([seg["text"] for seg in segments_for_api]),
            "language": result["metadata"].get("language_detected", language),
            "duration": result["metadata"]["duration"],
            "metadata": {
                "model_version": "whisperx-base",
                "processing_time": processing_time,
                "unique_speakers": result["metadata"]["unique_speakers"],
                "total_segments": result["metadata"]["total_segments"],
                "language_requested": result["metadata"].get("language_requested", language),
                "language_detected": result["metadata"].get("language_detected", language),
                "processing_mode": "targeted" if language != "auto" else "auto-detect",
                "processed_at": datetime.now().isoformat()
            }
        }

        # 완료 콜백 전송 (모든 데이터는 result 안에)
        await send_callback(
            job_id,
            "completed",
            100,
            "분석 완료",
            result=final_result,
            callback_base_url=callback_base_url,
        )

        logger.info(
            f"✅ 분석 완료 - job_id: {job_id}, 처리시간: {processing_time:.2f}초"
        )

        # Job 상태 업데이트
        jobs[job_id] = {
            "status": "completed",
            "processing_time": processing_time,
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"처리 실패 - job_id: {job_id}, error: {str(e)}")

        error_code = (
            "DOWNLOAD_ERROR" if "download" in str(e).lower() else "PROCESSING_ERROR"
        )
        await send_callback(
            job_id,
            "failed",
            0,
            error_message=str(e),
            error_code=error_code,
            callback_base_url=callback_base_url,
        )

        jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat(),
        }

    finally:
        # 임시 파일 정리 (다운로드된 파일만)
        if video_path and Path(video_path).exists() and "/tmp" in video_path:
            try:
                Path(video_path).unlink()
                logger.info(f"임시 파일 정리: {video_path}")
            except:
                pass


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """백엔드 호환 동기 전사 API"""
    start_time = datetime.now()

    try:
        logger.info(f"전사 요청 시작 - video_path: {request.video_path}")

        # 가상의 job_id 생성 (진행상황 추적용)
        job_id = str(uuid.uuid4())

        # Progress milestones (오디오 우선 처리)
        if request.audio_path:
            progress_steps = [
                (5, "오디오 파일 검증 중..."),
                (20, "음성 구간 분석 중..."),
                (65, "음성을 텍스트로 변환 중..."),
                (95, "감정 분석 중..."),
                (100, "분석 완료"),
            ]
        else:
            # 하위 호환성: 비디오에서 오디오 추출이 필요한 경우
            progress_steps = [
                (5, "비디오 파일 검증 중..."),
                (15, "오디오 추출 중..."),
                (25, "음성 구간 분석 중..."),
                (65, "음성을 텍스트로 변환 중..."),
                (95, "감정 분석 중..."),
                (100, "분석 완료"),
            ]

        # 첫 진행상황 업데이트
        await send_callback(
            job_id, "processing", progress_steps[0][0], progress_steps[0][1]
        )

        # 오디오 파일 준비 (오디오 우선, 비디오는 fallback)
        audio_s3_path = request.audio_path or request.video_path  # 하위 호환성
        file_extension = (
            ".wav" if request.audio_path else ".mp4"
        )  # 오디오면 .wav, 비디오면 .mp4

        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False
        ) as temp_file:
            try:
                # S3에서 오디오/비디오 파일 다운로드 시도
                s3_client.download_file(S3_BUCKET, audio_s3_path, temp_file.name)
                if request.audio_path:
                    logger.info(f"S3에서 오디오 다운로드 완료: {request.audio_path}")
                else:
                    logger.info(
                        f"S3에서 비디오 다운로드 완료 (오디오 추출 필요): {request.video_path}"
                    )
                actual_file_path = temp_file.name
            except Exception as s3_error:
                logger.warning(f"S3 다운로드 실패, 로컬 경로 사용: {s3_error}")
                actual_file_path = audio_s3_path

            # 진행상황 업데이트 (첫 번째 단계 이후 분석 전까지)
            analysis_start_index = (
                2 if request.audio_path else 3
            )  # 오디오는 2단계부터, 비디오는 3단계부터
            for progress, message in progress_steps[1:analysis_start_index]:
                await send_callback(job_id, "processing", progress, message)

            try:
                # 오디오/비디오 처리 실행
                result = await process_audio_core(
                    actual_file_path, language=request.language
                )

                # 감정 분석 진행상황
                await send_callback(
                    job_id, "processing", progress_steps[4][0], progress_steps[4][1]
                )

                processing_time = (datetime.now() - start_time).total_seconds()

                # 간소화된 결과 생성
                detailed_result = {
                    "success": True,
                    "segments": result["segments"],
                    "speakers": result["speakers"],
                    "metadata": {
                        **result["metadata"],
                        "processing_time": processing_time,
                        "processed_at": datetime.now().isoformat(),
                    },
                    "processing_time": processing_time,
                    "error": None,
                    "error_code": None,
                }

                # 완료 진행상황
                await send_callback(
                    job_id, "processing", progress_steps[5][0], progress_steps[5][1]
                )

                logger.info(f"전사 완료 - 처리시간: {processing_time:.2f}초")
                logger.info(f"반환할 세그먼트 수: {len(detailed_result['segments'])}")

                return detailed_result

            except Exception as analysis_error:
                logger.error(f"분석 실패: {analysis_error}")
                processing_time = (datetime.now() - start_time).total_seconds()

                return {
                    "success": False,
                    "error": f"분석 중 오류가 발생했습니다: {str(analysis_error)}",
                    "error_code": "ANALYSIS_ERROR",
                    "processing_time": processing_time,
                }

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"전사 요청 실패 - Error: {str(e)}")

        return {
            "success": False,
            "error": f"요청 처리 실패: {str(e)}",
            "error_code": "REQUEST_ERROR",
            "processing_time": processing_time,
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


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
    logger.info(f"   백엔드 URL: {BACKEND_URL if BACKEND_URL else 'Not configured'}")
    logger.info(f"   콜백 활성화: {ENABLE_CALLBACKS}")

    # GPU 확인
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.device_count()}개 사용 가능")
            for i in range(torch.cuda.device_count()):
                logger.info(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
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
