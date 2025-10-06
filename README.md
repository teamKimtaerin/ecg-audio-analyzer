# 🎵 ECG Audio Analyzer ML Server

> WhisperX 기반 고성능 음성 분석 및 화자 분리 ML 서버

<br/>

## 🔗 바로가기

- **API 서버**: `http://localhost:8080`
- **Health Check**: `http://localhost:8080/health`
- **API 문서**: `http://localhost:8080/docs` (FastAPI 자동 문서)

<br/>

## 📖 목차

1. [프로젝트 소개](#-프로젝트-소개)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#️-기술-스택)
4. [시스템 아키텍처](#️-시스템-아키텍처)
5. [설치 및 실행](#-설치-및-실행)
6. [API 사용법](#-api-사용법)

<br/>

## 📌 프로젝트 소개

### 🤔 개발 배경
- 영상 콘텐츠에서 **정확한 음성 인식**과 **화자 구분**이 필요한 상황이 증가
- 기존 솔루션들의 **처리 속도**와 **정확도** 한계
- **실시간 진행률 추적**과 **GPU 최적화**가 가능한 고성능 서버 필요

### 💡 서비스 목표
- **WhisperX** 기반 고정확도 음성-텍스트 변환
- **Pyannote.audio** 활용 정밀한 화자 분리
- **실시간 음향 특성 분석** (피치, 볼륨, 스펙트럴 특성)
- **AWS GPU 인스턴스** 최적화로 10분 영상을 30초 내 처리

<br/>

## ✨ 주요 기능

### 1. 🎙️ 음성-텍스트 변환 (STT)
- **WhisperX large-v2** 모델 사용으로 높은 정확도
- **다국어 지원** (자동 감지 또는 언어 지정)
- **단어 단위 타임스탬프** 정확한 동기화

### 2. 👥 화자 분리 (Speaker Diarization)
- **Pyannote.audio 3.1** 기반 고정확도 화자 구분
- **다중 화자 동시 처리** 가능
- 각 세그먼트별 **화자 ID 자동 할당**
- 화자별 **전체 발화 시간 통계** 제공

### 3. 🎵 음향 특성 분석
- **MFCC** (Mel-frequency cepstral coefficients)
- **피치 분석** (Pitch/F0 detection)
- **볼륨 분석** (RMS, dB level)
- **스펙트럴 특성** (Spectral centroid, rolloff)

### 4. 📊 실시간 진행률 콜백
- **비동기 처리**로 긴 영상도 안정적 처리
- **실시간 진행률** (0-100%) 및 상태 메시지
- **에러 핸들링**과 자동 복구 메커니즘
- **RESTful 콜백**으로 클라이언트 연동

<br/>

## 🛠️ 기술 스택

| 구분 | 기술 |
| --- | --- |
| **ML 프레임워크** | `PyTorch 1.13+` `WhisperX` `Pyannote.audio` `Librosa` |
| **API 서버** | `FastAPI` `Uvicorn` `Pydantic` `AsyncIO` |
| **GPU 최적화** | `CUDA 11.7+` `cuDNN` `TensorRT` `Mixed Precision` |
| **클라우드 인프라** | `AWS EC2 (T4 GPU)` `S3` `Docker` |


<br/>

## ⚙️ 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐ 
│   클라이언트       │───▶│   FastAPI 서버    │
│   (Backend)     │    │  (ml_api_server) │    
└─────────────────┘    └──────────────────┘    
                                ▼
                     ┌──────────────────┐
                     │   콜백 시스템       │
                     │ (Progress Track) │
                     └──────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML 처리 파이프라인                           │
├─────────────────┬──────────────┬───────────────┬────────────────┤
│  오디오 추출       │  음성 인식     │   화자 분리     │  특성 분석        │
│ (AudioExtractor)│ (WhisperX)   │ (Pyannote)    │ (Acoustic)     │
│                 │              │               │                │
│ • MP4→WAV 변환   │ • STT 처리    │ • 화자 구분     │ • MFCC         │
│ • S3/URL 지원    │ • 타임스탬프    │ • 세그먼트 분할  │ • Pitch        │
│ • GPU 가속       │ • 다국어 지원   │ • 화자 통계     │ • Volume       │
└─────────────────┴──────────────┴───────────────┴────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │     결과 JSON 생성       │
                    │ • 세그먼트 정보           │
                    │ • 화자 매핑              │
                    │ • 음향 특성              │
                    │ • 메타데이터              │
                    └────────────────────────┘
```

### 📁 주요 컴포넌트

- **`ml_api_server.py`** - FastAPI 기반 메인 서버
- **`src/pipeline/manager.py`** - 전체 처리 파이프라인 관리
- **`src/models/speech_recognizer.py`** - WhisperX 음성 인식
- **`src/services/audio_extractor.py`** - 오디오 추출 및 전처리
- **`src/services/acoustic_analyzer.py`** - 음향 특성 분석
- **`src/utils/gpu_optimizer.py`** - GPU 메모리 최적화

<br/>

## 🚀 설치 및 실행

### 1. 사전 요구사항
- **Python 3.7 이상** (3.10+ 권장)
- **NVIDIA GPU** (T4, V100 등) + CUDA 11.7+
- **FFmpeg** 설치 필수
- **8GB+ GPU 메모리** 권장

### 2. 설치 과정

```bash
# 1. Repository 클론
git clone [저장소 주소]
cd ecg-audio-analyzer

# 2. Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 기본 의존성 설치
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. WhisperX 설치 (GitHub에서 직접)
pip install git+https://github.com/m-bain/whisperx.git@v3.1.1

# 5. GPU 환경 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 서버 실행

```bash
# 개발 서버 실행
python ml_api_server.py --log-level debug

# 프로덕션 서버 실행
python ml_api_server.py --host 0.0.0.0 --port 8080

# Docker 실행 (GPU 지원)
docker build -t ecg-analyzer .
docker run --gpus all -p 8080:8080 -v $(pwd)/output:/app/output ecg-analyzer
```

<br/>

## 📡 API 사용법

### 1. 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/transcribe` | 음성 분석 (동기) |
| `POST` | `/api/upload-video/process-video` | 음성 분석 (비동기 + 콜백) |
| `GET` | `/jobs/{job_id}` | 작업 상태 조회 |

### 2. 음성 분석 API

**POST `/transcribe`**

```bash
# S3 파일 분석
curl -X POST "http://localhost:8080/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "s3://bucket-name/video.mp4",
    "language": "en",
    "job_id": "task-123"
  }'
```

**요청 파라미터:**
```json
{
  "video_path": "string",      // 필수: 파일 경로 또는 URL
  "audio_path": "string",      // 선택: 오디오 파일 직접 지정
  "language": "string",        // 선택: 언어 코드 (기본값: "auto")
  "job_id": "string"          // 선택: 작업 ID (콜백용)
}
```

### 3. 응답 데이터 구조

```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.23,
      "speaker": {
        "speaker_id": "SPEAKER_00"
      },
      "text": "안녕하세요, 반갑습니다.",
      "words": [
        {
          "word": "안녕하세요",
          "start": 0.12,
          "end": 1.45,
          "acoustic_features": {
            "volume_db": -18.5,
            "pitch_hz": 145.2,
            "spectral_centroid": 1250.4
          }
        }
      ]
    }
  ],
  "speakers": {
    "SPEAKER_00": {
      "total_duration": 45.67
    },
    "SPEAKER_01": {
      "total_duration": 23.12
    }
  },
  "text": "전체 텍스트 내용...",
  "language": "ko",
  "duration": 68.79,
  "metadata": {
    "processing_time": 12.34,
    "model_versions": {
      "whisperx": "large-v2",
      "diarization": "pyannote/speaker-diarization-3.1"
    }
  }
}
```

### 4. 비동기 처리 및 콜백

**POST `/api/upload-video/process-video`**

- 긴 영상 처리용 비동기 API
- 실시간 진행률을 콜백으로 전송
- 작업 완료 후 최종 결과 콜백

**진행률 콜백 예제:**
```json
{
  "job_id": "task-123",
  "status": "processing",
  "progress": 65,
  "message": "화자 분리 진행 중...",
  "result": null
}
```

**완료 콜백 예제:**
```json
{
  "job_id": "task-123",
  "status": "completed",
  "progress": 100,
  "message": "분석 완료",
  "result": { /* 위의 응답 데이터 구조 */ }
}
```

<br/>
