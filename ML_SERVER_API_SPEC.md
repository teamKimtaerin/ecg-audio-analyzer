# ML Server API 통합 명세서

**ECG Audio Analyzer ML Server API 완전 가이드**

---

## 📋 목차

1. [개요](#개요)
2. [서버 정보](#서버-정보)
3. [API 엔드포인트](#api-엔드포인트)
4. [요청/응답 모델](#요청응답-모델)
5. [콜백 메커니즘](#콜백-메커니즘)
6. [데이터 변환 로직](#데이터-변환-로직)
7. [에러 처리](#에러-처리)
8. [통합 체크리스트](#통합-체크리스트)
9. [문제 해결 가이드](#문제-해결-가이드)

---

## 개요

### 🎯 목적
ML Server는 비디오/오디오 파일의 음성 분석을 수행하고 백엔드 시스템과 실시간 콜백을 통해 통신합니다.

### 🔧 핵심 기능
- **비동기 비디오 처리**: S3 URL에서 비디오 다운로드 → 음성 추출 → 분석 → 결과 콜백
- **동기 전사 처리**: 직접 파일 업로드를 통한 즉시 분석
- **실시간 진행상황**: 처리 단계별 진행률 콜백 전송
- **언어 최적화**: 지정 언어별 최적화된 모델 사용

---

## 서버 정보

### 🌐 서버 환경
- **Private EC2**: `10.0.10.42:8001` (VPC 내부 접근만 가능)
- **Public 접근**: AWS Systems Manager Session Manager 사용
- **GPU**: NVIDIA T4 (G4dn.xlarge) 또는 V100 (P3.2xlarge)

### 🔗 CORS 설정
```python
allow_origins = [
    "http://localhost:3000",  # 프론트엔드 개발
    "http://localhost:8000",  # 백엔드 개발
    "http://ecg-project-pipeline-dev-alb-1703405864.us-east-1.elb.amazonaws.com"  # Fargate 백엔드
]
```

---

## API 엔드포인트

### 1. 비동기 비디오 처리 (주 엔드포인트)

```http
POST /api/upload-video/process-video
```

**설명**: S3 비디오 URL을 받아 백그라운드에서 처리하고 콜백으로 결과 전송

#### 요청
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_url": "https://s3.amazonaws.com/bucket/video.mp4",
    "fastapi_base_url": "https://ho-it.site",
    "enable_gpu": true,
    "emotion_detection": true,
    "language": "ko",
    "max_workers": 4
}
```

#### 응답 (즉시)
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "message": "비디오 처리가 시작되었습니다",
    "status_url": "/jobs/550e8400-e29b-41d4-a716-446655440000",
    "estimated_time": 300
}
```

### 2. 동기 전사 처리

```http
POST /transcribe
```

**설명**: 파일을 직접 업로드하여 즉시 분석 결과 반환

#### 요청
```json
{
    "video_path": "s3://bucket/video.mp4",
    "audio_path": "s3://bucket/audio.wav",  // 우선순위: audio_path > video_path
    "enable_gpu": true,
    "emotion_detection": true,
    "language": "en"
}
```

#### 응답
```json
{
    "success": true,
    "segments": [...],
    "speakers": {...},
    "metadata": {
        "filename": "video.mp4",
        "duration": 120.5,
        "total_segments": 45,
        "unique_speakers": 2,
        "processing_time": 32.1,
        "language_requested": "en",
        "language_detected": "en",
        "processing_mode": "targeted",
        "processed_at": "2024-01-15T10:30:00Z"
    },
    "processing_time": 32.1,
    "error": null,
    "error_code": null
}
```

### 3. 헬스체크

```http
GET /health
```

#### 응답
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### 4. 작업 상태 조회

```http
GET /jobs/{job_id}
```

#### 응답
```json
{
    "status": "completed",
    "processing_time": 45.2,
    "completed_at": "2024-01-15T10:30:00Z"
}
```

---

## 요청/응답 모델

### 📥 ProcessVideoRequest
```python
class ProcessVideoRequest(BaseModel):
    job_id: str                           # ✅ 필수 - UUID 형태 권장
    video_url: str                        # ✅ 필수 - S3 URL 또는 공개 URL
    fastapi_base_url: Optional[str]       # ⚠️ 선택 - 동적 콜백 URL (기본: BACKEND_URL)
    enable_gpu: bool = True               # ⚠️ 선택 - GPU 사용 여부
    emotion_detection: bool = True        # ⚠️ 선택 - 감정 분석 (현재 미구현)
    language: str = "auto"                # ⚠️ 선택 - "auto", "ko", "en", "ja", "zh"
    max_workers: int = 4                  # ⚠️ 선택 - 최대 워커 수
```

### 📤 ProcessVideoResponse
```python
class ProcessVideoResponse(BaseModel):
    job_id: str                           # 요청의 job_id 그대로 반환
    status: str                           # "processing" (고정)
    message: str                          # "비디오 처리가 시작되었습니다"
    status_url: Optional[str]             # "/jobs/{job_id}"
    estimated_time: Optional[int] = 300   # 예상 처리 시간 (초)
```

### 📥 TranscribeRequest
```python
class TranscribeRequest(BaseModel):
    video_path: str                       # 하위호환용 (deprecated)
    audio_path: Optional[str] = None      # ✅ 우선순위 높음 - S3 오디오 파일
    video_url: Optional[str] = None       # 비디오 URL
    enable_gpu: bool = True
    emotion_detection: bool = True
    language: str = "en"
```

### 📤 BackendTranscribeResponse
```python
class BackendTranscribeResponse(BaseModel):
    success: bool                         # 처리 성공 여부
    metadata: Optional[Dict[str, Any]]    # 메타데이터 (아래 참조)
    speakers: Optional[Dict[str, Any]]    # 화자 통계
    segments: Optional[list]              # 분석 결과 세그먼트
    processing_time: float                # 처리 시간 (초)
    error: Optional[str] = None           # 에러 메시지
    error_code: Optional[str] = None      # 에러 코드
```

---

## 콜백 메커니즘

### 🔄 콜백 URL
```
POST {fastapi_base_url}/api/upload-video/result

// 예시:
POST https://ho-it.site/api/upload-video/result
```

### 📊 콜백 페이로드 구조

#### ⚠️ 중요: 백엔드가 기대하는 정확한 구조

**백엔드는 최상위 레벨에 이 7개 필드만 기대합니다:**
1. `job_id` (필수)
2. `status` (필수)
3. `progress` (선택)
4. `message` (선택)
5. `result` (선택)
6. `error_message` (선택)
7. `error_code` (선택)

#### 진행 상황 콜백
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "progress": 50,
    "message": "음성을 텍스트로 변환 중..."
}
```

#### 완료 콜백
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": 100,
    "result": {
        // 🔴 모든 분석 결과는 result 객체 안에 포함!
        "segments": [
            {
                "start_time": 1.5,
                "end_time": 3.2,
                "text": "안녕하세요",
                "speaker": {
                    "speaker_id": "SPEAKER_00"
                },
                "words": [
                    {
                        "word": "안녕하세요",
                        "start": 1.5,
                        "end": 1.8,
                        "acoustic_features": {
                            "volume_db": -20.0,
                            "pitch_hz": 150.0,
                            "spectral_centroid": 1500.0
                        }
                    }
                ]
            }
        ],
        "word_segments": [  // 📌 추가됨: 단어 단위 세그먼트
            {
                "word": "안녕하세요",
                "start_time": 1.5,
                "end_time": 1.8,
                "speaker_id": "SPEAKER_00",
                "confidence": 0.95
            }
        ],
        "speakers": {  // 📌 result 내부로 이동됨
            "SPEAKER_00": {
                "total_duration": 45.2,
                "segment_count": 12
            },
            "SPEAKER_01": {
                "total_duration": 38.7,
                "segment_count": 9
            }
        },
        "text": "전체 전사 결과 텍스트...",
        "language": "ko",
        "duration": 120.5,
        "metadata": {
            "model_version": "whisperx-base",
            "processing_time": 45.2,
            "unique_speakers": 2,
            "total_segments": 35,
            "language_requested": "ko",
            "language_detected": "ko",
            "processing_mode": "targeted"
        }
    }
}
```

#### 실패 콜백
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "failed",
    "error_message": "오디오 추출 실패",
    "error_code": "AUDIO_EXTRACTION_ERROR"
}
```

### 🎯 진행 단계
1. **10%**: "비디오 다운로드 완료"
2. **25%**: "음성 구간 감지 중..."
3. **40%**: "화자 식별 중..."
4. **60%**: "음성을 텍스트로 변환 중..."
5. **75%**: "감정 분석 중..."
6. **90%**: "결과 정리 중..."
7. **100%**: "분석 완료" (result 포함)

### 🛡️ 중복 방지 메커니즘
- `completed_jobs` 세트로 완료된 작업 추적
- `failed_jobs` 세트로 실패한 작업 추적
- 동일 상태 중복 콜백 자동 차단

---

## 데이터 변환 로직

### 🔄 WhisperX → API 응답 변환

#### 세그먼트 구조 변환
```python
# WhisperX 원본
{
    "start": 1.5,
    "end": 3.2,
    "text": "안녕하세요",
    "speaker": "SPEAKER_00",
    "words": [...]
}

# API 응답 변환
{
    "start_time": 1.5,
    "end_time": 3.2,
    "speaker": {"speaker_id": "SPEAKER_00"},
    "text": "안녕하세요",
    "words": [
        {
            "word": "안녕하세요",
            "start": 1.5,
            "end": 1.8,
            "acoustic_features": {  // 📌 중첩 객체로 변경됨
                "volume_db": -20.0,
                "pitch_hz": 150.0,
                "spectral_centroid": 1500.0
            }
        }
    ]
}
```

### 🎵 음향 특성 처리
```python
# 각 단어별 음향 특성
"acoustic_features": {
    "volume_db": -20.0,        # 볼륨 (데시벨)
    "pitch_hz": 150.0,         # 피치 (헤르츠)
    "spectral_centroid": 1500.0 # 스펙트럴 중심 주파수
}
```

### 🗣️ 화자 통계
```python
"speakers": {
    "SPEAKER_00": {
        "total_duration": 45.2,    # 총 발화 시간 (초)
        "segment_count": 12        # 세그먼트 수
    },
    "SPEAKER_01": {
        "total_duration": 38.7,
        "segment_count": 9
    }
}
```

### 🌍 언어 최적화
```python
# language="auto" (자동 감지)
"metadata": {
    "language_requested": "auto",
    "language_detected": "ko",
    "processing_mode": "auto-detect"
}

# language="ko" (한국어 지정)
"metadata": {
    "language_requested": "ko",
    "language_detected": "ko",
    "processing_mode": "targeted"
}
```

---

## 에러 처리

### 🚨 에러 코드 체계
- `DOWNLOAD_ERROR`: 비디오 다운로드 실패
- `AUDIO_EXTRACTION_ERROR`: 오디오 추출 실패
- `PROCESSING_ERROR`: ML 처리 실패
- `MODEL_LOADING_ERROR`: 모델 로딩 실패
- `GPU_MEMORY_ERROR`: GPU 메모리 부족

### 📤 에러 응답 예시
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "failed",
    "error_message": "비디오 파일을 다운로드할 수 없습니다: 404 Not Found",
    "error_code": "DOWNLOAD_ERROR"
}
```

### 🔄 재시도 로직
- 일시적 네트워크 오류: 자동 재시도 (최대 3회)
- GPU 메모리 부족: CPU 모드로 자동 전환
- 모델 로딩 실패: 다른 모델 크기로 재시도

---

## 통합 체크리스트

### ✅ 백엔드 팀 확인사항

#### 요청 검증
- [ ] `job_id`가 UUID 형태인가?
- [ ] `video_url`이 접근 가능한 URL인가?
- [ ] `fastapi_base_url`이 올바른 콜백 도메인인가?
- [ ] `language` 값이 지원하는 언어인가? (`auto`, `ko`, `en`, `ja`, `zh`)

#### 콜백 엔드포인트 준비
- [ ] `POST /api/upload-video/result` 엔드포인트가 구현되어 있는가?
- [ ] 콜백 페이로드의 7개 필드를 모두 처리할 수 있는가?
- [ ] `result` 객체 내부의 세그먼트 구조를 파싱할 수 있는가?
- [ ] 동일 `job_id`의 중복 콜백을 처리할 수 있는가?

#### 응답 처리
- [ ] `processing` 상태의 진행률을 UI에 표시하는가?
- [ ] `completed` 상태에서 `result` 데이터를 저장하는가?
- [ ] `failed` 상태에서 에러를 적절히 처리하는가?

### ✅ ML 서버 운영팀 확인사항

#### 서버 상태
- [ ] EC2 인스턴스가 정상 작동 중인가?
- [ ] GPU 메모리가 충분한가? (권장: 16GB)
- [ ] S3 접근 권한이 설정되어 있는가?
- [ ] HF_TOKEN이 설정되어 있는가?

#### 모델 상태
- [ ] WhisperX 모델이 정상 로딩되는가?
- [ ] 화자 분리 모델이 정상 작동하는가?
- [ ] 언어별 최적화 설정이 적용되고 있는가?

---

## 문제 해결 가이드

### 🐛 일반적인 문제들

#### 1. 콜백이 전송되지 않음
**원인**: `fastapi_base_url`이 잘못되었거나 백엔드 엔드포인트가 없음
```bash
# 확인 방법
curl -X POST https://ho-it.site/api/upload-video/result \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","status":"processing","progress":50}'
```

#### 2. 분석 결과가 비어있음
**원인**: 오디오 추출 실패 또는 음성이 없는 파일
```python
# 로그 확인
grep "audio_extraction_failed" /logs/ml_server.log
grep "whisperx_pipeline_no_result" /logs/ml_server.log
```

#### 3. GPU 메모리 부족
**원인**: 이전 작업의 GPU 메모리가 정리되지 않음
```bash
# GPU 메모리 확인
nvidia-smi

# 서버 재시작 (메모리 정리)
pkill -f ml_api_server.py
python ml_api_server.py --host 0.0.0.0 --port 8001
```

#### 4. 언어 감지 오류
**원인**: 음성 품질이 낮거나 지원하지 않는 언어
```python
# 지원 언어 확인
supported_languages = ["auto", "ko", "en", "ja", "zh"]
```

### 🔧 디버깅 도구

#### 로그 모니터링
```bash
# 실시간 로그 확인
tail -f /logs/ml_server.log | grep "job_id"

# 특정 작업 추적
grep "550e8400-e29b-41d4-a716-446655440000" /logs/ml_server.log
```

#### 헬스체크
```bash
# 서버 상태 확인
curl http://10.0.10.42:8001/health

# 작업 상태 확인
curl http://10.0.10.42:8001/jobs/{job_id}
```

### 📞 연락처 및 지원

#### 개발팀 연락처
- **ML 서버**: [ML팀 연락처]
- **백엔드**: [백엔드팀 연락처]
- **DevOps**: [인프라팀 연락처]

#### 긴급 상황 대응
1. 서버 다운: AWS Systems Manager로 접속하여 재시작
2. GPU 메모리 오류: 서버 재시작 또는 CPU 모드 전환
3. 콜백 실패: 백엔드 엔드포인트 상태 확인

---

## 📝 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0.0 | 2024-01-15 | 초기 버전 작성 |
| 1.1.0 | 2024-01-15 | 언어 최적화 기능 추가 |
| 1.2.0 | 2024-01-15 | 콜백 구조 백엔드 호환성 수정 |

---

**📌 이 문서는 ML Server와 Backend 간의 API 통신 명세를 정의합니다.
변경 사항이 있을 때마다 반드시 양쪽 팀에 공유하고 업데이트해주세요.**