# ML 서버 설정 및 실행 가이드

## 🚨 문제 해결됨!

**이전 문제**: `ML 서버 연결 실패: Cannot connect to host localhost:8080 ssl:default [Connection refused]`

**해결책**: 환경변수 설정 수정 및 올바른 실행 방법 적용

---

## 🚀 빠른 시작

### 1. ML 서버 실행
```bash
# 간편한 실행 (권장)
./start_ml_server.sh

# 또는 직접 실행
export BACKEND_URL=http://localhost:8000
python ml_api_server.py --host 0.0.0.0 --port 8080
```

### 2. 서버 확인
```bash
# Health check
curl http://localhost:8080/health

# API 문서 확인
open http://localhost:8080/docs
```

### 3. 통합 테스트
```bash
# 통합 테스트 실행
python test_integration.py
```

---

## 📋 환경 설정

### 필수 환경변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `BACKEND_URL` | `http://localhost:8000` | ECG Backend URL |
| `ECG_BACKEND_URL` | `http://localhost:8000` | 대체 Backend URL |
| `ML_SERVER_PORT` | `8080` | ML 서버 포트 |

### 환경변수 설정 방법

#### 방법 1: 스크립트 사용 (권장)
```bash
./start_ml_server.sh  # 자동으로 환경변수 설정
```

#### 방법 2: 수동 설정
```bash
export BACKEND_URL=http://localhost:8000
export PYTHONPATH=/Users/ahntaeju/project/ecg-audio-analyzer
python ml_api_server.py --port 8080
```

#### 방법 3: .env 파일 사용
```bash
# .env 파일 생성
cat > .env << EOF
BACKEND_URL=http://localhost:8000
ECG_BACKEND_URL=http://localhost:8000
ML_SERVER_PORT=8080
PYTHONPATH=/Users/ahntaeju/project/ecg-audio-analyzer
EOF

# 환경변수 로드 후 실행
source .env
python ml_api_server.py --port 8080
```

---

## 🔗 API 엔드포인트

### 1. Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "service": "model-server"
}
```

### 2. 비디오 처리 요청 (ECG Backend용)
```http
POST /api/upload-video/process-video
Content-Type: application/json
User-Agent: ECS-FastAPI-Backend/1.0

Request:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_url": "https://s3.amazonaws.com/bucket/video.mp4"
}

Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "message": "Processing started",
  "estimated_time": 300
}
```

### 3. 기타 엔드포인트
- `GET /` - 서버 정보 및 사용 가능한 엔드포인트
- `GET /docs` - Swagger API 문서
- `POST /transcribe` - 동기 전사 API (레거시)

---

## 📡 콜백 시스템

ML 서버는 비디오 처리 과정에서 ECG Backend로 진행률과 결과를 전송합니다.

### 진행률 업데이트
```http
POST {BACKEND_URL}/api/v1/ml/ml-results
Content-Type: application/json
User-Agent: ML-Server/1.0

{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 65,
  "message": "Analyzing speech segments..."
}
```

### 완료 결과 전송
```http
POST {BACKEND_URL}/api/v1/ml/ml-results
Content-Type: application/json

{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result": {
    "metadata": {
      "filename": "video.mp4",
      "duration": 143.39,
      "total_segments": 25,
      "unique_speakers": 4
    },
    "segments": [
      {
        "start_time": 4.908,
        "end_time": 8.754,
        "speaker": {"speaker_id": "SPEAKER_01"},
        "text": "You know, we should all do. Go see a musical.",
        "words": [
          {
            "word": "You",
            "start": 4.908,
            "end": 4.988,
            "volume_db": -19.87,
            "pitch_hz": 851.09
          }
        ]
      }
    ]
  }
}
```

---

## 🧪 테스트 방법

### 1. 기본 연결 테스트
```bash
# ML 서버 Health Check
curl -X GET http://localhost:8080/health

# 예상 응답
{"status":"healthy","service":"model-server"}
```

### 2. 비디오 처리 테스트
```bash
# 테스트 요청
curl -X POST "http://localhost:8080/api/upload-video/process-video" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ECS-FastAPI-Backend/1.0" \
  -d '{
    "job_id": "test-123",
    "video_url": "https://example.com/test.mp4"
  }'

# 예상 응답
{
  "job_id": "test-123",
  "status": "accepted",
  "message": "Processing started",
  "estimated_time": 300
}
```

### 3. 통합 테스트 스크립트
```bash
python test_integration.py
```

---

## 🐛 문제 해결

### 문제 1: Connection Refused
```
ML 서버 연결 실패: Cannot connect to host localhost:8080
```

**해결책:**
1. ML 서버가 실행 중인지 확인
2. 포트 8080이 사용 가능한지 확인
3. 환경변수가 올바른지 확인

```bash
# 포트 사용 확인
lsof -i :8080

# 프로세스 종료 (필요시)
lsof -ti:8080 | xargs kill -9

# ML 서버 재시작
./start_ml_server.sh
```

### 문제 2: 콜백 전송 실패
```
⚠️ Progress update failed: 404
```

**해결책:**
1. ECG Backend URL 확인
2. ECG Backend가 실행 중인지 확인
3. 콜백 엔드포인트 `/api/v1/ml/ml-results` 존재 확인

```bash
# Backend URL 확인
echo $BACKEND_URL

# Backend 연결 테스트
curl -X GET $BACKEND_URL/health
```

### 문제 3: 환경변수 오류
```
Backend URL 설정이 AWS URL로 되어 있음
```

**해결책:**
```bash
# 올바른 환경변수 설정
export BACKEND_URL=http://localhost:8000
export ECG_BACKEND_URL=http://localhost:8000

# 또는 start script 사용
./start_ml_server.sh
```

### 문제 4: Python 모듈 오류
```
ModuleNotFoundError: No module named 'src'
```

**해결책:**
```bash
# PYTHONPATH 설정
export PYTHONPATH=/Users/ahntaeju/project/ecg-audio-analyzer:$PYTHONPATH

# 프로젝트 루트에서 실행
cd /Users/ahntaeju/project/ecg-audio-analyzer
python ml_api_server.py --port 8080
```

---

## 📊 로그 확인

### 주요 로그 메시지

#### 성공적인 시작
```
🚀 ECG Audio Analyzer ML API 서버 시작
   호스트: 0.0.0.0:8080
   백엔드 URL: http://localhost:8000
   GPU: 1개 사용 가능
```

#### 비디오 처리 요청
```
INFO - 비디오 처리 요청 접수 - job_id: test-123, video_url: https://...
INFO - Backend URL 설정: http://localhost:8000
```

#### 콜백 전송
```
DEBUG - 콜백 전송 - URL: http://localhost:8000/api/v1/ml/ml-results, 페이로드: {...}
INFO - ✅ Progress updated: 오디오 추출 중... (25%)
```

---

## 🔧 개발 팁

### 1. 디버그 모드 실행
```bash
python ml_api_server.py --port 8080 --log-level debug
```

### 2. 다른 포트로 실행
```bash
python ml_api_server.py --port 8081
```

### 3. 다른 Backend URL 사용
```bash
export BACKEND_URL=http://192.168.1.100:8000
python ml_api_server.py --port 8080
```

### 4. GPU 없이 실행
```bash
export CUDA_VISIBLE_DEVICES=""
python ml_api_server.py --port 8080
```

---

## ✅ 체크리스트

### 실행 전 확인사항
- [ ] Python 3.8+ 설치됨
- [ ] 필수 패키지 설치됨 (`pip install fastapi uvicorn boto3`)
- [ ] 포트 8080이 사용 가능함
- [ ] 프로젝트 경로가 올바름

### 통합 테스트 전 확인사항
- [ ] ML 서버가 8080 포트에서 실행 중
- [ ] ECG Backend가 8000 포트에서 실행 중 (선택사항)
- [ ] 환경변수가 올바르게 설정됨
- [ ] Health check 성공

### 프로덕션 배포 전 확인사항
- [ ] 모든 통합 테스트 통과
- [ ] 실제 비디오 파일로 전체 플로우 테스트
- [ ] 에러 처리 시나리오 테스트
- [ ] 로그 레벨 적절히 설정
- [ ] 보안 설정 검토

---

## 📞 지원

문제가 지속되면 다음을 확인하세요:

1. **로그 파일**: 상세한 오류 메시지
2. **환경변수**: `echo $BACKEND_URL` 등
3. **네트워크**: 포트 및 방화벽 설정
4. **통합 테스트**: `python test_integration.py`

이 가이드를 따르면 ML 서버와 ECG Backend 간의 통신 문제가 해결됩니다.