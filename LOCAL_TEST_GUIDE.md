# 🧪 로컬 테스트 가이드

ECG Audio Analyzer 시스템을 로컬에서 테스트하기 위한 완벽한 가이드입니다.

## 📡 시스템 아키텍처

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Client     │      │  API Server │      │  ML Server  │
│  Port 3000  │─────▶│  Port 8000  │─────▶│  Port 8080  │
└─────────────┘      └─────────────┘      └─────────────┘
     (1)                  (2)                  (3)
  파일 업로드         S3 키 전달           분석 수행
     ◀────────────────────────────────────────┘
                    (4) 결과 반환
```

## 🚀 Quick Start

### 1️⃣ ML Server 시작 (터미널 1)
```bash
cd /Users/ahntaeju/project/ecg-audio-analyzer
./run_local_ml_server.sh
```
또는 직접 실행:
```bash
BACKEND_URL=http://localhost:8000 python ml_api_server.py --port 8080
```

### 2️⃣ API Server 시작 (터미널 2)
```bash
# API 서버 프로젝트 디렉토리에서
export ML_SERVER_URL=http://localhost:8080
npm run dev  # 또는 해당 서버의 시작 명령
```

### 3️⃣ Client 시작 (터미널 3)
```bash
# 클라이언트 프로젝트 디렉토리에서
export API_BASE_URL=http://localhost:8000
npm run dev  # 또는 해당 클라이언트의 시작 명령
```

## 🧪 테스트 방법

### 사전 준비: AWS 설정 확인
```bash
./check_aws_setup.sh  # AWS 자격증명 및 S3 접근 확인
```

### 방법 1: S3 통합 테스트 (권장)
```bash
./test_with_s3.sh  # S3 업로드 포함 완전한 테스트
```

### 방법 2: 빠른 API 테스트
```bash
./test_local_api.sh  # S3에 파일이 이미 있다고 가정
```

### 방법 3: 수동 CURL 테스트

#### ML 서버 헬스체크
```bash
curl http://localhost:8080/health
```

#### API에서 ML로 분석 요청 (핵심 명령)
```bash
# S3 키 형식 사용 (실제 S3에서 다운로드)
curl -X POST "http://localhost:8080/request-process?video_key=uploads/test123/video.mp4"
```

응답 예시:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

## 📊 통신 플로우

### 1. 클라이언트 → API Server
```javascript
// 클라이언트에서 파일 업로드
POST http://localhost:8000/api/upload-video
FormData: { video: file.mp4 }
```

### 2. API Server → ML Server
```bash
# API 서버가 S3 업로드 후 ML 서버에 요청
POST http://localhost:8080/request-process?video_key=uploads/uuid/video.mp4
```

### 3. ML Server 처리
- S3에서 비디오 다운로드
- WhisperX로 음성 인식
- 화자 분리 수행
- 감정 분석 실행

### 4. ML Server → API Server
```json
POST http://localhost:8000/api/upload-video/results
{
  "job_id": "uuid",
  "video_key": "uploads/uuid/video.mp4",
  "status": "completed",
  "success": true,
  "results": {
    "transcript": [...],
    "emotions": [...],
    "metadata": {...}
  }
}
```

### 5. API Server → 클라이언트
WebSocket 또는 Polling으로 결과 전달

## ⚙️ 환경 설정

### ML Server (ml_api_server.py)
- `BACKEND_URL`: API 서버 URL (기본값: http://localhost:8000)
- `S3_BUCKET`: AWS S3 버킷 이름
- `AWS_ACCESS_KEY_ID`: AWS 액세스 키
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 키

### API Server
- `ML_SERVER_URL`: ML 서버 URL (http://localhost:8080)
- S3 설정 (업로드용)

### Client
- `API_BASE_URL`: API 서버 URL (http://localhost:8000)

## 🐛 문제 해결

### ML 서버가 시작되지 않을 때
```bash
# 의존성 확인
pip install -r requirements.txt

# GPU 사용 불가 시 CPU 모드로 실행
python ml_api_server.py --port 8080 --log-level debug
```

### S3 연결 오류
- AWS 자격증명 확인: `aws configure list`
- 버킷 접근 권한 확인
- 로컬 테스트 시 실제 S3 버킷 필요

### 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i :8080
lsof -i :8000
lsof -i :3000

# 프로세스 종료
kill -9 <PID>
```

## 📝 테스트 시나리오

### 시나리오 1: 단순 헬스체크
1. ML 서버 시작
2. `curl http://localhost:8080/health`
3. 응답 확인

### 시나리오 2: 엔드투엔드 테스트
1. 모든 서버 시작 (ML, API, Client)
2. 클라이언트에서 파일 업로드
3. 콘솔 로그로 진행 상황 확인
4. 결과 수신 확인

### 시나리오 3: ML 서버 단독 테스트
1. ML 서버만 시작
2. `friends.mp4` 파일로 직접 테스트
```bash
curl -X POST "http://localhost:8080/request-process?video_key=friends.mp4"
```

## 📊 예상 결과

정상 작동 시:
- ML 서버: "비동기 분석 요청 시작" 로그
- 백그라운드에서 분석 진행
- API 서버로 결과 전송 시도
- 전체 처리 시간: 10분 비디오 기준 약 30초

## 🔗 관련 파일

- `run_local_ml_server.sh`: ML 서버 실행 스크립트
- `test_local_api.sh`: API 테스트 스크립트
- `ml_api_server.py`: ML 서버 메인 코드
- `friends.mp4`: 테스트용 샘플 비디오