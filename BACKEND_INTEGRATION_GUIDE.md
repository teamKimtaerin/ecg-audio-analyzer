# Backend Server 통합 가이드

## 개요
ML Server의 API 변경사항에 맞춰 Backend Server에서 수정해야 할 부분들을 정리한 문서입니다.

## ML Server 변경사항 요약

### 1. 콜백 URL 변경
- **기존**: `/api/v1/ml/ml-results`
- **변경**: `/api/upload-video/result`

### 2. 요청/응답 구조 개선
- **ProcessVideoRequest**: 추가 파라미터 지원
- **ProcessVideoResponse**: status_url 필드 추가
- **결과 데이터**: 백엔드 기대 형식으로 구조 변경

## Backend Server 필수 수정사항

### 1. ML Server 호출 시 추가 파라미터 전송

**파일**: `ml_video.py` (또는 ML Server 호출 부분)

```python
# 기존 코드
payload = {
    "job_id": job_id,
    "video_url": video_url
}

# 수정된 코드
payload = {
    "job_id": job_id,
    "video_url": video_url,
    "fastapi_base_url": FASTAPI_BASE_URL,  # 동적 콜백 URL 제공
    "language": language or "auto",         # 언어 설정 (기본값: auto)
    "enable_gpu": True,                     # GPU 사용 여부
    "emotion_detection": True,              # 감정 분석 여부  
    "max_workers": 4                        # 최대 워커 수
}
```

**환경변수 추가**:
```bash
FASTAPI_BASE_URL=http://your-backend-server:8000  # Backend Server 주소
```

### 2. ML Server 응답 처리 수정

**새로운 응답 형식**:
```python
response = {
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "processing",              # "accepted" → "processing"으로 변경
    "message": "비디오 처리가 시작되었습니다",
    "status_url": "/api/upload-video/status/{job_id}",  # 새로 추가된 필드
    "estimated_time": 300
}
```

**Backend 처리 코드 수정**:
```python
# ML Server 응답 처리
if response.status_code == 200:
    result = response.json()
    
    # 새로운 필드 처리
    status_url = result.get("status_url")  # 선택적 처리
    
    # Job 상태를 "processing"으로 업데이트
    await update_job_status(
        job_id=result["job_id"],
        status=result["status"],  # "processing"
        message=result["message"]
    )
else:
    # 에러 처리 강화
    await update_job_status(
        job_id, "failed", 
        error_message=f"ML Server returned {response.status_code}"
    )
```

### 3. 콜백 엔드포인트 확인

ML Server는 이제 `/api/upload-video/result`로 콜백을 전송합니다.

**확인사항**:
- 해당 엔드포인트가 올바르게 구현되어 있는지 확인
- 콜백 데이터 구조가 ML Server가 전송하는 형식과 일치하는지 확인

**ML Server 콜백 데이터 구조**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing|completed|failed",
  "progress": 0-100,
  "message": "처리 상태 메시지",
  "result": {  // status가 "completed"인 경우에만 포함
    "text": "전체 transcription 텍스트",
    "segments": [...],
    "language": "en|ko|auto",
    "duration": 120.5,
    "metadata": {
      "model_version": "whisperx-base",
      "processing_time": 45.2,
      "unique_speakers": 3,
      "total_segments": 42
    }
  },
  "error_message": "에러 메시지",  // status가 "failed"인 경우에만 포함
  "error_code": "ERROR_CODE"     // status가 "failed"인 경우에만 포함
}
```

### 4. 타임아웃 설정 증가

**환경변수 수정**:
```bash
# 기존
ML_API_TIMEOUT=30

# 권장 변경
ML_API_TIMEOUT=300  # 5분 (대용량 비디오 처리 고려)
```

**코드 적용**:
```python
ML_API_TIMEOUT = int(os.getenv("ML_API_TIMEOUT", "300"))  # 기본 5분

# ML Server 호출 시
response = await http_client.post(
    f"{MODEL_SERVER_URL}/api/upload-video/process-video",
    json=payload,
    timeout=ML_API_TIMEOUT  # 적용
)
```

### 5. 에러 핸들링 개선

```python
try:
    response = await call_ml_server(payload)
    
    if response.status_code != 200:
        # ML Server가 200이 아닌 상태 반환 시 처리
        error_detail = response.json() if response.content else {}
        await update_job_status(
            job_id, "failed",
            error_message=f"ML Server error: {error_detail.get('message', 'Unknown error')}",
            error_code=error_detail.get('error', {}).get('code', 'ML_SERVER_ERROR')
        )
        return
        
except requests.exceptions.Timeout:
    await update_job_status(
        job_id, "failed",
        error_message="ML Server processing timeout",
        error_code="TIMEOUT_ERROR"
    )
    
except requests.exceptions.RequestException as e:
    await update_job_status(
        job_id, "failed",
        error_message=f"ML Server connection error: {str(e)}",
        error_code="CONNECTION_ERROR"
    )
```

## 권장 구현 순서

### Phase 1: 필수 수정사항 (즉시 적용)
1. ✅ 콜백 URL이 `/api/upload-video/result`로 설정되어 있는지 확인
2. ✅ ML Server 호출 시 `fastapi_base_url` 파라미터 추가
3. ✅ 타임아웃 설정을 300초로 증가

### Phase 2: 응답 처리 개선
1. ✅ `status_url` 필드 처리 추가
2. ✅ `status` 필드가 "processing"으로 변경된 것 반영
3. ✅ 에러 핸들링 강화

### Phase 3: 추가 기능 활용
1. 🔄 `language` 파라미터를 사용자 설정에 따라 전달
2. 🔄 `enable_gpu`, `emotion_detection` 등 설정 옵션 추가
3. 🔄 새로운 결과 데이터 구조 활용 (metadata 정보 등)

## 테스트 가이드

### 1. 로컬 테스트
```bash
# ML Server 헬스체크
curl http://ml-server:8080/health

# Backend → ML Server 통신 테스트
curl -X POST http://backend-server:8000/api/upload-video/request-process \
  -H "Content-Type: application/json" \
  -d '{"fileKey": "test-video.mp4"}'
```

### 2. 통합 테스트
1. 실제 비디오 파일로 전체 파이프라인 테스트
2. 콜백 수신 확인
3. 결과 데이터 구조 검증
4. 에러 시나리오 테스트 (타임아웃, 연결 실패 등)

### 3. 모니터링 포인트
- ML Server 응답 시간
- 콜백 수신 성공률  
- 전체 처리 완료율
- 에러 발생 빈도 및 유형

## 호환성 정보

### 하위 호환성
- 기존 콜백 URL(`/api/v1/ml/ml-results`)은 더 이상 사용되지 않음
- 새로운 파라미터들은 모두 선택적이므로 기존 요청도 동작함

### API 버전
- ML Server API 버전: v1.1
- 변경 날짜: 2024-12-28
- 호환성 보장: 2025년 3월까지

## 문제 해결

### 자주 발생하는 문제

**1. 콜백이 수신되지 않음**
- 콜백 URL이 올바른지 확인: `/api/upload-video/result`
- `fastapi_base_url`이 올바르게 전달되었는지 확인
- 네트워크 연결성 확인

**2. 타임아웃 에러**
- `ML_API_TIMEOUT` 환경변수를 300초로 설정
- 대용량 비디오의 경우 더 긴 타임아웃 고려

**3. 잘못된 응답 구조**
- ML Server 버전이 최신인지 확인
- 콜백 데이터 파싱 코드 점검

### 로그 확인
```bash
# ML Server 로그
sudo journalctl -u ml-server -f

# Backend 로그에서 ML Server 관련 로그 필터링
grep "ML Server\|process-video" /var/log/backend.log
```

---

**문의사항**: ML Server 관련 이슈는 개발팀에 문의하세요.
**업데이트**: 이 문서는 ML Server 변경사항에 따라 지속적으로 업데이트됩니다.