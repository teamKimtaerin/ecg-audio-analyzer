# EC2 배포 가이드

## 🚀 신속 배포 (datetime 오류 수정 적용)

### EC2 인스턴스 정보
- **IP**: 54.197.171.76
- **Instance Type**: G4dn.xlarge
- **User**: ubuntu
- **GPU**: NVIDIA Tesla T4

### 배포 방법

#### 1. SSH 접속
```bash
ssh ubuntu@54.197.171.76
```

#### 2. 배포 스크립트 실행
```bash
cd ~/ecg-audio-analyzer
./deploy_to_ec2.sh
```

### 수동 배포 (단계별)

#### 1. 코드 업데이트
```bash
cd ~/ecg-audio-analyzer
git fetch origin
git checkout fix/gpu-processing-and-cleanup
git pull origin fix/gpu-processing-and-cleanup
```

#### 2. 기존 서버 중지
```bash
pkill -f "python ml_api_server.py"
```

#### 3. 가상환경 활성화
```bash
source venv/bin/activate
```

#### 4. ML 서버 시작 (수정된 명령어)
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH nohup python ml_api_server.py --host 0.0.0.0 --port 8080 > server.log 2>&1 &
```

### 배포 확인

#### 1. 프로세스 확인
```bash
ps aux | grep ml_api_server
```

#### 2. Health Check
```bash
curl http://localhost:8080/health
```

#### 3. 로그 확인
```bash
tail -f server.log
```

### 문제 해결

#### 서버가 시작되지 않는 경우
```bash
# 로그 확인
tail -50 server.log

# 포트 사용 확인
netstat -tlnp | grep 8080

# GPU 상태 확인
nvidia-smi
```

#### datetime 오류가 계속 발생하는 경우
```bash
# 코드 버전 확인
git log --oneline -5

# 최신 수정사항이 포함되어 있는지 확인
grep "process_start_time" ml_api_server.py
```

## 🛠️ 수정된 내용

### datetime 변수 충돌 해결
- `start_time` → `process_start_time` (프로세스 시작 시간)
- `start_time`/`end_time` → `seg_start`/`seg_end` (세그먼트 타이밍)
- `start_time`/`end_time` → `word_start`/`word_end` (단어 타이밍)
- `start_time` → `transcribe_start_time` (전사 함수 시작 시간)

### 수정된 파일
- `ml_api_server.py`: datetime 변수 충돌 해결

## 🔗 유용한 링크

- **Health Check**: http://54.197.171.76:8080/health
- **API 문서**: http://54.197.171.76:8080/docs
- **Metrics**: http://54.197.171.76:8080/metrics

## 📞 지원

배포 중 문제가 발생하면:
1. `server.log` 파일 확인
2. GPU 메모리 상태 확인 (`nvidia-smi`)
3. 디스크 공간 확인 (`df -h`)
4. 네트워크 연결 확인 (`curl` 테스트)