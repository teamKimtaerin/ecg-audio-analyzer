# EC2 Deployment Configuration Guide

## 🚀 EC2 Instance Setup

### Instance Details
- **Public IP**: 54.197.171.76
- **Instance Type**: g4dn.xlarge (NVIDIA T4 GPU)
- **Instance ID**: `i-xxxxxxxxx` (⚠️ **NEEDS VERIFICATION**)
- **User**: ubuntu

### Directory Structure
```
~/ecg-audio-analyzer          # Git-connected directory (sync from GitHub)
~/ecg-audio-analyzer.backup   # Stable running directory (server runs from here)
```

### Deployment Flow
1. GitHub Actions pulls latest code to `~/ecg-audio-analyzer`
2. Code is synced from `~/ecg-audio-analyzer` to `~/ecg-audio-analyzer.backup`
3. Docker build and server restart happens in `~/ecg-audio-analyzer.backup`

## GitHub Secrets Configuration

### Required Secrets
1. **EC2_INSTANCE_ID**: `i-xxxxxxxxx` ⚠️ **NEEDS TO BE SET WITH CORRECT VALUE**
2. **AWS_ACCESS_KEY_ID**: AWS access key for EC2/SSM access
3. **AWS_SECRET_ACCESS_KEY**: AWS secret key for EC2/SSM access
4. **AWS_ACCOUNT_ID**: AWS account ID for ECR (if needed)

### How to Find EC2 Instance ID
```bash
# On the EC2 instance:
curl -s http://169.254.169.254/latest/meta-data/instance-id

# Or via AWS CLI:
aws ec2 describe-instances --filters "Name=ip-address,Values=54.197.171.76" --query 'Reservations[].Instances[].InstanceId' --output text
```

## Manual Deployment

### SSH 접속
```bash
ssh ubuntu@54.197.171.76
```

### 배포 스크립트 실행 (Updated)
```bash
cd ~/ecg-audio-analyzer
./deploy_to_ec2.sh  # Now uses main branch and proper directory sync
```

### 수동 배포 (단계별)

#### 1. 코드 업데이트 (Fixed)
```bash
cd ~/ecg-audio-analyzer
git fetch origin
git checkout main  # Changed from fix/gpu-processing-and-cleanup
git pull origin main
```

#### 2. 기존 서버 중지
```bash
pkill -f "python ml_api_server.py"
```

#### 3. 백업 디렉토리 동기화
```bash
rsync -av --exclude='.git' --exclude='venv' ~/ecg-audio-analyzer/ ~/ecg-audio-analyzer.backup/
```

#### 4. 백업 디렉토리에서 서버 시작
```bash
cd ~/ecg-audio-analyzer.backup
source venv/bin/activate
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
tail -f ~/ecg-audio-analyzer.backup/server.log
```

### 문제 해결

#### 서버가 시작되지 않는 경우
```bash
# 로그 확인
tail -50 ~/ecg-audio-analyzer.backup/server.log

# 포트 사용 확인
netstat -tlnp | grep 8080

# GPU 상태 확인
nvidia-smi
```

## Current Status
✅ Deploy script fixed for correct branch and directory structure
✅ GitHub Actions workflow updated for proper paths
⚠️ EC2_INSTANCE_ID in GitHub Secrets needs verification and update

## Next Steps
1. **Verify the actual EC2 instance ID** (should be `i-xxxxxxxxx` format)
2. **Update GitHub Secrets** with correct EC2_INSTANCE_ID
3. **Test deployment pipeline**

### How to Update GitHub Secrets
1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Update `EC2_INSTANCE_ID` with the correct instance ID from metadata service
3. Ensure other AWS credentials are properly set

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