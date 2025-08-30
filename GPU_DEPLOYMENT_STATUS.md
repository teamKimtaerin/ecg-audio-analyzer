# GPU 배포 현황 및 다음 단계

## 📊 현재 상황 (2025-08-30)

### ✅ 완료된 작업
- **CPU 최적화 구현**: WhisperX float32, 배치 최적화, 청크 병렬 처리
- **GPU 배포 스크립트 완료**: CloudFormation, Docker, 자동화 스크립트
- **성능 개선**: Speech recognition 복구, 256단어 추출, 50개 자막 세그먼트
- **AWS 설정**: ECR 저장소 생성, Key Pair (ecg-key) 생성 완료

### ⏳ 대기 중인 작업
- **AWS GPU 인스턴스 한도 증가**: 현재 0개 → 8 vCPUs 요청 필요
- **예상 승인 기간**: 1-2일

## 🚀 GPU 한도 승인 후 진행 단계

### 1단계: 한도 확인
```bash
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-DB2E81BA \
    --region us-east-1
```

### 2단계: 자동 배포 실행
```bash
./deployment/aws-deploy.sh -k ecg-key
```

### 3단계: 성능 테스트
```bash
# SSH 접속
ssh -i ~/.ssh/ecg-key.pem ubuntu@YOUR_GPU_INSTANCE_IP

# GPU 확인
nvidia-smi

# 성능 테스트
python -m src.cli analyze friends.mp4 --gpu --verbose
```

## 📈 예상 성능 향상

| 항목 | 현재 CPU | GPU 예상 | 개선율 |
|------|----------|----------|--------|
| 전체 처리시간 | 170.6초 | 30-50초 | **3-5배** |
| Speaker Diarization | ~138초 | ~20초 | **7배** |
| Speech Recognition | ~23초 | ~5초 | **5배** |

## 🔧 준비 완료된 구성요소

- ✅ `deployment/aws-deploy.sh`: 완전 자동화 배포
- ✅ `docker/Dockerfile.gpu`: CUDA 12.9.1 + PyTorch 최적화
- ✅ `aws/cloudformation.yml`: GPU 인프라 정의
- ✅ AWS ECR 저장소: `084828586938.dkr.ecr.us-east-1.amazonaws.com/ecg-audio-analyzer`
- ✅ Key Pair: `ecg-key` (생성 완료)

## 📞 AWS Support 요청 정보

**요청할 내용:**
- Service: Amazon EC2
- Type: Service limit increase  
- Limit: Running On-Demand G and VT instances
- Region: us-east-1 (또는 ap-northeast-2)
- New limit: 8 vCPUs (g4dn.2xlarge 1개 실행 가능)

**비용 예상:**
- g4dn.2xlarge: $0.75/시간
- 월 운영 (24시간): ~$540
- 테스트 (4시간): ~$3

---
**다음 작업 시점**: AWS Support 승인 완료 후
**실행 명령어**: `./deployment/aws-deploy.sh -k ecg-key`