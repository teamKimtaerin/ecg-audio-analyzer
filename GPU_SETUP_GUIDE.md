# GPU Setup Guide - ECG Audio Analyzer

## AWS Deep Learning AMI (권장 방법)

현재 사용 중인 **"Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)"**에서 GPU를 활성화하는 방법입니다.

### ✅ 이미 설치된 구성요소

Deep Learning AMI에는 다음이 이미 설치되어 있습니다:
- NVIDIA 드라이버 (R570)
- CUDA Toolkit (12.8)
- cuDNN (9.10)
- Docker
- NVIDIA Container Toolkit (1.17.4+)

### 🚀 간단한 해결책

**추가 설치 없이** 단순히 Docker 실행 시 `--gpus all` 플래그를 추가하면 됩니다:

```bash
# 기존 명령 (CPU 모드)
docker run --rm -p 8080:8080 ... ecg-analyzer:v47

# GPU 활성화 명령
docker run --rm --gpus all -p 8080:8080 ... ecg-analyzer:v47
```

### 📋 사용 방법

#### 1. 환경 변수 설정
```bash
export HF_TOKEN=your_huggingface_token
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
```

#### 2. GPU 활성화 스크립트 실행
```bash
# EC2 인스턴스에서
./run_gpu_dlami.sh
```

### 🧪 GPU 테스트 명령어

#### 1. NVIDIA 드라이버 확인
```bash
nvidia-smi
```

#### 2. Docker GPU 지원 테스트
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 3. ECG Analyzer에서 GPU 인식 확인
컨테이너 로그에서 다음과 같은 메시지를 확인:
```
GPU: 사용 가능 (CUDA 모드)
GPU info: Tesla T4
```

### 🔧 문제 해결

#### GPU가 인식되지 않는 경우:

1. **Docker 데몬 재시작**
   ```bash
   sudo systemctl restart docker
   ```

2. **NVIDIA 서비스 상태 확인**
   ```bash
   sudo systemctl status nvidia-persistenced
   ```

3. **CUDA 호환성 라이브러리 (Container Toolkit 1.17.4+)**
   ```bash
   # 여러 CUDA 버전 사용 시
   export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
   ```

### ⚡ 성능 예상

| 모드 | 143초 오디오 처리 시간 | 성능 개선 |
|------|---------------------|-----------|
| CPU  | ~4-5분              | 기준      |
| GPU  | ~30-60초            | 5-10배 빠름 |

### 📁 관련 파일

- `run_gpu_dlami.sh` - Deep Learning AMI용 GPU 실행 스크립트
- `setup_ec2_gpu.sh` - 일반 EC2 인스턴스용 GPU 설정 스크립트

### 🎯 인스턴스 타입별 권장사항

| 인스턴스 타입 | vCPU | GPU | 메모리 | 용도 |
|--------------|------|-----|--------|------|
| G4dn.xlarge  | 4    | 1   | 16GB   | 개발/테스트 |
| G4dn.2xlarge | 8    | 1   | 32GB   | 프로덕션 (현재 사용) |
| P3.2xlarge   | 8    | 1   | 61GB   | 고성능 처리 |

### 💡 팁

1. **첫 실행 시 모델 다운로드로 인한 지연 정상**
2. **GPU 메모리 부족 시 자동으로 CPU 모드로 전환**
3. **컨테이너 재시작 시에도 모델 캐시 유지됨**

---

## 일반 EC2 인스턴스 (수동 설정 필요)

일반 Ubuntu EC2 인스턴스에서 GPU를 설정해야 하는 경우:

### 1. NVIDIA 드라이버 설치
```bash
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2. Docker 설치
```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker
```

### 3. NVIDIA Container Toolkit 설치
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 4. 자동 설정 스크립트 사용
```bash
./setup_ec2_gpu.sh
```

---

**🎯 권장사항: Deep Learning AMI 사용 시 추가 설정 없이 `--gpus all` 플래그만 추가하면 됩니다!**