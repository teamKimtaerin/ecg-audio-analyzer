# ECG Audio Analyzer - ML API Server
# NVIDIA CUDA 기반 GPU 가속 이미지
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsox-dev \
    libsndfile1 \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3 /usr/bin/python

# pip 업그레이드
RUN python -m pip install --upgrade pip

# 작업 디렉터리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치 (캐시 최적화)
COPY requirements.txt requirements-gpu.txt ./

# 기본 의존성 설치 (배치별로 설치해서 메모리 효율성 향상)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.0 \
    pydantic==2.5.0 \
    python-dotenv==1.0.0 \
    structlog==23.2.0 \
    rich==13.7.0 \
    typer==0.9.0

# 오디오 처리 라이브러리 설치
RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    pydub==0.25.1 \
    ffmpeg-python==0.2.0 \
    yt-dlp==2023.12.30

# PyTorch 및 GPU 가속 라이브러리 설치
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    transformers==4.36.0 \
    scikit-learn==1.3.2 \
    numpy==1.24.3 \
    scipy==1.11.4

# WhisperX 설치 (가장 까다로운 패키지)
RUN pip install --no-cache-dir whisperx>=3.0.0

# 나머지 requirements 설치
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir -r requirements-gpu.txt || true

# 애플리케이션 코드 복사
COPY . .

# 권한 설정
RUN chmod +x ml_api_server.py

# 포트 노출
EXPOSE 8080

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 서버 실행
CMD ["python", "ml_api_server.py", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]