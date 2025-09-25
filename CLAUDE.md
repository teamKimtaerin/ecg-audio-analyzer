# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**High-performance audio analysis pipeline** for dynamic subtitle generation with speaker diarization and emotion detection. Extracts audio from MP4 files/URLs, performs WhisperX-based speech recognition + speaker diarization, and generates comprehensive JSON metadata with acoustic features. Optimized for AWS GPU instances with 30-second target processing time for 10-minute videos.

### Architecture Flow
1. **Input Processing**: Client uploads file → S3 storage → ML server receives URL/path
2. **Audio Processing**: Video/audio extraction → WhisperX pipeline → Speaker diarization
3. **Analysis Pipeline**: Acoustic feature extraction → Progress callbacks → Result aggregation
4. **Output**: Structured JSON with segments, speakers, timestamps, and acoustic features

**Key Features:**
- Unified WhisperX pipeline for speech recognition + speaker diarization
- Real-time acoustic feature extraction (MFCC, pitch, volume, spectral features)
- FastAPI ML server with real-time progress callbacks
- GPU-optimized processing with automatic CPU fallback
- S3 integration with presigned URL support

## Architecture

### Core Components
- **PipelineManager** (`src/pipeline/manager.py`) - Central orchestrator with async operations and GPU resource management
- **WhisperXPipeline** (`src/models/speech_recognizer.py`) - Unified speech recognition + speaker diarization using WhisperX
- **AudioExtractor** (`src/services/audio_extractor.py`) - MP4/URL → WAV conversion with ffmpeg and yt-dlp
- **FastAcousticAnalyzer** (`src/services/acoustic_analyzer.py`) - Real-time acoustic feature extraction
- **ML API Server** (`ml_api_server.py`) - FastAPI server with progress callbacks for ECS backend integration

### Utility Modules
- **GPUOptimizer** (`src/utils/gpu_optimizer.py`) - GPU memory management and device allocation
- **AudioCleaner** (`src/utils/audio_cleaner.py`) - Audio preprocessing and cleanup utilities
- **Logger** (`src/utils/logger.py`) - Structured logging configuration

### Configuration
- **`config/base_settings.py`** - Core performance settings, file handling, memory limits
- **`config/model_configs.py`** - ML model parameters optimized for GPU (FP16, batch sizes)
- **`config/aws_settings.py`** - AWS GPU instance optimization settings

## Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install WhisperX from GitHub (required for speech recognition)
pip install git+https://github.com/m-bain/whisperx.git@v3.1.1

# EC2-specific requirements (for deployment)
pip install -r requirements-ec2.txt
```

### Running ML API Server
```bash
# Production server
python ml_api_server.py --host 0.0.0.0 --port 8080

# Development server with logging
python ml_api_server.py --log-level debug

# Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/transcribe -H "Content-Type: application/json" -d '{"video_path":"test.mp4"}'

# Test with job ID and language
curl -X POST http://localhost:8080/transcribe -H "Content-Type: application/json" -d '{
  "video_path": "s3://bucket/video.mp4",
  "job_id": "test-123",
  "language": "en"
}'
```

### Development & Testing
```bash
# Code formatting and linting
black src/ config/ ml_api_server.py
ruff check src/ config/ ml_api_server.py --fix

# Security scanning
bandit -r src/ config/ ml_api_server.py

# Clean artifacts and temporary files
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "._*" -type f -delete  # Remove macOS metadata files

# Run using pipeline manager directly
PYTHONPATH=. python -m src.pipeline.manager --help

# Test individual components
PYTHONPATH=. python -c "
from src.models.speech_recognizer import WhisperXPipeline
from src.services.acoustic_analyzer import FastAcousticAnalyzer
print('Components loaded successfully')
"
```

### Testing
```bash
# Test ML API server endpoints
python ml_api_server.py --host 0.0.0.0 --port 8080

# Test with local video files (results saved to output/)
curl -X POST http://localhost:8080/transcribe \
  -H "Content-Type: application/json" \
  -d '{"video_path":"notebook.mp4", "language":"en"}'

# Monitor output directory
ls -la output/
```

### Docker & Deployment
```bash
# Build GPU container (if Dockerfile exists)
docker build -t ecg-analyzer .

# Run container with GPU support
docker run --gpus all -p 8080:8080 -v $(pwd)/output:/app/output ecg-analyzer

# Deploy to EC2 (manual process - see Production Deployment section)
# Requires: GPU instance, NVIDIA drivers, Docker with nvidia-runtime
```

## Development Guidelines

### Service Implementation Pattern
```python
class ServiceName:
    def __init__(self, config: Config, gpu_device: str = "cuda:0"):
        self.config = config
        self.device = gpu_device
        self._model = None

    def load_model(self) -> None:
        """Load and cache model in GPU memory"""

    def process(self, input_data: Any) -> ServiceResult:
        """Main processing method - single responsibility"""
```

### Progress Callback Integration
The pipeline supports real-time progress callbacks for long-running operations:
```python
async def progress_callback(progress: int, message: str):
    # Called during processing with 0-100 progress and descriptive message
    await send_callback(job_id, "processing", progress, message)

pipeline = create_pipeline(language="en", progress_callback=progress_callback)
```

### GPU Memory Management
- **Models use ~4-6GB total GPU memory**
- pyannote/speaker-diarization-3.1 (~96MB, 1-2GB GPU)
- WhisperX base (~290MB, 1-2GB GPU)
- FP16 precision enabled for optimization
- Automatic fallback to CPU on GPU errors

### Error Handling
- GPU memory errors trigger batch size reduction or CPU fallback
- Services continue processing despite individual failures
- Automatic cleanup of GPU memory and temporary files
- Structured logging with performance metrics

### AWS Infrastructure
- **Primary**: G4dn.xlarge (NVIDIA T4, 16GB VRAM)
- **High Performance**: P3.2xlarge (NVIDIA V100, 16GB VRAM)
- Docker: nvidia-docker2 runtime for containerized deployment
- S3 integration for file storage
- CloudWatch monitoring enabled

## Performance Targets
- 30 seconds for 10-minute video on P3.2xlarge
- >80% GPU utilization during ML inference  
- <6GB total GPU memory usage
- 99.5%+ successful processing rate

## API Integration

### FastAPI ML Server Endpoints
- **POST `/api/upload-video/process-video`** - Async video processing with progress callbacks to backend
- **POST `/transcribe`** - Synchronous transcription with detailed results (saves to output/)
- **GET `/health`** - Health check endpoint
- **GET `/jobs/{job_id}`** - Job status inquiry

### Input Formats Supported
- Local file paths: `"path/to/video.mp4"`
- S3 URLs: `"s3://bucket-name/key"`
- S3 HTTPS URLs: `"https://bucket.s3.region.amazonaws.com/key"`
- S3 Presigned URLs (with query parameters)
- YouTube URLs (via yt-dlp)

### Progress Callback System
The ML server sends real-time progress updates to the backend via HTTP callbacks to `/api/upload-video/result`:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 75,
  "message": "화자 식별 중...",
  "result": null
}
```

**Final Result Structure** (when status="completed"):
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.0,
      "speaker": {"speaker_id": "SPEAKER_00"},
      "text": "Hello, how are you?",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "acoustic_features": {
            "volume_db": -20.0,
            "pitch_hz": 150.0,
            "spectral_centroid": 1500.0
          }
        }
      ]
    }
  ],
  "word_segments": [...],
  "speakers": {"SPEAKER_00": {"total_duration": 83.36}},
  "text": "Full transcription text...",
  "language": "en",
  "duration": 143.36,
  "metadata": {...}
}
```

## Troubleshooting

### Common Issues
- **GPU Memory Errors**: Pipeline automatically falls back to CPU with reduced batch sizes
- **WhisperX Installation**: Must install from GitHub: `pip install git+https://github.com/m-bain/whisperx.git@v3.1.1`
- **WhisperX Loading**: Large models cached in GPU memory (~2-4GB), first load may be slow
- **S3 Access**: Ensure AWS credentials are configured via environment variables or IAM roles
- **FFmpeg Missing**: Install ffmpeg for audio/video processing: `brew install ffmpeg` (macOS)
- **macOS Files**: Remove `._*` metadata files that cause issues on Linux: `find . -name "._*" -delete`
- **Output Files**: Results are saved to `output/` directory with `_analysis.json` suffix

### Debug Commands
```bash
# Test WhisperX installation
PYTHONPATH=. python -c "from src.models.speech_recognizer import WhisperXPipeline; print('OK')"

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Verify AWS credentials
python -c "import boto3; print(boto3.Session().get_credentials())"

# Monitor ML API server logs
python ml_api_server.py --log-level debug
```

## Production Deployment Target

### GPU Environment (Final Goal)
**Target Infrastructure**: AWS EC2 g4dn.2xlarge
- **Instance**: i-03169309168d4d268 (ecg-audio-production)
- **GPU**: NVIDIA T4 16GB
- **CPU**: 8 vCPUs, 32GB RAM
- **OS**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)

**Performance Expectations**:
- 5-10x speed improvement over CPU
- Target: <20 seconds for 60-second video processing
- GPU memory utilization: <12GB for models
- Batch processing: 16-32 for optimal GPU usage

### Current Development Status
- **Phase**: CPU optimization and testing
- **Environment**: Local development (macOS)
- **Focus**: Speaker diarization accuracy improvements
- **Next**: Deploy optimized solution to GPU environment