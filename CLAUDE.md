# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**High-performance audio analysis pipeline** for dynamic subtitle generation. Extracts audio from MP4 files/URLs, performs speaker diarization, speech recognition with WhisperX, and generates comprehensive JSON metadata with acoustic features. Optimized for AWS GPU instances (G4dn.xlarge/P3.2xlarge) with 30-second target processing time for 10-minute videos.

### Objective
- Client uploads a file.
- API Server receives the file and sends the S3 key.
- ML Server performs the analysis.
- Results are returned back to the client.

**Key Features:**
- Unified WhisperX pipeline for speech recognition + speaker diarization
- Real-time acoustic feature extraction (MFCC, pitch, volume)
- FastAPI ML server for ECS backend integration
- GPU-optimized processing with automatic CPU fallback
- Comprehensive error handling and performance monitoring

## Architecture

### Core Components
- **PipelineManager** (`src/pipeline/manager.py`) - Central orchestrator with async operations and GPU resource management
- **WhisperXPipeline** (`src/models/speech_recognizer.py`) - Unified speech recognition + speaker diarization using WhisperX
- **AudioExtractor** (`src/services/audio_extractor.py`) - MP4/URL → WAV conversion with ffmpeg and yt-dlp
- **FastAcousticAnalyzer** (`src/services/acoustic_analyzer.py`) - Real-time acoustic feature extraction
- **ML API Server** (`ml_api_server.py`) - FastAPI server with progress callbacks for ECS backend integration
- **CallbackPipelineManager** (`src/services/callback_pipeline.py`) - Pipeline with real-time progress reporting

### Configuration
- **`config/base_settings.py`** - Core performance settings, file handling, memory limits
- **`config/model_configs.py`** - ML model parameters optimized for GPU (FP16, batch sizes)
- **`config/aws_settings.py`** - AWS GPU instance optimization settings

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt  # For GPU acceleration
```

### Running ML API Server
```bash
# Production server
python ml_api_server.py --host 0.0.0.0 --port 8080

# Development server with auto-restart  
./start_ml_server.sh

# Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/transcribe -H "Content-Type: application/json" -d '{"video_path":"test.mp4"}'
```

### Development & Testing
```bash
# Code formatting and linting
black src/ config/ ml_api_server.py
ruff check src/ config/ ml_api_server.py --fix

# Security scanning
bandit -r src/ config/ ml_api_server.py

# Clean artifacts
find /Users/ahntaeju/project/ecg-audio-analyzer -name "__pycache__" -type d -exec rm -rf {} +
```

### Docker & Deployment
```bash
# Build GPU container
docker build -f Dockerfile -t ecg-analyzer .

# Deploy to AWS
./deployment/aws-deploy.sh -k your-keypair

# Run container
docker run --gpus all -p 8080:8080 ecg-analyzer
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
- **POST `/api/upload-video/process-video`** - Main video processing with progress callbacks
- **POST `/transcribe`** - Synchronous transcription with detailed results
- **POST `/request-process`** - Legacy async processing endpoint
- **GET `/health`** - Health check for load balancers

### Progress Callback System
The ML server sends real-time progress updates to the backend via HTTP callbacks:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 75,
  "message": "감정 분석 중...",
  "result": null
}
```

## Troubleshooting

### Common Issues
- **GPU Memory Errors**: Pipeline automatically falls back to CPU with reduced batch sizes
- **WhisperX Loading**: Large models cached in GPU memory (~2-4GB), first load may be slow  
- **S3 Access**: Ensure AWS credentials and bucket permissions are configured
- **FFmpeg Missing**: Install ffmpeg for audio/video processing: `brew install ffmpeg` (macOS)