# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**High-performance audio analysis pipeline** for dynamic subtitle generation. Extracts audio from MP4 files/URLs, performs speaker diarization, speech recognition with WhisperX, and generates JSON metadata. Optimized for AWS GPU instances (G4dn.xlarge/P3.2xlarge) with 30-second target processing time for 10-minute videos.

## Architecture

### Core Components
- **PipelineManager** (`src/pipeline/manager.py`) - Central orchestrator with async operations and GPU resource management
- **WhisperXPipeline** (`src/models/speech_recognizer.py`) - Unified speech recognition + speaker diarization
- **AudioExtractor** (`src/services/audio_extractor.py`) - MP4/URL → WAV conversion with duration validation
- **ResultSynthesizer** (`src/services/result_synthesizer.py`) - JSON output generation with performance metrics
- **ML API Server** (`ml_api_server.py`) - FastAPI server providing HTTP API for ECS backend integration

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

### Running Analysis
```bash
# CLI analysis
python main.py analyze input.mp4 --gpu --workers 4
python main.py analyze "https://youtube.com/watch?v=..." --workers 2

# Via module
python -m src.cli analyze video.mp4 --workers 4 --verbose
PYTHONPATH=/Users/ahntaeju/project/ecg-audio-analyzer python -m src.cli analyze video.mp4

# ML API server
python ml_api_server.py  # Starts FastAPI server on port 8000
```

### Development
```bash
# Code formatting
black src/ config/
ruff check src/ config/ --fix

# Clean artifacts
find /Users/ahntaeju/project/ecg-audio-analyzer -name "__pycache__" -type d -exec rm -rf {} +
```

### Docker & Deployment
```bash
# Build GPU container
docker-compose -f docker-compose.aws.yml build

# Deploy to AWS
./deployment/aws-deploy.sh -k your-keypair

# Local development
docker-compose up --build
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