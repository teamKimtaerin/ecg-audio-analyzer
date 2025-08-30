# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository appears to be intended for an ECG (electrocardiogram) audio analysis project. Based on the `.gitignore` file, this is expected to be a Python-based application.

## Development Setup

Since this is currently an empty repository, the development setup will need to be determined once source files are added. The `.gitignore` suggests this will be a Python project that may use various package managers (pip, poetry, UV, pdm, or pixi).

## Architecture

The codebase architecture has not yet been established. This section should be updated once the project structure is implemented.

## Commands

Development commands will need to be determined based on the chosen Python project structure and dependencies. Common commands for Python projects typically include:

- Running the application
- Installing dependencies 
- Running tests
- Linting and formatting
- Building/packaging

This section should be updated with specific commands once the project is implemented.

# Claude Instructions for ECG Audio Analyzer Development

## 🎯 Project Context
You are helping develop a **high-performance audio analysis pipeline** for dynamic subtitle generation. This system extracts audio from MP4 files, performs speaker diarization, emotion analysis, and acoustic feature extraction, then outputs comprehensive JSON metadata.

## 🏗️ Architecture Principles

### Single Responsibility Principle (SRP)
- **Each module has ONE clear purpose** and one reason to change
- **AudioExtractor**: Only handles MP4/URL → WAV conversion
- **SpeakerDiarizer**: Only handles speaker identification and segmentation
- **EmotionAnalyzer**: Only handles emotion classification from audio
- **AcousticAnalyzer**: Only handles acoustic feature extraction
- **PipelineManager**: Only handles workflow orchestration
- **ResultSynthesizer**: Only handles JSON output generation

### Performance-First Design
- **Speed is the primary concern** - optimize for AWS GPU instances
- **Target**: 10-minute video analyzed in 30 seconds on P3.2xlarge
- **Parallel processing** wherever possible
- **GPU acceleration** for all ML inference
- **Memory efficiency** for large file handling
- **Async operations** for I/O-bound tasks

## 🚀 AWS GPU Infrastructure

### Target Deployment Environment
```yaml
Instance: EC2 P3.2xlarge (NVIDIA V100) or G4dn.2xlarge (NVIDIA T4)
Memory: 16GB+ RAM
Storage: EBS GP3 with high IOPS
CUDA: Version 11.8+ with cuDNN
Docker: nvidia-docker2 for containerization
```

### Performance Optimization Guidelines
1. **Use GPU acceleration** for all ML models (pyannote, transformers, etc.)
2. **Implement batch processing** for multiple audio segments
3. **Use async/await** for file I/O operations
4. **Cache model weights** in GPU memory when possible
5. **Implement chunked processing** for long audio files
6. **Monitor GPU utilization** and optimize batch sizes

## 🔧 Technical Implementation Guidelines

### Code Structure Requirements
```python
# Each service class should follow this pattern:
class ServiceName:
    def __init__(self, config: Config, gpu_device: str = "cuda:0"):
        self.config = config
        self.device = gpu_device
        self._model = None
        
    def load_model(self) -> None:
        """Load and cache model in GPU memory"""
        
    def process(self, input_data: Any) -> ServiceResult:
        """Main processing method - single responsibility"""
        
    def batch_process(self, batch_data: List[Any]) -> List[ServiceResult]:
        """Optimized batch processing for multiple inputs"""
```

### Performance Monitoring
- **Include performance metrics** in all major functions
- **Log GPU memory usage** before/after model inference
- **Track processing time** for each pipeline stage
- **Monitor CPU/GPU utilization** throughout pipeline
- **Implement graceful degradation** if GPU unavailable

### Error Handling Strategy
```python
# Implement robust error handling with performance consideration
try:
    result = gpu_intensive_operation(data)
except torch.cuda.OutOfMemoryError:
    # Fallback to smaller batch size or CPU processing
    result = fallback_operation(data)
except Exception as e:
    # Log error with context, continue pipeline if possible
    logger.error(f"Stage failed: {e}", extra={"stage": "emotion_analysis"})
```

## 📋 Development Priorities

### Phase 1: Core Pipeline (SRP Focus)
1. Implement each service as a separate, testable class
2. Create pipeline manager with clear orchestration logic
3. Add comprehensive logging and performance monitoring
4. Implement basic GPU acceleration

### Phase 2: Performance Optimization
1. Optimize GPU memory usage and batch processing
2. Implement async operations for I/O
3. Add model caching and warm-up procedures
4. Performance benchmarking and bottleneck identification

### Phase 3: AWS Integration
1. Containerize with nvidia-docker
2. Implement S3 integration for input/output
3. Add CloudWatch monitoring
4. Auto-scaling configuration

### Phase 4: Production Readiness
1. Comprehensive error handling and recovery
2. API endpoints for remote processing
3. Queue management for batch jobs
4. Performance tuning and optimization

## 🎛️ Configuration Management

### Settings Structure
```python
# config/aws_settings.py
@dataclass
class AWSConfig:
    instance_type: str = "p3.2xlarge"
    gpu_memory_limit: float = 0.8  # Use 80% of GPU memory
    batch_size: int = 8
    concurrent_workers: int = 4
    s3_bucket: str = "ecg-audio-processing"
    
# config/model_configs.py
@dataclass
class ModelConfig:
    speaker_model: str = "pyannote/speaker-diarization"
    emotion_model: str = "facebook/wav2vec2-emotion"
    device: str = "cuda:0"
    precision: str = "fp16"  # Use half precision for speed
```

## 🔍 Testing Strategy

### Unit Tests (Each Service)
- Test each service in isolation
- Mock external dependencies
- Performance benchmarks for each component
- GPU memory usage validation

### Integration Tests
- End-to-end pipeline testing
- Error handling validation
- Performance regression tests
- AWS deployment verification

## 💡 Key Reminders for Claude

1. **Prioritize speed and efficiency** in all code suggestions
2. **Maintain strict SRP** - one responsibility per class/module
3. **Always consider GPU optimization** when suggesting ML code
4. **Include performance monitoring** in code examples
5. **Design for AWS GPU instances** - not local development
6. **Use async/parallel processing** wherever beneficial
7. **Implement proper error handling** without breaking the pipeline
8. **Focus on production-ready code** rather than prototypes

## 🎯 Success Metrics

- **Processing Speed**: 30 seconds for 10-minute video on P3.2xlarge
- **Accuracy**: 97%+ speaker diarization, 88%+ emotion classification
- **Scalability**: Handle 10+ concurrent video analyses
- **Reliability**: 99.5%+ successful processing rate
- **Cost Efficiency**: Optimize GPU utilization above 80%

---
**Remember: This is a production system designed for speed and accuracy. Every design decision should optimize for performance while maintaining clean, maintainable code structure.**