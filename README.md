# ECG Audio Analyzer

🎵 **High-Performance Audio Analysis Library** for dynamic subtitle generation with speaker diarization and emotion detection.

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://pypi.org/project/ecg-audio-analyzer/)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/speed-30x_realtime-red.svg)](#performance)

## ✨ Features

- **🎯 Speaker Diarization**: Identify and segment multiple speakers in audio
- **😊 Emotion Analysis**: Real-time emotion detection for each speaker segment  
- **🔊 Acoustic Features**: Advanced audio feature extraction (MFCC, spectral features)
- **⚡ High Performance**: 30x+ real-time processing speed on CPU, 100x+ on GPU
- **🐳 Docker Ready**: Complete containerization for scalable deployment
- **☁️ AWS Integration**: CloudFormation, S3, CloudWatch, ECR support
- **📊 Rich Output**: Comprehensive JSON results with confidence scores
- **🔧 Easy Integration**: Simple API for use in other projects

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install ecg-audio-analyzer

# With GPU support
pip install ecg-audio-analyzer[gpu]

# With AWS integration
pip install ecg-audio-analyzer[aws]

# Development installation
pip install ecg-audio-analyzer[dev]
```

### Basic Usage

```python
from ecg_audio_analyzer import analyze_audio, AnalysisConfig

# Simple analysis
result = await analyze_audio("path/to/video.mp4")
print(f"Found {result.unique_speakers} speakers")

# Advanced configuration
config = AnalysisConfig(
    enable_gpu=True,
    emotion_detection=True,
    detailed_features=True,
    segment_length=5.0
)

result = await analyze_audio("video.mp4", config=config)

# Access results
for segment in result.segments:
    print(f"Speaker {segment.speaker.speaker_id}: "
          f"{segment.emotion.emotion} "
          f"({segment.start_time:.1f}s - {segment.end_time:.1f}s)")
```

### Command Line Interface

```bash
# Analyze a video file
ecg-analyze video.mp4 --output results.json

# With GPU acceleration
ecg-analyze video.mp4 --gpu --detailed-features

# Process YouTube URL
ecg-analyze "https://youtube.com/watch?v=..." --emotion-detection
```

## 📊 Performance

| Input Duration | Processing Time | Speed Ratio | Memory Usage |
|----------------|-----------------|-------------|--------------|
| 30 seconds     | 2.1s            | 14.3x       | < 500MB      |
| 2 minutes      | 4.3s            | 33.2x       | < 1GB        |
| 10 minutes     | 18s             | 33.3x       | < 2GB        |

*Benchmarked on MacBook Air M2 (CPU) and AWS g4dn.2xlarge (GPU)*

## 📋 Supported Formats

- **Input**: MP4, WAV, YouTube URLs, audio streams
- **Output**: JSON, dict objects for programmatic use
- **Languages**: English (more coming soon)

## 🏗️ Architecture

Built with **Single Responsibility Principle (SRP)** for maximum maintainability:

```
ecg_audio_analyzer/
├── api.py              # Public API interface
├── services/           # Individual analysis services
│   ├── audio_extractor.py
│   ├── speaker_diarizer.py
│   └── emotion_analyzer.py
├── pipeline/           # Service orchestration
├── models/             # Pydantic data models
└── utils/              # Shared utilities
```

## 🐳 Docker Deployment

```bash
# Build GPU-optimized container
docker build -f docker/Dockerfile.gpu -t ecg-analyzer .

# Run analysis service
docker run --gpus all -v ./data:/data ecg-analyzer python -c "
from ecg_audio_analyzer import analyze_audio_sync
result = analyze_audio_sync('/data/video.mp4')
print(result.to_json())
"
```

## ☁️ AWS Deployment

Complete AWS infrastructure with one command:

```bash
# Deploy to AWS with GPU instances
./deployment/aws-deploy.sh -k your-keypair

# Includes:
# - Auto Scaling Group with GPU instances
# - S3 bucket for file storage
# - CloudWatch monitoring
# - ECR container registry
```

## 🔧 Configuration

```python
config = AnalysisConfig(
    # Performance
    enable_gpu=True,
    max_workers=4,
    sample_rate=22050,
    
    # Features
    speaker_diarization=True,
    emotion_detection=True,
    acoustic_features=True,
    detailed_features=False,
    
    # Segmentation
    segment_length=5.0,
    min_segment_length=1.0,
    confidence_threshold=0.5,
    
    # Language
    language="en",
)
```

## 📊 Output Format

```json
{
  "metadata": {
    "filename": "video.mp4",
    "duration": 143.36,
    "unique_speakers": 2,
    "total_segments": 29,
    "processing_time": 4.32,
    "dominant_emotion": "happy"
  },
  "speakers": {
    "speaker_01": {
      "total_duration": 83.36,
      "segment_count": 17,
      "avg_confidence": 0.85,
      "emotions": ["happy", "neutral", "sad"]
    }
  },
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.0,
      "speaker": {
        "speaker_id": "speaker_01",
        "confidence": 0.87
      },
      "emotion": {
        "emotion": "happy",
        "confidence": 0.92
      },
      "acoustic_features": {
        "spectral_centroid": 2413.96,
        "energy": 0.045
      }
    }
  ]
}
```

## 🤝 Integration Examples

### FastAPI Service

```python
# separate_api_project/main.py
from fastapi import FastAPI, UploadFile
from ecg_audio_analyzer import analyze_audio_sync, AnalysisConfig

app = FastAPI()

@app.post("/analyze")
async def analyze_file(file: UploadFile):
    config = AnalysisConfig(enable_gpu=True)
    result = analyze_audio_sync(file.file, config=config)
    return result.to_dict()
```

### Jupyter Notebook

```python
# Install in notebook
!pip install ecg-audio-analyzer

# Analyze audio
from ecg_audio_analyzer import analyze_audio_sync
result = analyze_audio_sync("sample.mp4")

# Visualize results
import matplotlib.pyplot as plt
speakers = [seg.speaker.speaker_id for seg in result.segments]
plt.hist(speakers)
plt.title(f"Speaker Distribution ({result.unique_speakers} speakers)")
```

## 📈 Roadmap

- [ ] Real-time streaming analysis
- [ ] Multi-language support (Spanish, French, etc.)
- [ ] Speech-to-text integration
- [ ] Advanced emotion models
- [ ] Kubernetes deployment
- [ ] REST API service template

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 💬 Support

- 📧 Email: team@ecg-audio.ai
- 🐛 Issues: [GitHub Issues](https://github.com/ecg-team/ecg-audio-analyzer/issues)
- 📖 Docs: [Documentation](https://ecg-audio-analyzer.readthedocs.io/)

---

**Made with ❤️ by the ECG Team** | High-performance audio analysis for the modern web