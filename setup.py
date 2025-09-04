#!/usr/bin/env python3
"""
Setup script for ECG Audio Analysis
Makes the package installable in development mode
"""

from setuptools import setup, find_packages

setup(
    name="ecg-audio-analyzer",
    version="1.0.0",
    description="High-performance audio analysis for dynamic subtitle generation",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "boto3>=1.26.0",
        "botocore>=1.29.0",
        "yt-dlp>=2023.7.6",
        "ffmpeg-python>=0.2.0",
        "psutil>=5.9.0",
        "aiofiles>=23.1.0",
        "asyncio-throttle>=1.0.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=1.10.0",
        "pydantic-settings>=2.0.0",
        "structlog>=23.1.0",
        "rich>=13.3.0",
        "typer>=0.9.0",
        "python-dotenv>=1.0.0",
        "toml>=0.10.2",
        "whisperx>=3.0.0",
    ],
    extras_require={
        "gpu": [
            "pyannote-audio>=3.0.0",
            "speechbrain>=0.5.0",
            "datasets>=2.12.0",
            "accelerate>=0.20.0",
            "opensmile>=2.4.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "ruff>=0.0.270",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecg-analyze=main:main",
        ],
    },
)
