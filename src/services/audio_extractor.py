"""
Audio Extraction Service
Single Responsibility: Convert MP4/URLs to analysis-ready audio format

Optimized for AWS GPU instances with hardware acceleration support.
"""

import asyncio
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import subprocess

import librosa
import soundfile as sf
from pydub import AudioSegment
import yt_dlp
import ffmpeg

from ..utils.logger import get_logger
from ...config.base_settings import BaseConfig, ProcessingConfig, ValidationConfig


@dataclass
class AudioExtractionResult:
    """Result of audio extraction process"""
    success: bool
    output_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 0
    channels: int = 0
    file_size_mb: float = 0.0
    extraction_time_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AudioExtractor:
    """
    High-performance audio extraction service with hardware acceleration.
    
    Single Responsibility: Convert various input formats to normalized WAV files
    suitable for speech analysis pipeline.
    """
    
    def __init__(self, 
                 config: BaseConfig,
                 processing_config: ProcessingConfig,
                 validation_config: ValidationConfig,
                 temp_dir: Optional[Path] = None,
                 enable_gpu_acceleration: bool = True):
        
        self.config = config
        self.processing_config = processing_config
        self.validation_config = validation_config
        self.logger = get_logger().bind_context(service="audio_extractor")
        self.enable_gpu_acceleration = enable_gpu_acceleration
        
        # Setup temporary directory
        self.temp_dir = temp_dir or config.temp_dir / "audio_extraction"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=processing_config.max_workers)
        
        # Check for required tools
        self._validate_dependencies()
        
        # Configure yt-dlp options
        self.ytdl_opts = {
            'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'noplaylist': True,
            'extract_flat': False,
            'writethumbnail': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
        
        self.logger.info("audio_extractor_initialized", 
                        temp_dir=str(self.temp_dir),
                        gpu_acceleration=enable_gpu_acceleration)
    
    def _validate_dependencies(self) -> None:
        """Validate required system dependencies"""
        required_tools = ['ffmpeg', 'ffprobe']
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            raise RuntimeError(f"Missing required tools: {missing_tools}")
        
        self.logger.info("dependencies_validated", tools=required_tools)
    
    def _get_ffmpeg_gpu_options(self) -> Dict[str, str]:
        """Get FFmpeg GPU acceleration options if available"""
        if not self.enable_gpu_acceleration:
            return {}
        
        try:
            # Test for NVIDIA GPU support
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return {
                    'hwaccel': 'cuda',
                    'hwaccel_output_format': 'cuda'
                }
        except FileNotFoundError:
            pass
        
        # Fallback to software decoding
        self.logger.info("gpu_acceleration_unavailable", fallback="software")
        return {}
    
    def _validate_audio_file(self, file_path: Path) -> bool:
        """Validate audio file meets quality requirements"""
        try:
            info = sf.info(str(file_path))
            
            # Check duration
            if info.duration < self.validation_config.min_duration_seconds:
                self.logger.warning("audio_too_short", 
                                  duration=info.duration,
                                  min_required=self.validation_config.min_duration_seconds)
                return False
            
            if info.duration > self.validation_config.max_duration_seconds:
                self.logger.warning("audio_too_long", 
                                  duration=info.duration,
                                  max_allowed=self.validation_config.max_duration_seconds)
                return False
            
            # Check sample rate
            if info.samplerate < self.validation_config.min_sample_rate:
                self.logger.warning("sample_rate_too_low", 
                                  sample_rate=info.samplerate,
                                  min_required=self.validation_config.min_sample_rate)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("audio_validation_failed", error=str(e))
            return False
    
    def _extract_from_url(self, url: str) -> AudioExtractionResult:
        """Extract audio from URL using yt-dlp"""
        with self.logger.performance_timer("url_download"):
            try:
                with yt_dlp.YoutubeDL(self.ytdl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        return AudioExtractionResult(
                            success=False,
                            error_message="Failed to extract video information"
                        )
                    
                    title = info.get('title', 'unknown')
                    duration = info.get('duration', 0)
                    
                    self.logger.info("url_info_extracted", 
                                   title=title, 
                                   duration=duration,
                                   url=url)
                    
                    # Validate duration before download
                    if duration and duration > self.validation_config.max_duration_seconds:
                        return AudioExtractionResult(
                            success=False,
                            error_message=f"Video too long: {duration}s > {self.validation_config.max_duration_seconds}s"
                        )
                    
                    # Download video
                    ydl.download([url])
                    
                    # Find downloaded file
                    downloaded_files = list(self.temp_dir.glob("*"))
                    if not downloaded_files:
                        return AudioExtractionResult(
                            success=False,
                            error_message="No file downloaded"
                        )
                    
                    video_path = max(downloaded_files, key=lambda p: p.stat().st_mtime)
                    
                    # Convert to audio
                    return self._convert_to_wav(video_path, cleanup_source=True)
                    
            except Exception as e:
                self.logger.error("url_extraction_failed", url=url, error=str(e))
                return AudioExtractionResult(
                    success=False,
                    error_message=f"URL extraction failed: {str(e)}"
                )
    
    def _convert_to_wav(self, 
                       input_path: Path, 
                       output_path: Optional[Path] = None,
                       cleanup_source: bool = False) -> AudioExtractionResult:
        """Convert input file to normalized WAV format"""
        
        with self.logger.performance_timer("audio_conversion"):
            try:
                if output_path is None:
                    output_path = self.temp_dir / f"{input_path.stem}_converted.wav"
                
                # Get GPU acceleration options
                gpu_opts = self._get_ffmpeg_gpu_options()
                
                # Build FFmpeg command with optimizations
                stream = ffmpeg.input(str(input_path), **gpu_opts)
                
                # Audio processing pipeline
                stream = ffmpeg.output(
                    stream,
                    str(output_path),
                    acodec='pcm_s16le',          # 16-bit PCM
                    ac=self.config.channels,     # Mono for speech
                    ar=self.config.sample_rate,  # Standard sample rate
                    af='volume=0.8',             # Normalize volume
                    loglevel='error'
                )
                
                # Run conversion
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Validate output
                if not self._validate_audio_file(output_path):
                    return AudioExtractionResult(
                        success=False,
                        error_message="Output validation failed"
                    )
                
                # Get file info
                info = sf.info(str(output_path))
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                
                # Cleanup source if requested
                if cleanup_source and input_path != output_path:
                    try:
                        input_path.unlink()
                    except Exception as e:
                        self.logger.warning("cleanup_failed", 
                                          file=str(input_path), 
                                          error=str(e))
                
                self.logger.info("audio_conversion_successful",
                               input_file=str(input_path),
                               output_file=str(output_path),
                               duration=info.duration,
                               sample_rate=info.samplerate,
                               channels=info.channels,
                               file_size_mb=file_size_mb)
                
                return AudioExtractionResult(
                    success=True,
                    output_path=output_path,
                    duration_seconds=info.duration,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    file_size_mb=file_size_mb,
                    metadata={
                        'original_file': str(input_path),
                        'conversion_method': 'ffmpeg',
                        'gpu_acceleration': bool(gpu_opts)
                    }
                )
                
            except Exception as e:
                self.logger.error("audio_conversion_failed", 
                                input_file=str(input_path), 
                                error=str(e))
                return AudioExtractionResult(
                    success=False,
                    error_message=f"Conversion failed: {str(e)}"
                )
    
    def extract_single(self, 
                      source: Union[str, Path],
                      output_path: Optional[Path] = None) -> AudioExtractionResult:
        """
        Extract audio from a single source (file or URL).
        
        Args:
            source: File path or URL to extract audio from
            output_path: Optional output path for the converted audio
            
        Returns:
            AudioExtractionResult with extraction details
        """
        
        with self.logger.performance_timer("extract_single", items_count=1):
            
            # Determine source type
            source_str = str(source)
            is_url = urlparse(source_str).scheme in ('http', 'https')
            
            self.logger.info("extraction_started", 
                           source=source_str, 
                           is_url=is_url,
                           output_path=str(output_path) if output_path else None)
            
            if is_url:
                return self._extract_from_url(source_str)
            else:
                source_path = Path(source)
                if not source_path.exists():
                    return AudioExtractionResult(
                        success=False,
                        error_message=f"File not found: {source}"
                    )
                
                # Check file size
                file_size_gb = source_path.stat().st_size / (1024**3)
                if file_size_gb > self.config.max_file_size_gb:
                    return AudioExtractionResult(
                        success=False,
                        error_message=f"File too large: {file_size_gb:.1f}GB > {self.config.max_file_size_gb}GB"
                    )
                
                return self._convert_to_wav(source_path, output_path)
    
    async def extract_batch(self, 
                           sources: List[Union[str, Path]],
                           output_dir: Optional[Path] = None) -> List[AudioExtractionResult]:
        """
        Extract audio from multiple sources in parallel.
        
        Args:
            sources: List of file paths or URLs
            output_dir: Optional directory for output files
            
        Returns:
            List of AudioExtractionResult objects
        """
        
        with self.logger.performance_timer("extract_batch", items_count=len(sources)):
            
            self.logger.info("batch_extraction_started", 
                           source_count=len(sources),
                           output_dir=str(output_dir) if output_dir else None)
            
            # Create output directory if specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create async tasks for each source
            async def extract_async(source, index):
                output_path = None
                if output_dir:
                    source_name = Path(str(source)).stem if not str(source).startswith('http') else f"url_{index}"
                    output_path = output_dir / f"{source_name}.wav"
                
                # Run extraction in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_pool, 
                    self.extract_single, 
                    source, 
                    output_path
                )
            
            # Execute all extractions concurrently
            tasks = [extract_async(source, i) for i, source in enumerate(sources)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error("batch_extraction_exception", 
                                    source_index=i, 
                                    error=str(result))
                    processed_results.append(
                        AudioExtractionResult(
                            success=False,
                            error_message=f"Exception: {str(result)}"
                        )
                    )
                else:
                    processed_results.append(result)
            
            # Log batch summary
            successful = sum(1 for r in processed_results if r.success)
            failed = len(processed_results) - successful
            
            self.logger.info("batch_extraction_completed", 
                           total=len(sources),
                           successful=successful,
                           failed=failed)
            
            return processed_results
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files created during extraction"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("temp_files_cleaned", temp_dir=str(self.temp_dir))
        except Exception as e:
            self.logger.error("cleanup_failed", error=str(e))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.thread_pool.shutdown(wait=True)
        if self.config.enable_cleanup:
            self.cleanup_temp_files()
        
        if exc_type is not None:
            self.logger.error("audio_extractor_exception", 
                            exception_type=str(exc_type),
                            exception_message=str(exc_val))