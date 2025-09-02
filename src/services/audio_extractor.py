"""
Audio Extraction Service - Simplified version
Convert MP4/URLs to analysis-ready audio format
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import subprocess

import soundfile as sf
import yt_dlp
import ffmpeg

from ..utils.logger import get_logger


@dataclass
class AudioExtractionResult:
    """Result of audio extraction"""
    success: bool
    output_path: Optional[Path] = None
    duration: float = 0.0
    sample_rate: int = 16000
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output_path': str(self.output_path) if self.output_path else None,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'error': self.error
        }


class AudioExtractor:
    """
    Simple audio extraction service
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 temp_dir: Optional[Path] = None):
        
        self.target_sr = target_sr
        self.logger = get_logger().bind_context(service="audio_extractor")
        
        # Setup temp directory
        self.temp_dir = Path(temp_dir or tempfile.gettempdir()) / "audio_extract"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check ffmpeg availability
        if not shutil.which('ffmpeg'):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        
        # Simple yt-dlp config
        self.ytdl_opts = {
            'format': 'best',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True
        }
        
        self.logger.info("audio_extractor_ready", temp_dir=str(self.temp_dir))
    
    def extract(self, 
                source: Union[str, Path],
                output_path: Optional[Path] = None) -> AudioExtractionResult:
        """
        Extract audio from file or URL
        
        Args:
            source: File path or URL
            output_path: Optional output path
            
        Returns:
            AudioExtractionResult
        """
        source_str = str(source)
        is_url = source_str.startswith(('http://', 'https://'))
        
        try:
            # Handle URL
            if is_url:
                video_path = self._download_url(source_str)
                if not video_path:
                    return AudioExtractionResult(
                        success=False,
                        error="Failed to download URL"
                    )
            else:
                video_path = Path(source)
                if not video_path.exists():
                    return AudioExtractionResult(
                        success=False,
                        error=f"File not found: {source}"
                    )
            
            # Convert to WAV
            if not output_path:
                output_path = self.temp_dir / f"{video_path.stem}_audio.wav"
            
            success = self._convert_to_wav(video_path, output_path)
            
            if not success:
                return AudioExtractionResult(
                    success=False,
                    error="Conversion failed"
                )
            
            # Get audio info
            info = sf.info(str(output_path))
            
            # Clean up downloaded file if from URL
            if is_url and video_path.exists():
                try:
                    video_path.unlink()
                except:
                    pass
            
            return AudioExtractionResult(
                success=True,
                output_path=output_path,
                duration=info.duration,
                sample_rate=info.samplerate
            )
            
        except Exception as e:
            self.logger.error("extraction_failed", source=source_str, error=str(e))
            return AudioExtractionResult(
                success=False,
                error=str(e)
            )
    
    def _download_url(self, url: str) -> Optional[Path]:
        """Download video from URL"""
        try:
            with yt_dlp.YoutubeDL(self.ytdl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None
                
                # Find downloaded file
                title = info.get('title', 'download').replace('/', '_')
                for ext in ['.mp4', '.webm', '.mkv', '.m4a', '.mp3']:
                    file_path = self.temp_dir / f"{title}{ext}"
                    if file_path.exists():
                        return file_path
                
                # Fallback: find most recent file
                files = list(self.temp_dir.glob("*"))
                if files:
                    return max(files, key=lambda p: p.stat().st_mtime)
                
                return None
                
        except Exception as e:
            self.logger.error("download_failed", url=url, error=str(e))
            return None
    
    def _convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert video/audio to WAV"""
        try:
            # Use ffmpeg-python for cleaner API
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le',        # 16-bit PCM
                ac=1,                       # Mono
                ar=self.target_sr,          # Target sample rate
                loglevel='error'
            )
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return output_path.exists()
            
        except ffmpeg.Error as e:
            self.logger.error("ffmpeg_error", error=str(e))
            return False
        except Exception as e:
            self.logger.error("conversion_error", error=str(e))
            return False
    
    def cleanup(self):
        """Clean up temp files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("cleanup_completed")
        except Exception as e:
            self.logger.warning("cleanup_failed", error=str(e))


# Convenience function
def extract_audio(source: Union[str, Path], 
                 output_path: Optional[Path] = None) -> AudioExtractionResult:
    """
    Quick audio extraction
    
    Args:
        source: File path or URL
        output_path: Optional output path
        
    Returns:
        AudioExtractionResult
    """
    extractor = AudioExtractor()
    try:
        return extractor.extract(source, output_path)
    finally:
        extractor.cleanup()