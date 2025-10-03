"""
Audio Extraction Service - Simplified version
Convert MP4/URLs to analysis-ready audio format
"""

# cSpell:ignore ytdl outtmpl noplaylist ffprobe acodec samplerate Pylance hwaccel

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

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
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "error": self.error,
        }


class AudioExtractor:
    """
    Simple audio extraction service
    """

    def __init__(
        self,
        target_sr: int = 16000,
        temp_dir: Optional[Path] = None,
    ):
        self.target_sr = target_sr
        self.logger = get_logger().bind_context(service="audio_extractor")

        # Setup temp directory
        self.temp_dir = Path(temp_dir or tempfile.gettempdir()) / "audio_extract"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Check ffmpeg availability
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

        # Simple yt-dlp config
        self.ytdl_opts = {
            "format": "best",
            "outtmpl": str(self.temp_dir / "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        self.logger.info("audio_extractor_ready", temp_dir=str(self.temp_dir))

    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available for ffmpeg"""
        try:
            # PyTorch를 사용하여 CUDA 가능 여부 확인
            import torch

            if not torch.cuda.is_available():
                return False

            # nvidia-ml-py를 사용하여 GPU 메모리 확인 (선택적)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # GPU 메모리가 충분한지 확인 (최소 1GB)
                if int(meminfo.free) < 1024 * 1024 * 1024:
                    self.logger.warning("GPU memory low, falling back to CPU")
                    return False
                return True
            except (ImportError, Exception):
                # pynvml이 없거나 오류가 있으면 기본적으로 CUDA만 확인
                return True
        except ImportError:
            # PyTorch가 없으면 CPU 사용
            return False

    def _get_video_duration(self, video_path: Union[str, Path]) -> float:
        """Get video duration using ffprobe"""
        try:
            probe = ffmpeg.probe(str(video_path))
            return float(probe["streams"][0]["duration"])
        except Exception as e:
            self.logger.warning("video_duration_probe_failed", error=str(e))
            return 0.0

    def _convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert video/audio to WAV using ffmpeg with GPU acceleration"""
        try:
            # 입력 파일 크기 확인
            if not input_path.exists():
                raise FileNotFoundError(f"Input file does not exist: {input_path}")

            input_size = input_path.stat().st_size
            if input_size == 0:
                raise ValueError(f"Input file is empty (0 bytes): {input_path}")

            self.logger.info(
                f"Converting {input_path} ({input_size/1024/1024:.1f}MB) to WAV"
            )

            # GPU 가속 가능 여부 확인
            gpu_available = self._check_gpu_available()

            # GPU 가속을 사용한 input stream 생성
            if gpu_available:
                self.logger.debug("Using GPU acceleration for video decoding")
                stream = ffmpeg.input(
                    str(input_path),
                    threads=1,  # GPU decoding에는 thread=1이 최적
                    hwaccel="cuda",
                )
            else:
                self.logger.debug("Using CPU for video decoding")
                stream = ffmpeg.input(str(input_path))

            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="pcm_s16le",
                ac=1,
                ar=self.target_sr,
                loglevel="error",  # 더 상세한 에러 메시지를 위해 'error'로 변경
            )

            # stderr 캡처를 위해 run 옵션 수정
            ffmpeg.run(
                stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

            # 출력 파일 크기 확인
            if not output_path.exists():
                raise RuntimeError(f"Output file was not created: {output_path}")

            output_size = output_path.stat().st_size
            if output_size == 0:
                raise RuntimeError(
                    f"Output file is empty after conversion: {output_path}"
                )

            self.logger.info(
                f"Conversion successful: {output_path} ({output_size/1024/1024:.1f}MB)"
            )
            return True

        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode("utf-8") if e.stderr else "No stderr output"
            self.logger.error(f"FFmpeg conversion failed: {stderr_output}")
            return False
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            return False

    def extract(
        self, source: Union[str, Path], output_path: Optional[Path] = None
    ) -> AudioExtractionResult:
        """
        Extract audio from file or URL

        Args:
            source: File path or URL
            output_path: Optional output path

        Returns:
            AudioExtractionResult
        """
        source_str = str(source)
        is_url = source_str.startswith(("http://", "https://"))

        try:
            # Handle URL
            if is_url:
                video_path = self._download_url(source_str)
                if not video_path:
                    return AudioExtractionResult(
                        success=False, error="Failed to download URL"
                    )
            else:
                video_path = Path(source)
                if not video_path.exists():
                    return AudioExtractionResult(
                        success=False, error=f"File not found: {source}"
                    )

            # Convert to WAV
            if not output_path:
                output_path = self.temp_dir / f"{video_path.stem}_audio.wav"

            # Convert video/audio to WAV
            success = self._convert_to_wav(video_path, output_path)

            if not success:
                return AudioExtractionResult(success=False, error="Conversion failed")

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
                sample_rate=info.samplerate,
            )

        except Exception as e:
            self.logger.error("extraction_failed", source=source_str, error=str(e))
            return AudioExtractionResult(success=False, error=str(e))

    def _download_url(self, url: str) -> Optional[Path]:
        """Download video from URL"""
        try:
            # Type ignore for Pylance false positive - yt_dlp.YoutubeDL accepts dict as first arg
            with yt_dlp.YoutubeDL(self.ytdl_opts) as ydl:  # type: ignore[reportArgumentType]
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None

                # Find downloaded file
                title = (info.get("title") or "download").replace("/", "_")
                for ext in [".mp4", ".webm", ".mkv", ".m4a", ".mp3"]:
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

    def cleanup(self):
        """Clean up temp files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("cleanup_completed")
        except Exception as e:
            self.logger.warning("cleanup_failed", error=str(e))


# Convenience function
def extract_audio(
    source: Union[str, Path], output_path: Optional[Path] = None
) -> AudioExtractionResult:
    extractor = AudioExtractor()
    try:
        return extractor.extract(source, output_path)
    finally:
        extractor.cleanup()
