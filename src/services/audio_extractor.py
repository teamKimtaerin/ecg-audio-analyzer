"""
Audio Extraction Service - Simplified version
Convert MP4/URLs to analysis-ready audio format
"""

# cSpell:ignore ytdl outtmpl noplaylist ffprobe acodec samplerate Pylance

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

import soundfile as sf
import ffmpeg
import boto3
from botocore.exceptions import ClientError
import os

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

        # Setup S3 client
        self.s3_client = None
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            )
        except Exception as e:
            self.logger.warning("s3_client_setup_failed", error=str(e))

        self.logger.info("audio_extractor_ready", temp_dir=str(self.temp_dir))

    def _get_video_duration(self, video_path: Union[str, Path]) -> float:
        """Get video duration using ffprobe"""
        try:
            probe = ffmpeg.probe(str(video_path))
            return float(probe["streams"][0]["duration"])
        except Exception as e:
            self.logger.warning("video_duration_probe_failed", error=str(e))
            return 0.0

    def _convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert video/audio to WAV using ffmpeg"""
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
            # Safe access to returncode attribute
            if hasattr(e, "returncode"):
                returncode = getattr(e, "returncode", "unknown")
                self.logger.error(
                    f"FFmpeg command failed with return code {returncode}"
                )
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
        is_url = source_str.startswith(("http://", "https://", "s3://"))

        try:
            # Handle URL
            if is_url:
                if source_str.startswith("s3://"):
                    video_path = self._download_from_s3(source_str)
                else:
                    return AudioExtractionResult(
                        success=False, error="Only S3 URLs are supported"
                    )

                if not video_path:
                    return AudioExtractionResult(
                        success=False, error="Failed to download from S3"
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

    def _download_from_s3(self, s3_url: str) -> Optional[Path]:
        """Download file from S3"""
        if not self.s3_client:
            self.logger.error("s3_client_not_available")
            return None

        try:
            # Parse S3 URL: s3://bucket/key
            if not s3_url.startswith("s3://"):
                self.logger.error("invalid_s3_url", url=s3_url)
                return None

            url_parts = s3_url[5:].split("/", 1)  # Remove s3:// prefix
            if len(url_parts) != 2:
                self.logger.error("malformed_s3_url", url=s3_url)
                return None

            bucket, key = url_parts

            # Extract file extension from key
            file_ext = Path(key).suffix or ".mp4"
            local_file = self.temp_dir / f"s3_download_{hash(s3_url)}{file_ext}"

            self.logger.info("downloading_from_s3", bucket=bucket, key=key, local_path=str(local_file))

            # Download file
            self.s3_client.download_file(bucket, key, str(local_file))

            if local_file.exists():
                self.logger.info("s3_download_success", local_path=str(local_file), size=local_file.stat().st_size)
                return local_file
            else:
                self.logger.error("s3_download_file_not_found", local_path=str(local_file))
                return None

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            self.logger.error("s3_download_client_error", url=s3_url, error_code=error_code, error=str(e))
            return None
        except Exception as e:
            self.logger.error("s3_download_failed", url=s3_url, error=str(e))
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
