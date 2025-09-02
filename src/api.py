"""
ECG Audio Analyzer - Public API

Simple and clean interface for external usage.
This module provides the main entry points for the ECG Audio Analyzer.
"""

import asyncio
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# Import internal modules
from .utils.logger import get_logger


@dataclass
class AnalysisConfig:
    """Simplified configuration for audio analysis"""
    # Core processing options
    enable_gpu: bool = False
    emotion_detection: bool = True
    language: str = "en"  # or "auto" for auto-detection
    
    # Output optimization
    optimize_for_subtitles: bool = True
    
    # Performance optimization
    max_workers: int = 4  # Number of parallel workers
    optimize_cpu: bool = True  # Enable CPU optimizations
    
    # Internal settings (optimized defaults)
    sample_rate: int = 16000  # Optimized for WhisperX
    segment_length: float = 5.0
    min_segment_length: float = 1.0
    confidence_threshold: float = 0.5
    
    # Features enabled by default
    speaker_diarization: bool = True
    acoustic_features: bool = True


@dataclass
class AcousticFeatures:
    """Acoustic features for an audio segment"""
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    energy: float
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    mfcc_mean: Optional[List[float]] = None


@dataclass
class EmotionInfo:
    """Emotion analysis result"""
    emotion: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


@dataclass
class SpeakerInfo:
    """Speaker information"""
    speaker_id: str
    confidence: float
    gender: Optional[str] = None
    age_group: Optional[str] = None


@dataclass 
class AudioSegment:
    """Individual audio segment with analysis results"""
    start_time: float
    end_time: float
    duration: float
    speaker: SpeakerInfo
    emotion: Optional[EmotionInfo] = None
    acoustic_features: Optional[AcousticFeatures] = None
    text: Optional[str] = None  # For future speech-to-text integration
    

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    # Metadata
    filename: str
    duration: float
    sample_rate: int
    processed_at: datetime
    processing_time: float
    
    # Analysis results
    segments: List[AudioSegment]
    speakers: Dict[str, Dict[str, Any]]
    
    # Summary statistics
    total_segments: int
    unique_speakers: int
    dominant_emotion: Optional[str]
    avg_confidence: float
    
    # Additional data
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "metadata": {
                "filename": self.filename,
                "duration": self.duration,
                "sample_rate": self.sample_rate,
                "processed_at": self.processed_at.isoformat(),
                "processing_time": self.processing_time,
                "total_segments": self.total_segments,
                "unique_speakers": self.unique_speakers,
                "dominant_emotion": self.dominant_emotion,
                "avg_confidence": self.avg_confidence,
                **self.metadata
            },
            "speakers": self.speakers,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "speaker": {
                        "speaker_id": seg.speaker.speaker_id,
                        "confidence": seg.speaker.confidence,
                        "gender": seg.speaker.gender,
                        "age_group": seg.speaker.age_group,
                    },
                    "emotion": {
                        "emotion": seg.emotion.emotion,
                        "confidence": seg.emotion.confidence,
                        "probabilities": seg.emotion.probabilities,
                    } if seg.emotion else None,
                    "acoustic_features": {
                        "spectral_centroid": seg.acoustic_features.spectral_centroid,
                        "spectral_rolloff": seg.acoustic_features.spectral_rolloff,
                        "zero_crossing_rate": seg.acoustic_features.zero_crossing_rate,
                        "energy": seg.acoustic_features.energy,
                        "pitch_mean": seg.acoustic_features.pitch_mean,
                        "pitch_std": seg.acoustic_features.pitch_std,
                        "mfcc_mean": seg.acoustic_features.mfcc_mean,
                    } if seg.acoustic_features else None,
                    "text": seg.text,
                }
                for seg in self.segments
            ]
        }
        
        # Add subtitle optimization data if available
        if hasattr(self, 'subtitle_data') and self.subtitle_data:
            result.update(self.subtitle_data)
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)


async def analyze_audio(
    input_path: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
    output_path: Optional[Union[str, Path]] = None
) -> AnalysisResult:
    """
    Analyze audio file with speaker diarization and emotion detection.
    
    Args:
        input_path: Path to audio/video file or YouTube URL
        config: Analysis configuration (optional)
        output_path: Path to save results (optional)
    
    Returns:
        AnalysisResult object with complete analysis
        
    Example:
        # Basic usage
        result = await analyze_audio("video.mp4")
        
        # Advanced usage
        config = AnalysisConfig(enable_gpu=True, detailed_features=True)
        result = await analyze_audio("video.mp4", config=config)
        print(f"Found {result.unique_speakers} speakers")
    """
    
    # Use default config if none provided
    if config is None:
        config = AnalysisConfig()
    
    # Apply CPU optimizations if enabled
    if config.optimize_cpu and not config.enable_gpu:
        from .utils.cpu_optimization import optimize_cpu_environment, configure_torch_threads
        optimize_cpu_environment(max_threads=config.max_workers)
        configure_torch_threads(num_threads=config.max_workers)
    
    logger = get_logger().bind_context(
        service="public_api",
        input_path=str(input_path),
        gpu_enabled=config.enable_gpu
    )
    
    logger.info("starting_audio_analysis")
    start_time = datetime.now()
    processing_start = start_time
    
    try:
        # Use real ML models for processing
        result = await _process_audio_with_real_models(input_path, config, logger)
        
        end_time = datetime.now()
        processing_time = (end_time - processing_start).total_seconds()
        
        # Update result with timing info
        result.processed_at = end_time
        result.processing_time = processing_time
        
        logger.info("audio_analysis_completed", 
                   processing_time=processing_time,
                   segments_count=len(result.segments),
                   speakers_count=result.unique_speakers)
        
        # Save result to output directory
        from .utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        
        if output_path:
            # User specified output path
            saved_path = output_manager.save_analysis_result(result, str(input_path), output_path)
        else:
            # Auto-generate output path in output/ directory  
            saved_path = output_manager.save_analysis_result(result, str(input_path))
        
        logger.info("results_saved", output_path=str(saved_path))
        
        return result
        
    except Exception as e:
        logger.error("audio_analysis_failed", error=str(e))
        raise


def analyze_audio_sync(
    input_path: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
    output_path: Optional[Union[str, Path]] = None
) -> AnalysisResult:
    """
    Synchronous version of analyze_audio.
    
    Args:
        input_path: Path to audio/video file or YouTube URL
        config: Analysis configuration (optional)
        output_path: Path to save results (optional)
    
    Returns:
        AnalysisResult object with complete analysis
    """
    try:
        # Try to run in existing event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, we need to use a different approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, analyze_audio(input_path, config, output_path))
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        return asyncio.run(analyze_audio(input_path, config, output_path))


async def _process_audio_with_real_models(
    input_path: Union[str, Path], 
    config: AnalysisConfig,
    logger
) -> AnalysisResult:
    """Real audio processing using ML models"""
    
    import librosa
    import numpy as np
    from collections import Counter
    from .models.emotion_analyzer import EmotionAnalyzer
    from .models.speech_recognizer import WhisperXPipeline
    from .utils.output_manager import get_output_manager
    
    # Preprocess audio for optimal quality
    logger.info("preprocessing_audio")
    from .utils.audio_cleaner import AudioCleaner
    audio_cleaner = AudioCleaner(target_sr=config.sample_rate)
    
    # Clean audio (creates temporary file optimized for analysis)
    cleaned_audio_path = audio_cleaner.clean_audio(input_path)
    logger.info("audio_preprocessed", cleaned_path=cleaned_audio_path)
    
    # Initialize models
    logger.info("initializing_models")
    emotion_analyzer = EmotionAnalyzer(device="cuda" if config.enable_gpu else "cpu") if config.emotion_detection else None
    whisperx_pipeline = WhisperXPipeline(
        model_size="base",
        device="cuda" if config.enable_gpu else "cpu",
        language=config.language if config.language != "auto" else None
    )
    
    # Perform unified WhisperX processing (speech recognition + speaker diarization)
    logger.info("performing_whisperx_unified_processing")
    try:
        whisperx_result = whisperx_pipeline.process_audio_with_diarization(
            cleaned_audio_path,
            min_speakers=2,
            max_speakers=8,  # Allow reasonable range for speaker detection
            sample_rate=config.sample_rate
        )
        logger.info("whisperx_processing_completed", 
                   segments_count=len(whisperx_result["segments"]),
                   language=whisperx_result.get("language", "unknown"))
    except Exception as e:
        logger.error("whisperx_processing_failed", error=str(e))
        raise
    
    # Convert WhisperX segments to our SpeakerSegment format
    from dataclasses import dataclass
    
    @dataclass
    class SpeakerSegment:
        start: float
        end: float
        duration: float
        speaker: str
        confidence: float
    
    speaker_segments = []
    speech_results = []
    
    for segment in whisperx_result["segments"]:
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        duration = end_time - start_time
        speaker_id = segment.get("speaker", "SPEAKER_00")
        text = segment.get("text", "").strip()
        
        # Create SpeakerSegment for compatibility
        speaker_seg = SpeakerSegment(
            start=start_time,
            end=end_time,
            duration=duration,
            speaker=speaker_id,
            confidence=0.9  # WhisperX provides high quality diarization
        )
        speaker_segments.append(speaker_seg)
        
        # Create SpeechResult equivalent for text
        from .models.speech_recognizer import SpeechResult
        speech_result = SpeechResult(
            text=text,
            confidence=0.9,  # WhisperX provides high quality transcription
            language=whisperx_result.get("language"),
            word_segments=segment.get("words", [])
        )
        speech_results.append(speech_result)
    
    # Perform emotion analysis on each segment
    emotion_results = []
    if config.emotion_detection and emotion_analyzer:
        logger.info("performing_emotion_analysis")
        try:
            segment_tuples = [(seg.start, seg.end) for seg in speaker_segments]
            emotion_results = emotion_analyzer.batch_analyze_segments(
                cleaned_audio_path, segment_tuples, config.sample_rate
            )
            logger.info("emotion_analysis_completed", results_count=len(emotion_results))
        except Exception as e:
            logger.error("emotion_analysis_failed", error=str(e))
            raise
    
    # Extract acoustic features if requested
    acoustic_features_list = []
    if config.acoustic_features:
        logger.info("extracting_acoustic_features")
        # Load audio for acoustic feature extraction
        import librosa
        y, sr = librosa.load(str(cleaned_audio_path), sr=config.sample_rate)
        
        for seg in speaker_segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            segment_audio = y[start_sample:end_sample]
            
            try:
                spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(segment_audio)
                energy = float(np.mean(np.abs(segment_audio)))
                
                features = AcousticFeatures(
                    spectral_centroid=float(np.mean(spectral_centroid)),
                    spectral_rolloff=float(np.mean(spectral_rolloff)),
                    zero_crossing_rate=float(np.mean(zero_crossing_rate)),
                    energy=energy
                )
                
                # Add detailed features (MFCC and pitch analysis)
                try:
                    mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                    pitch = librosa.piptrack(y=segment_audio, sr=sr, threshold=0.1)[0]
                    pitch_values = pitch[pitch > 0]
                    
                    features.mfcc_mean = [float(np.mean(mfcc[i, :])) for i in range(mfcc.shape[0])]
                    features.pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else None
                    features.pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else None
                except:
                    pass  # Skip detailed features if extraction fails
                        
                acoustic_features_list.append(features)
            except:
                acoustic_features_list.append(None)
    
    # Create AudioSegment objects
    segments = []
    for i, speaker_seg in enumerate(speaker_segments):
        # Get emotion info
        emotion_info = None
        if i < len(emotion_results) and config.emotion_detection:
            emotion_result = emotion_results[i]
            emotion_info = EmotionInfo(
                emotion=emotion_result.emotion,
                confidence=emotion_result.confidence,
                probabilities=emotion_result.probabilities
            )
        
        # Get acoustic features
        acoustic_features = None
        if i < len(acoustic_features_list) and config.acoustic_features:
            acoustic_features = acoustic_features_list[i]
        
        # Get speech recognition result
        text = None
        if i < len(speech_results):
            speech_result = speech_results[i]
            text = speech_result.text if speech_result.text.strip() else None
        
        # Create speaker info
        speaker_info = SpeakerInfo(
            speaker_id=speaker_seg.speaker,
            confidence=speaker_seg.confidence
        )
        
        # Create segment
        segment = AudioSegment(
            start_time=speaker_seg.start,
            end_time=speaker_seg.end,
            duration=speaker_seg.duration,
            speaker=speaker_info,
            emotion=emotion_info,
            acoustic_features=acoustic_features,
            text=text
        )
        
        segments.append(segment)
    
    # Calculate statistics
    unique_speakers = len(set(seg.speaker.speaker_id for seg in segments))
    avg_confidence = sum(seg.speaker.confidence for seg in segments) / len(segments) if segments else 0
    
    # Find dominant emotion
    dominant_emotion = None
    if config.emotion_detection and emotion_results:
        emotions_list = [seg.emotion.emotion for seg in segments if seg.emotion]
        dominant_emotion = Counter(emotions_list).most_common(1)[0][0] if emotions_list else None
    
    # Create speaker summary
    speakers_dict = {}
    for speaker_id in set(seg.speaker.speaker_id for seg in segments):
        speaker_segments = [seg for seg in segments if seg.speaker.speaker_id == speaker_id]
        speakers_dict[speaker_id] = {
            "total_duration": sum(seg.duration for seg in speaker_segments),
            "segment_count": len(speaker_segments),
            "avg_confidence": sum(seg.speaker.confidence for seg in speaker_segments) / len(speaker_segments),
            "emotions": list(set(seg.emotion.emotion for seg in speaker_segments if seg.emotion)) if config.emotion_detection else []
        }
    
    # Optimize segments for subtitles
    subtitle_data = None
    if hasattr(config, 'optimize_for_subtitles') and config.optimize_for_subtitles:
        from .utils.subtitle_optimizer import SubtitleOptimizer
        logger.info("optimizing_for_subtitles")
        
        subtitle_optimizer = SubtitleOptimizer()
        
        # Convert segments to format expected by optimizer
        segments_for_optimization = []
        for seg in segments:
            segments_for_optimization.append({
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text or "",
                "speaker": {"speaker_id": seg.speaker.speaker_id, "confidence": seg.speaker.confidence}
            })
        
        optimized_subtitles = subtitle_optimizer.optimize_segments(segments_for_optimization)
        subtitle_data = subtitle_optimizer.generate_subtitle_json(optimized_subtitles)
        logger.info("subtitle_optimization_completed", optimized_segments=len(optimized_subtitles))
    
    # Calculate actual duration from segments (fix duration calculation issue)
    if segments:
        # Use the maximum end_time from segments as the actual duration
        actual_duration = max(seg.end_time for seg in segments)
        logger.info("duration_calculated_from_segments", 
                   calculated_duration=actual_duration,
                   segments_count=len(segments))
    else:
        # Fallback: load audio to get duration
        import librosa
        y, sr = librosa.load(str(cleaned_audio_path), sr=config.sample_rate)
        actual_duration = len(y) / sr
        logger.info("duration_calculated_from_audio_fallback", duration=actual_duration)
    
    # Create result
    result = AnalysisResult(
        filename=Path(input_path).name,
        duration=actual_duration,
        sample_rate=config.sample_rate,
        processed_at=datetime.now(),  # Will be updated by caller
        processing_time=0,  # Will be updated by caller
        segments=segments,
        speakers=speakers_dict,
        total_segments=len(segments),
        unique_speakers=unique_speakers,
        dominant_emotion=dominant_emotion,
        avg_confidence=avg_confidence,
        metadata={
            "processing_mode": "real_ml_models",
            "config": {
                "enable_gpu": config.enable_gpu,
                "segment_length": config.segment_length,
                "language": config.language,
                "unified_model": "whisperx-base-with-diarization",
                "emotion_model": emotion_analyzer.model_name if emotion_analyzer else None
            },
            "subtitle_optimization": subtitle_data is not None
        }
    )
    
    # Store subtitle data in result for JSON output
    if subtitle_data:
        result.subtitle_data = subtitle_data
    
    # Cleanup temporary audio file
    try:
        import os
        if os.path.exists(cleaned_audio_path) and str(input_path) != cleaned_audio_path:
            os.unlink(cleaned_audio_path)
            logger.info("temp_audio_cleaned", temp_file=cleaned_audio_path)
    except Exception as e:
        logger.warning("temp_cleanup_failed", error=str(e))
    
    return result


