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
class WordSegment:
    """Individual word with acoustic analysis for dynamic subtitles"""
    word: str
    start: float
    end: float
    confidence: float
    volume_db: Optional[float] = None
    pitch_hz: Optional[float] = None
    harmonics_ratio: Optional[float] = None
    spectral_centroid: Optional[float] = None


@dataclass 
class AudioSegment:
    """Individual audio segment with analysis results"""
    start_time: float
    end_time: float
    duration: float
    speaker: SpeakerInfo
    emotion: Optional[EmotionInfo] = None
    acoustic_features: Optional[AcousticFeatures] = None
    text: Optional[str] = None
    words: Optional[List[WordSegment]] = None  # Word-level acoustic data for dynamic subtitles
    

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
    
    # Dynamic subtitle statistics (global acoustic baselines)
    volume_statistics: Optional[Dict[str, float]] = None
    pitch_statistics: Optional[Dict[str, Any]] = None
    harmonics_statistics: Optional[Dict[str, float]] = None
    spectral_statistics: Optional[Dict[str, float]] = None
    
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
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.confidence,
                            "volume_db": word.volume_db,
                            "pitch_hz": word.pitch_hz,
                            "harmonics_ratio": word.harmonics_ratio,
                            "spectral_centroid": word.spectral_centroid
                        }
                        for word in seg.words
                    ] if seg.words else None,
                }
                for seg in self.segments
            ]
        }
        
        # Add global acoustic statistics for dynamic subtitles
        if self.volume_statistics:
            result["volume_statistics"] = self.volume_statistics
        if self.pitch_statistics:
            result["pitch_statistics"] = self.pitch_statistics
        if self.harmonics_statistics:
            result["harmonics_statistics"] = self.harmonics_statistics
        if self.spectral_statistics:
            result["spectral_statistics"] = self.spectral_statistics
        
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
    from pathlib import Path
    from .services.speaker_diarizer import SpeakerDiarizer
    from .services.emotion_analyzer import EmotionAnalyzer
    from .models.speech_recognizer import SpeechRecognizer
    from .utils.output_manager import get_output_manager
    
    # Preprocess audio for optimal quality
    logger.info("preprocessing_audio")
    from .utils.audio_cleaner import AudioCleaner
    audio_cleaner = AudioCleaner(target_sr=config.sample_rate)
    
    # Clean audio (creates temporary file optimized for analysis)
    cleaned_audio_path = audio_cleaner.process(input_path, "temp")
    logger.info("audio_preprocessed", cleaned_path=cleaned_audio_path)
    
    # Initialize models with optimized configurations
    logger.info("initializing_models")
    from config.model_configs import SpeakerDiarizationConfig
    speaker_config = SpeakerDiarizationConfig()
    speaker_config.device = "cuda" if config.enable_gpu else "cpu"
    speaker_diarizer = SpeakerDiarizer(config=speaker_config, device=speaker_config.device)
    emotion_analyzer = None
    if config.emotion_detection:
        from config.model_configs import EmotionAnalysisConfig
        emotion_config = EmotionAnalysisConfig(device="cuda" if config.enable_gpu else "cpu")
        emotion_analyzer = EmotionAnalyzer(config=emotion_config)
        emotion_analyzer.load_models()  # Explicitly load models to prevent fallback
    speech_recognizer = SpeechRecognizer(device="cuda" if config.enable_gpu else "cpu")
    
    # Load preprocessed audio for basic info
    logger.info("loading_preprocessed_audio")
    y, sr = librosa.load(str(cleaned_audio_path), sr=config.sample_rate)
    duration = len(y) / sr
    
    # Perform speaker diarization on cleaned audio
    logger.info("performing_speaker_diarization")
    try:
        diarization_result = speaker_diarizer.diarize_audio(cleaned_audio_path)
        if diarization_result.success:
            speaker_segments = diarization_result.segments
            logger.info("speaker_diarization_completed", 
                       segments_count=len(speaker_segments),
                       speakers_count=diarization_result.total_speakers)
        else:
            raise Exception(diarization_result.error_message or "Diarization failed")
    except Exception as e:
        logger.error("speaker_diarization_failed", error=str(e))
        raise RuntimeError(f"Speaker diarization failed: {str(e)}. No fallback mode allowed.")
    
    # Perform emotion analysis on each segment
    emotion_results = []
    if config.emotion_detection and emotion_analyzer:
        logger.info("performing_emotion_analysis")
        try:
            # Create segments in the format expected by services EmotionAnalyzer
            segment_tuples = [(seg.start_time, seg.end_time, seg.speaker_id) for seg in speaker_segments]
            emotion_results = emotion_analyzer.analyze_batch(
                Path(cleaned_audio_path), segment_tuples
            )
            logger.info("emotion_analysis_completed", results_count=len(emotion_results))
        except Exception as e:
            logger.error("emotion_analysis_failed", error=str(e))
            raise RuntimeError(f"Emotion analysis failed: {str(e)}. No fallback mode allowed.")
    
    # Perform speech recognition on each segment
    speech_results = []
    logger.info("performing_speech_recognition")
    try:
        segment_tuples = [(seg.start_time, seg.end_time) for seg in speaker_segments]
        speech_results = speech_recognizer.batch_transcribe_segments(
            cleaned_audio_path, segment_tuples, config.sample_rate
        )
        logger.info("speech_recognition_completed", results_count=len(speech_results))
    except Exception as e:
        logger.error("speech_recognition_failed", error=str(e))
        raise RuntimeError(f"Speech recognition failed: {str(e)}. No fallback mode allowed.")
    
    # Extract acoustic features if requested
    acoustic_features_list = []
    if config.acoustic_features:
        logger.info("extracting_acoustic_features")
        for seg in speaker_segments:
            start_sample = int(seg.start_time * sr)
            end_sample = int(seg.end_time * sr)
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
                
                # Add detailed features if requested
                if config.detailed_features:
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
                emotion=emotion_result.primary.value,  # Convert enum to string
                confidence=emotion_result.confidence,
                probabilities=emotion_result.all_scores.model_dump()  # Convert pydantic model to dict
            )
        
        # Get acoustic features
        acoustic_features = None
        if i < len(acoustic_features_list) and config.acoustic_features:
            acoustic_features = acoustic_features_list[i]
        
        # Get speech recognition result and word-level acoustic data
        text = None
        words = None
        if i < len(speech_results):
            speech_result = speech_results[i]
            text = speech_result.text if speech_result.text.strip() else None
            
            # Extract word-level acoustic data for dynamic subtitles
            if speech_result.word_segments:
                words = []
                for word_data in speech_result.word_segments:
                    if isinstance(word_data, dict):
                        word_segment = WordSegment(
                            word=word_data.get("word", ""),
                            start=word_data.get("start", 0.0),
                            end=word_data.get("end", 0.0),
                            confidence=word_data.get("confidence", 0.0),
                            volume_db=word_data.get("volume_db"),
                            pitch_hz=word_data.get("pitch_hz"),
                            harmonics_ratio=word_data.get("harmonics_ratio"),
                            spectral_centroid=word_data.get("spectral_centroid")
                        )
                        words.append(word_segment)
        
        # Create speaker info
        speaker_info = SpeakerInfo(
            speaker_id=speaker_seg.speaker_id,
            confidence=speaker_seg.confidence
        )
        
        # Create segment
        segment = AudioSegment(
            start_time=speaker_seg.start_time,
            end_time=speaker_seg.end_time,
            duration=speaker_seg.duration,
            speaker=speaker_info,
            emotion=emotion_info,
            acoustic_features=acoustic_features,
            text=text,
            words=words
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
    
    # Calculate global acoustic statistics for dynamic subtitles
    volume_stats = None
    pitch_stats = None
    harmonics_stats = None
    spectral_stats = None
    
    try:
        # Collect all word-level acoustic data
        all_volumes = []
        all_pitches = []
        all_harmonics = []
        all_spectral_centroids = []
        
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    if word.volume_db is not None and word.volume_db > -60:
                        all_volumes.append(word.volume_db)
                    if word.pitch_hz is not None and word.pitch_hz > 0:
                        all_pitches.append(word.pitch_hz)
                    if word.harmonics_ratio is not None and word.harmonics_ratio > 0:
                        all_harmonics.append(word.harmonics_ratio)
                    if word.spectral_centroid is not None and word.spectral_centroid > 0:
                        all_spectral_centroids.append(word.spectral_centroid)
        
        # Calculate statistics using numpy
        import numpy as np
        
        if all_volumes:
            volumes_array = np.array(all_volumes)
            volume_stats = {
                "global_min_db": float(np.min(volumes_array)),
                "global_max_db": float(np.max(volumes_array)),
                "global_mean_db": float(np.mean(volumes_array)),
                "baseline_db": float(np.percentile(volumes_array, 50)),
                "whisper_threshold_db": float(np.percentile(volumes_array, 25)),
                "loud_threshold_db": float(np.percentile(volumes_array, 75))
            }
        
        if all_pitches:
            pitches_array = np.array(all_pitches)
            pitch_stats = {
                "global_min_hz": float(np.min(pitches_array)),
                "global_max_hz": float(np.max(pitches_array)),
                "global_mean_hz": float(np.mean(pitches_array)),
                "baseline_range": {
                    "min_hz": float(np.percentile(pitches_array, 33)),
                    "max_hz": float(np.percentile(pitches_array, 67))
                }
            }
        
        if all_harmonics:
            harmonics_array = np.array(all_harmonics)
            harmonics_stats = {
                "global_min_ratio": float(np.min(harmonics_array)),
                "global_max_ratio": float(np.max(harmonics_array)),
                "global_mean_ratio": float(np.mean(harmonics_array)),
                "baseline_ratio": float(np.percentile(harmonics_array, 50))
            }
        
        if all_spectral_centroids:
            spectral_array = np.array(all_spectral_centroids)
            spectral_stats = {
                "global_min_hz": float(np.min(spectral_array)),
                "global_max_hz": float(np.max(spectral_array)),
                "global_mean_hz": float(np.mean(spectral_array)),
                "baseline_hz": float(np.percentile(spectral_array, 50))
            }
        
        logger.info("global_statistics_calculated",
                   volume_samples=len(all_volumes),
                   pitch_samples=len(all_pitches),
                   harmonics_samples=len(all_harmonics))
                   
    except Exception as e:
        logger.error("global_statistics_calculation_failed", error=str(e))
    
    # Create result
    result = AnalysisResult(
        filename=Path(input_path).name,
        duration=duration,
        sample_rate=sr,
        processed_at=datetime.now(),  # Will be updated by caller
        processing_time=0,  # Will be updated by caller
        segments=segments,
        speakers=speakers_dict,
        total_segments=len(segments),
        unique_speakers=unique_speakers,
        dominant_emotion=dominant_emotion,
        avg_confidence=avg_confidence,
        volume_statistics=volume_stats,
        pitch_statistics=pitch_stats,
        harmonics_statistics=harmonics_stats,
        spectral_statistics=spectral_stats,
        metadata={
            "processing_mode": "real_ml_models",
            "config": {
                "enable_gpu": config.enable_gpu,
                "segment_length": config.segment_length,
                "language": config.language,
                "speaker_model": speaker_diarizer.config.model_name,
                "emotion_model": emotion_analyzer.config.model_name if emotion_analyzer else None,
                "speech_model": "whisperx-base"
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


def _create_fallback_segments(y, sr, segment_length, min_segment_length):
    """Create fallback segments when speaker diarization fails"""
    from .services.speaker_diarizer import SpeakerSegment
    import numpy as np
    import librosa
    
    duration = len(y) / sr
    num_segments = int(duration // segment_length) + (1 if duration % segment_length > 0 else 0)
    
    segments = []
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    median_centroid = np.median(spectral_centroid)
    
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        
        if end_time - start_time < min_segment_length:
            continue
        
        # Simple speaker assignment based on spectral centroid
        start_frame = int(start_time * len(spectral_centroid) / duration)
        end_frame = int(end_time * len(spectral_centroid) / duration)
        
        if start_frame < end_frame and end_frame <= len(spectral_centroid):
            seg_centroid = np.mean(spectral_centroid[start_frame:end_frame])
        else:
            seg_centroid = median_centroid
        
        speaker_id = "speaker_01" if seg_centroid <= median_centroid else "speaker_02"
        confidence = 0.6 + 0.3 * abs(seg_centroid - median_centroid) / (np.max(spectral_centroid) - np.min(spectral_centroid))
        confidence = min(0.9, max(0.3, confidence))
        
        segment = SpeakerSegment(
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            confidence=confidence
        )
        
        segments.append(segment)
    
    return segments