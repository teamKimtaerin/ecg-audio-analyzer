"""
Acoustic Feature Analyzer Service
Single Responsibility: Extract audio characteristics for subtitle styling

High-performance acoustic feature extraction using OpenSMILE and custom signal processing.
"""

import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import subprocess
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, stats
from scipy.signal import hilbert, find_peaks
import parselmouth
from parselmouth.praat import call

from ..utils.logger import get_logger
from ...config.model_configs import AcousticAnalysisConfig
from ..models.output_models import AudioFeatures, VolumeCategory


@dataclass 
class RawAcousticFeatures:
    """Raw acoustic features extracted from audio"""
    # Time domain features
    rms_energy: float
    rms_db: float
    zero_crossing_rate: float
    amplitude_max: float
    amplitude_mean: float
    amplitude_std: float
    
    # Frequency domain features
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flux: float
    
    # Pitch features
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_slope: float
    
    # Energy features
    short_time_energy: List[float]
    energy_entropy: float
    
    # MFCC features
    mfcc_coefficients: List[float]
    
    # Prosodic features
    speaking_rate_estimate: float
    silence_ratio: float
    voiced_ratio: float
    
    # Dynamic features
    energy_dynamics: float
    pitch_dynamics: float
    spectral_dynamics: float
    
    # Quality indicators
    snr_estimate: float
    harmonic_ratio: float
    
    # Processing metadata
    segment_duration: float
    sample_rate: int
    processing_time: float


@dataclass
class VoiceActivitySegment:
    """Voice activity detection segment"""
    start_time: float
    end_time: float
    confidence: float
    energy_level: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SignalProcessor:
    """Advanced signal processing for acoustic feature extraction"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = get_logger().bind_context(component="signal_processor")
    
    def detect_voice_activity(self, 
                            audio: np.ndarray, 
                            frame_length: int = 2048,
                            hop_length: int = 512) -> List[VoiceActivitySegment]:
        """Detect voice activity in audio signal"""
        
        # Calculate short-time energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Calculate spectral centroid for voicing detection
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate,
            hop_length=hop_length
        )[0]
        
        # Adaptive thresholding
        energy_threshold = np.percentile(energy, 20)
        centroid_threshold = np.percentile(spectral_centroid, 30)
        
        # Combine energy and spectral features for VAD
        voice_activity = (energy > energy_threshold) & (spectral_centroid > centroid_threshold)
        
        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(voice_activity)), 
            sr=self.sample_rate, 
            hop_length=hop_length
        )
        
        # Extract voice segments
        segments = []
        in_speech = False
        start_time = 0.0
        
        for i, (is_voice, time_stamp, energy_val) in enumerate(zip(voice_activity, times, energy)):
            if is_voice and not in_speech:
                # Start of speech segment
                start_time = time_stamp
                in_speech = True
            elif not is_voice and in_speech:
                # End of speech segment
                if time_stamp - start_time > 0.1:  # Minimum 100ms segments
                    segments.append(VoiceActivitySegment(
                        start_time=start_time,
                        end_time=time_stamp,
                        confidence=0.8,  # Simple confidence score
                        energy_level=float(energy_val)
                    ))
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech:
            segments.append(VoiceActivitySegment(
                start_time=start_time,
                end_time=times[-1],
                confidence=0.8,
                energy_level=float(energy[-1])
            ))
        
        return segments
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features from audio"""
        
        try:
            # Convert to Parselmouth Sound object
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                sound = parselmouth.Sound(temp_file.name)
                os.unlink(temp_file.name)
            
            # Extract pitch
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            # Extract intensity
            intensity = sound.to_intensity(time_step=0.01)
            intensity_values = intensity.values[0]
            
            # Calculate prosodic features
            features = {
                'pitch_mean': float(np.mean(pitch_values)) if len(pitch_values) > 0 else 150.0,
                'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 20.0,
                'pitch_min': float(np.min(pitch_values)) if len(pitch_values) > 0 else 100.0,
                'pitch_max': float(np.max(pitch_values)) if len(pitch_values) > 0 else 300.0,
                'intensity_mean': float(np.mean(intensity_values)),
                'intensity_std': float(np.std(intensity_values)),
                'voiced_ratio': len(pitch_values) / len(pitch.selected_array['frequency'])
            }
            
            # Calculate pitch slope (fundamental frequency trend)
            if len(pitch_values) > 10:
                time_points = np.arange(len(pitch_values))
                slope, _, _, _, _ = stats.linregress(time_points, pitch_values)
                features['pitch_slope'] = float(slope)
            else:
                features['pitch_slope'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.warning("prosodic_feature_extraction_failed", error=str(e))
            return {
                'pitch_mean': 150.0,
                'pitch_std': 20.0,
                'pitch_min': 100.0,
                'pitch_max': 300.0,
                'pitch_slope': 0.0,
                'intensity_mean': 60.0,
                'intensity_std': 10.0,
                'voiced_ratio': 0.7
            }
    
    def estimate_speaking_rate(self, 
                             audio: np.ndarray,
                             voice_segments: List[VoiceActivitySegment]) -> float:
        """Estimate speaking rate from voice activity"""
        
        if not voice_segments:
            return 0.0
        
        # Calculate syllable-like events using energy peaks
        hop_length = 512
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Find peaks in energy corresponding to syllable nuclei
        energy_smooth = signal.savgol_filter(energy, 5, 2)
        peaks, _ = find_peaks(
            energy_smooth, 
            height=np.percentile(energy_smooth, 60),
            distance=int(0.1 * self.sample_rate / hop_length)  # Minimum 100ms between syllables
        )
        
        # Calculate speaking rate
        total_speech_time = sum(seg.duration for seg in voice_segments)
        if total_speech_time > 0:
            speaking_rate = len(peaks) / total_speech_time
        else:
            speaking_rate = 0.0
        
        # Reasonable bounds for speaking rate (syllables per second)
        return max(0.0, min(10.0, speaking_rate))
    
    def calculate_spectral_dynamics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate spectral dynamics and variation"""
        
        # Short-time Fourier transform
        stft = librosa.stft(audio, hop_length=512)
        magnitude = np.abs(stft)
        
        # Spectral features over time
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
        
        # Calculate dynamics (variation over time)
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_flux': float(np.mean(np.diff(spectral_centroids)**2))
        }
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        
        try:
            # Simple SNR estimation using energy distribution
            energy = audio ** 2
            
            # Assume noise is in the lower 20% of energy values
            noise_threshold = np.percentile(energy, 20)
            signal_threshold = np.percentile(energy, 80)
            
            noise_energy = np.mean(energy[energy <= noise_threshold])
            signal_energy = np.mean(energy[energy >= signal_threshold])
            
            if noise_energy > 0:
                snr_linear = signal_energy / noise_energy
                snr_db = 10 * np.log10(snr_linear)
            else:
                snr_db = 40.0  # High SNR if no noise detected
            
            return max(0.0, min(60.0, snr_db))  # Reasonable bounds
            
        except Exception:
            return 20.0  # Default reasonable SNR


class AcousticAnalyzer:
    """
    High-performance acoustic feature analyzer for subtitle styling.
    
    Single Responsibility: Extract comprehensive audio characteristics including
    energy, pitch, spectral, and prosodic features for dynamic subtitle generation.
    """
    
    def __init__(self, 
                 config: AcousticAnalysisConfig,
                 enable_parallel_processing: bool = True,
                 temp_dir: Optional[Path] = None):
        
        self.config = config
        self.enable_parallel_processing = enable_parallel_processing
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "ecg_acoustic_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger().bind_context(service="acoustic_analyzer")
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor(self.config.sample_rate)
        
        # Thread pool for parallel processing
        if enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
        else:
            self.thread_pool = None
        
        # Processing statistics
        self.total_segments_processed = 0
        self.total_processing_time = 0.0
        
        # Check OpenSMILE availability
        self.opensmile_available = self._check_opensmile_availability()
        
        self.logger.info("acoustic_analyzer_initialized",
                        opensmile_available=self.opensmile_available,
                        parallel_processing=enable_parallel_processing,
                        temp_dir=str(self.temp_dir))
    
    def _check_opensmile_availability(self) -> bool:
        """Check if OpenSMILE is available on the system"""
        try:
            result = subprocess.run(['SMILExtract', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            self.logger.warning("opensmile_not_available", 
                              message="OpenSMILE not found, using fallback feature extraction")
            return False
    
    def _extract_opensmile_features(self, audio_path: Path) -> Dict[str, float]:
        """Extract features using OpenSMILE"""
        
        if not self.opensmile_available:
            return {}
        
        try:
            # Create temporary output file
            output_file = self.temp_dir / f"features_{time.time()}.csv"
            
            # OpenSMILE command for basic feature extraction
            cmd = [
                'SMILExtract',
                '-C', 'config/IS13_ComParE.conf',  # Use ComParE feature set
                '-I', str(audio_path),
                '-csvoutput', str(output_file),
                '-timestampcsv', '1',
                '-headercsv', '1'
            ]
            
            # Run OpenSMILE
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_file.exists():
                # Parse CSV output
                import pandas as pd
                df = pd.read_csv(output_file, delimiter=';')
                
                # Extract relevant features
                features = {}
                if len(df) > 0:
                    row = df.iloc[0]  # Take first row
                    
                    # Map OpenSMILE features to our feature names
                    feature_mapping = {
                        'mfcc_mean': 'mfcc[1]',
                        'energy_mean': 'rms_energy_sma',
                        'zcr_mean': 'zcr_sma',
                        'spectral_centroid': 'spectralCentroidSma_sma',
                        'pitch_mean': 'F0_sma'
                    }
                    
                    for our_name, opensmile_name in feature_mapping.items():
                        if opensmile_name in row:
                            features[our_name] = float(row[opensmile_name])
                
                # Cleanup
                output_file.unlink()
                
                return features
            
        except Exception as e:
            self.logger.warning("opensmile_extraction_failed", error=str(e))
        
        return {}
    
    def _extract_basic_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract basic acoustic features using librosa and scipy"""
        
        features = {}
        
        try:
            # Time domain features
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_energy'] = float(np.mean(rms))
            features['rms_db'] = float(20 * np.log10(np.mean(rms) + 1e-8))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Amplitude statistics
            features['amplitude_max'] = float(np.max(np.abs(audio)))
            features['amplitude_mean'] = float(np.mean(np.abs(audio)))
            features['amplitude_std'] = float(np.std(np.abs(audio)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # MFCC features (first 3 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            features['mfcc_1'] = float(np.mean(mfccs[0]))
            features['mfcc_2'] = float(np.mean(mfccs[1]))
            features['mfcc_3'] = float(np.mean(mfccs[2]))
            
            # Energy entropy
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, axis=0)
            frame_energies = np.sum(frames**2, axis=0)
            frame_energies = frame_energies / (np.sum(frame_energies) + 1e-8)
            entropy = -np.sum(frame_energies * np.log(frame_energies + 1e-8))
            features['energy_entropy'] = float(entropy)
            
        except Exception as e:
            self.logger.warning("basic_feature_extraction_failed", error=str(e))
            # Return default values on failure
            features.update({
                'rms_energy': 0.02,
                'rms_db': -30.0,
                'zero_crossing_rate': 0.05,
                'amplitude_max': 0.5,
                'amplitude_mean': 0.1,
                'amplitude_std': 0.08,
                'spectral_centroid': 2000.0,
                'spectral_bandwidth': 1500.0,
                'spectral_rolloff': 4000.0,
                'mfcc_1': 12.0,
                'mfcc_2': -8.0,
                'mfcc_3': 4.0,
                'energy_entropy': 5.0
            })
        
        return features
    
    def _categorize_volume(self, rms_energy: float, amplitude_max: float) -> VolumeCategory:
        """Categorize volume level based on energy metrics"""
        
        # Dynamic thresholds based on both RMS and peak amplitude
        energy_score = rms_energy * 0.7 + amplitude_max * 0.3
        
        if energy_score < 0.03:
            return VolumeCategory.LOW
        elif energy_score < 0.06:
            return VolumeCategory.MEDIUM
        elif energy_score < 0.12:
            return VolumeCategory.HIGH
        else:
            return VolumeCategory.EMPHASIS
    
    def extract_features(self, 
                        audio_path: Path,
                        start_time: float,
                        end_time: float,
                        speaker_id: str) -> AudioFeatures:
        """
        Extract comprehensive acoustic features from audio segment.
        
        Args:
            audio_path: Path to audio file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            speaker_id: Speaker identifier
            
        Returns:
            AudioFeatures object with comprehensive acoustic analysis
        """
        
        start_processing = time.time()
        
        with self.logger.performance_timer("acoustic_feature_extraction", items_count=1):
            
            try:
                # Load audio segment
                duration = end_time - start_time
                offset = start_time
                
                audio, sample_rate = librosa.load(
                    str(audio_path),
                    sr=self.config.sample_rate,
                    offset=offset,
                    duration=duration,
                    mono=True
                )
                
                # Extract basic features
                basic_features = self._extract_basic_features(audio, sample_rate)
                
                # Extract prosodic features
                prosodic_features = self.signal_processor.extract_prosodic_features(audio)
                
                # Voice activity detection
                voice_segments = self.signal_processor.detect_voice_activity(audio)
                
                # Calculate additional metrics
                silence_ratio = 1.0 - sum(seg.duration for seg in voice_segments) / duration
                silence_ratio = max(0.0, min(1.0, silence_ratio))
                
                # Speaking rate estimation
                speaking_rate = self.signal_processor.estimate_speaking_rate(audio, voice_segments)
                
                # Spectral dynamics
                spectral_dynamics = self.signal_processor.calculate_spectral_dynamics(audio)
                
                # SNR estimation
                snr = self.signal_processor.estimate_snr(audio)
                
                # Volume categorization
                volume_category = self._categorize_volume(
                    basic_features['rms_energy'], 
                    basic_features['amplitude_max']
                )
                
                # Create AudioFeatures object
                features = AudioFeatures(
                    rms_energy=basic_features['rms_energy'],
                    rms_db=basic_features['rms_db'],
                    pitch_mean=prosodic_features['pitch_mean'],
                    pitch_variance=prosodic_features['pitch_std']**2,
                    speaking_rate=speaking_rate,
                    amplitude_max=basic_features['amplitude_max'],
                    silence_ratio=silence_ratio,
                    spectral_centroid=spectral_dynamics['spectral_centroid_mean'],
                    zcr=basic_features['zero_crossing_rate'],
                    mfcc=[basic_features['mfcc_1'], basic_features['mfcc_2'], basic_features['mfcc_3']],
                    volume_category=volume_category
                )
                
                # Update statistics
                processing_time = time.time() - start_processing
                self.total_segments_processed += 1
                self.total_processing_time += processing_time
                
                self.logger.debug("acoustic_features_extracted",
                                segment_duration=duration,
                                speaker_id=speaker_id,
                                volume_category=volume_category.value,
                                speaking_rate=speaking_rate,
                                processing_time=processing_time)
                
                return features
                
            except Exception as e:
                self.logger.error("acoustic_feature_extraction_failed",
                                audio_path=str(audio_path),
                                start_time=start_time,
                                end_time=end_time,
                                error=str(e))
                
                # Return default features on failure
                return AudioFeatures(
                    rms_energy=0.03,
                    rms_db=-25.0,
                    pitch_mean=180.0,
                    pitch_variance=400.0,
                    speaking_rate=3.5,
                    amplitude_max=0.5,
                    silence_ratio=0.3,
                    spectral_centroid=2000.0,
                    zcr=0.05,
                    mfcc=[12.0, -8.0, 4.0],
                    volume_category=VolumeCategory.MEDIUM
                )
    
    def extract_batch_features(self,
                              audio_path: Path,
                              segments: List[Tuple[float, float, str]],
                              max_concurrent: int = None) -> List[AudioFeatures]:
        """
        Extract features for multiple segments concurrently.
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time, speaker_id) tuples
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of AudioFeatures in same order as input segments
        """
        
        if max_concurrent is None:
            max_concurrent = min(len(segments), 4)
        
        with self.logger.performance_timer("acoustic_batch_extraction", items_count=len(segments)):
            
            self.logger.info("acoustic_batch_analysis_started",
                           segment_count=len(segments),
                           max_concurrent=max_concurrent,
                           audio_file=str(audio_path))
            
            if not self.enable_parallel_processing or not self.thread_pool or len(segments) == 1:
                # Process sequentially
                results = []
                for start_time, end_time, speaker_id in segments:
                    result = self.extract_features(audio_path, start_time, end_time, speaker_id)
                    results.append(result)
                return results
            
            # Process in parallel
            results = [None] * len(segments)
            
            def process_segment(index, start_time, end_time, speaker_id):
                try:
                    return index, self.extract_features(
                        audio_path, start_time, end_time, speaker_id
                    )
                except Exception as e:
                    self.logger.error("batch_acoustic_analysis_failed",
                                    index=index,
                                    start_time=start_time,
                                    error=str(e))
                    
                    # Return default features as fallback
                    return index, AudioFeatures(
                        rms_energy=0.03,
                        rms_db=-25.0,
                        pitch_mean=180.0,
                        pitch_variance=400.0,
                        speaking_rate=3.5,
                        amplitude_max=0.5,
                        silence_ratio=0.3,
                        spectral_centroid=2000.0,
                        zcr=0.05,
                        mfcc=[12.0, -8.0, 4.0],
                        volume_category=VolumeCategory.MEDIUM
                    )
            
            # Submit tasks
            futures = []
            for i, (start_time, end_time, speaker_id) in enumerate(segments):
                future = self.thread_pool.submit(
                    process_segment, i, start_time, end_time, speaker_id
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error("batch_future_failed", error=str(e))
            
            # Fill any None results with defaults
            for i, result in enumerate(results):
                if result is None:
                    results[i] = AudioFeatures(
                        rms_energy=0.03,
                        rms_db=-25.0,
                        pitch_mean=180.0,
                        pitch_variance=400.0,
                        speaking_rate=3.5,
                        amplitude_max=0.5,
                        silence_ratio=0.3,
                        spectral_centroid=2000.0,
                        zcr=0.05,
                        mfcc=[12.0, -8.0, 4.0],
                        volume_category=VolumeCategory.MEDIUM
                    )
            
            self.logger.info("acoustic_batch_analysis_completed",
                           segment_count=len(segments),
                           avg_processing_time=self.total_processing_time / max(1, self.total_segments_processed))
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = (
            self.total_processing_time / max(1, self.total_segments_processed)
        )
        
        return {
            'total_segments_processed': self.total_segments_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_per_segment': avg_processing_time,
            'segments_per_second': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'opensmile_available': self.opensmile_available,
            'parallel_processing_enabled': self.enable_parallel_processing
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Shutdown thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            # Clean up temporary files
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
            
            self.logger.info("acoustic_analyzer_cleanup_completed")
            
        except Exception as e:
            self.logger.error("acoustic_analyzer_cleanup_failed", error=str(e))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error("acoustic_analyzer_exception",
                            exception_type=str(exc_type),
                            exception_message=str(exc_val))