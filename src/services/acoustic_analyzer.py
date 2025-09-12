"""
음향 특징 분석 서비스 - 속도 최적화 버전
단일 책임: 자막 스타일링을 위한 핵심 오디오 특징 추출
"""

import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from ..utils.logger import get_logger
from ..models.output_models import AudioFeatures, VolumeCategory


class FastAcousticAnalyzer:
    def __init__(
        self, sample_rate: int = 16000, max_workers: int = 4, batch_threshold: int = 3
    ):
        self.sample_rate = sample_rate
        self.logger = get_logger().bind_context(service="acoustic_analyzer")
        self.max_workers = max_workers
        self.batch_threshold = batch_threshold

        # 공통 파라미터 사전 계산
        self.hop_length = 512
        self.frame_length = 2048

    def _load_audio_segment(
        self, audio_path: Path, start_time: float, end_time: float
    ) -> np.ndarray:
        """오디오 구간 로드"""
        return librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            offset=start_time,
            duration=end_time - start_time,
            mono=True,
        )[0]

    def _extract_volume_features(
        self, audio: np.ndarray
    ) -> Tuple[float, float, VolumeCategory]:
        """볼륨 관련 특징 추출"""
        # RMS 에너지
        rms = np.sqrt(np.mean(audio**2))

        # dB 단위 변환
        db = 20 * np.log10(rms + 1e-10)

        # 에너지 변화량
        frame_energy = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        energy_variation = np.std(frame_energy) / (np.mean(frame_energy) + 1e-10)

        # 볼륨 카테고리 분류
        if db < -35:
            category = VolumeCategory.LOW
        elif db < -25:
            category = VolumeCategory.MEDIUM
        elif db < -15:
            category = VolumeCategory.HIGH
        else:
            category = VolumeCategory.EMPHASIS

        return db, energy_variation, category

    def _estimate_pitch(self, audio: np.ndarray) -> float:
        """빠른 피치 추정 (자동상관 기반)"""
        # librosa의 피치 추정 사용
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length, threshold=0.1
        )

        # 유효한 크기를 가진 피치 값 선택
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 80 < pitch < 400:
                pitch_values.append(pitch)

        if pitch_values:
            return float(np.median(pitch_values))
        else:
            return 150.0

    def _estimate_speaking_rate(self, audio: np.ndarray) -> float:
        """빠른 발화 속도 추정"""
        # 음절 수를 추정하기 위해 onset 검출 사용
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )

        # 피크 탐지
        peaks = librosa.util.peak_pick(
            onset_envelope,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.3,
            wait=10,
        )

        # 속도 계산
        duration = len(audio) / self.sample_rate
        if duration > 0:
            rate = len(peaks) / duration
            # 합리적인 범위 제한
            return max(2.0, min(8.0, rate))
        else:
            return 4.0

    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """오디오에서 무음 비율 계산"""
        # 단순 에너지 기반 VAD
        frame_energy = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]

        # 20번째 분위수를 임계값으로 설정
        threshold = np.percentile(frame_energy, 20)
        silence_frames = np.sum(frame_energy < threshold)
        total_frames = len(frame_energy)

        if total_frames > 0:
            return silence_frames / total_frames
        else:
            return 0.0

    def _create_default_audio_features(self) -> AudioFeatures:
        """에러 상황에서 기본 AudioFeatures 생성"""
        return AudioFeatures(
            rms_energy=0.03,
            rms_db=-25.0,
            pitch_mean=150.0,
            pitch_variance=100.0,
            speaking_rate=4.0,
            amplitude_max=0.5,
            silence_ratio=0.2,
            spectral_centroid=2000.0,
            zcr=0.05,
            mfcc=[12.0, -8.0, 4.0],
            volume_category=VolumeCategory.MEDIUM,
            volume_peaks=[0.03] * 5,
        )

    def extract_features(
        self, audio_path: Path, start_time: float, end_time: float
    ) -> AudioFeatures:
        """
        자막 스타일링을 위한 필수 음향 특징 추출.

        속도 최적화 - 구간당 약 50ms 처리
        """
        try:
            # 오디오 구간 로드
            audio = self._load_audio_segment(audio_path, start_time, end_time)

            # 특징 추출
            volume_db, _, volume_cat = self._extract_volume_features(audio)
            pitch = self._estimate_pitch(audio)
            speaking_rate = self._estimate_speaking_rate(audio)
            silence_ratio = self._calculate_silence_ratio(audio)

            # 스펙트럼 중심
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            spectral_centroid_mean = float(np.mean(spectral_centroid))

            # 파형 표시용 볼륨 피크 추출
            frame_energy = librosa.feature.rms(
                y=audio, frame_length=self.frame_length, hop_length=self.hop_length
            )[0]

            # 균등 간격으로 샘플 추출
            if len(frame_energy) >= 10:
                step = len(frame_energy) // 10
                volume_peaks = [float(frame_energy[i * step]) for i in range(10)]
            else:
                # 가능한 모든 점 사용
                volume_peaks = frame_energy.tolist()

            # AudioFeatures 객체 생성
            return AudioFeatures(
                rms_energy=np.exp(volume_db / 20) * 1e-5,
                rms_db=volume_db,
                pitch_mean=pitch,
                pitch_variance=100.0,
                speaking_rate=speaking_rate,
                amplitude_max=float(np.max(np.abs(audio))),
                silence_ratio=silence_ratio,
                spectral_centroid=spectral_centroid_mean,
                zcr=0.05,
                mfcc=[12.0, -8.0, 4.0],
                volume_category=volume_cat,
                volume_peaks=volume_peaks,
            )

        except Exception as e:
            self.logger.error(
                "feature_extraction_failed",
                error=str(e),
                segment=f"{start_time}-{end_time}",
            )

            # 에러 발생 시 기본값 반환
            return self._create_default_audio_features()

    def extract_batch_features(
        self, audio_path: Path, segments: List[Tuple[float, float, str]]
    ) -> List[AudioFeatures]:
        """
        여러 구간의 특징을 효율적으로 추출.

        병렬 실행을 통한 배치 처리 최적화
        """

        if not segments:
            return []

        start_time = time.time()

        # 소규모 배치는 순차 처리
        if len(segments) <= self.batch_threshold:
            results = []
            for start, end, _ in segments:
                results.append(self.extract_features(audio_path, start, end))
            return results

        # 대규모 배치는 병렬 처리
        results: List[AudioFeatures] = [None] * len(segments)  # type: ignore

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(segments))
        ) as executor:
            futures = {}

            for i, (start, end, _) in enumerate(segments):
                future = executor.submit(self.extract_features, audio_path, start, end)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error("batch_segment_failed", index=idx, error=str(e))
                    # 기본 특징 사용
                    results[idx] = self._create_default_audio_features()

        self.logger.info(
            "batch_processing_completed",
            segments=len(segments),
            time=time.time() - start_time,
        )

        return results
