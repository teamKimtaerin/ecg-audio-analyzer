# ECG Audio Analyzer API JSON Response Structures

이 문서는 ECG Audio Analyzer ML API의 JSON 응답 구조를 설명합니다. API는 비디오/오디오 파일을 분석하여 화자 인식, 음성 인식, 단어별 음향 특성을 제공합니다.

## 공통 응답 구조

모든 API 응답은 다음과 같은 기본 구조를 가집니다:

```json
{
  "success": true,
  "metadata": { /* 메타데이터 */ },
  "speakers": { /* 화자 정보 */ },
  "segments": [ /* 세그먼트 배열 */ ],
  "processing_time": 157.649908,
  "error": null,
  "error_code": null
}
```

### 메타데이터 (metadata)

```json
{
  "filename": "friends.mp4",
  "duration": 143.4,
  "sample_rate": 16000,
  "processed_at": "2025-09-11T16:28:35.586328Z",
  "processing_time": 157.649908,
  "total_segments": 25,
  "unique_speakers": 4,
  "processing_mode": "real_ml_models",
  "config": {
    "enable_gpu": false,
    "segment_length": 5.0,
    "language": "en",
    "unified_model": "whisperx-base-with-diarization"
  },
  "subtitle_optimization": true
}
```

### 화자 정보 (speakers)

```json
{
  "SPEAKER_00": {
    "total_duration": 25.677999999999994,
    "segment_count": 8
  },
  "SPEAKER_01": {
    "total_duration": 28.625,
    "segment_count": 10
  }
}
```

## Option C: 객체 기반 구조 (현재 기본)

각 단어가 개별 객체로 저장되는 전통적인 구조입니다.

### 세그먼트 구조

```json
{
  "start_time": 4.908,
  "end_time": 8.754,
  "duration": 3.845999999999999,
  "speaker": {
    "speaker_id": "SPEAKER_03"
  },
  "acoustic_features": {
    "volume_db": -22.255,
    "pitch_hz": 225.587,
    "spectral_centroid": 2423.728,
    "zero_crossing_rate": 0.05,
    "pitch_mean": 225.587,
    "pitch_std": 10.0,
    "mfcc_mean": [12.0, -8.0, 4.0]
  },
  "text": "You know, we should all do. Go see a musical.",
  "words": [
    {
      "word": "You",
      "start": 4.908,
      "end": 4.988,
      "volume_db": -17.254,
      "pitch_hz": 389.397,
      "spectral_centroid": 1877.97
    },
    {
      "word": "know,",
      "start": 5.008,
      "end": 5.168,
      "volume_db": -17.753,
      "pitch_hz": 150.0,
      "spectral_centroid": 1597.92
    }
  ]
}
```

### Option C 특징

**장점:**
- 📖 **높은 가독성**: 각 단어의 정보가 명확하게 구분
- 🔧 **쉬운 접근**: `segment.words[0].volume_db` 형태로 직관적 접근
- 🛠️ **개발 편의성**: 대부분의 JSON 라이브러리에서 자연스럽게 처리
- 📝 **자기 설명적**: 필드명만으로도 데이터 의미 파악 가능

**단점:**
- 💾 **큰 파일 크기**: 키 이름이 반복되어 파일이 큼
- 🐌 **느린 파싱**: 많은 객체 생성으로 인한 성능 저하
- 💻 **메모리 사용량 증가**: 각 단어마다 별도 객체 생성

**사용 사례:**
- 웹 애플리케이션 프론트엔드
- 프로토타이핑 및 개발
- 소량의 데이터 처리

## Option A: 병렬 배열 구조 (최적화)

단어 데이터를 병렬 배열로 저장하여 효율성을 극대화한 구조입니다.

### 세그먼트 구조

```json
{
  "start_time": 4.908,
  "end_time": 8.754,
  "duration": 3.845999999999999,
  "speaker": {
    "speaker_id": "SPEAKER_03"
  },
  "acoustic_features": {
    "volume_db": -22.255,
    "pitch_hz": 225.587,
    "spectral_centroid": 2423.728,
    "zero_crossing_rate": 0.05,
    "pitch_mean": 225.587,
    "pitch_std": 10.0,
    "mfcc_mean": [12.0, -8.0, 4.0]
  },
  "text": "You know, we should all do. Go see a musical.",
  "words": ["You", "know,", "we", "should", "all", "do.", "Go", "see", "a", "musical."],
  "word_times": [
    [4.908, 4.988], [5.008, 5.168], [5.188, 5.269], [5.289, 5.449],
    [5.649, 5.809], [5.869, 6.05], [6.11, 7.752], [7.812, 8.033],
    [8.113, 8.153], [8.273, 8.754]
  ],
  "word_acoustics": {
    "volume_db": [-17.254, -17.753, -17.501, -25.599, -17.479, -19.976, -26.05, -22.294, -30.129, -26.643],
    "pitch_hz": [389.397, 150.0, 357.81, 299.009, 225.884, 374.49, 179.402, 299.253, 241.547, 221.589],
    "spectral_centroid": [1877.97, 1597.92, 2168.705, 3231.783, 1898.108, 1910.971, 2415.008, 3496.49, 1808.347, 2441.09]
  }
}
```

### Option A 특징

**장점:**
- 🚀 **빠른 성능**: 배열 접근이 객체 접근보다 빠름
- 💾 **작은 파일 크기**: 키 중복 제거로 ~40% 크기 감소
- 💻 **메모리 효율**: 동일 타입 데이터의 연속 저장
- 📊 **분석 최적화**: NumPy/Pandas로 직접 변환 가능
- ⚡ **대량 처리**: 수천 개 단어도 효율적 처리

**단점:**
- 🤔 **복잡한 접근**: `word_acoustics.volume_db[index]` 형태로 접근
- 📏 **배열 길이 관리**: 모든 배열 길이가 일치해야 함
- 🔧 **개발 복잡성**: 인덱스 기반 접근으로 실수 가능성

**사용 사례:**
- 대용량 데이터 처리
- 실시간 분석 시스템
- 데이터 분석 및 머신러닝
- 네트워크 대역폭이 제한적인 환경

## 데이터 접근 방법 비교

### Option C (객체 기반)
```javascript
// 첫 번째 단어의 볼륨 가져오기
const firstWordVolume = segment.words[0].volume_db;

// 모든 단어의 볼륨 가져오기
const allVolumes = segment.words.map(word => word.volume_db);
```

### Option A (병렬 배열)
```javascript
// 첫 번째 단어의 볼륨 가져오기
const firstWordVolume = segment.word_acoustics.volume_db[0];

// 모든 단어의 볼륨 가져오기 (이미 배열)
const allVolumes = segment.word_acoustics.volume_db;
```

## 음향 특성 필드 설명

### 세그먼트 레벨 음향 특성
- `volume_db`: 평균 음량 (데시벨)
- `pitch_hz`: 평균 피치 (헤르츠)
- `spectral_centroid`: 스펙트럼 중심 주파수
- `zero_crossing_rate`: 영교차율
- `pitch_mean`: 피치 평균값
- `pitch_std`: 피치 표준편차
- `mfcc_mean`: MFCC 계수 평균 (3차원 배열)

### 단어 레벨 음향 특성
- `volume_db`: 단어별 음량 (데시벨)
- `pitch_hz`: 단어별 피치 (헤르츠)
- `spectral_centroid`: 단어별 스펙트럼 중심 주파수

## API 엔드포인트

### POST /transcribe
비디오/오디오 파일을 분석하여 전사 결과와 음향 특성을 반환합니다.

**요청 예시:**
```json
{
  "video_path": "friends.mp4",
  "enable_gpu": false,
  "language": "en",
  "output_path": "output/result.json"
}
```

**응답 형태:** Option C 또는 Option A 구조 (설정에 따라)

## 성능 비교

| 구조 | 파일 크기 | 파싱 속도 | 메모리 사용량 | 가독성 | 개발 편의성 |
|------|-----------|-----------|---------------|--------|-------------|
| Option C | 100% | 100% | 100% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Option A | 60% | 150% | 70% | ⭐⭐⭐ | ⭐⭐⭐ |

## 선택 가이드

### Option C를 선택하세요:
- 🌐 웹 프론트엔드 개발
- 🛠️ 빠른 프로토타이핑
- 👥 팀원들의 JSON 경험이 적은 경우
- 📖 코드 가독성이 최우선인 경우

### Option A를 선택하세요:
- 📊 대용량 데이터 분석
- ⚡ 성능이 중요한 실시간 시스템
- 💰 네트워크 비용 절감이 필요한 경우
- 🔬 머신러닝/데이터 사이언스 용도

---

**문서 작성일:** 2025-09-12  
**API 버전:** v1.0  
**최종 업데이트:** JSON 구조 최적화 완료