# Frontend 통합 가이드 - ML 서버 결과 처리

## 📋 ML 서버가 제공하는 결과 구조

ML 서버는 분석 완료 시 Backend로 다음과 같은 JSON을 전송합니다:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result": {
    "metadata": {
      "filename": "video.mp4",
      "duration": 143.39,
      "total_segments": 25,
      "unique_speakers": 4
    },
    "segments": [
      {
        "start_time": 4.908,
        "end_time": 8.754,
        "speaker": {
          "speaker_id": "SPEAKER_01"
        },
        "text": "You know, we should all do. Go see a musical.",
        "words": [
          {
            "word": "You",
            "start": 4.908,
            "end": 4.988,
            "volume_db": -19.87,
            "pitch_hz": 851.09
          },
          {
            "word": "know",
            "start": 5.0,
            "end": 5.2,
            "volume_db": -22.1,
            "pitch_hz": 420.5
          }
        ]
      }
    ]
  }
}
```

## 🔧 Frontend에서 구현해야 할 코드

### 1. useTranscriptionPolling.ts 수정

```typescript
// src/hooks/useTranscriptionPolling.ts

case 'completed':
  updateProgress(100, 'completed')
  
  // ✅ 결과 처리 로직 추가
  if (jobStatus.results && jobStatus.results.segments) {
    // segments를 clips로 변환
    const clips = convertSegmentsToClips(jobStatus.results.segments)
    
    // Editor store에 저장
    if (editorStore) {
      editorStore.setClips(clips)
      editorStore.setSpeakers(extractSpeakers(jobStatus.results.segments))
    }
    
    // 완료 콜백 호출
    if (onTranscriptionComplete) {
      onTranscriptionComplete(jobStatus.results)
    }
    
    // 성공 메시지 표시
    toast.success('트랜스크립션이 완료되었습니다!')
  }
  
  // 폴링 중지
  if (intervalRef.current) {
    clearInterval(intervalRef.current)
    intervalRef.current = null
  }
  break
```

### 2. 변환 유틸리티 함수 추가

```typescript
// src/utils/transcriptionConverter.ts

export interface MLSegment {
  start_time: number
  end_time: number
  speaker: {
    speaker_id: string
  }
  text: string
  words: Array<{
    word: string
    start: number
    end: number
    volume_db?: number
    pitch_hz?: number
  }>
}

export interface EditorClip {
  id: string
  startTime: number
  endTime: number
  text: string
  speaker: string
  words: Array<{
    word: string
    start: number
    end: number
    volume_db: number
    pitch_hz: number
  }>
  // 애니메이션용 추가 필드
  avgVolume?: number
  avgPitch?: number
}

export function convertSegmentsToClips(segments: MLSegment[]): EditorClip[] {
  return segments.map((segment, index) => {
    // 평균 볼륨과 피치 계산
    const avgVolume = segment.words.length > 0
      ? segment.words.reduce((sum, w) => sum + (w.volume_db || -20), 0) / segment.words.length
      : -20
    
    const avgPitch = segment.words.length > 0
      ? segment.words.reduce((sum, w) => sum + (w.pitch_hz || 200), 0) / segment.words.length
      : 200
    
    return {
      id: `clip-${Date.now()}-${index}`,
      startTime: segment.start_time,
      endTime: segment.end_time,
      text: segment.text,
      speaker: segment.speaker.speaker_id,
      words: segment.words.map(word => ({
        word: word.word,
        start: word.start,
        end: word.end,
        volume_db: word.volume_db || -20,
        pitch_hz: word.pitch_hz || 200
      })),
      avgVolume,
      avgPitch
    }
  })
}

export function extractSpeakers(segments: MLSegment[]): string[] {
  const speakerSet = new Set<string>()
  segments.forEach(segment => {
    speakerSet.add(segment.speaker.speaker_id)
  })
  return Array.from(speakerSet).sort()
}
```

### 3. transcriptionStore 확장

```typescript
// src/stores/transcriptionStore.ts

import { create } from 'zustand'

interface TranscriptionStore {
  // 기존 필드들...
  jobId: string | null
  status: string
  progress: number
  
  // ✅ 추가 필드
  results: any | null
  onCompleteCallback: ((results: any) => void) | null
  
  // 기존 메서드들...
  setJobId: (id: string) => void
  updateProgress: (progress: number, status: string) => void
  
  // ✅ 추가 메서드
  setResults: (results: any) => void
  onComplete: (callback: (results: any) => void) => void
  clearResults: () => void
}

export const useTranscriptionStore = create<TranscriptionStore>((set) => ({
  // 기존 상태...
  jobId: null,
  status: 'idle',
  progress: 0,
  
  // ✅ 새 상태
  results: null,
  onCompleteCallback: null,
  
  // 기존 액션...
  setJobId: (id) => set({ jobId: id }),
  updateProgress: (progress, status) => set({ progress, status }),
  
  // ✅ 새 액션
  setResults: (results) => set({ results }),
  onComplete: (callback) => set({ onCompleteCallback: callback }),
  clearResults: () => set({ results: null, jobId: null, progress: 0, status: 'idle' })
}))
```

### 4. Editor 페이지 연동

```typescript
// src/pages/Editor.tsx

import { useEffect } from 'react'
import { useTranscriptionStore } from '@/stores/transcriptionStore'
import { useEditorStore } from '@/stores/editorStore'
import { convertSegmentsToClips } from '@/utils/transcriptionConverter'

export function Editor() {
  const { results, onComplete } = useTranscriptionStore()
  const { setClips, setSpeakers } = useEditorStore()
  
  useEffect(() => {
    // 트랜스크립션 완료 시 자동으로 clips 업데이트
    onComplete((transcriptionResults) => {
      if (transcriptionResults?.segments) {
        const clips = convertSegmentsToClips(transcriptionResults.segments)
        setClips(clips)
        
        // 화자 목록 업데이트
        const speakers = [...new Set(clips.map(c => c.speaker))]
        setSpeakers(speakers)
        
        // 성공 알림
        toast.success(`${clips.length}개의 자막이 생성되었습니다!`)
      }
    })
  }, [onComplete, setClips, setSpeakers])
  
  // ... 나머지 Editor 컴포넌트 코드
}
```

## 🎯 화자별 색상 매핑

```typescript
// src/utils/speakerColors.ts

const SPEAKER_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Teal
  '#45B7D1', // Blue
  '#96CEB4', // Green
  '#FFEAA7', // Yellow
  '#DDA0DD', // Plum
  '#98D8C8', // Mint
  '#FFB6C1', // Pink
]

export function getSpeakerColor(speakerId: string, allSpeakers: string[]): string {
  const index = allSpeakers.indexOf(speakerId)
  return SPEAKER_COLORS[index % SPEAKER_COLORS.length]
}
```

## 📊 단어별 애니메이션 활용

ML 서버가 제공하는 `volume_db`와 `pitch_hz` 값을 활용한 애니메이션:

```typescript
// src/components/AnimatedSubtitle.tsx

interface AnimatedSubtitleProps {
  word: string
  volume_db: number  // -60 ~ 0 범위
  pitch_hz: number   // 50 ~ 500 범위
}

export function AnimatedSubtitle({ word, volume_db, pitch_hz }: AnimatedSubtitleProps) {
  // 볼륨에 따른 크기 계산 (큰 소리일수록 큰 글자)
  const scale = 1 + (volume_db + 30) / 60  // 0.5 ~ 1.5 범위
  
  // 피치에 따른 색상 계산 (높은 음일수록 밝은 색)
  const brightness = 50 + (pitch_hz / 500) * 50  // 50 ~ 100 범위
  
  return (
    <span
      style={{
        transform: `scale(${scale})`,
        filter: `brightness(${brightness}%)`,
        transition: 'all 0.3s ease',
        display: 'inline-block',
        margin: '0 2px'
      }}
    >
      {word}
    </span>
  )
}
```

## ✅ 테스트 체크리스트

1. **비디오 업로드**
   - [ ] 파일 선택 및 업로드 성공
   - [ ] 진행률 표시 정상 작동

2. **처리 중**
   - [ ] 진행률 0% → 100% 업데이트
   - [ ] 상태 메시지 표시

3. **완료 후**
   - [ ] 자막 리스트에 segments 표시
   - [ ] 화자별 색상 구분
   - [ ] 타임라인에 자막 표시
   - [ ] 편집 기능 작동

4. **애니메이션**
   - [ ] 볼륨에 따른 글자 크기 변화
   - [ ] 피치에 따른 밝기 변화

## 🚨 문제 해결

### Q: 결과를 받았는데 자막이 표시되지 않음
A: Browser DevTools에서 다음 확인:
1. Network 탭에서 job-status API 응답에 results 필드 있는지 확인
2. Console에서 convertSegmentsToClips 함수 호출 결과 확인
3. Redux DevTools에서 editorStore의 clips 상태 확인

### Q: 화자 구분이 안 됨
A: segments의 speaker.speaker_id 필드가 올바른지 확인

### Q: 애니메이션이 작동하지 않음
A: words 배열의 volume_db, pitch_hz 값이 숫자인지 확인

## 📞 지원

ML 서버 관련 문제는 다음 파일 참조:
- `ml_api_server.py`: API 엔드포인트 구현
- `ML_SERVER_SETUP.md`: 서버 실행 가이드
- `test_integration.py`: 통합 테스트 스크립트