#!/bin/bash
# ML 서버 시작 스크립트
# ECG Backend와의 통합을 위한 환경 설정 및 서버 시작

set -e

echo "🚀 ECG Audio Analyzer ML Server 시작 중..."
echo "=================================="

# 프로젝트 루트 디렉토리 설정
export PYTHONPATH="/Users/ahntaeju/project/ecg-audio-analyzer:$PYTHONPATH"
cd "/Users/ahntaeju/project/ecg-audio-analyzer"

# 환경 변수 설정
export BACKEND_URL="http://localhost:8000"        # ECG Backend URL
export ECG_BACKEND_URL="http://localhost:8000"     # 대체 환경변수명
export ML_SERVER_PORT=8080                         # ML 서버 포트
export LOG_LEVEL="info"                            # 로그 레벨

# 기본 설정 출력
echo "📋 설정 정보:"
echo "   - ML 서버 포트: $ML_SERVER_PORT"
echo "   - ECG Backend URL: $BACKEND_URL"
echo "   - 프로젝트 경로: $(pwd)"
echo "   - Python 경로: $PYTHONPATH"
echo "   - 로그 레벨: $LOG_LEVEL"
echo ""

# Python 의존성 확인
echo "🔍 Python 환경 확인 중..."
python3 -c "import fastapi, uvicorn, boto3; print('✅ 필수 패키지 확인 완료')" || {
    echo "❌ 필수 패키지가 설치되지 않았습니다."
    echo "다음 명령어로 설치하세요: pip install fastapi uvicorn boto3 requests"
    exit 1
}

# 포트 사용 여부 확인
echo "🔍 포트 $ML_SERVER_PORT 사용 여부 확인 중..."
if lsof -Pi :$ML_SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️ 포트 $ML_SERVER_PORT이 이미 사용 중입니다."
    echo "다음 명령어로 프로세스를 종료하세요:"
    echo "   lsof -ti:$ML_SERVER_PORT | xargs kill -9"
    echo ""
    echo "계속하시겠습니까? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "ML 서버 시작을 취소했습니다."
        exit 1
    fi
fi

# GPU 확인 (선택사항)
echo "🔍 GPU 환경 확인 중..."
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ GPU 사용 가능: {torch.cuda.device_count()}개 디바이스')
        for i in range(torch.cuda.device_count()):
            print(f'   - GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('⚠️ GPU 사용 불가 - CPU 모드로 실행됩니다')
except ImportError:
    print('⚠️ PyTorch 미설치 - CPU 모드로 실행됩니다')
" 2>/dev/null || echo "⚠️ GPU 확인 실패 - CPU 모드로 실행됩니다"

echo ""
echo "🚀 ML 서버 시작 중..."
echo "   - 서버 URL: http://localhost:$ML_SERVER_PORT"
echo "   - Health Check: http://localhost:$ML_SERVER_PORT/health"
echo "   - API 문서: http://localhost:$ML_SERVER_PORT/docs"
echo ""
echo "중지하려면 Ctrl+C를 누르세요."
echo "=================================="

# ML 서버 실행
exec python3 ml_api_server.py \
    --host 0.0.0.0 \
    --port $ML_SERVER_PORT \
    --log-level $LOG_LEVEL \
    --workers 1