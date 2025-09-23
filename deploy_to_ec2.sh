#!/bin/bash
# ECG Audio Analyzer - EC2 배포 스크립트
# EC2 인스턴스(54.197.171.76)에서 실행할 스크립트

set -e

echo "🚀 ECG Audio Analyzer ML 서버 배포 시작..."
echo "📅 $(date)"

# 프로젝트 디렉토리로 이동
cd ~/ecg-audio-analyzer

# 현재 실행 중인 서버 중지
echo "🛑 기존 ML 서버 중지..."
pkill -f "python ml_api_server.py" || echo "실행 중인 서버가 없습니다."
sleep 2

# Git 상태 확인 및 최신 코드 가져오기
echo "📦 최신 코드 업데이트..."
git fetch origin
git checkout main
git pull origin main

# 서버는 backup 디렉토리에서 계속 실행
cd ~/ecg-audio-analyzer.backup

# 가상환경 활성화
echo "🐍 가상환경 활성화..."
source venv/bin/activate

# 의존성 확인 및 업데이트
echo "📚 의존성 확인..."
pip install -r requirements.txt --quiet

# 이전 로그 백업
if [ -f server.log ]; then
    mv server.log server_backup_$(date +%Y%m%d_%H%M%S).log
fi

# ML 서버 시작 (LD_LIBRARY_PATH 설정 포함)
echo "🚀 ML 서버 시작..."
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH nohup python ml_api_server.py --host 0.0.0.0 --port 8080 > server.log 2>&1 &

echo "⏳ 서버 시작 대기 중..."
sleep 10

# 서버 상태 확인
if pgrep -f "ml_api_server.py" > /dev/null; then
    echo "✅ ML 서버가 성공적으로 시작되었습니다!"
    echo "📊 프로세스 ID: $(pgrep -f ml_api_server.py)"

    # Health check
    echo "🔍 Health check 수행..."
    if curl -s http://localhost:8080/health > /dev/null; then
        echo "✅ Health check 통과 - 서버가 정상 작동 중입니다!"
    else
        echo "⚠️ Health check 실패 - 서버는 실행 중이지만 응답하지 않습니다."
    fi

    echo ""
    echo "📋 배포 완료 정보:"
    echo "- 서버 URL: http://54.197.171.76:8080"
    echo "- Health check: http://54.197.171.76:8080/health"
    echo "- 로그 확인: tail -f ~/ecg-audio-analyzer.backup/server.log"
    echo "- 서버 중지: pkill -f 'python ml_api_server.py'"

else
    echo "❌ ML 서버 시작에 실패했습니다!"
    echo "📄 최근 로그:"
    tail -20 server.log
    exit 1
fi

echo "🎉 배포가 완료되었습니다!"