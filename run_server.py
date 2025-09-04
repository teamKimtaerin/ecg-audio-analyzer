#!/usr/bin/env python3
"""
오디오 분석 서버 실행 스크립트
"""

import uvicorn
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🎵 ECG Audio Analysis Server Starting...")
    print("📡 Server will be available at: http://localhost:8080")
    print("📋 API Documentation: http://localhost:8080/docs")

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        access_log=True,
    )
