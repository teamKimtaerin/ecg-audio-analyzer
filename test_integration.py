#!/usr/bin/env python3
"""
통합 테스트 스크립트 - ECG Backend와 ML 서버 간 통신 검증
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any
import requests
import aiohttp

# 설정
ML_SERVER_URL = "http://localhost:8080"
ECG_BACKEND_URL = "http://localhost:8000"
TEST_TIMEOUT = 30  # seconds


class Colors:
    """터미널 컬러 출력"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(message: str, status: str = "info"):
    """상태 메시지 출력"""
    color = {
        "success": Colors.GREEN,
        "error": Colors.RED,
        "warning": Colors.YELLOW,
        "info": Colors.BLUE,
    }.get(status, Colors.BLUE)

    symbol = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
        status, "ℹ️"
    )

    print(f"{color}{symbol} {message}{Colors.END}")


class IntegrationTester:
    """통합 테스트 실행기"""

    def __init__(self):
        self.ml_server_url = ML_SERVER_URL
        self.backend_url = ECG_BACKEND_URL
        self.test_job_id = str(uuid.uuid4())
        self.test_results = []

    def run_all_tests(self):
        """모든 테스트 실행"""
        print(f"{Colors.BOLD}ECG Backend ↔ ML Server 통합 테스트{Colors.END}")
        print("=" * 60)
        print(f"ML 서버: {self.ml_server_url}")
        print(f"ECG Backend: {self.backend_url}")
        print(f"테스트 시작: {datetime.now().isoformat()}")
        print()

        # 테스트 순서
        tests = [
            ("ML 서버 헬스체크", self.test_ml_health),
            ("ECG Backend 연결 확인", self.test_backend_connection),
            ("ML 서버 API 구조 확인", self.test_ml_api_structure),
            ("비디오 처리 요청 테스트", self.test_video_processing),
            ("콜백 URL 테스트", self.test_callback_functionality),
            ("에러 처리 테스트", self.test_error_handling),
        ]

        for test_name, test_func in tests:
            print_status(f"테스트 시작: {test_name}", "info")
            try:
                result = test_func()
                if result:
                    print_status(f"테스트 성공: {test_name}", "success")
                    self.test_results.append((test_name, True, None))
                else:
                    print_status(f"테스트 실패: {test_name}", "error")
                    self.test_results.append((test_name, False, "테스트 실패"))
            except Exception as e:
                print_status(f"테스트 예외: {test_name} - {str(e)}", "error")
                self.test_results.append((test_name, False, str(e)))
            print()

        self.print_summary()

    def test_ml_health(self) -> bool:
        """ML 서버 헬스체크 테스트"""
        try:
            response = requests.get(f"{self.ml_server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   응답: {json.dumps(data, ensure_ascii=False)}")
                return data.get("status") == "healthy"
            else:
                print(f"   HTTP 상태: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"   연결 실패: ML 서버가 {self.ml_server_url}에서 실행되지 않음")
            return False
        except Exception as e:
            print(f"   예외 발생: {str(e)}")
            return False

    def test_backend_connection(self) -> bool:
        """ECG Backend 연결 확인 (선택사항)"""
        try:
            # 간단한 ping 시도
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            print(f"   ECG Backend 응답: {response.status_code}")
            return True
        except requests.exceptions.ConnectionError:
            print_status("ECG Backend 연결 불가 - 테스트는 계속 진행", "warning")
            return True  # Backend 없어도 ML 서버 테스트는 가능
        except Exception as e:
            print(f"   Backend 확인 중 예외: {str(e)}")
            return True

    def test_ml_api_structure(self) -> bool:
        """ML 서버 API 구조 확인"""
        try:
            # Root endpoint 확인
            response = requests.get(f"{self.ml_server_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   서비스: {data.get('service')}")
                print(f"   버전: {data.get('version')}")

                # 필수 엔드포인트 확인
                endpoints = data.get("endpoints", {})
                required_endpoints = ["process-video-api", "health"]

                missing = [ep for ep in required_endpoints if ep not in endpoints]
                if missing:
                    print(f"   누락된 엔드포인트: {missing}")
                    return False

                print(f"   사용 가능한 엔드포인트: {list(endpoints.keys())}")
                return True
            return False
        except Exception as e:
            print(f"   API 구조 확인 실패: {str(e)}")
            return False

    def test_video_processing(self) -> bool:
        """비디오 처리 요청 테스트"""
        test_payload = {
            "job_id": self.test_job_id,
            "video_url": "https://example.com/test-video.mp4",  # 테스트용 URL
        }

        try:
            response = requests.post(
                f"{self.ml_server_url}/api/upload-video/process-video",
                json=test_payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ECS-FastAPI-Backend/1.0",
                },
                timeout=10,
            )

            print(f"   HTTP 상태: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   응답: {json.dumps(data, ensure_ascii=False, indent=2)}")

                # 필수 필드 확인
                required_fields = ["job_id", "status", "message"]
                for field in required_fields:
                    if field not in data:
                        print(f"   누락된 필드: {field}")
                        return False

                if data["job_id"] == self.test_job_id and data["status"] == "accepted":
                    return True

            print(f"   응답 내용: {response.text}")
            return False

        except Exception as e:
            print(f"   비디오 처리 요청 실패: {str(e)}")
            return False

    def test_callback_functionality(self) -> bool:
        """콜백 기능 테스트"""
        print("   콜백은 실제 ECG Backend가 있어야 완전히 테스트 가능")
        print("   ML 서버 코드에서 콜백 URL 확인 중...")

        # ML 서버 코드에서 콜백 설정 확인
        try:
            # 서버 정보에서 backend_url 확인
            response = requests.get(f"{self.ml_server_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                backend_url = data.get("backend_url")
                if backend_url:
                    print(f"   설정된 Backend URL: {backend_url}")
                    expected_callback_url = f"{backend_url}/api/v1/ml/ml-results"
                    print(f"   예상 콜백 URL: {expected_callback_url}")

                    if backend_url == self.backend_url:
                        return True
                    else:
                        print_status(
                            f"Backend URL 불일치: {backend_url} vs {self.backend_url}",
                            "warning",
                        )
                        return False
        except Exception as e:
            print(f"   콜백 설정 확인 실패: {str(e)}")

        return False

    def test_error_handling(self) -> bool:
        """에러 처리 테스트"""
        # 잘못된 요청으로 에러 처리 확인
        invalid_payload = {"job_id": "test"}  # video_url 누락

        try:
            response = requests.post(
                f"{self.ml_server_url}/api/upload-video/process-video",
                json=invalid_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            print(f"   HTTP 상태: {response.status_code}")

            # 400 또는 422 상태 코드를 기대
            if response.status_code in [400, 422]:
                print(f"   에러 응답: {response.text}")
                return True
            else:
                print(f"   예상하지 못한 응답: {response.text}")
                return False

        except Exception as e:
            print(f"   에러 처리 테스트 실패: {str(e)}")
            return False

    def print_summary(self):
        """테스트 결과 요약"""
        print("=" * 60)
        print(f"{Colors.BOLD}테스트 결과 요약{Colors.END}")
        print()

        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)

        for test_name, success, error in self.test_results:
            status = "success" if success else "error"
            print_status(f"{test_name}", status)
            if error:
                print(f"     오류: {error}")

        print()
        if passed == total:
            print_status(f"모든 테스트 통과: {passed}/{total}", "success")
        else:
            print_status(f"테스트 결과: {passed}/{total} 통과", "warning")

        print()
        print("🚀 ML 서버 실행 방법:")
        print("   ./start_ml_server.sh")
        print()
        print("🌐 실제 테스트 URL:")
        print(f"   Health Check: {self.ml_server_url}/health")
        print(f"   API 문서: {self.ml_server_url}/docs")
        print()
        print("📋 다음 단계:")
        if passed < total:
            print("   1. ML 서버가 실행되고 있는지 확인")
            print("   2. 환경변수가 올바르게 설정되어 있는지 확인")
            print("   3. 포트 충돌이 없는지 확인")
        else:
            print("   1. ECG Backend 실행")
            print("   2. 실제 비디오 파일로 통합 테스트")
            print("   3. 프로덕션 환경에서 최종 검증")


def main():
    """메인 함수"""
    tester = IntegrationTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
