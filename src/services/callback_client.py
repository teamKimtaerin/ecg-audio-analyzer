"""
FastAPI API 클라이언트 모듈

EC2 ML 서버에서 ECS FastAPI 백엔드의 API를 호출하는 HTTP 클라이언트
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
from dataclasses import dataclass, asdict
from enum import Enum


class ProcessingStatus(Enum):
    """처리 상태"""

    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CallbackPayload:
    """콜백 페이로드 구조"""

    job_id: str
    status: ProcessingStatus
    progress: float  # 0.0 ~ 1.0
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FastAPIClient:
    """ECS FastAPI 백엔드의 API를 호출하는 HTTP 클라이언트"""

    def __init__(
        self,
        fastapi_base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            fastapi_base_url: FastAPI 백엔드의 기본 URL (예: http://fastapi-server:8000)
            timeout: HTTP 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        """
        self.fastapi_base_url = fastapi_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)

    async def send_ml_results(self, payload: CallbackPayload) -> bool:
        """
        FastAPI 백엔드의 /api/v1/ml-results 엔드포인트로 결과 전송

        Args:
            payload: 전송할 ML 분석 결과 데이터

        Returns:
            bool: 전송 성공 여부
        """

        # API 엔드포인트 URL 구성
        api_url = f"{self.fastapi_base_url}/api/v1/ml-results"
        # Enum을 문자열로 변환
        payload_dict = asdict(payload)
        payload_dict["status"] = payload.status.value

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ECG-ML-Server/1.0",
            "Accept": "application/json",
            "Connection": "close",
        }

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # 연결 풀과 커넥터 설정 최적화
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )

                timeout_config = aiohttp.ClientTimeout(
                    total=self.timeout, connect=10.0, sock_read=20.0
                )

                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout_config,
                    headers={"User-Agent": "ECG-ML-Server/1.0"},
                ) as session:

                    self.logger.info(
                        f"콜백 전송 시도 {attempt + 1}/{self.max_retries + 1} - Job ID: {payload.job_id}"
                    )

                    async with session.post(
                        api_url,
                        json=payload_dict,
                        headers={k: v for k, v in headers.items() if k != "User-Agent"},
                    ) as response:

                        if 200 <= response.status < 300:
                            self.logger.info(
                                f"콜백 전송 성공 - Job ID: {payload.job_id}, Status: {response.status}"
                            )
                            return True

                        elif response.status in [400, 422]:
                            # 클라이언트 오류는 재시도하지 않음
                            error_text = await response.text()
                            self.logger.error(
                                f"콜백 클라이언트 오류 (재시도 안함) - Status: {response.status}, "
                                f"Job ID: {payload.job_id}, Response: {error_text[:500]}"
                            )
                            return False

                        elif response.status >= 500:
                            # 서버 오류는 재시도
                            error_text = await response.text()
                            last_error = (
                                f"서버 오류 {response.status}: {error_text[:200]}"
                            )
                            self.logger.warning(
                                f"콜백 서버 오류 (재시도 예정) - Status: {response.status}, "
                                f"Job ID: {payload.job_id}, 시도: {attempt + 1}"
                            )
                        else:
                            # 기타 상태 코드
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text[:200]}"
                            self.logger.warning(
                                f"콜백 예상치 못한 응답 - Status: {response.status}, "
                                f"Job ID: {payload.job_id}, 시도: {attempt + 1}"
                            )

            except asyncio.TimeoutError as e:
                last_error = f"타임아웃: {str(e)}"
                self.logger.warning(
                    f"콜백 타임아웃 - Job ID: {payload.job_id}, 시도: {attempt + 1}"
                )

            except aiohttp.ClientConnectorError as e:
                last_error = f"연결 오류: {str(e)}"
                self.logger.warning(
                    f"콜백 연결 오류 - Job ID: {payload.job_id}, 시도: {attempt + 1}, 오류: {str(e)}"
                )

            except aiohttp.ServerDisconnectedError as e:
                last_error = f"서버 연결 끊김: {str(e)}"
                self.logger.warning(
                    f"콜백 서버 연결 끊김 - Job ID: {payload.job_id}, 시도: {attempt + 1}"
                )

            except aiohttp.ClientError as e:
                last_error = f"클라이언트 오류: {str(e)}"
                self.logger.warning(
                    f"콜백 클라이언트 오류 - Job ID: {payload.job_id}, 시도: {attempt + 1}, 오류: {str(e)}"
                )

            except Exception as e:
                last_error = f"예상치 못한 오류: {str(e)}"
                self.logger.error(
                    f"콜백 전송 중 예상치 못한 오류 - Job ID: {payload.job_id}, 시도: {attempt + 1}, 오류: {str(e)}"
                )

            # 마지막 시도가 아니면 지수 백오프로 대기
            if attempt < self.max_retries:
                wait_time = self.retry_delay * (2**attempt)
                self.logger.info(
                    f"콜백 재시도 대기 - Job ID: {payload.job_id}, 대기 시간: {wait_time}초"
                )
                await asyncio.sleep(wait_time)

        self.logger.error(
            f"콜백 전송 최종 실패 - Job ID: {payload.job_id}, 마지막 오류: {last_error}"
        )
        return False

    async def send_progress_update(
        self, job_id: str, progress: float, message: str = None
    ) -> bool:
        """진행 상황 업데이트 전송"""
        payload = CallbackPayload(
            job_id=job_id,
            status=ProcessingStatus.PROCESSING,
            progress=progress,
            results={"message": message} if message else None,
        )
        return await self.send_ml_results(payload)

    async def send_completion(self, job_id: str, results: Dict[str, Any]) -> bool:
        """처리 완료 결과 전송"""
        payload = CallbackPayload(
            job_id=job_id,
            status=ProcessingStatus.COMPLETED,
            progress=1.0,
            results=results,
        )
        return await self.send_ml_results(payload)

    async def send_error(
        self, job_id: str, error_message: str, progress: float = 0.0
    ) -> bool:
        """오류 상태 전송"""
        payload = CallbackPayload(
            job_id=job_id,
            status=ProcessingStatus.FAILED,
            progress=progress,
            error_message=error_message,
        )
        return await self.send_ml_results(payload)


# 싱글톤 FastAPI 클라이언트 인스턴스 (전역 사용)
fastapi_client: Optional[FastAPIClient] = None


def initialize_fastapi_client(fastapi_base_url: str, **kwargs) -> FastAPIClient:
    """FastAPI 클라이언트 초기화"""
    global fastapi_client
    fastapi_client = FastAPIClient(fastapi_base_url, **kwargs)
    return fastapi_client


def get_fastapi_client() -> Optional[FastAPIClient]:
    """FastAPI 클라이언트 인스턴스 반환"""
    return fastapi_client
