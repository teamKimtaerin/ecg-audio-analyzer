import os
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Any
import warnings

import torch
import torch.nn
import torch.cuda
import torch.backends.cudnn as cudnn

from .logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")


@dataclass
class GPUConfig:
    enable_amp: bool = True
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    memory_fraction: float = 0.9


class GPUOptimizer:
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.logger = get_logger().bind_context(component="gpu_optimizer")

        self.device = self._setup_device()
        self._apply_optimizations()

        self.logger.info(
            "gpu_optimizer_initialized",
            device=str(self.device),
            amp_enabled=self.config.enable_amp,
            tf32_enabled=self.config.enable_tf32,
        )

    def _setup_device(self) -> torch.device:
        """GPU 디바이스 설정, 없으면 CPU로 대체"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(0)
            self.logger.info(
                "gpu_detected",
                name=props.name,
                memory_gb=props.total_memory / 1024**3,
                compute_capability=f"{props.major}.{props.minor}",
            )
            return device
        else:
            self.logger.warning("gpu_not_available_using_cpu")
            return torch.device("cpu")

    def _apply_optimizations(self) -> None:
        """기본 GPU 최적화 적용"""
        if not self.device.type == "cuda":
            return

        # 메모리 사용 비율 설정
        if self.config.memory_fraction < 1.0:
            try:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                self.logger.info(
                    "memory_fraction_set", fraction=self.config.memory_fraction
                )
            except Exception as e:
                self.logger.warning("memory_fraction_failed", error=str(e))

        # Ampere GPU에서 TF32 활성화
        if self.config.enable_tf32 and hasattr(
            torch.backends.cuda.matmul, "allow_tf32"
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("tf32_enabled")

        # CuDNN 최적화
        if self.config.enable_cudnn_benchmark:
            cudnn.benchmark = True
            cudnn.deterministic = False
            self.logger.info("cudnn_benchmark_enabled")

        # 메모리 풀 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    @contextmanager
    def inference_mode(self, model: torch.nn.Module, enable_amp: Optional[bool] = None):
        """최적화된 추론용 컨텍스트 매니저"""
        if enable_amp is None:
            enable_amp = self.config.enable_amp and self.device.type == "cuda"

        original_training = model.training

        try:
            model.eval()

            with torch.no_grad():
                if enable_amp:
                    with torch.cuda.amp.autocast():
                        yield model
                else:
                    yield model
        finally:
            model.train(original_training)

    def optimize_model(
        self, model: torch.nn.Module, model_name: str
    ) -> torch.nn.Module:
        """추론을 위한 모델 최적화"""
        try:
            # 디바이스로 이동
            optimized_model = model.to(self.device)
            self.logger.info(
                "model_moved_to_device", model_name=model_name, device=str(self.device)
            )

            # 최신 GPU에서 half precision 활성화
            if (
                self.device.type == "cuda"
                and self.config.enable_amp
                and torch.cuda.get_device_capability(0)[0] >= 7
            ):  # Volta+
                optimized_model = optimized_model.half()
                self.logger.info("model_converted_to_half", model_name=model_name)

            return optimized_model

        except Exception as e:
            self.logger.error(
                "model_optimization_failed", model_name=model_name, error=str(e)
            )
            return model

    def get_memory_info(self) -> Dict[str, Any]:
        """현재 GPU 메모리 정보 가져오기"""
        if self.device.type != "cuda":
            return {"device": "cpu", "gpu_available": False}

        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(0) / 1024**2

            if hasattr(torch.cuda, "mem_get_info"):
                free, total = torch.cuda.mem_get_info(0)
                free_mb = free / 1024**2
                total_mb = total / 1024**2
            else:
                props = torch.cuda.get_device_properties(0)
                total_mb = props.total_memory / 1024**2
                free_mb = total_mb - reserved

            return {
                "device": str(self.device),
                "gpu_available": True,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "free_mb": round(free_mb, 2),
                "total_mb": round(total_mb, 2),
                "usage_percent": (
                    round((allocated / total_mb) * 100, 1) if total_mb > 0 else 0
                ),
            }

        except Exception as e:
            self.logger.warning("memory_info_failed", error=str(e))
            return {"device": str(self.device), "gpu_available": True, "error": str(e)}

    def clear_cache(self) -> None:
        """GPU 메모리 캐시 비우기"""
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                self.logger.info("gpu_cache_cleared")
            except Exception as e:
                self.logger.error("cache_clearing_failed", error=str(e))

    def cleanup(self) -> None:
        """GPU 리소스 정리"""
        self.clear_cache()
        self.logger.info("gpu_optimizer_cleanup_completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 전역 인스턴스
_gpu_optimizer: Optional[GPUOptimizer] = None


def get_gpu_optimizer(config: Optional[GPUConfig] = None) -> GPUOptimizer:
    """전역 GPU Optimizer 인스턴스 가져오기 또는 생성"""
    global _gpu_optimizer

    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer(config)

    return _gpu_optimizer


def optimize_for_inference(
    model: torch.nn.Module, model_name: str, config: Optional[GPUConfig] = None
) -> torch.nn.Module:
    """모델을 추론용으로 최적화하는 편의 함수"""
    optimizer = get_gpu_optimizer(config)
    return optimizer.optimize_model(model, model_name)
