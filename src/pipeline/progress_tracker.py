"""
Progress Tracker - Pipeline progress tracking and estimation
Single Responsibility: Track and calculate pipeline progress
"""

from dataclasses import dataclass, field
from typing import List, Dict
from ..utils.logger import get_logger


@dataclass
class PipelineProgress:
    """Pipeline progress tracking"""

    current_stage: str = "initialization"
    completed_stages: List[str] = field(default_factory=list)
    total_stages: int = 0
    progress_percentage: float = 0.0
    error_count: int = 0
    warnings_count: int = 0


class ProgressTracker:
    """Track pipeline progress and provide status updates"""

    def __init__(self, stage_config: Dict[str, Dict]):
        self.logger = get_logger().bind_context(component="progress_tracker")
        self.progress = PipelineProgress()
        self.stage_config = stage_config
        self._initialize_total_stages()

        self.logger.info(
            "progress_tracker_initialized", total_stages=self.progress.total_stages
        )

    def _initialize_total_stages(self):
        """Initialize total required stages count"""
        self.progress.total_stages = len(
            [
                stage
                for stage, config in self.stage_config.items()
                if config.get("required", True)
            ]
        )

    def _calculate_progress_percentage(self) -> float:
        """Calculate progress percentage based on completed required stages"""
        if self.progress.total_stages == 0:
            return 100.0

        completed_required = len(
            [
                stage
                for stage in self.progress.completed_stages
                if stage in self.stage_config
                and self.stage_config[stage].get("required", True)
            ]
        )
        return (completed_required / self.progress.total_stages) * 100

    def update_stage(self, stage_name: str):
        """Update current stage"""
        self.progress.current_stage = stage_name
        self.progress.progress_percentage = self._calculate_progress_percentage()

        self.logger.info(
            "progress_updated",
            stage=stage_name,
            progress=self.progress.progress_percentage,
            completed_stages=len(self.progress.completed_stages),
        )

    def mark_stage_completed(self, stage_name: str):
        """Mark a stage as completed"""
        if stage_name not in self.progress.completed_stages:
            self.progress.completed_stages.append(stage_name)
            self.logger.info("stage_completed", stage=stage_name)
            self.update_stage(stage_name)

    def add_error(self):
        """Increment error count"""
        self.progress.error_count += 1
        self.logger.warning("error_recorded", total_errors=self.progress.error_count)

    def add_warning(self):
        """Increment warning count"""
        self.progress.warnings_count += 1
        self.logger.info(
            "warning_recorded", total_warnings=self.progress.warnings_count
        )

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed"""
        return stage_name in self.progress.completed_stages

    def get_progress(self) -> PipelineProgress:
        """Get current progress state"""
        return self.progress

    def is_completed(self) -> bool:
        """Check if all required stages are completed"""
        required_stages = [
            stage
            for stage, config in self.stage_config.items()
            if config.get("required", True)
        ]
        return all(stage in self.progress.completed_stages for stage in required_stages)

    def reset(self):
        """Reset progress tracking"""
        self.progress = PipelineProgress()
        self._initialize_total_stages()
        self.logger.info("progress_reset")
