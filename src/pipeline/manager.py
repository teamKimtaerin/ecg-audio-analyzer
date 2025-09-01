"""
Pipeline Manager - Service Orchestration
Single Responsibility: Coordinate entire analysis workflow efficiently

High-performance orchestration with async operations, GPU management, and error recovery.
"""

import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import torch
import psutil

from ..services.audio_extractor import AudioExtractor, AudioExtractionResult
from ..services.speaker_diarizer import SpeakerDiarizer, DiarizationResult
from ..models.output_models import (
    CompleteAnalysisResult, 
    AnalysisMetadata, 
    PerformanceStats,
    create_empty_analysis_result
)
from ..utils.logger import get_logger
from ...config.base_settings import BaseConfig, ProcessingConfig
from ...config.aws_settings import AWSConfig
from ...config.model_configs import SpeakerDiarizationConfig


@dataclass
class PipelineStage:
    """Individual pipeline stage definition"""
    name: str
    service: Any
    required: bool = True
    gpu_intensive: bool = False
    parallel_capable: bool = False
    estimated_time_ratio: float = 1.0  # Relative time compared to audio extraction
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PipelineProgress:
    """Pipeline progress tracking"""
    current_stage: str = "initialization"
    completed_stages: List[str] = field(default_factory=list)
    total_stages: int = 0
    progress_percentage: float = 0.0
    estimated_completion_time: Optional[float] = None
    error_count: int = 0
    warnings_count: int = 0


@dataclass
class ResourceUsage:
    """System resource usage tracking"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    active_threads: int = 0
    peak_memory_mb: float = 0.0
    
    def update_peak_memory(self, current_memory: float):
        """Update peak memory if current is higher"""
        if current_memory > self.peak_memory_mb:
            self.peak_memory_mb = current_memory


class GPUResourceManager:
    """Manage GPU resources and queue processing"""
    
    def __init__(self, max_concurrent_gpu_tasks: int = 2):
        self.max_concurrent_gpu_tasks = max_concurrent_gpu_tasks
        self.active_gpu_tasks = 0
        self.gpu_semaphore = asyncio.Semaphore(max_concurrent_gpu_tasks)
        self.logger = get_logger().bind_context(component="gpu_resource_manager")
    
    @asynccontextmanager
    async def acquire_gpu_resource(self):
        """Context manager for GPU resource acquisition"""
        async with self.gpu_semaphore:
            self.active_gpu_tasks += 1
            self.logger.debug("gpu_resource_acquired", 
                            active_tasks=self.active_gpu_tasks,
                            max_tasks=self.max_concurrent_gpu_tasks)
            
            try:
                yield
            finally:
                self.active_gpu_tasks -= 1
                self.logger.debug("gpu_resource_released", 
                                active_tasks=self.active_gpu_tasks)
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "utilization": 0.0}
        
        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
            
            # Try to get utilization if nvidia-ml-py3 available
            utilization = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu / 100.0
            except ImportError:
                pass
            
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "utilization": utilization
            }
        except Exception:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "utilization": 0.0}


class PipelineManager:
    """
    High-performance pipeline orchestration manager.
    
    Single Responsibility: Coordinate all analysis services efficiently with
    async operations, resource management, and comprehensive error handling.
    """
    
    def __init__(self,
                 base_config: BaseConfig,
                 processing_config: ProcessingConfig,
                 aws_config: Optional[AWSConfig] = None,
                 speaker_config: Optional[SpeakerDiarizationConfig] = None):
        
        self.base_config = base_config
        self.processing_config = processing_config
        self.aws_config = aws_config
        self.speaker_config = speaker_config or SpeakerDiarizationConfig()
        
        self.logger = get_logger().bind_context(component="pipeline_manager")
        
        # Resource management
        self.gpu_manager = GPUResourceManager(
            max_concurrent_gpu_tasks=aws_config.concurrent_workers if aws_config else 2
        )
        self.thread_pool = ThreadPoolExecutor(
            max_workers=processing_config.max_workers
        )
        # Process pool for CPU-intensive tasks (better for ML inference)
        cpu_cores = psutil.cpu_count(logical=False) or 4
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(cpu_cores, processing_config.max_workers)
        )
        
        # Progress tracking
        self.progress = PipelineProgress()
        self.resource_usage = ResourceUsage()
        
        # Service instances (lazy initialization)
        self._audio_extractor: Optional[AudioExtractor] = None
        self._speaker_diarizer: Optional[SpeakerDiarizer] = None
        
        # Pipeline stages definition
        self.stages = self._initialize_pipeline_stages()
        self.progress.total_stages = len([s for s in self.stages.values() if s.required])
        
        self.logger.info("pipeline_manager_initialized",
                        total_stages=self.progress.total_stages,
                        gpu_workers=self.gpu_manager.max_concurrent_gpu_tasks,
                        thread_workers=processing_config.max_workers)
        
        # Initialize with optional model preloading
        self._models_preloaded = False
    
    def _initialize_pipeline_stages(self) -> Dict[str, PipelineStage]:
        """Initialize pipeline stages configuration"""
        return {
            "audio_extraction": PipelineStage(
                name="audio_extraction",
                service="audio_extractor",
                required=True,
                gpu_intensive=False,
                parallel_capable=True,
                estimated_time_ratio=1.0,
                dependencies=[]
            ),
            "speaker_diarization": PipelineStage(
                name="speaker_diarization",
                service="speaker_diarizer",
                required=True,
                gpu_intensive=True,
                parallel_capable=False,
                estimated_time_ratio=2.0,
                dependencies=["audio_extraction"]
            ),
            "emotion_analysis": PipelineStage(
                name="emotion_analysis",
                service="emotion_analyzer",
                required=False,  # Optional for MVP
                gpu_intensive=True,
                parallel_capable=True,
                estimated_time_ratio=1.5,
                dependencies=["speaker_diarization"]
            ),
            "acoustic_analysis": PipelineStage(
                name="acoustic_analysis",
                service="acoustic_analyzer",
                required=True,  # Optional for MVP
                gpu_intensive=False,
                parallel_capable=True,
                estimated_time_ratio=0.8,
                dependencies=["audio_extraction"]
            ),
            "result_synthesis": PipelineStage(
                name="result_synthesis",
                service="result_synthesizer",
                required=True,
                gpu_intensive=True,
                parallel_capable=False,
                estimated_time_ratio=0.2,
                dependencies=["speaker_diarization"]
            )
        }
    
    def _get_audio_extractor(self) -> AudioExtractor:
        """Get or create audio extractor instance"""
        if self._audio_extractor is None:
            self._audio_extractor = AudioExtractor(
                config=self.base_config,
                processing_config=self.processing_config,
                validation_config=self.base_config,  # Using base config for validation
                enable_gpu_acceleration=self.aws_config.cuda_device.startswith('cuda') if self.aws_config else False
            )
        return self._audio_extractor
    
    def _get_speaker_diarizer(self) -> SpeakerDiarizer:
        """Get or create speaker diarizer instance"""
        if self._speaker_diarizer is None:
            device = self.aws_config.cuda_device if self.aws_config else "cpu"
            self._speaker_diarizer = SpeakerDiarizer(
                config=self.speaker_config,
                device=device,
                enable_caching=True
            )
        return self._speaker_diarizer
    
    async def preload_models(self, enable_warmup: bool = True) -> Dict[str, Any]:
        """
        Preload models to reduce first-time processing latency
        
        Args:
            enable_warmup: Whether to warm up models with dummy data
            
        Returns:
            Dictionary with preloading results and timing
        """
        if self._models_preloaded:
            return {"status": "already_preloaded"}
        
        self.logger.info("preloading_pipeline_models", warmup=enable_warmup)
        
        start_time = time.time()
        
        try:
            # Get model manager from services
            from ..models.model_manager import get_model_manager
            model_manager = get_model_manager()
            
            # Preload models in parallel using thread pool
            loop = asyncio.get_event_loop()
            
            preload_results = await loop.run_in_executor(
                self.thread_pool,
                model_manager.preload_models,
                True,  # load_speaker
                True,  # load_whisperx
                False  # load_emotion (optional)
            )
            
            # Warm up models if requested
            warmup_results = {}
            if enable_warmup:
                warmup_results = await loop.run_in_executor(
                    self.thread_pool,
                    model_manager.warmup_models
                )
            
            preload_time = time.time() - start_time
            self._models_preloaded = True
            
            self.logger.info("models_preloaded_successfully",
                           preload_time=preload_time,
                           models_loaded=list(preload_results.keys()),
                           warmup_enabled=enable_warmup)
            
            return {
                "status": "success",
                "preload_time": preload_time,
                "models": preload_results,
                "warmup_times": warmup_results
            }
            
        except Exception as e:
            self.logger.error("model_preloading_failed", error=str(e))
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _update_progress(self, stage_name: str, percentage: float = None):
        """Update pipeline progress"""
        self.progress.current_stage = stage_name
        
        if percentage is not None:
            self.progress.progress_percentage = percentage
        else:
            # Calculate based on completed stages
            completed_required = len([s for s in self.progress.completed_stages 
                                    if self.stages.get(s, {}).required])
            total_required = len([s for s in self.stages.values() if s.required])
            self.progress.progress_percentage = (completed_required / total_required) * 100
        
        self.logger.info("pipeline_progress_updated",
                        stage=stage_name,
                        progress=self.progress.progress_percentage,
                        completed_stages=len(self.progress.completed_stages))
    
    def _update_resource_usage(self):
        """Update resource usage statistics"""
        try:
            # System resources
            process = psutil.Process()
            self.resource_usage.cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            self.resource_usage.memory_mb = current_memory
            self.resource_usage.update_peak_memory(current_memory)
            
            # GPU resources
            gpu_info = self.gpu_manager.get_gpu_memory_info()
            self.resource_usage.gpu_memory_mb = gpu_info["allocated_mb"]
            self.resource_usage.gpu_utilization = gpu_info["utilization"]
            
            # Thread count
            self.resource_usage.active_threads = process.num_threads()
            
        except Exception as e:
            self.logger.warning("resource_monitoring_failed", error=str(e))
    
    async def _run_audio_extraction(self, source: Union[str, Path]) -> AudioExtractionResult:
        """Run audio extraction stage"""
        self._update_progress("audio_extraction")
        
        with self.logger.performance_timer("audio_extraction_stage"):
            extractor = self._get_audio_extractor()
            
            # Run in thread pool for I/O intensive operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                extractor.extract_single,
                source
            )
            
            if result.success:
                self.progress.completed_stages.append("audio_extraction")
                self.logger.info("audio_extraction_completed", 
                               output_path=str(result.output_path),
                               duration=result.duration_seconds)
            else:
                self.progress.error_count += 1
                self.logger.error("audio_extraction_failed", 
                                error=result.error_message)
            
            return result
    
    async def _run_speaker_diarization(self, audio_path: Path) -> DiarizationResult:
        """Run speaker diarization stage"""
        self._update_progress("speaker_diarization")
        
        # Use GPU resource manager for GPU-intensive task
        async with self.gpu_manager.acquire_gpu_resource():
            with self.logger.performance_timer("speaker_diarization_stage"):
                
                diarizer = self._get_speaker_diarizer()
                
                # Load model if not already loaded
                if diarizer.pipeline is None:
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        diarizer.load_model
                    )
                
                # Run diarization
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    diarizer.diarize_audio,
                    audio_path
                )
                
                if result.success:
                    self.progress.completed_stages.append("speaker_diarization")
                    self.logger.info("speaker_diarization_completed",
                                   total_speakers=result.total_speakers,
                                   total_segments=len(result.segments),
                                   processing_time=result.processing_time)
                else:
                    self.progress.error_count += 1
                    self.logger.error("speaker_diarization_failed",
                                    error=result.error_message)
                
                return result
    
    async def _create_basic_result(self, 
                                  source: Union[str, Path],
                                  extraction_result: AudioExtractionResult,
                                  diarization_result: DiarizationResult) -> CompleteAnalysisResult:
        """Create basic analysis result from available data"""
        
        filename = Path(source).name if isinstance(source, (str, Path)) else "unknown"
        duration = extraction_result.duration_seconds or 0.0
        
        # Create basic metadata
        metadata = AnalysisMetadata(
            filename=filename,
            duration=duration,
            total_speakers=diarization_result.total_speakers,
            processing_time=f"00:00:{int(extraction_result.extraction_time_seconds + diarization_result.processing_time):02d}",
            gpu_acceleration=self.aws_config.cuda_device.startswith('cuda') if self.aws_config else False
        )
        
        # Create performance stats
        self._update_resource_usage()
        performance_stats = PerformanceStats(
            gpu_utilization=self.resource_usage.gpu_utilization,
            peak_memory_mb=int(self.resource_usage.peak_memory_mb),
            avg_processing_fps=duration / (extraction_result.extraction_time_seconds + diarization_result.processing_time) if duration > 0 else 0.0,
            bottleneck_stage="speaker_diarization" if diarization_result.processing_time > extraction_result.extraction_time_seconds else "audio_extraction"
        )
        
        # For MVP, create basic result with available data
        result = CompleteAnalysisResult(
            metadata=metadata,
            speakers={},  # Will be populated by ResultSynthesizer
            timeline=[],  # Will be populated by ResultSynthesizer  
            performance_stats=performance_stats
        )
        
        return result
    
    async def process_single(self, 
                           source: Union[str, Path],
                           output_path: Optional[Path] = None) -> CompleteAnalysisResult:
        """
        Process a single audio/video file through the complete pipeline.
        
        Args:
            source: Input file path or URL
            output_path: Optional output path for results
            
        Returns:
            CompleteAnalysisResult with comprehensive analysis
        """
        
        start_time = time.time()
        source_str = str(source)
        
        with self.logger.performance_timer("complete_pipeline", items_count=1):
            
            self.logger.info("pipeline_started", 
                           source=source_str,
                           output_path=str(output_path) if output_path else None)
            
            try:
                # Stage 1: Audio Extraction
                extraction_result = await self._run_audio_extraction(source)
                if not extraction_result.success:
                    return create_empty_analysis_result(
                        filename=Path(source_str).name,
                        duration=0.0
                    )
                
                # Stage 2: Speaker Diarization
                diarization_result = await self._run_speaker_diarization(extraction_result.output_path)
                if not diarization_result.success:
                    return create_empty_analysis_result(
                        filename=Path(source_str).name,
                        duration=extraction_result.duration_seconds or 0.0
                    )
                
                # For MVP: Create basic result (emotion analysis and acoustic features will be added later)
                self._update_progress("result_synthesis")
                result = await self._create_basic_result(source, extraction_result, diarization_result)
                
                # Mark synthesis as completed
                self.progress.completed_stages.append("result_synthesis")
                
                # Save result if output path specified
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        f.write(result.model_dump_json_formatted())
                    
                    self.logger.info("results_saved", output_path=str(output_path))
                
                total_time = time.time() - start_time
                
                self.logger.info("pipeline_completed",
                               source=source_str,
                               total_time=total_time,
                               total_speakers=result.metadata.total_speakers,
                               success=True)
                
                return result
                
            except Exception as e:
                self.progress.error_count += 1
                error_msg = str(e)
                self.logger.error("pipeline_failed",
                                source=source_str,
                                error=error_msg,
                                traceback=traceback.format_exc())
                
                # Return error result
                result = create_empty_analysis_result(
                    filename=Path(source_str).name,
                    duration=0.0
                )
                result.metadata.processing_time = f"00:00:{int(time.time() - start_time):02d}"
                
                return result
    
    async def process_batch(self, 
                          sources: List[Union[str, Path]],
                          output_dir: Optional[Path] = None,
                          max_concurrent: int = None) -> List[CompleteAnalysisResult]:
        """
        Process multiple files concurrently.
        
        Args:
            sources: List of input file paths or URLs
            output_dir: Optional directory for output files
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of CompleteAnalysisResult objects
        """
        
        if max_concurrent is None:
            max_concurrent = self.processing_config.max_workers
        
        with self.logger.performance_timer("batch_pipeline", items_count=len(sources)):
            
            self.logger.info("batch_processing_started",
                           source_count=len(sources),
                           max_concurrent=max_concurrent,
                           output_dir=str(output_dir) if output_dir else None)
            
            # Create output directory if specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(source, index):
                async with semaphore:
                    output_path = None
                    if output_dir:
                        source_name = Path(str(source)).stem
                        output_path = output_dir / f"{source_name}_analysis.json"
                    
                    return await self.process_single(source, output_path)
            
            # Process all sources concurrently
            tasks = [process_with_semaphore(source, i) for i, source in enumerate(sources)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error("batch_processing_exception",
                                    source_index=i,
                                    error=str(result))
                    processed_results.append(
                        create_empty_analysis_result(
                            filename=Path(str(sources[i])).name,
                            duration=0.0
                        )
                    )
                else:
                    processed_results.append(result)
            
            # Log batch summary
            successful = sum(1 for r in processed_results if r.metadata.total_speakers > 0)
            failed = len(processed_results) - successful
            
            self.logger.info("batch_processing_completed",
                           total=len(sources),
                           successful=successful,
                           failed=failed)
            
            return processed_results
    
    def get_progress(self) -> PipelineProgress:
        """Get current pipeline progress"""
        return self.progress
    
    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        self._update_resource_usage()
        return self.resource_usage
    
    def cleanup(self):
        """Clean up pipeline resources"""
        try:
            # Clean up services
            if self._audio_extractor:
                if hasattr(self._audio_extractor, 'cleanup_temp_files'):
                    self._audio_extractor.cleanup_temp_files()
            
            if self._speaker_diarizer:
                self._speaker_diarizer.cleanup()
            
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("pipeline_cleanup_completed")
            
        except Exception as e:
            self.logger.error("pipeline_cleanup_failed", error=str(e))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error("pipeline_manager_exception",
                            exception_type=str(exc_type),
                            exception_message=str(exc_val))