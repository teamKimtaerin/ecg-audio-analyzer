"""
AWS Configuration Settings
Optimized for EC2 P3/G4 GPU instances with high-performance processing
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class AWSConfig:
    """AWS infrastructure configuration for GPU-accelerated processing"""
    
    # Instance configuration
    instance_type: str = "p3.2xlarge"  # NVIDIA V100 GPU
    region: str = "us-east-1"
    availability_zone: Optional[str] = None
    
    # GPU settings
    gpu_memory_limit: float = 0.8      # Use 80% of GPU memory
    cuda_device: str = "cuda:0"
    enable_mixed_precision: bool = True  # Use FP16 for speed
    gpu_batch_size: int = 8
    
    # Performance optimization
    concurrent_workers: int = 4         # Parallel processing workers
    max_queue_size: int = 100          # Maximum batch job queue size
    enable_gpu_monitoring: bool = True
    gpu_utilization_target: float = 0.85  # Target 85% GPU utilization
    
    # Storage configuration
    s3_bucket: str = field(default_factory=lambda: os.getenv("ECG_S3_BUCKET", "ecg-audio-processing"))
    s3_input_prefix: str = "input/"
    s3_output_prefix: str = "output/"
    s3_temp_prefix: str = "temp/"
    
    # EBS storage settings
    ebs_volume_size: int = 500         # GB
    ebs_volume_type: str = "gp3"
    ebs_iops: int = 3000
    ebs_throughput: int = 250          # MB/s
    
    # Networking
    enhanced_networking: bool = True
    placement_group: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    
    # Auto Scaling
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 5
    scale_up_threshold: float = 0.8    # Scale up when queue 80% full
    scale_down_threshold: float = 0.2  # Scale down when queue 20% full
    cooldown_period: int = 300         # 5 minutes cooldown
    
    # Cost optimization
    enable_spot_instances: bool = False  # Set True for cost savings
    spot_instance_types: List[str] = field(default_factory=lambda: ["g4dn.2xlarge", "p3.2xlarge"])
    max_spot_price: float = 1.0
    
    def __post_init__(self):
        """Validate AWS configuration"""
        if not self.s3_bucket:
            raise ValueError("S3 bucket name is required")
        
        # Set CUDA device based on availability
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count == 0:
                self.cuda_device = "cpu"
            else:
                self.cuda_device = f"cuda:0"
        else:
            self.cuda_device = "cpu"


@dataclass
class CloudWatchConfig:
    """CloudWatch monitoring configuration"""
    
    # Metrics configuration
    namespace: str = "ECG/AudioAnalysis"
    enable_custom_metrics: bool = True
    metric_resolution: int = 60        # 1 minute resolution
    
    # Performance metrics
    track_processing_time: bool = True
    track_gpu_utilization: bool = True
    track_memory_usage: bool = True
    track_throughput: bool = True
    track_error_rates: bool = True
    
    # Alarms
    enable_alarms: bool = True
    high_error_rate_threshold: float = 0.05    # 5% error rate
    low_gpu_utilization_threshold: float = 0.3  # 30% GPU utilization
    high_memory_threshold: float = 0.9          # 90% memory usage
    
    # Logs
    log_group_name: str = "/aws/ec2/ecg-audio-analysis"
    log_retention_days: int = 30
    enable_structured_logs: bool = True


@dataclass  
class S3Config:
    """S3 storage configuration for audio processing"""
    
    # Transfer acceleration
    enable_transfer_acceleration: bool = True
    multipart_threshold: int = 64 * 1024 * 1024  # 64MB
    multipart_chunksize: int = 16 * 1024 * 1024  # 16MB
    max_concurrency: int = 10
    
    # Lifecycle management
    enable_lifecycle_rules: bool = True
    temp_file_expiration_days: int = 1
    output_file_transition_days: int = 30  # Move to IA after 30 days
    
    # Security
    enable_encryption: bool = True
    kms_key_id: Optional[str] = None
    enable_versioning: bool = False  # Disable for temp files
    
    # Performance optimization
    request_payer: str = "BucketOwner"
    enable_crc32c_checksums: bool = True