"""
AWS Utilities
S3, EC2, and CloudWatch integration for high-performance audio analysis
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from .logger import get_logger


@dataclass
class S3UploadResult:
    """Result of S3 upload operation"""
    success: bool
    s3_key: str
    bucket: str
    upload_time: float
    file_size: int
    error_message: Optional[str] = None
    
    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.s3_key}"


@dataclass
class S3DownloadResult:
    """Result of S3 download operation"""
    success: bool
    local_path: Path
    s3_key: str
    bucket: str
    download_time: float
    file_size: int
    error_message: Optional[str] = None


@dataclass
class CloudWatchMetric:
    """CloudWatch custom metric"""
    namespace: str
    metric_name: str
    value: float
    unit: str = "Count"
    dimensions: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class S3Manager:
    """High-performance S3 operations for audio analysis pipeline"""
    
    def __init__(self, 
                 bucket_name: str,
                 region: str = "us-east-1",
                 enable_transfer_acceleration: bool = True,
                 max_concurrent_uploads: int = 10):
        
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 not available. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region = region
        self.enable_transfer_acceleration = enable_transfer_acceleration
        self.max_concurrent_uploads = max_concurrent_uploads
        
        self.logger = get_logger().bind_context(component="s3_manager")
        
        # Initialize S3 client with optimizations
        self._init_s3_client()
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_uploads)
        
        self.logger.info("s3_manager_initialized",
                        bucket=bucket_name,
                        region=region,
                        transfer_acceleration=enable_transfer_acceleration)
    
    def _init_s3_client(self):
        """Initialize S3 client with optimizations"""
        try:
            # Configure session with optimizations
            session = boto3.Session()
            
            config = boto3.session.Config(
                region_name=self.region,
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                },
                max_pool_connections=50,
                # Enable transfer acceleration if requested
                s3={
                    'addressing_style': 'virtual' if self.enable_transfer_acceleration else 'auto'
                }
            )
            
            self.s3_client = session.client('s3', config=config)
            
            # Enable transfer acceleration if requested
            if self.enable_transfer_acceleration:
                try:
                    self.s3_client.put_bucket_accelerate_configuration(
                        Bucket=self.bucket_name,
                        AccelerateConfiguration={'Status': 'Enabled'}
                    )
                    self.logger.info("transfer_acceleration_enabled")
                except ClientError as e:
                    self.logger.warning("transfer_acceleration_failed", error=str(e))
            
            # Validate bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info("s3_bucket_access_verified", bucket=self.bucket_name)
            
        except NoCredentialsError:
            self.logger.error("aws_credentials_not_found")
            raise RuntimeError("AWS credentials not found")
        except ClientError as e:
            self.logger.error("s3_client_initialization_failed", error=str(e))
            raise
    
    def upload_file(self, 
                    local_path: Union[str, Path], 
                    s3_key: str,
                    metadata: Optional[Dict[str, str]] = None,
                    content_type: Optional[str] = None) -> S3UploadResult:
        """
        Upload file to S3 with optimizations.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata
            content_type: Optional content type
            
        Returns:
            S3UploadResult with upload details
        """
        
        start_time = time.time()
        local_path = Path(local_path)
        
        try:
            if not local_path.exists():
                return S3UploadResult(
                    success=False,
                    s3_key=s3_key,
                    bucket=self.bucket_name,
                    upload_time=0.0,
                    file_size=0,
                    error_message="Local file not found"
                )
            
            file_size = local_path.stat().st_size
            
            # Determine content type
            if content_type is None:
                if local_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    content_type = 'audio/wav' if local_path.suffix.lower() == '.wav' else 'audio/mpeg'
                elif local_path.suffix.lower() == '.json':
                    content_type = 'application/json'
                else:
                    content_type = 'application/octet-stream'
            
            # Prepare extra args
            extra_args = {
                'ContentType': content_type
            }
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Use multipart upload for large files
            if file_size > 100 * 1024 * 1024:  # 100MB
                extra_args['Config'] = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=64 * 1024 * 1024,  # 64MB
                    max_concurrency=10,
                    multipart_chunksize=16 * 1024 * 1024,  # 16MB
                    use_threads=True
                )
            
            # Upload file
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            upload_time = time.time() - start_time
            
            self.logger.info("s3_file_uploaded",
                           local_path=str(local_path),
                           s3_key=s3_key,
                           file_size_mb=file_size / 1024 / 1024,
                           upload_time=upload_time)
            
            return S3UploadResult(
                success=True,
                s3_key=s3_key,
                bucket=self.bucket_name,
                upload_time=upload_time,
                file_size=file_size
            )
            
        except Exception as e:
            upload_time = time.time() - start_time
            self.logger.error("s3_upload_failed",
                            local_path=str(local_path),
                            s3_key=s3_key,
                            error=str(e))
            
            return S3UploadResult(
                success=False,
                s3_key=s3_key,
                bucket=self.bucket_name,
                upload_time=upload_time,
                file_size=local_path.stat().st_size if local_path.exists() else 0,
                error_message=str(e)
            )
    
    def download_file(self, 
                     s3_key: str, 
                     local_path: Union[str, Path]) -> S3DownloadResult:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            S3DownloadResult with download details
        """
        
        start_time = time.time()
        local_path = Path(local_path)
        
        try:
            # Create parent directories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            
            download_time = time.time() - start_time
            file_size = local_path.stat().st_size
            
            self.logger.info("s3_file_downloaded",
                           s3_key=s3_key,
                           local_path=str(local_path),
                           file_size_mb=file_size / 1024 / 1024,
                           download_time=download_time)
            
            return S3DownloadResult(
                success=True,
                local_path=local_path,
                s3_key=s3_key,
                bucket=self.bucket_name,
                download_time=download_time,
                file_size=file_size
            )
            
        except Exception as e:
            download_time = time.time() - start_time
            self.logger.error("s3_download_failed",
                            s3_key=s3_key,
                            local_path=str(local_path),
                            error=str(e))
            
            return S3DownloadResult(
                success=False,
                local_path=local_path,
                s3_key=s3_key,
                bucket=self.bucket_name,
                download_time=download_time,
                file_size=0,
                error_message=str(e)
            )
    
    def upload_batch(self, 
                    file_mappings: List[tuple],
                    progress_callback: Optional[callable] = None) -> List[S3UploadResult]:
        """
        Upload multiple files concurrently.
        
        Args:
            file_mappings: List of (local_path, s3_key) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of S3UploadResult objects
        """
        
        with self.logger.performance_timer("s3_batch_upload", items_count=len(file_mappings)):
            
            self.logger.info("s3_batch_upload_started", file_count=len(file_mappings))
            
            # Submit upload tasks
            futures = []
            for local_path, s3_key in file_mappings:
                future = self.thread_pool.submit(self.upload_file, local_path, s3_key)
                futures.append(future)
            
            # Collect results
            results = []
            completed = 0
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(file_mappings))
                        
                except Exception as e:
                    self.logger.error("batch_upload_future_failed", error=str(e))
                    # Create error result
                    results.append(S3UploadResult(
                        success=False,
                        s3_key="unknown",
                        bucket=self.bucket_name,
                        upload_time=0.0,
                        file_size=0,
                        error_message=str(e)
                    ))
                    completed += 1
            
            # Log summary
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            self.logger.info("s3_batch_upload_completed",
                           total=len(file_mappings),
                           successful=successful,
                           failed=failed)
            
            return results
    
    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = response.get('Contents', [])
            return objects
            
        except Exception as e:
            self.logger.error("s3_list_objects_failed", prefix=prefix, error=str(e))
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info("s3_object_deleted", s3_key=s3_key)
            return True
        except Exception as e:
            self.logger.error("s3_delete_failed", s3_key=s3_key, error=str(e))
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)
        self.logger.info("s3_manager_cleanup_completed")


class CloudWatchMetrics:
    """CloudWatch metrics publisher for monitoring"""
    
    def __init__(self, namespace: str = "ECG/AudioAnalysis", region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 not available")
        
        self.namespace = namespace
        self.region = region
        self.logger = get_logger().bind_context(component="cloudwatch_metrics")
        
        # Initialize CloudWatch client
        self.cw_client = boto3.client('cloudwatch', region_name=region)
        
        # Metrics buffer for batch publishing
        self.metrics_buffer: List[CloudWatchMetric] = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 20  # CloudWatch limit
        
        # Background publishing
        self.publishing_thread = None
        self.should_publish = True
        
        self.logger.info("cloudwatch_metrics_initialized", namespace=namespace)
    
    def put_metric(self, 
                   metric_name: str, 
                   value: float, 
                   unit: str = "Count",
                   dimensions: Optional[Dict[str, str]] = None):
        """Add metric to buffer for publishing"""
        
        metric = CloudWatchMetric(
            namespace=self.namespace,
            metric_name=metric_name,
            value=value,
            unit=unit,
            dimensions=dimensions or {}
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            
            # Flush buffer if it's full
            if len(self.metrics_buffer) >= self.max_buffer_size:
                self._flush_metrics()
    
    def _flush_metrics(self):
        """Flush metrics buffer to CloudWatch"""
        if not self.metrics_buffer:
            return
        
        try:
            # Prepare metric data for CloudWatch
            metric_data = []
            
            for metric in self.metrics_buffer:
                data = {
                    'MetricName': metric.metric_name,
                    'Value': metric.value,
                    'Unit': metric.unit,
                    'Timestamp': metric.timestamp
                }
                
                if metric.dimensions:
                    data['Dimensions'] = [
                        {'Name': k, 'Value': v} for k, v in metric.dimensions.items()
                    ]
                
                metric_data.append(data)
            
            # Send to CloudWatch
            self.cw_client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
            
            self.logger.debug("cloudwatch_metrics_published", count=len(metric_data))
            self.metrics_buffer.clear()
            
        except Exception as e:
            self.logger.error("cloudwatch_metrics_publish_failed", error=str(e))
    
    def flush(self):
        """Manually flush all buffered metrics"""
        with self.buffer_lock:
            self._flush_metrics()
    
    def start_background_publishing(self, interval: float = 60.0):
        """Start background metrics publishing"""
        def publishing_loop():
            while self.should_publish:
                time.sleep(interval)
                with self.buffer_lock:
                    if self.metrics_buffer:
                        self._flush_metrics()
        
        if not self.publishing_thread or not self.publishing_thread.is_alive():
            self.publishing_thread = threading.Thread(target=publishing_loop, daemon=True)
            self.publishing_thread.start()
            self.logger.info("background_metrics_publishing_started", interval=interval)
    
    def stop_background_publishing(self):
        """Stop background publishing and flush remaining metrics"""
        self.should_publish = False
        if self.publishing_thread:
            self.publishing_thread.join(timeout=5.0)
        
        # Flush remaining metrics
        self.flush()
        self.logger.info("background_metrics_publishing_stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_background_publishing()


class EC2InstanceManager:
    """EC2 instance management utilities"""
    
    def __init__(self, region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 not available")
        
        self.region = region
        self.logger = get_logger().bind_context(component="ec2_manager")
        
        # Initialize EC2 client
        self.ec2_client = boto3.client('ec2', region_name=region)
        
        self.logger.info("ec2_manager_initialized", region=region)
    
    def get_instance_info(self) -> Dict[str, Any]:
        """Get current EC2 instance information"""
        try:
            # Try to get instance metadata
            import requests
            
            # Get instance ID
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=2
            )
            instance_id = response.text
            
            # Get instance details
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            
            instance = response['Reservations'][0]['Instances'][0]
            
            return {
                'instance_id': instance_id,
                'instance_type': instance.get('InstanceType'),
                'availability_zone': instance.get('Placement', {}).get('AvailabilityZone'),
                'private_ip': instance.get('PrivateIpAddress'),
                'public_ip': instance.get('PublicIpAddress'),
                'launch_time': instance.get('LaunchTime'),
                'state': instance.get('State', {}).get('Name')
            }
            
        except Exception as e:
            self.logger.warning("instance_info_unavailable", error=str(e))
            return {
                'instance_id': 'unknown',
                'instance_type': 'unknown',
                'availability_zone': 'unknown',
                'running_on_ec2': False
            }
    
    def set_instance_tags(self, instance_id: str, tags: Dict[str, str]) -> bool:
        """Set tags on EC2 instance"""
        try:
            self.ec2_client.create_tags(
                Resources=[instance_id],
                Tags=[{'Key': k, 'Value': v} for k, v in tags.items()]
            )
            
            self.logger.info("instance_tags_set", instance_id=instance_id, tags=tags)
            return True
            
        except Exception as e:
            self.logger.error("instance_tags_failed", error=str(e))
            return False


class AWSIntegration:
    """Complete AWS integration for ECG Audio Analysis"""
    
    def __init__(self, 
                 s3_bucket: str,
                 region: str = "us-east-1",
                 enable_metrics: bool = True):
        
        if not AWS_AVAILABLE:
            raise RuntimeError("AWS integration requires boto3")
        
        self.s3_bucket = s3_bucket
        self.region = region
        self.logger = get_logger().bind_context(component="aws_integration")
        
        # Initialize AWS services
        self.s3_manager = S3Manager(s3_bucket, region)
        
        if enable_metrics:
            self.metrics = CloudWatchMetrics(region=region)
            self.metrics.start_background_publishing()
        else:
            self.metrics = None
        
        self.ec2_manager = EC2InstanceManager(region)
        
        # Get instance information
        self.instance_info = self.ec2_manager.get_instance_info()
        
        self.logger.info("aws_integration_initialized",
                        s3_bucket=s3_bucket,
                        region=region,
                        metrics_enabled=enable_metrics,
                        instance_type=self.instance_info.get('instance_type'))
    
    def upload_results(self, 
                      local_results_dir: Path, 
                      s3_prefix: str = "results/") -> List[S3UploadResult]:
        """Upload all results to S3"""
        
        # Find all JSON files in results directory
        json_files = list(local_results_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("no_results_found", results_dir=str(local_results_dir))
            return []
        
        # Create file mappings
        file_mappings = []
        for json_file in json_files:
            s3_key = f"{s3_prefix}{json_file.name}"
            file_mappings.append((json_file, s3_key))
        
        # Upload batch
        results = self.s3_manager.upload_batch(file_mappings)
        
        # Send metrics
        if self.metrics:
            successful = sum(1 for r in results if r.success)
            self.metrics.put_metric("ResultsUploaded", successful, "Count")
            
            if successful < len(results):
                failed = len(results) - successful
                self.metrics.put_metric("ResultsUploadFailed", failed, "Count")
        
        return results
    
    def send_processing_metrics(self, 
                               duration: float, 
                               files_processed: int,
                               speakers_found: int,
                               gpu_utilization: Optional[float] = None):
        """Send processing metrics to CloudWatch"""
        
        if not self.metrics:
            return
        
        dimensions = {
            'InstanceType': self.instance_info.get('instance_type', 'unknown'),
            'AvailabilityZone': self.instance_info.get('availability_zone', 'unknown')
        }
        
        # Processing metrics
        self.metrics.put_metric("ProcessingDuration", duration, "Seconds", dimensions)
        self.metrics.put_metric("FilesProcessed", files_processed, "Count", dimensions)
        self.metrics.put_metric("SpeakersFound", speakers_found, "Count", dimensions)
        
        if gpu_utilization is not None:
            self.metrics.put_metric("GPUUtilization", gpu_utilization, "Percent", dimensions)
        
        # Throughput
        if duration > 0:
            throughput = files_processed / duration
            self.metrics.put_metric("ProcessingThroughput", throughput, "Count/Second", dimensions)
    
    def cleanup(self):
        """Clean up AWS resources"""
        if self.s3_manager:
            self.s3_manager.cleanup()
        
        if self.metrics:
            self.metrics.cleanup()
        
        self.logger.info("aws_integration_cleanup_completed")


# Global AWS integration instance
_global_aws_integration: Optional[AWSIntegration] = None


def get_aws_integration(s3_bucket: str, region: str = "us-east-1") -> Optional[AWSIntegration]:
    """Get or create global AWS integration instance"""
    global _global_aws_integration
    
    if not AWS_AVAILABLE:
        return None
    
    if _global_aws_integration is None:
        try:
            _global_aws_integration = AWSIntegration(s3_bucket, region)
        except Exception as e:
            logger = get_logger()
            logger.warning("aws_integration_failed", error=str(e))
            return None
    
    return _global_aws_integration