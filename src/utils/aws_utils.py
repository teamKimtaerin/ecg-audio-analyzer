"""
AWS Utilities - Simplified S3 and CloudWatch integration
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None

from .logger import get_logger


@dataclass
class S3Result:
    """Result of S3 operation"""
    success: bool
    key: str
    bucket: str
    operation_time: float
    file_size: int = 0
    error: Optional[str] = None
    local_path: Optional[Path] = None
    
    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class S3Client:
    """Simplified S3 operations"""
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 not installed")
        
        self.bucket = bucket
        self.logger = get_logger().bind_context(component="s3")
        
        # Simple client configuration
        config = Config(
            region_name=region,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=10
        )
        
        try:
            self.client = boto3.client('s3', config=config)
            self.client.head_bucket(Bucket=bucket)  # Verify access
            self.logger.info("s3_initialized", bucket=bucket)
        except (NoCredentialsError, ClientError) as e:
            self.logger.error("s3_init_failed", error=str(e))
            raise RuntimeError(f"S3 initialization failed: {e}")
        
        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=5)
    
    def upload(self, 
               local_path: Union[str, Path], 
               s3_key: str,
               metadata: Optional[Dict[str, str]] = None) -> S3Result:
        """Upload file to S3"""
        start = time.time()
        local_path = Path(local_path)
        
        if not local_path.exists():
            return S3Result(
                success=False, 
                key=s3_key, 
                bucket=self.bucket,
                operation_time=0, 
                error="File not found"
            )
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Auto-detect content type
            if local_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                extra_args['ContentType'] = 'audio/wav'
            elif local_path.suffix.lower() == '.json':
                extra_args['ContentType'] = 'application/json'
            
            self.client.upload_file(
                str(local_path), 
                self.bucket, 
                s3_key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            return S3Result(
                success=True,
                key=s3_key,
                bucket=self.bucket,
                operation_time=time.time() - start,
                file_size=local_path.stat().st_size
            )
            
        except Exception as e:
            self.logger.error("upload_failed", key=s3_key, error=str(e))
            return S3Result(
                success=False,
                key=s3_key,
                bucket=self.bucket,
                operation_time=time.time() - start,
                error=str(e)
            )
    
    def download(self, s3_key: str, local_path: Union[str, Path]) -> S3Result:
        """Download file from S3"""
        start = time.time()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client.download_file(self.bucket, s3_key, str(local_path))
            
            return S3Result(
                success=True,
                key=s3_key,
                bucket=self.bucket,
                operation_time=time.time() - start,
                file_size=local_path.stat().st_size,
                local_path=local_path
            )
            
        except Exception as e:
            self.logger.error("download_failed", key=s3_key, error=str(e))
            return S3Result(
                success=False,
                key=s3_key,
                bucket=self.bucket,
                operation_time=time.time() - start,
                error=str(e)
            )
    
    def batch_upload(self, files: List[tuple]) -> List[S3Result]:
        """Upload multiple files concurrently"""
        futures = [
            self._executor.submit(self.upload, local_path, s3_key)
            for local_path, s3_key in files
        ]
        
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                self.logger.error("batch_upload_error", error=str(e))
        
        return results
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List object keys with prefix"""
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    keys.extend([obj['Key'] for obj in page['Contents']])
            return keys
            
        except Exception as e:
            self.logger.error("list_failed", error=str(e))
            return []
    
    def delete(self, s3_key: str) -> bool:
        """Delete object"""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception as e:
            self.logger.error("delete_failed", key=s3_key, error=str(e))
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)


class MetricsClient:
    """Simplified CloudWatch metrics"""
    
    def __init__(self, namespace: str = "AudioAnalysis", region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 not installed")
        
        self.namespace = namespace
        self.logger = get_logger().bind_context(component="metrics")
        
        try:
            self.client = boto3.client('cloudwatch', region_name=region)
        except Exception as e:
            self.logger.error("metrics_init_failed", error=str(e))
            raise
        
        # Simple buffer for batching
        self._buffer = deque(maxlen=20)  # CloudWatch limit
        self._lock = threading.Lock()
        self._flush_timer = None
    
    def put(self, 
            name: str, 
            value: float, 
            unit: str = "Count",
            dimensions: Optional[Dict[str, str]] = None):
        """Add metric to buffer"""
        metric = {
            'MetricName': name,
            'Value': value,
            'Unit': unit,
            'Timestamp': time.time()
        }
        
        if dimensions:
            metric['Dimensions'] = [
                {'Name': k, 'Value': str(v)} for k, v in dimensions.items()
            ]
        
        with self._lock:
            self._buffer.append(metric)
            if len(self._buffer) >= 20:
                self._flush()
            else:
                self._schedule_flush()
    
    def _flush(self):
        """Send buffered metrics"""
        if not self._buffer:
            return
        
        try:
            metrics = list(self._buffer)
            self._buffer.clear()
            
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
        except Exception as e:
            self.logger.error("metrics_flush_failed", error=str(e))
    
    def _schedule_flush(self):
        """Schedule flush after delay"""
        if self._flush_timer:
            self._flush_timer.cancel()
        
        self._flush_timer = threading.Timer(60.0, self.flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def flush(self):
        """Manually flush metrics"""
        with self._lock:
            self._flush()
    
    def cleanup(self):
        """Cleanup resources"""
        if self._flush_timer:
            self._flush_timer.cancel()
        self.flush()


class SimpleAWSClient:
    """Unified simplified AWS client"""
    
    def __init__(self, bucket: str, region: str = "us-east-1", enable_metrics: bool = True):
        self.logger = get_logger().bind_context(component="aws")
        
        if not AWS_AVAILABLE:
            self.logger.warning("aws_unavailable")
            self.s3 = None
            self.metrics = None
            return
        
        try:
            self.s3 = S3Client(bucket, region)
            self.metrics = MetricsClient(region=region) if enable_metrics else None
            
            # Get instance metadata if on EC2
            self.instance_info = self._get_instance_metadata()
            
            self.logger.info("aws_client_ready", 
                           bucket=bucket,
                           metrics=enable_metrics,
                           instance_type=self.instance_info.get('instance_type'))
        except Exception as e:
            self.logger.error("aws_init_failed", error=str(e))
            raise
    
    def _get_instance_metadata(self) -> Dict[str, str]:
        """Try to get EC2 instance metadata"""
        try:
            import requests
            # Quick timeout to avoid hanging on non-EC2 environments
            resp = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-type',
                timeout=0.5
            )
            if resp.status_code == 200:
                return {'instance_type': resp.text, 'on_ec2': True}
        except:
            pass
        
        return {'instance_type': 'local', 'on_ec2': False}
    
    def upload_results(self, results_dir: Path, prefix: str = "results/") -> List[S3Result]:
        """Upload all JSON results from directory"""
        if not self.s3:
            return []
        
        files = list(results_dir.glob("*.json"))
        if not files:
            return []
        
        file_mappings = [(f, f"{prefix}{f.name}") for f in files]
        results = self.s3.batch_upload(file_mappings)
        
        # Send metrics
        if self.metrics:
            successful = sum(1 for r in results if r.success)
            self.metrics.put("ResultsUploaded", successful)
            
            if successful < len(results):
                self.metrics.put("ResultsUploadFailed", len(results) - successful)
        
        return results
    
    def track_processing(self, 
                        duration: float, 
                        files: int, 
                        speakers: int):
        """Track processing metrics"""
        if not self.metrics:
            return
        
        dims = {'InstanceType': self.instance_info.get('instance_type', 'unknown')}
        
        self.metrics.put("ProcessingDuration", duration, "Seconds", dims)
        self.metrics.put("FilesProcessed", files, "Count", dims)
        self.metrics.put("SpeakersFound", speakers, "Count", dims)
        
        if duration > 0:
            self.metrics.put("Throughput", files / duration, "Count/Second", dims)
    
    def cleanup(self):
        """Cleanup all resources"""
        if self.s3:
            self.s3.cleanup()
        if self.metrics:
            self.metrics.cleanup()


# Convenience functions
def get_aws_client(bucket: str, region: str = "us-east-1") -> Optional[SimpleAWSClient]:
    """Get AWS client if available"""
    if not AWS_AVAILABLE:
        return None
    
    try:
        return SimpleAWSClient(bucket, region)
    except Exception:
        return None