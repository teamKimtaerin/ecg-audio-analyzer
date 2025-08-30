# ECG Audio Analysis - Docker GPU Deployment

This directory contains Docker configuration files for deploying the ECG Audio Analysis system on AWS GPU instances with NVIDIA CUDA support.

## Files Overview

- **Dockerfile.gpu**: Multi-stage GPU-optimized Docker container
- **docker-compose.aws.yml**: Docker Compose configuration for AWS deployment
- **docker-entrypoint.sh**: Container initialization and health check script
- **healthcheck.sh**: Application health check script
- **.env.template**: Environment variable template

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp docker/.env.template docker/.env

# Edit with your AWS credentials and settings
nano docker/.env
```

### 2. Build Container

```bash
# Build GPU-optimized container
docker build -f docker/Dockerfile.gpu -t ecg-audio-analyzer:gpu .

# Or use Docker Compose
docker-compose -f docker-compose.aws.yml build
```

### 3. Deploy on AWS GPU Instance

```bash
# Start the service
docker-compose -f docker-compose.aws.yml up -d

# Check logs
docker-compose -f docker-compose.aws.yml logs -f

# Check health
docker-compose -f docker-compose.aws.yml exec ecg-audio-analyzer /usr/local/bin/healthcheck.sh
```

## AWS Integration

The container is designed for seamless integration with AWS services:

- **S3**: Automatic file upload/download with transfer acceleration
- **CloudWatch**: Metrics and logging integration
- **ECR**: Container registry for deployment
- **IAM**: Role-based access control

## GPU Support

The container includes full NVIDIA GPU support:

- **CUDA 11.8**: Latest stable CUDA runtime
- **cuDNN**: Deep learning GPU acceleration
- **PyTorch GPU**: GPU-accelerated PyTorch with CUDA support
- **Memory Management**: Automatic GPU memory optimization

## Environment Variables

### Required
- `AWS_DEFAULT_REGION`: AWS region
- `ECG_S3_BUCKET`: S3 bucket for file storage
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

### Optional
- `INSTANCE_ID`: EC2 instance identifier
- `INSTANCE_TYPE`: EC2 instance type
- `LOG_LEVEL`: Application log level (INFO, DEBUG, etc.)
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## Performance Tuning

The container includes optimizations for AWS GPU instances:

- **Multi-threading**: Optimized thread counts for different instance types
- **Memory Management**: GPU and CPU memory optimization
- **I/O Performance**: Optimized file handling and network transfers
- **Model Caching**: Pre-loaded models to reduce startup time

## Health Checks

The container includes comprehensive health checks:

- **Application Health**: Main process monitoring
- **GPU Health**: NVIDIA GPU accessibility
- **Resource Health**: Memory and disk usage monitoring
- **AWS Health**: S3 and CloudWatch connectivity

## Monitoring

Built-in monitoring includes:

- **CloudWatch Metrics**: Custom application metrics
- **CloudWatch Logs**: Structured application logs
- **Prometheus**: Optional metrics export
- **Health Endpoints**: HTTP health check endpoints

## Security

Security features include:

- **Non-root User**: Application runs as unprivileged user
- **Resource Limits**: CPU and memory constraints
- **Network Security**: Minimal port exposure
- **Credential Management**: Secure AWS credential handling

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **AWS Credentials**
   ```bash
   # Test AWS connectivity
   aws sts get-caller-identity
   
   # Check IAM permissions
   aws s3 ls s3://your-bucket-name
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Check logs for bottlenecks
   docker logs ecg-audio-analyzer
   ```

### Debug Mode

```bash
# Start container in debug mode
docker-compose -f docker-compose.aws.yml exec ecg-audio-analyzer bash

# Run health check manually
/usr/local/bin/healthcheck.sh

# Test application manually
python main.py --help
```

## Production Deployment

For production deployment:

1. Use the AWS deployment script: `deployment/aws-deploy.sh`
2. Configure Auto Scaling for high availability
3. Set up CloudWatch alarms for monitoring
4. Use ECR for container registry
5. Implement blue-green deployment strategy

## Resource Requirements

### Minimum (Development)
- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Storage**: 100GB SSD

### Recommended (Production)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA V100/A100 (32GB+ VRAM)
- **Storage**: 500GB+ NVMe SSD

### AWS Instance Types
- **Development**: g4dn.xlarge
- **Production**: g4dn.2xlarge or p3.2xlarge
- **High Performance**: p3.8xlarge or p4d.24xlarge