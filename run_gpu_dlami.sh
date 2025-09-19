#!/bin/bash

# ECG Audio Analyzer GPU Runner for AWS Deep Learning AMI
# Optimized for "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" instances
# Instance Type: G4dn.2xlarge (or other GPU instances)

set -e

echo "🚀 ECG Audio Analyzer - GPU Mode (Deep Learning AMI)"
echo "📍 Instance: G4dn.2xlarge with Deep Learning AMI"
echo "🎯 Target: GPU acceleration with --gpus all flag"
echo ""

# Check if running on EC2 instance
if ! curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
    echo "⚠️  Warning: Not running on EC2 instance"
else
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    echo "📋 Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
fi

# Check required environment variables
echo "🔐 Checking environment variables..."
missing_vars=()

if [[ -z "$HF_TOKEN" ]]; then
    missing_vars+=("HF_TOKEN")
fi

if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    missing_vars+=("AWS_ACCESS_KEY_ID")
fi

if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    missing_vars+=("AWS_SECRET_ACCESS_KEY")
fi

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "❌ Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "💡 Set them with:"
    echo "   export HF_TOKEN=your_huggingface_token"
    echo "   export AWS_ACCESS_KEY_ID=your_aws_key"
    echo "   export AWS_SECRET_ACCESS_KEY=your_aws_secret"
    exit 1
fi

echo "✅ All required environment variables are set"

# Quick GPU check (optional, non-blocking)
echo ""
echo "🧪 Quick GPU check..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi >/dev/null 2>&1; then
        echo "✅ NVIDIA drivers working"
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo "🎮 GPU: $GPU_NAME (Count: $GPU_COUNT)"
    else
        echo "⚠️  NVIDIA drivers installed but not responding"
        echo "   Continuing anyway (Docker might still work)"
    fi
else
    echo "⚠️  nvidia-smi not found"
    echo "   Continuing anyway (Deep Learning AMI should have drivers)"
fi

# Test Docker GPU support
echo ""
echo "🐳 Testing Docker GPU support..."
if sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ Docker GPU support confirmed"
else
    echo "❌ Docker GPU support test failed"
    echo "   This may indicate missing NVIDIA Container Toolkit"
    echo "   But attempting to run ECG analyzer anyway..."
fi

# Run ECG Audio Analyzer with GPU support
echo ""
echo "🚀 Starting ECG Audio Analyzer with GPU support..."
echo "📡 Server will be available at: http://localhost:8080"
echo "🔍 Health check: http://localhost:8080/health"
echo ""
echo "🎯 GPU Mode: ENABLED (--gpus all)"
echo "⚡ Expected performance boost: 5-10x faster than CPU mode"
echo ""

# Set default values
S3_BUCKET_NAME="${S3_BUCKET_NAME:-ecg-audio-analyzer-production-audio-084828586938}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "🔧 Configuration:"
echo "   S3 Bucket: $S3_BUCKET_NAME"
echo "   AWS Region: $AWS_DEFAULT_REGION"
echo "   HF Token: ${HF_TOKEN:0:10}***"
echo ""

# Run the container with GPU support
exec docker run --rm --gpus all \
    -p 8080:8080 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e S3_BUCKET_NAME="${S3_BUCKET_NAME}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    ecg-analyzer:v47