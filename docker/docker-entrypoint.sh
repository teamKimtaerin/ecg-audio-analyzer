#!/bin/bash
# ECG Audio Analysis - Docker Container Entrypoint Script
# Handles initialization, health checks, and graceful shutdown

set -e

# Function to log with timestamp
log() {
    echo "[$(date -Iseconds)] ECG-ENTRYPOINT: $1"
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            log "GPU detected and accessible"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
            return 0
        else
            log "WARNING: nvidia-smi command failed"
            return 1
        fi
    else
        log "WARNING: nvidia-smi not found"
        return 1
    fi
}

# Function to wait for AWS credentials
wait_for_aws_credentials() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if aws sts get-caller-identity >/dev/null 2>&1; then
            log "AWS credentials verified successfully"
            return 0
        fi
        
        log "Waiting for AWS credentials... (attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log "WARNING: AWS credentials not available after $max_attempts attempts"
    return 1
}

# Function to download models if needed
download_models() {
    log "Checking model availability..."
    
    # Create models directory if it doesn't exist
    mkdir -p /app/models
    
    # Check if we have write access to models directory
    if [ ! -w /app/models ]; then
        log "WARNING: No write access to models directory"
        return 1
    fi
    
    # Pre-download models to avoid first-run delays
    python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from services.speaker_diarizer import SpeakerDiarizer
    from services.emotion_analyzer import EmotionAnalyzer
    
    print('Pre-loading models...')
    
    # Initialize services to trigger model downloads
    diarizer = SpeakerDiarizer()
    emotion_analyzer = EmotionAnalyzer()
    
    print('Models ready')
except Exception as e:
    print(f'Model pre-loading failed: {e}')
    # Don't fail startup for model loading issues
" || log "Model pre-loading encountered issues (will retry on first use)"
}

# Function to run health check
health_check() {
    log "Running initial health check..."
    
    if /usr/local/bin/healthcheck.sh; then
        log "Health check passed"
        return 0
    else
        log "Health check failed"
        return 1
    fi
}

# Function to handle shutdown gracefully
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Remove temporary files
    rm -f /tmp/.app_healthy
    rm -rf /tmp/ecg_*
    
    # Clean up temporary processing files
    find /app/temp -type f -name "*.wav" -o -name "*.tmp" | head -100 | xargs -r rm -f
    
    log "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Main execution
main() {
    log "Starting ECG Audio Analysis container..."
    
    # Print environment information
    log "Environment: ${ENVIRONMENT:-development}"
    log "AWS Region: ${AWS_DEFAULT_REGION:-us-east-1}"
    log "S3 Bucket: ${ECG_S3_BUCKET:-not-set}"
    log "Instance Type: ${INSTANCE_TYPE:-unknown}"
    log "Log Level: ${LOG_LEVEL:-INFO}"
    
    # Check GPU availability
    if ! check_gpu; then
        log "WARNING: GPU not available - performance will be degraded"
    fi
    
    # Wait for AWS credentials in production
    if [ "${ENVIRONMENT}" = "production" ]; then
        if ! wait_for_aws_credentials; then
            log "WARNING: Starting without AWS credentials"
        fi
    fi
    
    # Download/verify models
    if [ "${PRELOAD_MODELS:-true}" = "true" ]; then
        download_models &
        MODEL_PID=$!
    fi
    
    # Create health check marker
    touch /tmp/.app_healthy
    
    # Start the application based on command line arguments
    if [ "$1" = "python" ] || [ "$1" = "main.py" ]; then
        log "Starting main application: $*"
        exec "$@"
    elif [ "$1" = "health" ]; then
        health_check
    elif [ "$1" = "shell" ] || [ "$1" = "bash" ]; then
        log "Starting interactive shell"
        exec /bin/bash
    else
        log "Starting with default configuration: $*"
        exec "$@"
    fi
}

# Run main function
main "$@"