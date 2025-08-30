#!/bin/bash
# Health check script for ECG Audio Analysis container

set -e

# Check if Python process is running
if ! pgrep -f "python.*main.py" > /dev/null; then
    echo "Main application process not running"
    exit 1
fi

# Check GPU availability (if expected)
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "GPU not accessible"
        exit 1
    fi
fi

# Check if application is responsive (simple file check)
if [ ! -f /tmp/.app_healthy ]; then
    # Try to create health marker
    python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from utils.logger import get_logger
    logger = get_logger('healthcheck')
    logger.info('Health check passed')
    with open('/tmp/.app_healthy', 'w') as f:
        f.write('healthy')
    print('Health check passed')
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "Container healthy"
exit 0