#!/bin/bash
# ECG Audio Analysis - EC2 Instance Setup Script
# Optimized for GPU instances with CUDA acceleration

set -e

# Logging setup
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "$(date): Starting ECG Audio Analysis instance setup"

# Update system packages
echo "$(date): Updating system packages"
apt-get update
apt-get upgrade -y

# Install essential packages
echo "$(date): Installing essential packages"
apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    awscli \
    htop \
    nvtop \
    docker.io \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libfftw3-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev

# Install NVIDIA Docker support
echo "$(date): Installing NVIDIA Docker support"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2

# Restart Docker with NVIDIA runtime
systemctl restart docker

# Verify NVIDIA Docker installation
echo "$(date): Verifying NVIDIA Docker installation"
if command -v nvidia-smi &> /dev/null && docker info | grep -q nvidia; then
    echo "NVIDIA Docker support confirmed"
    nvidia-smi
else
    echo "Warning: NVIDIA Docker support may not be fully configured"
fi

# Install Python dependencies
echo "$(date): Setting up Python environment"
pip3 install --upgrade pip setuptools wheel

# Install AWS CloudWatch agent
echo "$(date): Installing CloudWatch agent"
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
cat << 'EOF' > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
{
    "metrics": {
        "namespace": "ECG/AudioAnalysis",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": true
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time",
                    "read_bytes",
                    "write_bytes",
                    "reads",
                    "writes"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/ecg-audio-analyzer/*.log",
                        "log_group_name": "/aws/ec2/ecg-audio-analyzer",
                        "log_stream_name": "{instance_id}-application",
                        "timezone": "UTC"
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/ecg-audio-analyzer",
                        "log_stream_name": "{instance_id}-user-data",
                        "timezone": "UTC"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
systemctl enable amazon-cloudwatch-agent
systemctl start amazon-cloudwatch-agent

# Create application directory structure
echo "$(date): Setting up application directories"
mkdir -p /opt/ecg-audio-analyzer
mkdir -p /opt/ecg-audio-analyzer/models
mkdir -p /opt/ecg-audio-analyzer/logs
mkdir -p /opt/ecg-audio-analyzer/temp
mkdir -p /opt/ecg-audio-analyzer/results

# Set up log directory permissions
mkdir -p /var/log/ecg-audio-analyzer
chown -R ubuntu:ubuntu /var/log/ecg-audio-analyzer
chown -R ubuntu:ubuntu /opt/ecg-audio-analyzer

# Performance optimizations
echo "$(date): Applying performance optimizations"

# Network optimizations
cat << EOF >> /etc/sysctl.conf
# Network performance tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.netdev_max_backlog = 5000
net.core.netdev_budget = 600
net.ipv4.tcp_congestion_control = bbr

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system optimizations
fs.file-max = 2097152
EOF

sysctl -p

# GPU-specific optimizations
if command -v nvidia-smi &> /dev/null; then
    echo "$(date): Applying GPU optimizations"
    
    # Set GPU persistence mode
    nvidia-smi -pm 1
    
    # Set GPU performance mode (if supported)
    nvidia-smi -ac $(nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits | tail -1 | tr ',' ' ') || true
    
    # Create GPU monitoring script
    cat << 'EOF' > /opt/ecg-audio-analyzer/gpu_monitor.sh
#!/bin/bash
while true; do
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader,nounits >> /var/log/ecg-audio-analyzer/gpu_metrics.log
    sleep 60
done
EOF
    chmod +x /opt/ecg-audio-analyzer/gpu_monitor.sh
    
    # Create systemd service for GPU monitoring
    cat << EOF > /etc/systemd/system/gpu-monitor.service
[Unit]
Description=GPU Monitoring Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/opt/ecg-audio-analyzer/gpu_monitor.sh
User=ubuntu
Group=ubuntu
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl enable gpu-monitor.service
    systemctl start gpu-monitor.service
fi

# Download and setup application from S3
echo "$(date): Setting up ECG Audio Analysis application"
cd /opt/ecg-audio-analyzer

# Download application code from S3
echo "$(date): Downloading application code from S3..."
ECG_S3_BUCKET="ecg-audio-analyzer-production-audio-084828586938"
aws s3 cp s3://${ECG_S3_BUCKET}/app/ecg-audio-analyzer.tar.gz . --region us-east-1
tar -xzf ecg-audio-analyzer.tar.gz
rm ecg-audio-analyzer.tar.gz

# Set permissions
chown -R ubuntu:ubuntu /opt/ecg-audio-analyzer

# Install application dependencies directly (no venv for system-wide access)
pip3 install --upgrade pip
pip3 install -e .

# Create environment configuration
cat << EOF > /opt/ecg-audio-analyzer/.env
# ECG Audio Analysis Environment Configuration
AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
ECG_S3_BUCKET=${ECG_S3_BUCKET:-ecg-audio-analysis-bucket}
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/opt/ecg-audio-analyzer/src

# Performance settings
OMP_NUM_THREADS=$(nproc)
MKL_NUM_THREADS=$(nproc)
OPENBLAS_NUM_THREADS=$(nproc)

# GPU settings
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/ecg-audio-analyzer
EOF

# Create application startup script
cat << 'EOF' > /opt/ecg-audio-analyzer/start_service.sh
#!/bin/bash
set -e

# Load environment
source /opt/ecg-audio-analyzer/.env
source /opt/ecg-audio-analyzer/venv/bin/activate

cd /opt/ecg-audio-analyzer

# Start the application
python main.py --gpu --workers 4 --aws-instance $(curl -s http://169.254.169.254/latest/meta-data/instance-type) --cloudwatch
EOF

chmod +x /opt/ecg-audio-analyzer/start_service.sh

# Create systemd service for the application
cat << EOF > /etc/systemd/system/ecg-audio-analyzer.service
[Unit]
Description=ECG Audio Analysis Service
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/ecg-audio-analyzer
ExecStart=/opt/ecg-audio-analyzer/start_service.sh
Restart=always
RestartSec=30
Environment=HOME=/home/ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable service (but don't start yet - wait for application deployment)
systemctl enable ecg-audio-analyzer.service

# Install monitoring tools
echo "$(date): Installing monitoring tools"
pip3 install psutil nvidia-ml-py3

# Create health check script
cat << 'EOF' > /opt/ecg-audio-analyzer/health_check.sh
#!/bin/bash
# Simple health check for the ECG Audio Analysis service

# Check if Python process is running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "Application process is running"
    exit 0
else
    echo "Application process is not running"
    
    # Try to start the service
    systemctl start ecg-audio-analyzer.service
    sleep 10
    
    # Check again
    if pgrep -f "python.*main.py" > /dev/null; then
        echo "Application started successfully"
        exit 0
    else
        echo "Failed to start application"
        exit 1
    fi
fi
EOF

chmod +x /opt/ecg-audio-analyzer/health_check.sh

# Setup log rotation
cat << EOF > /etc/logrotate.d/ecg-audio-analyzer
/var/log/ecg-audio-analyzer/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
    notifempty
    su ubuntu ubuntu
}
EOF

# Create cleanup script for temporary files
cat << 'EOF' > /opt/ecg-audio-analyzer/cleanup_temp.sh
#!/bin/bash
# Clean up temporary files older than 1 hour
find /opt/ecg-audio-analyzer/temp -type f -mmin +60 -delete 2>/dev/null || true
find /tmp -name "ecg_*" -type f -mmin +60 -delete 2>/dev/null || true
EOF

chmod +x /opt/ecg-audio-analyzer/cleanup_temp.sh

# Add cleanup to cron
echo "0 * * * * /opt/ecg-audio-analyzer/cleanup_temp.sh" | crontab -u ubuntu -

# Create instance metadata
cat << EOF > /opt/ecg-audio-analyzer/instance_info.json
{
    "instance_id": "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)",
    "instance_type": "$(curl -s http://169.254.169.254/latest/meta-data/instance-type)",
    "availability_zone": "$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)",
    "region": "$(curl -s http://169.254.169.254/latest/meta-data/placement/region)",
    "setup_completed": "$(date -Iseconds)",
    "cuda_available": $(command -v nvidia-smi >/dev/null 2>&1 && echo "true" || echo "false")
}
EOF

# Download test files and run GPU performance test
echo "$(date): Downloading test files and running GPU performance test"
mkdir -p /opt/ecg-audio-analyzer/test-files
aws s3 cp s3://${ECG_S3_BUCKET}/test-files/friends.mp4 /opt/ecg-audio-analyzer/test-files/ --region us-east-1

# Run GPU performance test as ubuntu user
echo "$(date): Starting GPU performance test..."
su - ubuntu -c "cd /opt/ecg-audio-analyzer && PYTHONPATH=/opt/ecg-audio-analyzer python3 -m src.cli analyze test-files/friends.mp4 --gpu --workers 4 --verbose" > /var/log/ecg-audio-analyzer/gpu-performance-test.log 2>&1 &

# Wait a bit for test to start
sleep 10

# Final setup
echo "$(date): Finalizing setup"
chown -R ubuntu:ubuntu /opt/ecg-audio-analyzer

# Signal CloudFormation that setup is complete
if command -v cfn-signal >/dev/null 2>&1; then
    echo "$(date): Signaling CloudFormation completion"
    /opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource AutoScalingGroup --region ${AWS::Region} || true
fi

echo "$(date): ECG Audio Analysis instance setup completed successfully"
echo "$(date): GPU performance test is running in the background"

# Display system information
echo "=== System Information ==="
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Availability Zone: $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
echo "CUDA Available: $(command -v nvidia-smi >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "Docker Available: $(command -v docker >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "==========================="