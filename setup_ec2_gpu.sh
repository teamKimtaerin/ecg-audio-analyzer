#!/bin/bash

# EC2 GPU Instance Setup Script for ECG Audio Analyzer
# Supports G4dn, P3, and other NVIDIA GPU instances
# Run with: bash setup_ec2_gpu.sh

set -e  # Exit on any error

echo "🚀 Starting EC2 GPU setup for ECG Audio Analyzer..."

# Check if running on EC2 instance
if ! curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
    echo "❌ This script should be run on an EC2 instance"
    exit 1
fi

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
echo "📋 Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

# Check if GPU instance type
if [[ ! $INSTANCE_TYPE =~ ^(p[2-5]|g[3-5]|gr6) ]]; then
    echo "⚠️  Warning: $INSTANCE_TYPE may not have GPU support"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Check and install NVIDIA drivers
echo "🔧 Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "📦 Installing NVIDIA drivers..."

    # Detect OS
    if [[ -f /etc/ubuntu-release ]]; then
        # Ubuntu
        sudo apt-get update
        sudo apt-get install -y ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall
    elif [[ -f /etc/amazon-linux-release ]]; then
        # Amazon Linux 2
        sudo yum update -y
        sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)

        # Install NVIDIA driver for Amazon Linux 2
        aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
        chmod +x NVIDIA-Linux-x86_64*.run
        sudo ./NVIDIA-Linux-x86_64*.run --silent
    else
        echo "❌ Unsupported OS. Please install NVIDIA drivers manually."
        exit 1
    fi

    echo "🔄 Rebooting required after driver installation..."
    echo "Please reboot and run this script again: sudo reboot"
    exit 0
else
    echo "✅ NVIDIA drivers already installed"
    nvidia-smi
fi

# 2. Install Docker if not present
echo "🐳 Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# 3. Install NVIDIA Container Toolkit
echo "🔧 Installing NVIDIA Container Toolkit..."

# Detect OS for package installation
if [[ -f /etc/ubuntu-release ]] || [[ -f /etc/debian_version ]]; then
    # Ubuntu/Debian
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-docker2

elif [[ -f /etc/amazon-linux-release ]] || [[ -f /etc/redhat-release ]]; then
    # Amazon Linux 2 / RHEL/CentOS
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

    sudo yum install -y nvidia-docker2
else
    echo "❌ Unsupported OS for automatic nvidia-docker2 installation"
    exit 1
fi

# 4. Configure Docker for GPU
echo "⚙️ Configuring Docker for GPU..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# 5. Handle G4dn specific configuration
if [[ $INSTANCE_TYPE =~ ^g4dn ]]; then
    echo "🔧 Applying G4dn specific configuration..."
    echo "options nvidia NVreg_EnableGpuFirmware=0" | sudo tee -a /etc/modprobe.d/nvidia.conf
    echo "⚠️  Reboot required for G4dn GPU firmware setting"
fi

# 6. Restart Docker
echo "🔄 Restarting Docker..."
sudo systemctl restart docker

# 7. Test GPU Docker integration
echo "🧪 Testing GPU Docker integration..."
if sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi; then
    echo "✅ GPU Docker integration successful!"
else
    echo "❌ GPU Docker integration failed"
    exit 1
fi

# 8. Set up environment for ECG Audio Analyzer
echo "📝 Setting up ECG Audio Analyzer environment..."

# Create docker run script with GPU support
cat > ~/run_ecg_analyzer_gpu.sh << 'EOF'
#!/bin/bash
# ECG Audio Analyzer GPU Docker Run Script

# Check required environment variables
if [[ -z "$HF_TOKEN" ]]; then
    echo "❌ HF_TOKEN environment variable is required"
    exit 1
fi

if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "❌ AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) are required"
    exit 1
fi

echo "🚀 Starting ECG Audio Analyzer with GPU support..."

docker run --rm --gpus all \
    -p 8080:8080 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e S3_BUCKET_NAME="${S3_BUCKET_NAME:-ecg-audio-analyzer-production-audio-084828586938}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
    ecg-analyzer:v47
EOF

chmod +x ~/run_ecg_analyzer_gpu.sh

echo "✅ EC2 GPU setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. If this is a G4dn instance, reboot with: sudo reboot"
echo "2. Set environment variables:"
echo "   export HF_TOKEN=your_huggingface_token"
echo "   export AWS_ACCESS_KEY_ID=your_aws_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_aws_secret"
echo "3. Run ECG Audio Analyzer: ~/run_ecg_analyzer_gpu.sh"
echo ""
echo "🔍 Verify GPU setup anytime with: nvidia-smi"
echo "🧪 Test Docker GPU: sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"