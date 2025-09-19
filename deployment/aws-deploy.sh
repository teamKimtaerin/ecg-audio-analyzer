#!/bin/bash
# ECG Audio Analysis - AWS Deployment Script
# Deploy complete infrastructure and application to AWS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROJECT_NAME="ecg-audio-analyzer"
ENVIRONMENT="production"
INSTANCE_TYPE="g4dn.2xlarge"
MIN_INSTANCES=1
MAX_INSTANCES=5
VOLUME_SIZE=500
REGION="us-east-1"

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --project-name NAME       Project name (default: ecg-audio-analyzer)"
    echo "  -e, --environment ENV         Environment name (default: production)"
    echo "  -i, --instance-type TYPE      EC2 instance type (default: g4dn.2xlarge)"
    echo "  -r, --region REGION           AWS region (default: us-east-1)"
    echo "  -k, --key-pair KEY            EC2 Key Pair name (required)"
    echo "  --min-instances N             Minimum instances (default: 1)"
    echo "  --max-instances N             Maximum instances (default: 5)"
    echo "  --volume-size SIZE            EBS volume size in GB (default: 500)"
    echo "  --dry-run                     Show what would be deployed without deploying"
    echo "  --skip-build                  Skip Docker image build"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -k my-key-pair"
    echo "  $0 -k my-key-pair -e staging -i g4dn.xlarge"
    echo "  $0 -k my-key-pair --dry-run"
}

check_requirements() {
    print_info "Checking requirements..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is required but not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check Docker (if not skipping build)
    if [[ "$SKIP_BUILD" != "true" ]] && ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check jq for JSON parsing
    if ! command -v jq &> /dev/null; then
        print_warning "jq is recommended for better output formatting"
    fi
    
    print_success "Requirements check passed"
}

validate_parameters() {
    print_info "Validating parameters..."
    
    if [[ -z "$KEY_PAIR" ]]; then
        print_error "EC2 Key Pair name is required (-k or --key-pair)"
        exit 1
    fi
    
    # Validate instance types
    valid_instances="g4dn.xlarge g4dn.2xlarge g4dn.4xlarge p3.2xlarge p3.8xlarge"
    if [[ ! " $valid_instances " =~ " $INSTANCE_TYPE " ]]; then
        print_error "Invalid instance type: $INSTANCE_TYPE"
        print_error "Valid types: $valid_instances"
        exit 1
    fi
    
    # Check if key pair exists
    if ! aws ec2 describe-key-pairs --key-names "$KEY_PAIR" --region "$REGION" &> /dev/null; then
        print_error "Key pair '$KEY_PAIR' not found in region '$REGION'"
        exit 1
    fi
    
    print_success "Parameters validation passed"
}

build_docker_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        print_info "Skipping Docker image build"
        return
    fi
    
    print_info "Building Docker image..."
    
    local image_tag="$PROJECT_NAME-$ENVIRONMENT:latest"
    
    # Build image
    docker build -f docker/Dockerfile.gpu -t "$image_tag" .
    
    # Tag for ECR (optional)
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_uri="$account_id.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME"
    
    print_info "Tagging image for ECR: $ecr_uri:$ENVIRONMENT"
    docker tag "$image_tag" "$ecr_uri:$ENVIRONMENT"
    
    print_success "Docker image built successfully"
}

deploy_infrastructure() {
    print_info "Deploying AWS infrastructure..."
    
    local stack_name="$PROJECT_NAME-$ENVIRONMENT"
    local template_file="aws/cloudformation.yml"
    
    if [[ ! -f "$template_file" ]]; then
        print_error "CloudFormation template not found: $template_file"
        exit 1
    fi
    
    # Parameters for CloudFormation
    local parameters=(
        "ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME"
        "ParameterKey=EnvironmentName,ParameterValue=$ENVIRONMENT"
        "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE"
        "ParameterKey=KeyPairName,ParameterValue=$KEY_PAIR"
        "ParameterKey=MinInstances,ParameterValue=$MIN_INSTANCES"
        "ParameterKey=MaxInstances,ParameterValue=$MAX_INSTANCES"
        "ParameterKey=VolumeSize,ParameterValue=$VOLUME_SIZE"
    )
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name "$stack_name" --region "$REGION" &> /dev/null; then
        print_info "Updating existing stack: $stack_name"
        action="update-stack"
    else
        print_info "Creating new stack: $stack_name"
        action="create-stack"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "DRY RUN: Would $action with parameters:"
        printf '%s\n' "${parameters[@]}"
        return
    fi
    
    # Deploy/Update stack
    local change_set_name="changeset-$(date +%Y%m%d-%H%M%S)"
    
    aws cloudformation create-change-set \
        --stack-name "$stack_name" \
        --change-set-name "$change_set_name" \
        --template-body "file://$template_file" \
        --parameters "${parameters[@]}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$REGION"
    
    # Wait for change set creation
    print_info "Waiting for change set creation..."
    aws cloudformation wait change-set-create-complete \
        --stack-name "$stack_name" \
        --change-set-name "$change_set_name" \
        --region "$REGION"
    
    # Show changes
    print_info "Changes to be applied:"
    aws cloudformation describe-change-set \
        --stack-name "$stack_name" \
        --change-set-name "$change_set_name" \
        --region "$REGION" \
        --query 'Changes[*].[Action,ResourceChange.LogicalResourceId,ResourceChange.ResourceType]' \
        --output table
    
    # Confirm execution
    echo -n "Execute change set? (y/N): "
    read -r confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_info "Deployment cancelled"
        aws cloudformation delete-change-set \
            --stack-name "$stack_name" \
            --change-set-name "$change_set_name" \
            --region "$REGION"
        exit 0
    fi
    
    # Execute change set
    aws cloudformation execute-change-set \
        --stack-name "$stack_name" \
        --change-set-name "$change_set_name" \
        --region "$REGION"
    
    # Wait for completion
    print_info "Waiting for stack deployment to complete..."
    if [[ "$action" == "create-stack" ]]; then
        aws cloudformation wait stack-create-complete --stack-name "$stack_name" --region "$REGION"
    else
        aws cloudformation wait stack-update-complete --stack-name "$stack_name" --region "$REGION"
    fi
    
    print_success "Infrastructure deployment completed"
}

get_stack_outputs() {
    local stack_name="$PROJECT_NAME-$ENVIRONMENT"
    
    print_info "Retrieving stack outputs..."
    
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output table
    
    # Save outputs to file
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' > "deployment/stack-outputs.json"
    
    print_success "Stack outputs saved to deployment/stack-outputs.json"
}

create_ecr_repository() {
    print_info "Creating ECR repository..."
    
    local repo_name="$PROJECT_NAME"
    
    # Check if repository exists
    if aws ecr describe-repositories --repository-names "$repo_name" --region "$REGION" &> /dev/null; then
        print_info "ECR repository already exists: $repo_name"
    else
        aws ecr create-repository \
            --repository-name "$repo_name" \
            --region "$REGION" \
            --image-scanning-configuration scanOnPush=true
        
        print_success "ECR repository created: $repo_name"
    fi
    
    # Get login command
    print_info "Getting ECR login token..."
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$account_id.dkr.ecr.$REGION.amazonaws.com"
}

push_docker_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        print_info "Skipping Docker image push"
        return
    fi
    
    print_info "Pushing Docker image to ECR..."
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_uri="$account_id.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "DRY RUN: Would push image to $ecr_uri:$ENVIRONMENT"
        return
    fi
    
    docker push "$ecr_uri:$ENVIRONMENT"
    
    print_success "Docker image pushed successfully"
}

deploy_application() {
    print_info "Deploying application to instances..."

    # This would typically involve:
    # 1. SSH to instances
    # 2. Pull latest Docker image
    # 3. Update configuration
    # 4. Restart services

    local account_id=$(aws sts get-caller-identity --query Account --output text)

    # For now, we'll just create a deployment script
    cat << EOF > deployment/update-instances.sh
#!/bin/bash
# Update script for EC2 instances
# Run this script on each EC2 instance to update the application

set -e

echo "🚀 Updating ECG Audio Analyzer with GPU support..."

# Check if using Deep Learning AMI
if [[ -f /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl ]]; then
    echo "✅ Deep Learning AMI detected - GPU support ready"
    GPU_SUPPORT="--gpus all"
else
    echo "⚠️  Standard AMI detected - checking GPU setup..."
    # Test GPU support
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU support confirmed"
        GPU_SUPPORT="--gpus all"
    else
        echo "❌ GPU support not available - running in CPU mode"
        GPU_SUPPORT=""
    fi
fi

echo "Pulling latest Docker image..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $account_id.dkr.ecr.$REGION.amazonaws.com
docker pull $account_id.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME:$ENVIRONMENT

echo "Stopping current service..."
sudo systemctl stop ecg-audio-analyzer || true

# Update systemd service with GPU support
cat << SYSTEMD_EOF > /tmp/ecg-audio-analyzer.service
[Unit]
Description=ECG Audio Analyzer ML Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
ExecStartPre=/usr/bin/docker pull $account_id.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME:$ENVIRONMENT
ExecStart=/usr/bin/docker run --rm \$GPU_SUPPORT \\
    -p 8080:8080 \\
    -e HF_TOKEN=\${HF_TOKEN} \\
    -e S3_BUCKET_NAME=\${S3_BUCKET_NAME} \\
    -e AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID} \\
    -e AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY} \\
    -e AWS_DEFAULT_REGION=\${AWS_DEFAULT_REGION} \\
    --name ecg-audio-analyzer \\
    $account_id.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME:$ENVIRONMENT
ExecStop=/usr/bin/docker stop ecg-audio-analyzer
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

# Replace GPU_SUPPORT placeholder
sed -i "s/\\\$GPU_SUPPORT/\$GPU_SUPPORT/g" /tmp/ecg-audio-analyzer.service

sudo mv /tmp/ecg-audio-analyzer.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "Starting updated service with GPU support..."
sudo systemctl start ecg-audio-analyzer

echo "Verifying service status..."
sudo systemctl status ecg-audio-analyzer --no-pager

echo "Checking GPU status in container..."
sleep 10
docker logs \$(docker ps --filter "name=ecg-audio-analyzer" --format "{{.ID}}") | grep -E "(GPU:|CUDA|Tesla)" || echo "No GPU logs found yet"

echo "✅ Application updated successfully with GPU support"
echo "🔍 Check logs: docker logs \$(docker ps --filter \"name=ecg-audio-analyzer\" --format \"{{.ID}}\")"
echo "🌐 Service available at: http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
EOF

    chmod +x deployment/update-instances.sh

    print_success "Application deployment script created: deployment/update-instances.sh"
    print_info "SSH to your instances and run this script to update the application"
    print_info "The script will automatically detect Deep Learning AMI and enable GPU support"
}

cleanup() {
    print_info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-name)
                PROJECT_NAME="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -i|--instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -k|--key-pair)
                KEY_PAIR="$2"
                shift 2
                ;;
            --min-instances)
                MIN_INSTANCES="$2"
                shift 2
                ;;
            --max-instances)
                MAX_INSTANCES="$2"
                shift 2
                ;;
            --volume-size)
                VOLUME_SIZE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    print_info "Starting ECG Audio Analysis deployment"
    print_info "Project: $PROJECT_NAME"
    print_info "Environment: $ENVIRONMENT"
    print_info "Instance Type: $INSTANCE_TYPE"
    print_info "Region: $REGION"
    print_info "Key Pair: $KEY_PAIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Execute deployment steps
    check_requirements
    validate_parameters
    
    if [[ "$SKIP_BUILD" != "true" ]]; then
        create_ecr_repository
        build_docker_image
        push_docker_image
    fi
    
    deploy_infrastructure
    get_stack_outputs
    deploy_application
    
    print_success "Deployment completed successfully!"
    print_info "Check the AWS Console for resource details and monitoring"
    
    # Display useful information
    echo ""
    echo "=== Deployment Summary ==="
    echo "Stack Name: $PROJECT_NAME-$ENVIRONMENT"
    echo "Region: $REGION"
    echo "Instance Type: $INSTANCE_TYPE"
    echo "Min/Max Instances: $MIN_INSTANCES/$MAX_INSTANCES"
    echo ""
    echo "Next steps:"
    echo "1. Check CloudFormation console for stack status"
    echo "2. Monitor instance health in EC2 console"
    echo "3. Check CloudWatch logs for application logs"
    echo "4. Use deployment/update-instances.sh to update running instances"
    echo "=========================="
}

# Run main function
main "$@"