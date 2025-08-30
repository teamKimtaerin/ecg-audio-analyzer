#!/bin/bash
# ECG Audio Analyzer - Package Build Script
# Build and prepare package for PyPI distribution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Main build function
build_package() {
    print_info "Starting ECG Audio Analyzer package build..."
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found. Run this script from the project root."
        exit 1
    fi
    
    # Clean previous builds
    print_info "Cleaning previous builds..."
    rm -rf dist/ build/ *.egg-info/
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    
    # Install build dependencies
    print_info "Installing build dependencies..."
    pip install --upgrade build twine wheel setuptools
    
    # Run tests
    print_info "Running package API tests..."
    if ! python test_package_api.py; then
        print_error "Package tests failed! Fix issues before building."
        exit 1
    fi
    
    # Build package
    print_info "Building source and wheel distributions..."
    python -m build
    
    # Check distributions
    print_info "Checking built distributions..."
    python -m twine check dist/*
    
    # Display build results
    print_success "Package build completed successfully!"
    print_info "Built distributions:"
    ls -la dist/
    
    print_info "Next steps:"
    echo "  1. Test install: pip install dist/*.whl"
    echo "  2. Upload to TestPyPI: twine upload --repository testpypi dist/*"
    echo "  3. Upload to PyPI: twine upload dist/*"
}

# Test installation function
test_installation() {
    print_info "Testing package installation..."
    
    # Create temporary virtual environment
    TEMP_ENV=$(mktemp -d)
    python -m venv "$TEMP_ENV"
    source "$TEMP_ENV/bin/activate"
    
    # Install the built package
    pip install dist/*.whl
    
    # Test import
    python -c "
import ecg_audio_analyzer
print(f'Successfully imported ECG Audio Analyzer v{ecg_audio_analyzer.get_version()}')

# Test basic functionality
from ecg_audio_analyzer import AnalysisConfig, get_system_info
config = AnalysisConfig()
info = get_system_info()
print(f'System test passed: GPU available = {info[\"gpu_available\"]}')
"
    
    # Test CLI
    ecg-analyze version
    ecg-analyze test
    
    # Cleanup
    deactivate
    rm -rf "$TEMP_ENV"
    
    print_success "Installation test passed!"
}

# Upload functions
upload_to_testpypi() {
    print_info "Uploading to TestPyPI..."
    
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        print_error "No distributions found. Run build first."
        exit 1
    fi
    
    python -m twine upload --repository testpypi dist/*
    print_success "Uploaded to TestPyPI!"
    print_info "Test install with: pip install --index-url https://test.pypi.org/simple/ ecg-audio-analyzer"
}

upload_to_pypi() {
    print_info "Uploading to PyPI..."
    
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        print_error "No distributions found. Run build first."
        exit 1
    fi
    
    echo -n "Are you sure you want to upload to PyPI? This cannot be undone. (y/N): "
    read -r confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_info "Upload cancelled."
        exit 0
    fi
    
    python -m twine upload dist/*
    print_success "Uploaded to PyPI!"
    print_info "Install with: pip install ecg-audio-analyzer"
}

# Command line argument parsing
case "${1:-build}" in
    build)
        build_package
        ;;
    test)
        test_installation
        ;;
    upload-test)
        upload_to_testpypi
        ;;
    upload)
        upload_to_pypi
        ;;
    all)
        build_package
        test_installation
        print_info "Ready for upload! Use 'upload-test' or 'upload' commands."
        ;;
    clean)
        print_info "Cleaning build artifacts..."
        rm -rf dist/ build/ *.egg-info/
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -type f -delete 2>/dev/null || true
        print_success "Cleaned build artifacts."
        ;;
    help)
        echo "ECG Audio Analyzer - Package Build Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  build       Build source and wheel distributions (default)"
        echo "  test        Test installation of built package"
        echo "  upload-test Upload to TestPyPI"
        echo "  upload      Upload to PyPI"
        echo "  all         Build and test (but don't upload)"
        echo "  clean       Clean build artifacts"
        echo "  help        Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac