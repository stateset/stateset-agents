#!/bin/bash

# Quick Publishing Script for StateSet Agents
# This script provides a simple interface for common publishing tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "pyproject.toml" ] || [ ! -f "stateset_agents/__init__.py" ]; then
        log_error "Not in the StateSet Agents project directory"
        log_error "Please run this script from the project root"
        exit 1
    fi
}

# Get current version
get_current_version() {
    python -c "import stateset_agents; print(stateset_agents.__version__)"
}

# Validate version format
validate_version() {
    local version=$1
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$ ]]; then
        log_error "Invalid version format: $version"
        log_error "Expected format: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-prerelease"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    command -v python >/dev/null 2>&1 || { log_error "Python is required but not installed"; exit 1; }
    command -v pip >/dev/null 2>&1 || { log_error "pip is required but not installed"; exit 1; }
    command -v git >/dev/null 2>&1 || { log_error "git is required but not installed"; exit 1; }
    
    # Check if build tools are installed
    python -c "import build" 2>/dev/null || { log_error "build package not installed. Run: pip install build"; exit 1; }
    python -c "import twine" 2>/dev/null || { log_error "twine package not installed. Run: pip install twine"; exit 1; }
    
    log_success "Prerequisites check passed"
}

# Build package
build_package() {
    log_info "Building package..."
    python -m build
    log_success "Package built successfully"
}

# Test package
test_package() {
    log_info "Testing package..."
    
    # Find the wheel file
    WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
    if [ -z "$WHEEL_FILE" ]; then
        log_error "No wheel file found in dist/"
        exit 1
    fi
    
    # Test installation
    pip install --force-reinstall --quiet "$WHEEL_FILE"
    
    # Test import
    python -c "import stateset_agents; print(f'✅ Package imported successfully: v{stateset_agents.__version__}')"
    
    log_success "Package test passed"
}

# Publish to TestPyPI
publish_test() {
    log_info "Publishing to TestPyPI..."
    
    if [ -z "$TEST_PYPI_API_TOKEN" ]; then
        log_warning "TEST_PYPI_API_TOKEN not set. Using interactive mode."
        log_warning "Set TEST_PYPI_API_TOKEN environment variable for automated publishing"
    fi
    
    python -m twine upload --repository testpypi dist/*
    log_success "Published to TestPyPI"
    
    log_info "Test installation:"
    log_info "pip install --index-url https://test.pypi.org/simple/ stateset-agents"
}

# Publish to PyPI
publish_production() {
    log_info "Publishing to PyPI..."
    
    if [ -z "$PYPI_API_TOKEN" ]; then
        log_warning "PYPI_API_TOKEN not set. Using interactive mode."
        log_warning "Set PYPI_API_TOKEN environment variable for automated publishing"
    fi
    
    # Confirmation
    echo
    log_warning "🚨 About to publish to PRODUCTION PyPI!"
    read -p "Are you sure? (yes/no): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Publishing cancelled"
        exit 0
    fi
    
    python -m twine upload dist/*
    log_success "Published to PyPI"
    
    log_info "Installation:"
    log_info "pip install stateset-agents"
}

# Create git tag
create_git_tag() {
    local version=$1
    local tag="v$version"
    
    log_info "Creating git tag: $tag"
    
    # Check if tag already exists
    if git tag -l | grep -q "^$tag$"; then
        log_warning "Tag $tag already exists"
        return
    fi
    
    # Create annotated tag
    git tag -a "$tag" -m "Release $tag"
    git push origin "$tag"
    
    log_success "Git tag created and pushed: $tag"
}

# Main menu
show_menu() {
    echo
    echo "🚀 StateSet Agents Publishing Script"
    echo "===================================="
    echo
    echo "Current version: $(get_current_version)"
    echo
    echo "Choose an option:"
    echo "1) Build package only"
    echo "2) Build and test package"
    echo "3) Publish to TestPyPI"
    echo "4) Publish to PyPI (Production)"
    echo "5) Full release (build + test + TestPyPI)"
    echo "6) Production release (build + test + PyPI + git tag)"
    echo "7) Quick test cycle (build + test + TestPyPI)"
    echo "8) Show current status"
    echo "9) Exit"
    echo
}

# Show status
show_status() {
    echo
    echo "📊 Current Status"
    echo "================="
    echo "Version: $(get_current_version)"
    echo "Git branch: $(git branch --show-current)"
    echo "Git status: $(git status --porcelain | wc -l) changes"
    echo "Dist files: $(ls dist/ 2>/dev/null | wc -l) files"
    echo
    
    # Check environment variables
    if [ -n "$TEST_PYPI_API_TOKEN" ]; then
        echo "✅ TestPyPI token configured"
    else
        echo "❌ TestPyPI token not configured"
    fi
    
    if [ -n "$PYPI_API_TOKEN" ]; then
        echo "✅ PyPI token configured"
    else
        echo "❌ PyPI token not configured"
    fi
    
    echo
}

# Main function
main() {
    check_directory
    check_prerequisites
    
    while true; do
        show_menu
        read -p "Enter choice (1-9): " choice
        
        case $choice in
            1)
                build_package
                ;;
            2)
                build_package
                test_package
                ;;
            3)
                build_package
                test_package
                publish_test
                ;;
            4)
                build_package
                test_package
                publish_production
                ;;
            5)
                build_package
                test_package
                publish_test
                ;;
            6)
                build_package
                test_package
                publish_production
                create_git_tag "$(get_current_version)"
                ;;
            7)
                build_package
                test_package
                publish_test
                ;;
            8)
                show_status
                ;;
            9)
                log_info "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid choice. Please enter 1-9."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
