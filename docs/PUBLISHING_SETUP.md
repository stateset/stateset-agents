# Publishing Setup Guide

This guide covers how to set up your environment for publishing StateSet Agents to PyPI, Docker Hub, and other distribution channels.

## �� PyPI Setup

### 1. Create PyPI Accounts

1. **TestPyPI** (for testing): https://test.pypi.org/
2. **PyPI** (production): https://pypi.org/

Create accounts on both platforms.

### 2. Generate API Tokens

#### TestPyPI Token
1. Go to https://test.pypi.org/manage/account/#api-tokens
2. Create a new API token with scope "Entire account"
3. Copy the token (starts with `pypi-`)

#### PyPI Token  
1. Go to https://pypi.org/manage/account/#api-tokens
2. Create a new API token with scope "Entire account"
3. Copy the token

### 3. Set Environment Variables

```bash
# For local development
export TEST_PYPI_API_TOKEN="your_test_pypi_token"
export PYPI_API_TOKEN="your_pypi_token"

# For GitHub Actions (add to repository secrets)
# TEST_PYPI_API_TOKEN
# PYPI_API_TOKEN
```

### 4. Configure twine

Create `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = YOUR_PYPI_TOKEN

[testpypi]
username = __token__
password = YOUR_TEST_PYPI_TOKEN
```

## 🐳 Docker Hub Setup

### 1. Create Docker Hub Account

1. Go to https://hub.docker.com/
2. Create account or login
3. Create repository: `stateset/agents`

### 2. Set Docker Credentials

```bash
# Login to Docker Hub
docker login

# For GitHub Actions (add to repository secrets)
# DOCKER_USERNAME
# DOCKER_PASSWORD
```

### 3. Configure Docker Buildx

```bash
# Enable buildx
docker buildx create --use
docker buildx inspect --bootstrap
```

## 📚 GitHub Setup

### 1. Repository Settings

1. Go to repository Settings → Pages
2. Set source to "GitHub Actions"
3. Set custom domain (optional): `docs.stateset.ai`

### 2. Repository Secrets

Add these secrets in Settings → Secrets and variables → Actions:

```bash
# PyPI
TEST_PYPI_API_TOKEN=your_test_token
PYPI_API_TOKEN=your_prod_token

# Docker
DOCKER_USERNAME=your_docker_username  
DOCKER_PASSWORD=your_docker_password

# GitHub
GITHUB_TOKEN=github_token (automatically available)
```

### 3. GitHub CLI Setup (Optional)

```bash
# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login
```

## 🏗️ Build Tools Setup

### 1. Install Build Dependencies

```bash
# Install build tools
pip install build twine

# Install GitHub CLI (optional)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### 2. Verify Setup

```bash
# Test PyPI connection
python -m twine upload --repository testpypi --verbose dist/* --dry-run

# Test Docker build
docker build -t test-build -f deployment/docker/Dockerfile .

# Test GitHub CLI
gh auth status
```

## 🚀 Publishing Workflow

### Development Workflow

1. **Local Development**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: add new feature"
   
   # Test locally
   make test-all
   make build
   make test-package
   ```

2. **Test Release**
   ```bash
   # Publish to TestPyPI
   make publish-test
   
   # Test installation
   pip install --index-url https://test.pypi.org/simple/ stateset-agents
   ```

3. **Production Release**
   ```bash
   # Tag version
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin v1.2.3
   
   # Create GitHub release (triggers CI/CD)
   gh release create v1.2.3 --generate-notes
   ```

### Automated Workflow

The CI/CD pipeline handles:

1. **Version bumping** - Automatic or manual version updates
2. **Package building** - Creates wheel and source distributions
3. **Publishing** - Uploads to PyPI/TestPyPI based on trigger
4. **Docker images** - Builds and pushes multi-platform images
5. **Documentation** - Deploys to GitHub Pages
6. **GitHub release** - Creates release with changelog

### Manual Publishing

For manual control:

```bash
# Full manual release
python scripts/publish.py --version 1.2.3

# Or step by step
make build
make test-package
make publish-test  # Test first
make publish       # Then production
make docker-release
```

## 📦 Package Metadata

### PyPI Configuration

Ensure `pyproject.toml` has correct metadata:

```toml
[project]
name = "stateset-agents"
version = "1.2.3"
description = "A comprehensive framework for training multi-turn AI agents"
readme = "README.md"
license = {text = "Business Source License 1.1"}
requires-python = ">=3.8"
authors = [
    {name = "StateSet Team", email = "team@stateset.ai"}
]
maintainers = [
    {name = "StateSet Team", email = "team@stateset.ai"}
]
keywords = ["ai", "agents", "reinforcement-learning", "machine-learning", "nlp"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    # Core dependencies
]
```

### Docker Labels

Add proper labels to Docker images:

```dockerfile
LABEL org.opencontainers.image.title="StateSet Agents"
LABEL org.opencontainers.image.description="A comprehensive framework for training multi-turn AI agents"
LABEL org.opencontainers.image.url="https://github.com/stateset/stateset-agents"
LABEL org.opencontainers.image.source="https://github.com/stateset/stateset-agents"
LABEL org.opencontainers.image.version="1.2.3"
LABEL org.opencontainers.image.created="2024-01-15T10:30:00Z"
LABEL org.opencontainers.image.licenses="Business Source License 1.1"
```

## 🔍 Testing Publishing

### Pre-Publishing Checks

```bash
# Run all tests
make test-all

# Check code quality
make lint

# Security scan
make security-scan

# Build documentation
make docs-build

# Test package build
make build
make test-package
```

### Post-Publishing Verification

```bash
# Test PyPI installation
pip install stateset-agents --index-url https://test.pypi.org/simple/

# Test production installation
pip install stateset-agents

# Test Docker image
docker run stateset/agents:latest --help

# Check documentation
curl -f https://docs.stateset.ai/
```

## 🆘 Troubleshooting

### Common Issues

#### PyPI Upload Errors
```bash
# Check token
echo $PYPI_API_TOKEN

# Test connection
python -m twine upload --repository pypi --verbose dist/* --dry-run

# Check package files
ls -la dist/
```

#### Docker Build Issues
```bash
# Check Docker setup
docker buildx ls

# Test basic build
docker build -t test -f deployment/docker/Dockerfile .

# Check disk space
df -h
```

#### GitHub Release Issues
```bash
# Check GitHub CLI auth
gh auth status

# Test release creation
gh release create v1.2.3-test --generate-notes --draft
```

### Rollback Procedures

#### PyPI Rollback
```bash
# Yank release (keeps it but marks as broken)
twine upload --skip-existing dist/* --yank

# Or delete (if very broken)
# Contact PyPI admins for deletion
```

#### Docker Rollback
```bash
# Remove broken tag
docker rmi stateset/agents:1.2.3

# Update latest to previous version
docker tag stateset/agents:1.2.2 stateset/agents:latest
docker push stateset/agents:latest
```

#### GitHub Rollback
```bash
# Delete release and tag
gh release delete v1.2.3
git tag -d v1.2.3
git push origin :refs/tags/v1.2.3
```

## 📞 Support

If you encounter issues with publishing:

1. Check this guide first
2. Review error messages carefully
3. Test with `--dry-run` flags first
4. Check GitHub Actions logs
5. Open an issue with full error details

## 🎯 Best Practices

- **Always test on TestPyPI first**
- **Use semantic versioning**
- **Keep changelog updated**
- **Tag releases properly**
- **Test installation in clean environment**
- **Document breaking changes**
- **Have rollback plan ready**

Remember: Publishing is permanent! Always test thoroughly before publishing to production PyPI.
