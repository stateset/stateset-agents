# Release Management Guide

This guide covers the complete process for releasing new versions of StateSet Agents.

## 📋 Release Process Overview

1. **Prepare Release** - Update version, changelog, and documentation
2. **Test Release** - Build, test, and validate package
3. **Publish Release** - Deploy to PyPI and create GitHub release
4. **Deploy Assets** - Publish Docker images and documentation
5. **Announce Release** - Update community and stakeholders

## 🎯 Version Numbering

StateSet Agents follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
└──┬──┴──┬──┴──┬──┴─ Patch version (bug fixes)
   └──┬──┴──┴───── Minor version (new features, backward compatible)
      └─────────── Major version (breaking changes)
```

### Examples
- `1.0.0` - Initial stable release
- `1.0.1` - Bug fix release
- `1.1.0` - New feature release
- `2.0.0` - Breaking change release
- `1.0.0-rc.1` - Release candidate
- `1.0.0-alpha.1` - Alpha release

## 🚀 Automated Release Process

### Quick Release Commands

```bash
# Patch release (1.0.0 -> 1.0.1)
make release-patch

# Minor release (1.0.0 -> 1.1.0)  
make release-minor

# Major release (1.0.0 -> 2.0.0)
make release-major

# Release candidate
make release-rc

# Custom version release
make release VERSION=1.2.3
```

### Manual Release Process

#### 1. Prepare Release Branch
```bash
# Create release branch
git checkout -b release/v1.2.3

# Update version numbers
python scripts/publish.py --version 1.2.3

# Update changelog
vim CHANGELOG.md
```

#### 2. Test Release
```bash
# Run full test suite
make test-all

# Build and test package
make build
make test-package

# Test with TestPyPI
make publish-test
```

#### 3. Create GitHub Release
```bash
# Tag and push
git add .
git commit -m "Release v1.2.3"
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3

# Create GitHub release
gh release create v1.2.3 \
  --title "StateSet Agents v1.2.3" \
  --notes-file RELEASE_NOTES.md \
  --latest
```

#### 4. Publish to PyPI
```bash
# Publish to production PyPI
make publish
```

## 📝 Release Checklist

### Pre-Release
- [ ] All tests pass (`make test-all`)
- [ ] Code quality checks pass (`make lint`)
- [ ] Security scan clean (`make security-scan`)
- [ ] Documentation updated (`make docs-build`)
- [ ] Changelog updated (`CHANGELOG.md`)
- [ ] Version numbers updated in all files
- [ ] Breaking changes documented
- [ ] Migration guide created (if needed)

### Release
- [ ] Package builds successfully (`make build`)
- [ ] Package installs correctly (`make test-package`)
- [ ] TestPyPI publication successful (`make publish-test`)
- [ ] PyPI publication successful (`make publish`)
- [ ] Docker images published
- [ ] GitHub release created
- [ ] Documentation deployed

### Post-Release
- [ ] Release announced on Discord/Slack
- [ ] Release blog post published
- [ ] Social media announcement
- [ ] Community forums updated
- [ ] Support channels notified

## 🔧 Release Commands

### Automated Commands
```bash
# Full release process
make release VERSION=1.2.3

# Patch release
make release-patch

# Test release only
make release-test

# Docker release
make docker-release
```

### Manual Commands
```bash
# Build package
python -m build

# Test package
pip install dist/*.whl
python -c "import stateset_agents; print(stateset_agents.__version__)"

# Publish to TestPyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*

# Create GitHub release
gh release create v1.2.3 --generate-notes
```

## 📦 Package Configuration

### PyPI Metadata
```toml
# pyproject.toml
[project]
name = "stateset-agents"
version = "1.2.3"
description = "A comprehensive framework for training multi-turn AI agents"
authors = [{name = "StateSet Team", email = "team@stateset.ai"}]
license = {text = "Business Source License 1.1"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: Other/Proprietary License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

### Docker Images
```bash
# Build and tag
docker build -t stateset/agents:1.2.3 -f deployment/docker/Dockerfile .
docker tag stateset/agents:1.2.3 stateset/agents:latest

# Push to registry
docker push stateset/agents:1.2.3
docker push stateset/agents:latest
```

## 🔒 Security Releases

For security-related releases:

1. **Coordinate privately** - Don't announce until fix is ready
2. **Use security advisory** - GitHub Security Advisories
3. **Patch silently** - Deploy fix before announcement
4. **Notify users** - Clear upgrade instructions
5. **Post-mortem** - Document incident and response

```bash
# Create security advisory
gh security-advisory create \
  --summary "Security vulnerability in agent authentication" \
  --description "Details of the vulnerability..." \
  --severity high \
  --cwe CWE-287
```

## 🌍 Distribution Channels

### PyPI
- **Primary distribution** - `pip install stateset-agents`
- **Test releases** - TestPyPI for validation
- **Source distribution** - Both wheel and source packages

### Docker Hub
- **Official images** - `stateset/agents`
- **GPU support** - `stateset/agents:gpu`
- **Version tags** - `stateset/agents:v1.2.3`

### GitHub
- **Source code** - GitHub repository
- **Releases** - Tagged releases with assets
- **Documentation** - GitHub Pages deployment

### Conda (Future)
- **Conda package** - `conda install stateset-agents`
- **Multi-platform** - Linux, macOS, Windows support

## 📊 Release Metrics

Track these metrics for each release:

- **Download count** - PyPI downloads
- **Docker pulls** - Docker Hub pulls
- **GitHub clones** - Repository clones
- **Issue resolution** - Bug fix rate
- **Community growth** - New contributors/stars

## 🆘 Rollback Procedures

If a release needs to be rolled back:

1. **Stop deployment** - Halt CI/CD pipelines
2. **Revert changes** - Git revert if needed
3. **Yank release** - `twine upload --skip-existing dist/*` with `--yank`
4. **Communicate** - Inform users of rollback
5. **Fix issues** - Address root cause
6. **Re-release** - Deploy fixed version

```bash
# Yank a release from PyPI
twine upload --skip-existing --yank dist/*
```

## 📞 Communication Templates

### Release Announcement
```markdown
# 🚀 StateSet Agents v1.2.3 Released!

We're excited to announce the release of StateSet Agents v1.2.3!

## ✨ What's New

- **New Feature**: Description of major features
- **Improvements**: Performance and usability enhancements
- **Bug Fixes**: Issues resolved

## 📦 Installation

```bash
pip install stateset-agents==1.2.3
```

## 📚 Documentation

Full documentation: https://docs.stateset.ai

## 🙏 Feedback

We'd love to hear your thoughts! Join our Discord or open a GitHub issue.
```

### Security Advisory
```markdown
# 🔒 Security Update: StateSet Agents v1.2.3

## Summary
A security vulnerability has been discovered and fixed in StateSet Agents.

## Impact
- **Severity**: High
- **Affected Versions**: < 1.2.3
- **CVSS Score**: 8.5

## Resolution
Upgrade immediately to v1.2.3:

```bash
pip install --upgrade stateset-agents
```

## Details
[Technical details of the vulnerability and fix]

## Timeline
- **Discovered**: [Date]
- **Fixed**: [Date]  
- **Released**: [Date]

## Contact
security@stateset.ai
```

## 🎯 Best Practices

### Release Frequency
- **Major releases**: 3-6 months
- **Minor releases**: 2-4 weeks
- **Patch releases**: As needed for critical fixes
- **Pre-releases**: Weekly for active development

### Quality Gates
- **Code coverage**: > 80%
- **Security scan**: Clean results
- **Performance**: No regressions
- **Compatibility**: Test across Python versions

### Communication
- **Advance notice**: 1-2 weeks for major releases
- **Clear changelog**: User-friendly change descriptions
- **Migration guide**: For breaking changes
- **Support channels**: Multiple ways to get help

### Automation
- **CI/CD**: Automated testing and publishing
- **Release notes**: Auto-generated from PRs
- **Docker builds**: Automated image creation
- **Documentation**: Auto-deployed on release
