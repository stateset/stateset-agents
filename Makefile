.PHONY: help install install-dev install-all test test-cov test-unit test-integration lint lint-fix format check-types clean docs build publish dev-setup pre-commit-install pre-commit-run

# Default target
help: ## Show this help message
@echo "Available commands:"
@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Installation
install: ## Install package with basic dependencies
pip install -e .

install-dev: ## Install package with development dependencies
pip install -e ".[dev]"

install-all: ## Install package with all optional dependencies
pip install -e ".[dev,api,examples,trl]"

# Testing
test: ## Run all tests
pytest

test-cov: ## Run tests with coverage report
pytest --cov-report=html
@echo "Coverage report generated in htmlcov/index.html"

test-unit: ## Run only unit tests
pytest -m "unit"

test-integration: ## Run only integration tests
pytest -m "integration"

test-slow: ## Run slow tests
pytest -m "slow"

# Code Quality
lint: ## Run all linters
ruff check .
flake8 .

lint-fix: ## Fix linting issues automatically
ruff check . --fix
black .
isort .

format: ## Format code with black and isort
black .
isort .

check-types: ## Run mypy type checking
check-types-script: ## Run custom type checking script
	python scripts/check_types.py
mypy .

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
pre-commit install

pre-commit-run: ## Run pre-commit on all files
pre-commit run --all-files

# Development setup
dev-setup: install-all pre-commit-install ## Set up development environment

# Documentation
docs: ## Build documentation
sphinx-build docs docs/_build/html

docs-serve: ## Build and serve documentation
docs-build: ## Build documentation
	sphinx-build docs docs/_build/html
docs-clean: ## Clean documentation build artifacts
	rm -rf docs/_build
docs-api: ## Generate API documentation
build: ## Build distribution package
	python -m build
test-package: ## Test built package
	pip install dist/*.whl && python -c "import stateset_agents; print(stateset_agents.__version__)"
publish-test: ## Publish to TestPyPI
	python scripts/publish.py --test
publish: ## Publish to PyPI
	python scripts/publish.py --production
release: ## Create full release
	python scripts/publish.py --version $(VERSION)
release-patch: ## Create patch release
	python scripts/publish.py --version patch
release-minor: ## Create minor release
	python scripts/publish.py --version minor
release-major: ## Create major release
	python scripts/publish.py --version major
docker-build-release: ## Build Docker images for release
	docker build -t stateset/agents:$(VERSION) -f deployment/docker/Dockerfile .
docker-push-release: ## Push Docker images for release
	docker push stateset/agents:$(VERSION)
docker-release: ## Build and push Docker images
quick-publish: ## Interactive publishing script
	./scripts/quick_publish.sh
	make docker-build-release docker-push-release
	sphinx-apidoc -f -o docs/api stateset_agents
benchmark: ## Run performance benchmarks
	python scripts/benchmark.py
sphinx-build docs docs/_build/html && cd docs/_build/html && python -m http.server 8000

# Building and Publishing
build: ## Build distribution packages
python -m build

publish-test: ## Publish to TestPyPI
twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
twine upload dist/*

# Cleaning
clean: ## Clean build artifacts and cache files
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .coverage
rm -rf htmlcov/
rm -rf .pytest_cache/
rm -rf .mypy_cache/
rm -rf __pycache__/
rm -rf */__pycache__/
rm -rf */*/__pycache__/
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Docker
docker-build: ## Build Docker image
docker build -t stateset-agents .

docker-run: ## Run Docker container
docker-dev: ## Run development environment
	docker-compose -f deployment/docker/docker-compose.dev.yml up stateset-agents-dev
docker-test: ## Run tests in Docker
	docker-compose -f deployment/docker/docker-compose.dev.yml --profile test up stateset-agents-test
docker-build-all: ## Build all Docker images
	docker-compose -f deployment/docker/docker-compose.yml build
docker-up: ## Start all services
	docker-compose -f deployment/docker/docker-compose.yml up -d
docker-down: ## Stop all services
	docker-compose -f deployment/docker/docker-compose.yml down
docker run -p 8000:8000 stateset-agents

# Utilities
check-deps: ## Check for outdated dependencies
pip list --outdated

update-deps: ## Update dependencies
pip install --upgrade -e ".[dev,api,examples,trl]"

# Quick development cycle
dev-test: lint-fix check-types test-unit ## Run quick development checks

# CI simulation
ci: dev-test test-cov ## Simulate CI pipeline locally
security-scan: ## Run security scanning tools
	bandit -r stateset_agents grpo_agent_framework || true
	safety check || true
	semgrep --config=auto . || true
