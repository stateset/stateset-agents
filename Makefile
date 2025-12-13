.PHONY: help install install-dev install-all test test-cov test-unit test-integration test-slow lint lint-fix format check-types check-types-script clean docs docs-build docs-clean docs-api docs-serve build test-package publish-test publish release release-patch release-minor release-major quick-publish benchmark dev-test ci security-scan docker-build docker-run docker-dev docker-test docker-build-all docker-up docker-down pre-commit-install pre-commit-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = \":.*?## \"}; {printf \"  %-20s %s\n\", $$1, $$2}'

# Installation
install: ## Install package with core dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

install-all: ## Install package with all optional dependencies
	pip install -e ".[dev,api,examples,training,trl]"

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

# Code quality
lint: ## Run linters
	ruff check .
	flake8 .

lint-fix: ## Auto-fix lint issues and format
	ruff check . --fix
	black .
	isort .

format: ## Format code with black and isort
	black .
	isort .

check-types: ## Run mypy type checking
	mypy .

check-types-script: ## Run custom type checking script
	python scripts/check_types.py

# Documentation
docs: ## Build documentation
	sphinx-build docs docs/_build/html

docs-build: ## Build documentation (alias)
	sphinx-build docs docs/_build/html

docs-clean: ## Clean documentation build artifacts
	rm -rf docs/_build

docs-api: ## Generate API documentation stubs
	sphinx-apidoc -f -o docs/api stateset_agents

docs-serve: ## Build and serve documentation locally
	sphinx-build docs docs/_build/html
	cd docs/_build/html && python -m http.server 8000

# Packaging
build: ## Build distribution packages
	python -m build --no-isolation || python setup.py sdist bdist_wheel

test-package: ## Install built wheel and smoke test
	pip install dist/*.whl && python -c "import stateset_agents; print(stateset_agents.__version__)"

publish-test: ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	twine upload dist/*

release: ## Create a release with custom version
	python scripts/publish.py --version $${VERSION}

release-patch: ## Create patch release
	python scripts/publish.py --version patch

release-minor: ## Create minor release
	python scripts/publish.py --version minor

release-major: ## Create major release
	python scripts/publish.py --version major

quick-publish: ## Run interactive publishing script
	./scripts/quick_publish.sh

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

# Docker helpers
docker-build: ## Build Docker image
	docker build -t stateset-agents .

docker-run: ## Run Docker container
	docker run -p 8000:8000 stateset-agents

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

# Development workflows
dev-test: ## Quick development checks (format, type, unit tests)
	$(MAKE) lint-fix
	$(MAKE) check-types
	$(MAKE) test-unit

ci: ## Simulate CI pipeline locally
	$(MAKE) dev-test
	$(MAKE) test-cov

security-scan: ## Run basic security scanning tools
	bandit -r stateset_agents grpo_agent_framework || true
	safety check || true
	semgrep --config=auto . || true

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Cleanup
clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .coverage htmlcov .pytest_cache .mypy_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[cod]" -delete
