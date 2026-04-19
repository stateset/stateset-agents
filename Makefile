.PHONY: help install install-dev install-all dev-setup test test-cov test-unit test-integration test-slow lint lint-fix format check-types check-types-script repo-hygiene clean docs docs-build docs-clean docs-api docs-serve build test-package publish-test publish release release-patch release-minor release-major require-release-branch quick-publish benchmark dev-test ci security-scan security-scan-strict publish-readiness docker-build docker-run docker-build-gateway docker-run-gateway docker-build-trainer docker-dev docker-test docker-build-all docker-up docker-down pre-commit-install pre-commit-run

PYTHON_BIN := $(shell command -v python3 >/dev/null 2>&1 && echo python3 || command -v python)
PACKAGE_VERSION := $(shell $(PYTHON_BIN) -c "import stateset_agents; print(stateset_agents.__version__)")
SPHINX_DOCS_ENV := API_REQUIRE_AUTH=false INFERENCE_BACKEND=stub

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Installation
install: ## Install package with core dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,api]"

install-all: ## Install package with all optional dependencies
	pip install -e ".[dev,api,examples,training,trl]"

dev-setup: ## Install development dependencies and pre-commit hooks
	$(MAKE) install-dev
	$(MAKE) pre-commit-install

# Testing
test: ## Run all tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=stateset_agents --cov-report=html --cov-report=term-missing
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

lint-fix: ## Auto-fix lint issues and format
	ruff check . --fix
	black .
	isort .

format: ## Format code with black and isort
	black .
	isort .

check-types: ## Run mypy type checking
	python scripts/check_types.py --all

check-types-script: ## Run custom type checking script
	python scripts/check_types.py

repo-hygiene: ## Ensure generated and backup artifacts are not tracked
	python scripts/check_repo_hygiene.py

# Documentation
docs: ## Build documentation
	$(SPHINX_DOCS_ENV) sphinx-build docs docs/_build/html

docs-build: ## Build documentation (alias)
	$(SPHINX_DOCS_ENV) sphinx-build docs docs/_build/html

docs-clean: ## Clean documentation build artifacts
	rm -rf docs/_build

docs-api: ## Generate API documentation stubs
	sphinx-apidoc -f -o docs/api stateset_agents

docs-serve: ## Build and serve documentation locally
	$(SPHINX_DOCS_ENV) sphinx-build docs docs/_build/html
	cd docs/_build/html && python -m http.server 8000

# Packaging
build: ## Build distribution packages
	python -m build --no-isolation || python setup.py sdist bdist_wheel

test-package: ## Install built wheel and smoke test
	pip install dist/*.whl && python -c "import stateset_agents; print(stateset_agents.__version__)"

require-release-branch: ## Ensure releases run only from sanctioned branches
	@if [ "$${SKIP_RELEASE_BRANCH_CHECK:-0}" != "1" ]; then \
		branch=$$(git rev-parse --abbrev-ref HEAD); \
		if [ "$$branch" = "HEAD" ] || [ -z "$$branch" ]; then \
			echo "Release checks require a local branch (detached HEAD detected)."; \
			exit 1; \
		fi; \
		case "$$branch" in \
			main|master|release/*) \
				;; \
			*) \
				echo "Refusing to publish from branch '$$branch'."; \
				echo "Use a main/master/release/* branch or set SKIP_RELEASE_BRANCH_CHECK=1 for override."; \
				exit 1; \
				;; \
		esac; \
	fi

publish-test: ## Publish to TestPyPI
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	$(PYTHON_BIN) -m twine upload --skip-existing --repository testpypi dist/*

publish: ## Publish to PyPI
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	$(PYTHON_BIN) -m twine upload --skip-existing dist/*

release: ## Create a release with custom version
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=<version|patch|minor|major>"; \
		echo "Example: make release VERSION=1.2.3"; \
		exit 1; \
	fi
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	python scripts/publish.py --skip-readiness --production --version $${VERSION}

release-patch: ## Create patch release
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	python scripts/publish.py --skip-readiness --production --version patch

release-minor: ## Create minor release
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	python scripts/publish.py --skip-readiness --production --version minor

release-major: ## Create major release
	$(MAKE) require-release-branch
	$(MAKE) publish-readiness
	python scripts/publish.py --skip-readiness --production --version major

quick-publish: ## Run interactive publishing script
	./scripts/quick_publish.sh

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

# Docker helpers
docker-build: ## Build Docker image
	docker build -t stateset-agents .

docker-run: ## Run Docker container
	docker run -p 8000:8000 stateset-agents

docker-build-gateway: ## Build FastAPI gateway image (deployment/docker/Dockerfile)
	docker build -f deployment/docker/Dockerfile -t stateset/stateset-agents-api:$(PACKAGE_VERSION) .

docker-run-gateway: ## Run FastAPI gateway locally (stub backend)
	docker run -p 8000:8000 \
	  -e API_ENVIRONMENT=development \
	  -e API_REQUIRE_AUTH=false \
	  -e INFERENCE_BACKEND=stub \
	  -e INFERENCE_DEFAULT_MODEL=moonshotai/Kimi-K2.5 \
	  stateset/stateset-agents-api:$(PACKAGE_VERSION)

docker-build-trainer: ## Build trainer image (deployment/docker/Dockerfile.trainer)
	docker build -f deployment/docker/Dockerfile.trainer -t stateset/stateset-agents-trainer:$(PACKAGE_VERSION) .

docker-dev: ## Run development environment
	docker compose -f deployment/docker/docker-compose.dev.yml up stateset-agents-api-dev

docker-test: ## Run tests in Docker
	docker compose -f deployment/docker/docker-compose.dev.yml --profile test up stateset-agents-test

docker-build-all: ## Build all Docker images
	docker compose -f deployment/docker/docker-compose.yml build

docker-up: ## Start all services
	docker compose -f deployment/docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f deployment/docker/docker-compose.yml down

# Development workflows
dev-test: ## Quick development checks (format, type, unit tests)
	$(MAKE) lint-fix
	$(MAKE) check-types
	$(MAKE) test-unit

ci: ## Simulate CI pipeline locally
	$(MAKE) repo-hygiene
	$(MAKE) lint
	$(MAKE) check-types
	$(MAKE) test-unit
	$(MAKE) test-cov

security-scan: ## Run basic security scanning tools
	bandit -r stateset_agents || true
	safety check || true
	semgrep --config=auto . || true

security-scan-strict: ## Run stricter security scanning (exit on high severity findings)
	bandit -r stateset_agents -f json -o bandit-report.json || true
	safety check --json > safety-report.json || true
	$(PYTHON_BIN) - <<-'PY'
		import json
		import sys
		from pathlib import Path

		bandit_path = Path("bandit-report.json")
		safety_path = Path("safety-report.json")

		if not bandit_path.exists() or not bandit_path.read_text().strip():
		    print("Bandit report not generated")
		    sys.exit(1)

		try:
		    bandit_payload = json.loads(bandit_path.read_text())
		except Exception as exc:
		    print(f"Bandit output parse failed: {exc}")
		    sys.exit(1)

		bandit_results = []
		if isinstance(bandit_payload, dict):
		    bandit_results = bandit_payload.get("results", [])
		elif isinstance(bandit_payload, list):
		    bandit_results = bandit_payload

		high_findings = [
		    item
		    for item in bandit_results
		    if str(item.get("issue_severity", "")).upper()
		    in {"MEDIUM", "HIGH", "CRITICAL"}
		]

		if high_findings:
		    for item in high_findings[:10]:
		        print(
		            f"Bandit: {item.get('filename')}:{item.get('line_number')} "
		            f"{item.get('test_id')} {item.get('issue_severity')}"
		        )
		    print(
		        f"Bandit: failing with {len(high_findings)} "
		        "medium/high/critical findings"
		    )
		    sys.exit(1)

		if not safety_path.exists() or not safety_path.read_text().strip():
		    print("Safety report not generated; ensure safety is installed")
		    sys.exit(1)

		try:
		    safety_payload = json.loads(safety_path.read_text())
		except Exception as exc:
		    print(f"Safety output parse failed: {exc}")
		    sys.exit(1)

		vulns = safety_payload.get(
		    "vulnerabilities",
		    safety_payload if isinstance(safety_payload, list) else [],
		)
		high = [
		    v
		    for v in vulns
		    if str(v.get("severity", "")).upper() in {"HIGH", "CRITICAL"}
		]

		if high:
		    for v in high[:10]:
		        print(
		            f"High severity vulnerability: "
		            f"{v.get('package_name', 'unknown')} {v.get('id', '')}"
		        )
		    sys.exit(1)

		sys.exit(0)
	PY

publish-readiness: ## Run pre-publish release readiness gate
	bash scripts/publish_readiness.sh

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
