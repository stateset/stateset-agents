.PHONY: help install install-dev install-all dev-setup test test-cov test-unit test-integration test-slow lint lint-fix format check-types check-types-script repo-hygiene clean docs docs-build docs-clean docs-api docs-serve build test-package publish-test publish release release-patch release-minor release-major require-release-branch quick-publish benchmark benchmark-smoke benchmark-phase0 benchmark-phase0-all benchmark-aggregate benchmark-aggregate-strict benchmark-plot benchmark-publish release-whitepaper-v1 release-whitepaper-v1-strict serve-trained starter-test smoke release-prep demo grade-transcript grade-batch grade-batch-summary prepare-sft sft-from-curated full-loop changelog-check new-version demo-curation demo-full-loop demo-all smoke-cli smoke-fast health dev-test ci security-scan security-scan-strict publish-readiness docker-build docker-run docker-build-gateway docker-run-gateway docker-build-trainer docker-dev docker-test docker-build-all docker-up docker-down pre-commit-install pre-commit-run

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

# Phase 0 / whitepaper-v1 empirical-results pipeline
smoke-fast: ## Quick inner-loop smoke: just the platform-pipeline unit tests (~20s, no integration tests)
	@python -m pytest tests/unit/test_gsm8k.py \
		tests/unit/test_reproducibility.py \
		tests/unit/test_aggregate_phase0_results.py \
		tests/unit/test_customer_support_bench.py \
		tests/unit/test_tool_calling_bench.py \
		tests/unit/test_task_adapters.py \
		tests/unit/test_scaffolding.py \
		tests/unit/test_chat_cli.py \
		tests/unit/test_grade_transcript.py \
		tests/unit/test_prepare_sft_dataset.py \
		tests/unit/test_sft_from_curated.py \
		tests/unit/test_recipe_cli.py \
		tests/unit/test_agent_config_peft_path.py \
		tests/unit/test_checkpoint_agent_registration.py \
		tests/unit/test_evaluate_batch.py \
		-q --no-header

smoke-cli: ## Verify every CLI subcommand's --help loads without error (catches arg-parsing regressions)
	@echo "==> Verifying CLI subcommands load"
	@FAILED=0; \
	for cmd in version doctor preflight serve evaluate init train init-config validate-config tour starter benchmark chat fine-tune recipe; do \
		if python -m stateset_agents.cli $$cmd --help > /dev/null 2>&1; then \
			printf "  ✓ %s\n" "$$cmd"; \
		else \
			printf "  ✗ %s — FAILED\n" "$$cmd"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	if [ $$FAILED -gt 0 ]; then \
		echo ""; \
		echo "❌ $$FAILED CLI subcommand(s) failed to load."; \
		exit 1; \
	fi; \
	echo ""; \
	echo "✓ All CLI subcommands load cleanly."

health: ## Comprehensive platform health check: smoke-cli + doctor + version + changelog + a 5s end-to-end demo
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║   StateSet Agents — Platform Health Check                              ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "▶ Step 1/5: Version + provenance"
	@python -m stateset_agents.cli version 2>&1 | grep -v "platform detected" | tail -12 | sed 's/^/  /'
	@echo ""
	@echo "▶ Step 2/5: CHANGELOG entry matches pyproject version"
	@$(MAKE) -s changelog-check
	@echo ""
	@echo "▶ Step 3/5: CLI subcommands load"
	@$(MAKE) -s smoke-cli
	@echo ""
	@echo "▶ Step 4/5: Doctor diagnostic"
	@python -m stateset_agents.cli doctor 2>&1 | grep -E "✅|❌|⚠️ |CUDA" | sed 's/^/  /' | head -10
	@echo ""
	@echo "▶ Step 5/5: 6-second end-to-end smoke (loads GSM8K, parses, seeds)"
	@$(MAKE) -s benchmark-smoke 2>&1 | tail -3 | sed 's/^/  /'
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║   ✓ Health check complete.                                             ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"

notebook-lint: ## Lint bundled notebooks for the foot-gun patterns from issue #16
	@python3 scripts/lint_notebooks.py

whitepaper-pdf: ## Build docs/WHITEPAPER.pdf from docs/WHITEPAPER.md via weasyprint
	@python3.10 scripts/build_whitepaper_pdf.py

getting-started-smoke: ## Run examples/getting_started/smoke.sh against the installed PyPI package
	@PYTHON=python3.10 examples/getting_started/smoke.sh

smoke: ## Run the full local smoke pipeline: unit tests + benchmark smoke + starter smoke + notebook validation
	@echo "==> Unit tests for benchmark + scaffolding pipelines"
	@python -m pytest tests/unit/test_gsm8k.py \
		tests/unit/test_reproducibility.py \
		tests/unit/test_aggregate_phase0_results.py \
		tests/unit/test_customer_support_bench.py \
		tests/unit/test_tool_calling_bench.py \
		tests/unit/test_task_adapters.py \
		tests/unit/test_scaffolding.py \
		-q --no-header
	@echo ""
	@echo "==> Benchmark pipeline smoke (no GPU)"
	@$(MAKE) -s benchmark-smoke
	@echo ""
	@echo "==> Starter scaffold smoke (all 4 templates)"
	@$(MAKE) -s starter-test
	@echo ""
	@echo "==> Notebook lint (foot-gun patterns from issue #16)"
	@$(MAKE) -s notebook-lint
	@echo ""
	@echo "✓ All smoke checks passed."

benchmark-smoke: ## Smoke test the GSM8K benchmark pipeline (no GPU, ~6s)
	python -m stateset_agents.cli benchmark smoke

benchmark-phase0: ## Run a single Phase 0 GSM8K benchmark (defaults: gspo, seed=42)
	python -m stateset_agents.cli benchmark phase0 \
		--trainer $(TRAINER) \
		--seed $(SEED) \
		--output benchmark_results/whitepaper_v1/$(TRAINER)_seed$(SEED)_qwen3_5_0_8b.json

benchmark-phase0-all: ## Run GRPO + GSPO + DAPO × 3 seeds = 9 benchmark runs (~6h on A100)
	@for trainer in grpo gspo dapo; do \
		for seed in 42 1337 2026; do \
			echo "=== $$trainer seed=$$seed ==="; \
			python -m stateset_agents.cli benchmark phase0 \
				--trainer $$trainer --seed $$seed \
				--output benchmark_results/whitepaper_v1/$${trainer}_seed$${seed}_qwen3_5_0_8b.json; \
		done; \
	done

benchmark-aggregate: ## Aggregate all *.json results into summary.md + summary.csv
	python -m stateset_agents.cli benchmark aggregate \
		--results-dir benchmark_results/whitepaper_v1

benchmark-aggregate-strict: ## Same as benchmark-aggregate, but exit 1 if gates fail (for CI)
	python -m stateset_agents.cli benchmark aggregate \
		--results-dir benchmark_results/whitepaper_v1 --strict

benchmark-plot: ## Generate PNG figures + text_plots.md from summary.csv
	python -m stateset_agents.cli benchmark plot \
		--results-dir benchmark_results/whitepaper_v1

benchmark-publish: benchmark-aggregate benchmark-plot ## Produce summary.md + figures in one shot
	@echo "Whitepaper-ready artifacts in benchmark_results/whitepaper_v1/"
	@ls benchmark_results/whitepaper_v1/ 2>/dev/null || true

release-whitepaper-v1: ## One-shot v1.0 whitepaper release packaging (aggregate + plot + §11.7 + manifest)
	python scripts/release_v1_whitepaper.py

release-whitepaper-v1-strict: ## Same but fails the build if benchmark gates aren't met
	python scripts/release_v1_whitepaper.py --strict

# Closing the loop: train → serve
starter-test: ## Materialize every starter template into /tmp and validate generated configs
	@rm -rf /tmp/stateset_starter_test
	@mkdir -p /tmp/stateset_starter_test
	@for tmpl in customer-support gsm8k-math tool-calling-agent minimal; do \
		echo "==> scaffolding $$tmpl"; \
		python -m stateset_agents.cli starter "$$tmpl" "/tmp/stateset_starter_test/$$tmpl" || exit 1; \
		test -f "/tmp/stateset_starter_test/$$tmpl/config.yaml" || (echo "FAIL: missing config.yaml" && exit 1); \
		test -f "/tmp/stateset_starter_test/$$tmpl/train.py" || (echo "FAIL: missing train.py" && exit 1); \
		python -c "import yaml,pathlib; cfg=yaml.safe_load(pathlib.Path('/tmp/stateset_starter_test/$$tmpl/config.yaml').read_text()); assert cfg, 'empty config'" || exit 1; \
		python -c "import ast,pathlib; ast.parse(pathlib.Path('/tmp/stateset_starter_test/$$tmpl/train.py').read_text())" || exit 1; \
	done
	@echo "==> client-name customization"
	@python -m stateset_agents.cli starter customer-support /tmp/stateset_starter_test/client_acme --client-name "Acme Corp" > /dev/null
	@grep -q "wandb_project: acme_corp" /tmp/stateset_starter_test/client_acme/config.yaml || (echo "FAIL: client-name customization broken" && exit 1)
	@echo ""
	@echo "✓ All 4 templates materialize, generate valid YAML configs and parseable train.py."
	@echo "✓ --client-name customization patches config.yaml correctly."

grade-transcript: ## Grade one saved chat transcript. Usage: make grade-transcript HISTORY=conv.jsonl REWARD=customer_support [CONTEXT=ctx.jsonl]
	@if [ -z "$(HISTORY)" ] || [ -z "$(REWARD)" ]; then \
		echo "Usage: make grade-transcript HISTORY=<jsonl> REWARD=<gsm8k|customer_support|tool_calling> [CONTEXT=<jsonl>] [OUTPUT=<md>]"; \
		exit 2; \
	fi
	python scripts/grade_transcript.py \
		--history $(HISTORY) \
		--reward $(REWARD) \
		$(if $(CONTEXT),--context-file $(CONTEXT),) \
		$(if $(OUTPUT),--output $(OUTPUT),)

grade-batch: ## Grade every *.jsonl under DIR. Usage: make grade-batch DIR=transcripts/ REWARD=customer_support [OUT_DIR=graded/] [CURATED=good_examples.jsonl] [THRESHOLD=0.7]
	@if [ -z "$(DIR)" ] || [ -z "$(REWARD)" ]; then \
		echo "Usage: make grade-batch DIR=<path> REWARD=<gsm8k|customer_support|tool_calling> [OUT_DIR=<path>] [CURATED=<jsonl>] [THRESHOLD=<float>]"; \
		exit 2; \
	fi
	@OUT="$(if $(OUT_DIR),$(OUT_DIR),$(DIR)/graded)"; \
	mkdir -p $$OUT; \
	rm -f "$(CURATED)"; \
	count=0; \
	for f in $(DIR)/*.jsonl; do \
		[ -f "$$f" ] || continue; \
		base=$$(basename $$f .jsonl); \
		echo "==> grading $$f"; \
		python scripts/grade_transcript.py \
			--history $$f --reward $(REWARD) \
			--output $$OUT/$$base.md --json \
			$(if $(CURATED),--output-curated $(CURATED) --threshold $(or $(THRESHOLD),0.7),) \
			|| exit 1; \
		count=$$((count + 1)); \
	done; \
	echo ""; \
	echo "✓ Graded $$count transcript(s) → $$OUT/"; \
	if [ -n "$(CURATED)" ] && [ -f "$(CURATED)" ]; then \
		echo "✓ Curated $$(wc -l < $(CURATED)) good example(s) → $(CURATED)"; \
	fi

grade-batch-summary: ## Produce one umbrella summary across all graded transcripts. Usage: make grade-batch-summary GRADED_DIR=transcripts/graded
	@if [ -z "$(GRADED_DIR)" ]; then \
		echo "Usage: make grade-batch-summary GRADED_DIR=<path> [OUTPUT=<md>]"; \
		exit 2; \
	fi
	python scripts/summarize_graded_batch.py \
		--graded-dir $(GRADED_DIR) \
		$(if $(OUTPUT),--output $(OUTPUT),)

full-loop: ## prepare-sft + sft-from-curated in one shot. Usage: make full-loop INPUT=curated.jsonl BASE_MODEL=Qwen/Qwen3.5-0.8B [OUTPUT_DIR=outputs/sft_v1] [MIN_SCORE=0.7] [DRY_RUN=1]
	@if [ -z "$(INPUT)" ] || [ -z "$(BASE_MODEL)" ]; then \
		echo "Usage: make full-loop INPUT=<curated.jsonl> BASE_MODEL=<hf-model> [OUTPUT_DIR=<dir>] [MIN_SCORE=<float>] [DRY_RUN=1]"; \
		exit 2; \
	fi
	@TMP_SFT="$$(mktemp -d)/sft_train.jsonl"; \
	echo "▶ Step 1/2: prepare-sft (curated → chat format)"; \
	python scripts/prepare_sft_dataset.py \
		--input $(INPUT) \
		--format chat \
		--output $$TMP_SFT \
		$(if $(MIN_SCORE),--min-score $(MIN_SCORE),--min-score 0.7) \
		--dedup --stats 2>&1 | tail -10; \
	echo ""; \
	echo "▶ Step 2/2: sft-from-curated (chat format → trained adapter)"; \
	python scripts/sft_from_curated.py \
		--dataset $$TMP_SFT \
		--base-model $(BASE_MODEL) \
		--output-dir $(or $(OUTPUT_DIR),outputs/sft_v1) \
		$(if $(NUM_EPOCHS),--num-epochs $(NUM_EPOCHS),) \
		$(if $(LORA_R),--lora-r $(LORA_R),) \
		$(if $(DRY_RUN),--dry-run,)

sft-from-curated: ## Run SFT on a prepared chat-format JSONL. Usage: make sft-from-curated DATASET=sft_train.jsonl BASE_MODEL=Qwen/Qwen3.5-0.8B [OUTPUT_DIR=outputs/sft_v1]
	@if [ -z "$(DATASET)" ] || [ -z "$(BASE_MODEL)" ]; then \
		echo "Usage: make sft-from-curated DATASET=<sft_train.jsonl> BASE_MODEL=<hf-model> [OUTPUT_DIR=<dir>] [NUM_EPOCHS=<n>] [LORA_R=<r>]"; \
		exit 2; \
	fi
	python scripts/sft_from_curated.py \
		--dataset $(DATASET) \
		--base-model $(BASE_MODEL) \
		--output-dir $(or $(OUTPUT_DIR),outputs/sft_v1) \
		$(if $(NUM_EPOCHS),--num-epochs $(NUM_EPOCHS),) \
		$(if $(LORA_R),--lora-r $(LORA_R),) \
		$(if $(DRY_RUN),--dry-run,)

prepare-sft: ## Convert curated.jsonl into SFT dataset format. Usage: make prepare-sft INPUT=curated.jsonl FORMAT=chat OUTPUT=sft.jsonl [MIN_SCORE=0.7] [DEDUP=1]
	@if [ -z "$(INPUT)" ] || [ -z "$(FORMAT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make prepare-sft INPUT=<curated.jsonl> FORMAT=<hf-trainer|chat|axolotl> OUTPUT=<out.jsonl> [MIN_SCORE=<float>] [DEDUP=1]"; \
		exit 2; \
	fi
	python scripts/prepare_sft_dataset.py \
		--input $(INPUT) --format $(FORMAT) --output $(OUTPUT) \
		$(if $(MIN_SCORE),--min-score $(MIN_SCORE),) \
		$(if $(DEDUP),--dedup,) \
		--stats

new-version: ## Bump version in pyproject.toml + __init__.py and prepend a CHANGELOG draft. Usage: make new-version VERSION=0.12.0
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make new-version VERSION=<x.y.z>"; \
		exit 2; \
	fi
	@CURRENT=$$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/'); \
	echo "==> Bumping $$CURRENT → $(VERSION)"; \
	sed -i.bak -E 's/^version = "[^"]+"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak; \
	sed -i.bak -E 's/^__version__ = "[^"]+"/__version__ = "$(VERSION)"/' stateset_agents/__init__.py && rm stateset_agents/__init__.py.bak; \
	echo "==> Prepending CHANGELOG draft"; \
	TODAY=$$(date +%Y-%m-%d); \
	awk -v VER="$(VERSION)" -v TODAY="$$TODAY" 'BEGIN { \
		printed=0 \
	} /^## \[/ && !printed { \
		print "## [" VER "] - " TODAY " — TODO: title"; \
		print ""; \
		print "### Added"; \
		print "- TODO: describe what was added in this release."; \
		print ""; \
		print "### Fixed"; \
		print "- TODO: describe bug fixes (or remove this section)."; \
		print ""; \
		printed=1 \
	} { print }' CHANGELOG.md > CHANGELOG.md.new && mv CHANGELOG.md.new CHANGELOG.md
	@echo ""
	@echo "✓ Version bumped to $(VERSION) in pyproject.toml + __init__.py"
	@echo "✓ CHANGELOG.md draft entry prepended — fill in TODOs before publishing."
	@echo ""
	@echo "Next steps:"
	@echo "  \$$EDITOR CHANGELOG.md           # fill in the draft entry"
	@echo "  make changelog-check            # verify the entry"
	@echo "  make release-prep               # smoke + build + twine check"

changelog-check: ## Fail if pyproject version isn't documented in CHANGELOG. Run in CI before releasing.
	@VER=$$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/'); \
	if [ -z "$$VER" ]; then \
		echo "Could not extract version from pyproject.toml"; exit 2; \
	fi; \
	if grep -qE "^## \[$$VER\]" CHANGELOG.md; then \
		echo "✓ CHANGELOG.md has an entry for version $$VER"; \
	else \
		echo "❌ Missing CHANGELOG entry for version $$VER"; \
		echo "   Add a section starting with '## [$$VER] - <date>' to CHANGELOG.md"; \
		exit 1; \
	fi

serve-trained: ## Serve a trained checkpoint via the FastAPI gateway. Usage: make serve-trained CHECKPOINT=path/to/adapter BASE_MODEL=Qwen/Qwen3.5-0.8B
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make serve-trained CHECKPOINT=<path> [BASE_MODEL=<name>] [PORT=<port>]"; exit 2; fi
	python -m stateset_agents.cli serve \
		--checkpoint $(CHECKPOINT) \
		$(if $(BASE_MODEL),--base-model $(BASE_MODEL),) \
		--port $(or $(PORT),8000)

# Default Makefile variables for benchmark targets — override with: make benchmark-phase0 TRAINER=dapo SEED=1337
TRAINER ?= gspo
SEED ?= 42

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
	$(MAKE) smoke

demo-all: ## Run all three platform demos in sequence (no GPU, ~12s total). Use for live screen shares.
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║   StateSet Agents — Triple Demo                                        ║"
	@echo "║   benchmark + curation + SFT-loop in 12 seconds, no GPU.               ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) -s demo
	@echo ""
	@echo ""
	@$(MAKE) -s demo-curation
	@echo ""
	@echo ""
	@$(MAKE) -s demo-full-loop
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║   ✓ All three demos completed.                                         ║"
	@echo "║                                                                       ║"
	@echo "║   This is the platform's full developer surface:                       ║"
	@echo "║     • make demo            — RL train + benchmark pipeline             ║"
	@echo "║     • make demo-curation   — chat → grade → curate                     ║"
	@echo "║     • make demo-full-loop  — curated → prepare → SFT adapter           ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"

demo-full-loop: ## Full-loop demo: synthetic curated.jsonl → prepare → SFT (dry run, no GPU, ~5s)
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║    StateSet Agents — Full-Loop Demo                                    ║"
	@echo "║    curated.jsonl → prepare-sft → sft-from-curated in 5 seconds.        ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@rm -rf /tmp/stateset_loop_demo
	@mkdir -p /tmp/stateset_loop_demo
	@echo "▶ Step 1/3: Stage a curated.jsonl (would come from grade-batch in practice)"
	@printf '%s\n' \
		'{"prompt": "I need a refund for my order", "response": "I'\''d be happy to help with your refund. Please share the order number.", "score": 0.91, "source": "s1.jsonl"}' \
		'{"prompt": "The app keeps crashing", "response": "I'\''m sorry to hear that. Which version are you on?", "score": 0.85, "source": "s2.jsonl"}' \
		'{"prompt": "How do I update my card?", "response": "Settings → Billing → Payment Methods.", "score": 0.86, "source": "s3.jsonl"}' \
		> /tmp/stateset_loop_demo/curated.jsonl
	@echo "  Wrote 3 curated examples → /tmp/stateset_loop_demo/curated.jsonl"
	@echo ""
	@echo "▶ Step 2/3: Convert to chat-format SFT dataset"
	@python scripts/prepare_sft_dataset.py \
		--input /tmp/stateset_loop_demo/curated.jsonl \
		--format chat \
		--output /tmp/stateset_loop_demo/sft_train.jsonl \
		--min-score 0.7 --dedup --stats 2>&1 | tail -7 | sed 's/^/  /'
	@echo ""
	@echo "▶ Step 3/3: SFT (dry-run because no GPU; on GPU this writes a LoRA adapter)"
	@python scripts/sft_from_curated.py \
		--dataset /tmp/stateset_loop_demo/sft_train.jsonl \
		--base-model Qwen/Qwen3.5-0.8B \
		--output-dir /tmp/stateset_loop_demo/sft_v1 \
		--dry-run 2>&1 | grep -E "Dataset size|Base model|Output dir|Epochs|LoRA r|First example|\[user\]|\[assistant\]" | sed 's/^/  /' | head -8
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ Full-loop demo complete.                                           ║"
	@echo "║                                                                       ║"
	@echo "║  On a GPU host, Step 3 produces /tmp/stateset_loop_demo/sft_v1/        ║"
	@echo "║  (a LoRA adapter) which loads via:                                     ║"
	@echo "║                                                                       ║"
	@echo "║    stateset-agents chat \\                                              ║"
	@echo "║      --model Qwen/Qwen3.5-0.8B \\                                       ║"
	@echo "║      --checkpoint /tmp/stateset_loop_demo/sft_v1                       ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"

demo-curation: ## End-to-end curation demo: chat (synthetic) → grade → curate → summary (no GPU, ~5s)
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║       StateSet Agents — Human-in-the-Loop Curation Demo                ║"
	@echo "║       chat → grade → curate → summary in 5 seconds, no GPU.            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@rm -rf /tmp/stateset_curate_demo
	@mkdir -p /tmp/stateset_curate_demo/transcripts
	@echo "▶ Step 1/4: Stage three transcripts of varying quality"
	@printf '%s\n%s\n' \
		'{"role": "user", "content": "I need a refund for my order"}' \
		'{"role": "assistant", "content": "I'\''d be happy to help with your refund. Please share the order number and I'\''ll process it right away."}' \
		> /tmp/stateset_curate_demo/transcripts/good.jsonl
	@printf '%s\n%s\n' \
		'{"role": "user", "content": "refund please"}' \
		'{"role": "assistant", "content": "ok"}' \
		> /tmp/stateset_curate_demo/transcripts/mediocre.jsonl
	@printf '%s\n%s\n' \
		'{"role": "user", "content": "refund"}' \
		'{"role": "assistant", "content": "That request is impossible to help with."}' \
		> /tmp/stateset_curate_demo/transcripts/bad.jsonl
	@ls /tmp/stateset_curate_demo/transcripts/ | sed 's/^/  /'
	@echo ""
	@echo "▶ Step 2/4: Stage contexts (intent ground truth)"
	@for f in good mediocre bad; do \
		echo '{"intent": "refund", "must_acknowledge": ["refund", "order"], "must_avoid": ["impossible"]}' \
			> /tmp/stateset_curate_demo/transcripts/$$f.context.jsonl; \
	done
	@echo "  3 context JSONLs"
	@echo ""
	@echo "▶ Step 3/4: Grade each transcript + curate good examples"
	@for f in good mediocre bad; do \
		python scripts/grade_transcript.py \
			--history /tmp/stateset_curate_demo/transcripts/$$f.jsonl \
			--context-file /tmp/stateset_curate_demo/transcripts/$$f.context.jsonl \
			--reward customer_support \
			--output /tmp/stateset_curate_demo/transcripts/$$f.graded.md --json \
			--output-curated /tmp/stateset_curate_demo/curated.jsonl --threshold 0.7 \
			2>&1 | grep "Curated\|Mean" | sed 's/^/  /' | head -2; \
	done
	@echo ""
	@echo "▶ Step 4/4: Curated training set"
	@if [ -f /tmp/stateset_curate_demo/curated.jsonl ]; then \
		echo "  $$(wc -l < /tmp/stateset_curate_demo/curated.jsonl) example(s) kept at threshold=0.7:"; \
		cat /tmp/stateset_curate_demo/curated.jsonl | python3 -c "import sys, json; [print(f'    score={d[\"score\"]:.2f}  source={d[\"source\"]}  prompt={d[\"prompt\"][:50]}…') for d in (json.loads(l) for l in sys.stdin if l.strip())]"; \
	else \
		echo "  (no examples passed the threshold)"; \
	fi
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ Curation flow complete. The good examples are now in:              ║"
	@echo "║    /tmp/stateset_curate_demo/curated.jsonl                             ║"
	@echo "║                                                                       ║"
	@echo "║  In production, repeat with real conversation logs and use the        ║"
	@echo "║  curated.jsonl as new training data for the next fine-tune pass.      ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"

demo: ## End-to-end demo: scaffold + smoke + synthetic-aggregate + plot + release (no GPU, ~30s)
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║       StateSet Agents — End-to-End Pipeline Demo                       ║"
	@echo "║       From scaffold to whitepaper artifacts in 30 seconds.             ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@rm -rf /tmp/stateset_demo
	@mkdir -p /tmp/stateset_demo
	@echo "▶ Step 1/5: Scaffold a customer-support project"
	@python -m stateset_agents.cli starter customer-support /tmp/stateset_demo/client-acme \
		--client-name "Acme Corp" 2>&1 | grep "Created\|Next steps\|Edit scenarios" | head -4 | sed 's/^/  /'
	@echo ""
	@echo "▶ Step 2/5: Run the benchmark pipeline smoke test"
	@python scripts/run_phase0_benchmark.py --trainer gspo --task gsm8k --smoke-test --output /tmp/stateset_demo/_smoke.json 2>&1 | grep "Smoke test\|Train:" | sed 's/^/  /' | head -3
	@echo ""
	@echo "▶ Step 3/5: Drop synthetic benchmark results (3 trainers × 3 seeds)"
	@mkdir -p /tmp/stateset_demo/benchmark_results/whitepaper_v1
	@for trainer in gspo grpo dapo; do \
		case $$trainer in gspo) FINAL=0.42 ;; grpo) FINAL=0.38 ;; dapo) FINAL=0.45 ;; esac; \
		for seed in 42 1337 2026; do \
			printf '{"trainer":"%s","task":"gsm8k","model":"Qwen/Qwen3.5-0.8B","seed":%s,"commit":"demo","timestamp":"2026-05-14T12:00:00Z","config":{"learning_rate":5e-6},"metrics":{"eval_pass_at_1":%s,"eval_pass_at_1_baseline":0.32,"wall_clock_seconds":2700,"peak_vram_mb":24317}}' \
				$$trainer $$seed $$FINAL > /tmp/stateset_demo/benchmark_results/whitepaper_v1/$${trainer}_seed$${seed}.json; \
		done; \
	done
	@echo "  Wrote 9 JSON files."
	@echo ""
	@echo "▶ Step 4/5: Aggregate + plot"
	@python scripts/aggregate_phase0_results.py --results-dir /tmp/stateset_demo/benchmark_results/whitepaper_v1 2>&1 | tail -3 | sed 's/^/  /'
	@python scripts/plot_phase0_results.py --results-dir /tmp/stateset_demo/benchmark_results/whitepaper_v1 --no-matplotlib 2>&1 | tail -1 | sed 's/^/  /'
	@echo ""
	@echo "▶ Step 5/5: Aggregated summary table"
	@head -12 /tmp/stateset_demo/benchmark_results/whitepaper_v1/summary.md | sed 's/^/  /'
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ Demo complete. The pipeline is operational end-to-end.             ║"
	@echo "║                                                                       ║"
	@echo "║  Artifacts in /tmp/stateset_demo:                                     ║"
	@echo "║    client-acme/             ← fork-and-go project (10 files)          ║"
	@echo "║    benchmark_results/        ← 9 result JSONs + summary.md/csv        ║"
	@echo "║                                                                       ║"
	@echo "║  On a real A100, replace synthetic results with:                       ║"
	@echo "║    make benchmark-phase0-all                                          ║"
	@echo "║    make release-whitepaper-v1                                         ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════╝"

release-prep: ## Final readiness check before publishing — smoke + build + twine check
	@echo "==> Running smoke umbrella"
	@$(MAKE) -s smoke
	@echo ""
	@echo "==> Building wheel + sdist"
	@rm -rf dist/
	@python -m build
	@echo ""
	@echo "==> Validating distribution metadata"
	@python -m twine check dist/*
	@echo ""
	@echo "==> Distribution artifacts:"
	@ls -lh dist/
	@echo ""
	@echo "✓ Release-ready. Next: \`make publish\` (or \`make publish-test\` first)."

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
