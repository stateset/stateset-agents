# API Hardening Implementation Plan (A+ push, Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the security/operability gaps in the FastAPI layer: unauthenticated `/api/lab` router, unbounded in-memory lab state, dead safety nets, per-pod rate limiting, and misleading duplicate surface (shims, root Rust Dockerfile, nightly vLLM tag).

**Architecture:** All changes are additive hardening inside `stateset_agents/api/` plus deployment metadata. No new services: the training-lab keeps its in-memory simulator but becomes opt-in, authenticated, and bounded. Rate limiting gains an optional Redis backend via the existing `distributed_cache` module, falling back to the current in-memory limiter.

**Tech Stack:** Python 3.10, FastAPI, pytest (tests/api/), Helm/Docker for deployment files.

## Global Constraints

- Secure-by-default: any new flag that exposes surface defaults OFF in production and may default ON only in development.
- Never weaken existing passing tests; new tests go in `tests/api/`.
- Follow existing patterns: config via `stateset_agents/api/config.py` typed getters; auth via `auth.py` dependencies (`require_auth_if_enabled`, `require_role`).
- Ruff (E,W,F,B,C4,UP) clean; conventional commits ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; do not push.
- Update `CHANGELOG.md` (Keep-a-Changelog, Unreleased section) in the task that makes each user-visible change.

---

### Task 1: Gate `/api/lab` behind auth and a feature flag

**Files:**
- Modify: `stateset_agents/api/routers/training_lab.py` (router construction, ~line 26)
- Modify: `stateset_agents/api/config.py` (add flag), `stateset_agents/api/main.py` (conditional include, ~line 207)
- Test: `tests/api/test_training_lab_auth.py` (new)

**Interfaces:**
- New config field `enable_training_lab: bool` read from env `API_ENABLE_TRAINING_LAB`; default: `True` in development, `False` otherwise (mirror how existing env-dependent defaults are computed in `config.py`).
- `main.py` includes the lab router only when the flag is on.
- The lab router is constructed with `dependencies=[Depends(require_auth_if_enabled)]` so every HTTP endpoint requires auth when auth is enabled; the WebSocket endpoint (~:1229) authenticates explicitly (WebSocket can't use the same HTTP dependency — validate the API key/JWT from the query param or first message, mirroring how `auth.py` validates, and close with code 4401 on failure).

- [ ] **Step 1: Write failing tests** — with auth enabled and a valid key configured: unauthenticated `GET /api/lab/experiments` → 401; authenticated → 200. With `API_ENABLE_TRAINING_LAB=false`: route returns 404 (not mounted). WebSocket without credentials is rejected. Build the app via the existing test factory used in `tests/api/` (read `tests/api/test_grpo_auth_regressions.py` for the fixture pattern).
- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement** (config flag + conditional mount + router dependencies + WS auth)
- [ ] **Step 4: Run new tests + `python -m pytest tests/api -q` — pass**
- [ ] **Step 5: Commit** `fix(api): require auth for training lab and gate it behind API_ENABLE_TRAINING_LAB`

---

### Task 2: Bound the training-lab in-memory state

**Files:**
- Modify: `stateset_agents/api/routers/training_lab.py` (module state, ~:32-35, creation endpoints, log/metric appends)
- Test: `tests/api/test_training_lab_limits.py` (new)

**Interfaces:**
- Module constants `MAX_EXPERIMENTS = 100`, `MAX_EPISODES_PER_EXPERIMENT = 1000`, `MAX_LOGS_PER_EXPERIMENT = 5000` (env-overridable via config is NOT required).
- Creating an experiment beyond `MAX_EXPERIMENTS` evicts the oldest **completed/stopped** experiment; if all are running, respond 429 with a clear detail message.
- Episode and log lists become bounded (`collections.deque(maxlen=...)` or explicit trim on append).
- Deleting/stopping an experiment cancels its background asyncio task (verify the task handle is stored; store it if not).

- [ ] **Step 1: Write failing tests** — creating MAX+1 experiments evicts oldest finished or 429s; logs trim to maxlen; stopping an experiment cancels its task (assert task.cancelled() or done()).
- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run new + existing api tests — pass**
- [ ] **Step 5: Commit** `fix(api): bound training-lab in-memory state and cancel background tasks on stop`

---

### Task 3: Enforce `config.validate()` at startup

**Files:**
- Modify: `stateset_agents/api/config.py` (`validate()` ~:346-360), `stateset_agents/api/main.py` (lifespan/startup)
- Test: `tests/api/test_config_validation_startup.py` (new)

**Interfaces:**
- `create_app()` (or the lifespan startup) calls `config.validate()`; each warning is logged at WARNING. In production (`environment == "production"`), the subset of findings that are security-critical (auth enabled but no API keys AND no JWT secret) raises `ConfigurationError` — check `validate()`'s current strings and classify.

- [ ] **Step 1: Failing tests** — building the app in development with no keys logs warnings (caplog); building in production with auth enabled and zero credential sources raises ConfigurationError.
- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests/api — pass**
- [ ] **Step 5: Commit** `fix(api): run config validation at startup; fail closed in production`

---

### Task 4: Rate limiting — key on identity, optional Redis backend

**Files:**
- Modify: `stateset_agents/api/middleware.py` (`SlidingWindowRateLimiter` ~:158-222)
- Modify: `stateset_agents/api/config.py` (add `rate_limit_backend` env `API_RATE_LIMIT_BACKEND`, values `memory` (default) | `redis`, and `rate_limit_redis_url` env `API_RATE_LIMIT_REDIS_URL`)
- Test: `tests/api/test_rate_limit_identity.py` (new)

**Interfaces:**
- Bucket key: hashed API key (reuse `_derive_api_user_id`-style hashing from `auth.py`) when an `Authorization`/`X-API-Key` credential is present; else client IP taking `X-Forwarded-For`'s first hop **only when** a new config `trust_proxy_headers` (env `API_TRUST_PROXY_HEADERS`, default False) is on.
- Redis backend: implement a `RedisSlidingWindowLimiter` with the same `is_allowed(key) -> tuple[bool, int]` interface using INCR/EXPIRE fixed-window (document the approximation) via lazy `redis.asyncio` import; on import/connection error, log once and fall back to the in-memory limiter. Do not add redis to core deps — document it as part of the `api` extra if not already present (check pyproject).

- [ ] **Step 1: Failing tests** — two different API keys don't share a bucket; same key from two "IPs" shares one; XFF ignored when trust flag off, honored when on; memory fallback used when backend=redis but redis unavailable (no crash).
- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests/api — pass**
- [ ] **Step 5: Commit** `feat(api): identity-keyed rate limiting with optional redis backend and proxy-trust flag`

---

### Task 5: Kill the misleading surface — shims, root Dockerfile, nightly vLLM tag, deprecations

**Files:**
- Delete: `stateset_agents/api/ultimate_grpo_service.py`, `stateset_agents/api/enhanced_ultimate_grpo_service.py` (verify nothing imports them: grep repo + tests first; if tests import them, convert those tests to import `stateset_agents.api.main` directly)
- Move: root `Dockerfile` → `deployment/docker/Dockerfile.rust-commerce-agent` with a 3-line header comment stating it builds the unrelated Rust commerce daemon from `src/main.rs`, and fix its HEALTHCHECK note; root gets NO Dockerfile (README's docker instructions point at `deployment/docker/Dockerfile` — verify and fix references: grep `README.md`, `Makefile`, `docker-compose.yml`, CI workflows for `Dockerfile` paths)
- Modify: `deployment/helm/stateset-agents/values.yaml:63` — replace vLLM `tag: "nightly"` with the latest stable vllm-openai release tag (check what version the code targets in `pyproject.toml`'s vllm extra; use e.g. `v0.9.x` matching it) and add a comment `# pin by digest in production overrides`
- Modify: `stateset_agents/api/grpo/__init__.py` — module-level `DeprecationWarning` ("secondary GRPO API app is deprecated; use stateset_agents.api.main") without breaking its tests (filter in those tests if needed)
- Modify: `stateset_agents/api/main.py:279,291` `datetime.utcnow()` → `datetime.now(timezone.utc)`; `main.py:9` deprecated `fastapi.middleware.cors` import → `fastapi.middleware.cors` current canonical path (`from starlette.middleware.cors import CORSMiddleware` or current FastAPI re-export); `auth.py:249` plain dict membership → constant-time comparison over stored keys (`any(hmac.compare_digest(k, candidate) for k in keys)`)
- Test: adjust affected tests; add `tests/api/test_no_legacy_shims.py` asserting the two shim modules are gone (`importlib.util.find_spec(...) is None`)

- [ ] **Step 1: Grep for all references to the deleted/moved files; write the shim-absence test (failing)**
- [ ] **Step 2: Implement all changes; update CHANGELOG (Removed/Changed/Fixed entries)**
- [ ] **Step 3: Run `python -m pytest tests/api -q` + `python scripts/check_repo_hygiene.py` — pass**
- [ ] **Step 4: Commit** `chore(api): remove legacy shims, relocate rust Dockerfile, pin vLLM tag, fix deprecated APIs, constant-time key check`

---

## Self-Review notes

- The review's finding that dashboard/mobile have no deployment path is deliberately deferred to Plan 3 (surface consolidation) — it is a product decision (ship or archive), not hardening.
- `api/grpo/` full removal is out of scope here (tested code, needs a deprecation cycle); Task 5 starts that cycle.
- Task 1 must land before Task 2's tests (they exercise authenticated routes); tasks otherwise independent.
