#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR"
BANDIT_REPORT_PATH="$REPORT_DIR/bandit-report.json"
SAFETY_REPORT_PATH="$REPORT_DIR/safety-report.json"
SUMMARY_PATH="$REPORT_DIR/publish-readiness-summary.json"
SAFETY_INPUT_PATH="$(mktemp /tmp/stateset-publish-safety.XXXXXX.txt)"
SMOKE_VENV=""
PYTHON_BIN="${PYTHON_BIN:-}"
cd "$ROOT_DIR"
START_TIME="$(date -u +%s)"
CURRENT_STEP="initialize"
GIT_SHA="${GITHUB_SHA:-unknown}"
GIT_REF="${GITHUB_REF_NAME:-unknown}"
CURRENT_BRANCH="${GITHUB_HEAD_REF:-unknown}"

json_escape() {
    printf '%s' "$1" | sed \
        -e 's/\\/\\\\/g' \
        -e 's/"/\\"/g' \
        -e ':a;N;$!ba;s/\n/\\n/g'
}

write_summary_fallback() {
    local status="$1"
    local step="$2"
    local end_time="$3"
    local failed_step_json="null"
    local failure_detail_json=""
    local detail

    if [ "$status" != "passed" ]; then
        failed_step_json="\"$step\""
        detail="${READINESS_FAILURE_DETAIL:-}"
        if [ -n "$detail" ]; then
            detail="$(json_escape "$detail")"
            failure_detail_json=",\"failure_detail\":\"$detail\""
        fi
    fi

    cat > "$SUMMARY_PATH" <<JSON
{
  "status": "$status",
  "failed_step": $failed_step_json,
  "generated_at_unix": $end_time,
  "duration_seconds": $((end_time - START_TIME)),
  "git": {
    "sha": "$GIT_SHA",
    "ref": "$GIT_REF",
    "branch": "$CURRENT_BRANCH"
  }$failure_detail_json
}
JSON
}

write_summary() {
    local rc="$1"
    local step="$2"
    local end_time
    local status

    end_time="$(date -u +%s)"
    if [ "${rc}" -eq 0 ]; then
        status="passed"
    else
        status="failed"
    fi

    if [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" - "$status" "$step" "$START_TIME" "$end_time" "$SUMMARY_PATH" "$GIT_SHA" "$GIT_REF" "$CURRENT_BRANCH" <<'PY'
import json
import sys
import os
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

status = sys.argv[1]
failed_step = sys.argv[2] if status != "passed" else None
start_time = int(sys.argv[3])
end_time = int(sys.argv[4])
summary_path = sys.argv[5]
git_sha = sys.argv[6]
git_ref = sys.argv[7]
git_branch = sys.argv[8]

summary = {
    "schema_version": 2,
    "kind": "stateset-publish-readiness-summary",
    "status": status,
    "failed_step": failed_step,
    "generated_at_unix": end_time,
    "duration_seconds": max(0, end_time - start_time),
    "git": {
        "sha": git_sha,
        "ref": git_ref,
        "branch": git_branch,
    },
}
if status == "passed":
    root = Path(summary_path).resolve().parent
    version_match = re.search(
        r'^version = "(\d+\.\d+\.\d+)"$',
        (root / "pyproject.toml").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if version_match is None:
        raise SystemExit("could not bind readiness summary to package version")

    def retained(path):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"readiness artifact is missing or symlinked: {path}")
        data = path.read_bytes()
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    distributions = sorted((root / "dist").iterdir())
    wheels = [path for path in distributions if path.suffix == ".whl"]
    sdists = [path for path in distributions if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(distributions) != 2:
        raise SystemExit("readiness requires exactly one wheel and one source archive")
    summary.update(
        {
            "framework_version": version_match.group(1),
            "checks": [
                "linters",
                "type_checks",
                "api_compatibility",
                "release_governance",
                "agent_quality_contract",
                "tests_with_coverage",
                "security_scans",
                "build",
                "twine_check",
                "isolated_wheel_smoke",
                "working_tree_clean",
            ],
            "working_tree_clean": True,
            "distributions": [retained(path) for path in [*wheels, *sdists]],
            "security_reports": [
                retained(root / "bandit-report.json"),
                retained(root / "safety-report.json"),
            ],
            "coverage_reports": [retained(root / "coverage.xml")],
            "coverage_percent": round(
                float(ET.parse(root / "coverage.xml").getroot().attrib["line-rate"])
                * 100,
                4,
            ),
        }
    )
failure_detail = os.environ.get("READINESS_FAILURE_DETAIL")
if failure_detail:
    summary["failure_detail"] = failure_detail

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY
    else
        write_summary_fallback "$status" "$step" "$end_time"
    fi
}

on_exit() {
    local exit_code="$?"
    rm -f "$SAFETY_INPUT_PATH"
    case "$SMOKE_VENV" in
        /tmp/stateset-wheel-smoke.*)
            rm -rf -- "$SMOKE_VENV"
            ;;
    esac
    if ! write_summary "$exit_code" "$CURRENT_STEP"; then
        echo "ERROR: could not write a complete publish-readiness summary" >&2
        exit_code=1
    fi
    trap - EXIT
    exit "$exit_code"
}
trap on_exit EXIT

CURRENT_STEP="preflight"
READINESS_FAILURE_DETAIL=""

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ] || ! "$PYTHON_BIN" -c \
  'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  READINESS_FAILURE_DETAIL="missing supported Python interpreter (requires >=3.10)"
  echo "ERROR: Python 3.10 or newer is required."
  exit 1
fi

REQUIRED_COMMANDS=(
  "$PYTHON_BIN"
  "git"
  "ruff"
  "black"
  "isort"
  "mypy"
  "bandit"
  "safety"
)

MISSING_COMMANDS=()
for cmd in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    MISSING_COMMANDS+=("$cmd")
  fi
done

if [ "${#MISSING_COMMANDS[@]}" -ne 0 ]; then
  READINESS_FAILURE_DETAIL="missing_tools: $(printf '%s ' "${MISSING_COMMANDS[@]}")"
  echo "ERROR: Missing required publish-readiness tooling:"
  printf '  - %s\n' "${MISSING_COMMANDS[@]}"
  echo "Install dependencies and retry: pip install -e \".[dev,api]\" ruff black isort mypy pytest bandit safety twine build"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import build, pytest, twine" >/dev/null 2>&1; then
  READINESS_FAILURE_DETAIL="missing_python_modules: build, pytest, or twine"
  echo "ERROR: Required Python modules are missing: build, pytest, or twine"
  echo "Install dependencies and retry: pip install build pytest twine"
  exit 1
fi

if [ "$CURRENT_BRANCH" = "unknown" ] && [ -n "${GITHUB_REF_NAME:-}" ]; then
  CURRENT_BRANCH="$GITHUB_REF_NAME"
fi
if [ "$CURRENT_BRANCH" = "unknown" ]; then
  if ! CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"; then
    CURRENT_BRANCH="unknown"
  fi
fi
if [ "$CURRENT_BRANCH" = "HEAD" ] || [ -z "$CURRENT_BRANCH" ]; then
  if ! CURRENT_BRANCH="$(git rev-parse --short HEAD 2>/dev/null)"; then
    CURRENT_BRANCH="unknown"
  fi
fi
if [ "${GIT_SHA}" = "unknown" ]; then
  if ! GIT_SHA="$(git rev-parse HEAD 2>/dev/null)"; then
    GIT_SHA="unknown"
  fi
fi

printf "\n==> Publish readiness checks for stateset-agents\n"

printf "\n[1/11] Running linters...\n"
CURRENT_STEP="linters"
ruff check .
black --check .
isort --check-only .

printf "\n[2/11] Running type checks...\n"
CURRENT_STEP="type_checks"
python scripts/check_types.py --all

printf "\n[3/11] Verifying stable v1 API compatibility...\n"
CURRENT_STEP="api_compatibility"
"$PYTHON_BIN" scripts/check_api_compatibility.py

printf "\n[4/11] Verifying release governance...\n"
CURRENT_STEP="release_governance"
"$PYTHON_BIN" scripts/check_release_governance.py

printf "\n[5/11] Verifying standard-agent benchmark contract...\n"
CURRENT_STEP="agent_quality_contract"
"$PYTHON_BIN" benchmarks/run_agent_quality_matrix.py \
  benchmarks/agent_quality_manifest.example.json \
  --output-dir /tmp/stateset-agent-quality-contract --dry-run

printf "\n[6/11] Running tests with coverage gate...\n"
CURRENT_STEP="tests_with_coverage"
# Gate value lives in pyproject.toml's [tool.coverage.report] fail_under and
# is honored automatically by pytest-cov. Avoid passing --cov-fail-under here
# so the gate has a single source of truth (see v0.15.3 ratchet correction).
"$PYTHON_BIN" -m pytest --cov=stateset_agents --cov-report=xml

printf "\n[7/11] Running security scans...\n"
CURRENT_STEP="security_scans"
bandit -c pyproject.toml -r stateset_agents -f json -o "$BANDIT_REPORT_PATH" || true
# --save-json writes the JSON straight to a file; piping `--json` stdout to
# a file (the previous form here) captures safety's banner/deprecation
# notice ahead of the payload too, corrupting a naive json.loads() the same
# way `make security-scan-strict` hit before it was fixed. Match the
# Makefile's invocation exactly so both paths behave identically. Safety's
# parser does not recognize the environment-marker form of cuda-toolkit even
# though pip does, so omit that entry from Safety's normalized scan input.
grep -v '^cuda-toolkit\[' requirements-dev-lock.txt > "$SAFETY_INPUT_PATH"
safety check -r "$SAFETY_INPUT_PATH" --save-json "$SAFETY_REPORT_PATH" \
  --no-prompt > /dev/null 2>&1 || true
# Route both reports through check_security_findings.py's lenient parser
# (raw_decode from the first '{', ignoring any surrounding banner text)
# instead of a second, stricter inline copy of this same parsing logic --
# one implementation for both `make security-scan-strict` and this script.
"$PYTHON_BIN" scripts/check_security_findings.py

printf "\n[8/11] Building package...\n"
CURRENT_STEP="build"
if [ -d dist ]; then
  rm -rf dist
fi
"$PYTHON_BIN" -m build --no-isolation

printf "\n[9/11] Verifying built distribution metadata...\n"
CURRENT_STEP="twine_check"
"$PYTHON_BIN" -m twine check dist/*

printf "\n[10/11] Installing and importing the built wheel in isolation...\n"
CURRENT_STEP="isolated_wheel_smoke"
SMOKE_VENV="$(mktemp -d /tmp/stateset-wheel-smoke.XXXXXX)"
# Reuse the dependency environment that already ran the test/security gates,
# but install StateSet itself only from the freshly built wheel.  A completely
# empty --no-deps venv cannot import the API (Pydantic is a declared runtime
# dependency), while resolving dependencies again here would make this smoke
# network-dependent and test the package index rather than our distribution.
"$PYTHON_BIN" -m venv --system-site-packages "$SMOKE_VENV"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$SMOKE_VENV/bin/python" -m pip install \
  --no-index --no-deps dist/*.whl
SMOKE_SITE_PACKAGES="$("$SMOKE_VENV/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
DEPENDENCY_SITE_PACKAGES="$("$PYTHON_BIN" -c \
  'import site; print("\n".join(site.getsitepackages()))')"
printf '%s\n' "$DEPENDENCY_SITE_PACKAGES" > \
  "$SMOKE_SITE_PACKAGES/stateset-readiness-dependencies.pth"
EXPECTED_VERSION="$("$PYTHON_BIN" -c \
  'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
(
  cd /tmp
  "$SMOKE_VENV/bin/python" -c \
    "import sys; from pathlib import Path; import stateset_agents, stateset_agents.api; assert stateset_agents.__version__ == sys.argv[1]; assert Path(stateset_agents.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); print(stateset_agents.__version__)" \
    "$EXPECTED_VERSION"
)

printf "\n[11/11] Verifying working tree is clean...\n"
CURRENT_STEP="working_tree_clean"
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Commit before releasing."
  git status --short
  exit 1
fi

CURRENT_STEP="publish-readiness-complete"
printf "\nPublish readiness checks passed.\n"
