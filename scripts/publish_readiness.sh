#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR"
BANDIT_REPORT_PATH="$REPORT_DIR/bandit-report.json"
SAFETY_REPORT_PATH="$REPORT_DIR/safety-report.json"
SUMMARY_PATH="$REPORT_DIR/publish-readiness-summary.json"
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

status = sys.argv[1]
failed_step = sys.argv[2] if status != "passed" else None
start_time = int(sys.argv[3])
end_time = int(sys.argv[4])
summary_path = sys.argv[5]
git_sha = sys.argv[6]
git_ref = sys.argv[7]
git_branch = sys.argv[8]

summary = {
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
    write_summary "$?" "$CURRENT_STEP"
}
trap on_exit EXIT

CURRENT_STEP="preflight"
READINESS_FAILURE_DETAIL=""

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    READINESS_FAILURE_DETAIL="missing python interpreter (python3 or python)"
    echo "ERROR: Neither python3 nor python is available on PATH."
    exit 1
  fi
fi

REQUIRED_COMMANDS=(
  "$PYTHON_BIN"
  "git"
  "ruff"
  "black"
  "isort"
  "mypy"
  "pytest"
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

if ! "$PYTHON_BIN" -c "import build, twine" >/dev/null 2>&1; then
  READINESS_FAILURE_DETAIL="missing_python_modules: build or twine"
  echo "ERROR: Required Python modules are missing: build, twine"
  echo "Install dependencies and retry: pip install build twine"
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

printf "\n[1/8] Running linters...\n"
CURRENT_STEP="linters"
ruff check .
black --check .
isort --check-only .

printf "\n[2/8] Running type checks...\n"
CURRENT_STEP="type_checks"
mypy --config-file mypy.ini

printf "\n[3/8] Running tests with coverage gate (70%)...\n"
CURRENT_STEP="tests_with_coverage"
pytest --cov=stateset_agents --cov-report=xml --cov-fail-under=70

printf "\n[4/8] Running security scans...\n"
CURRENT_STEP="security_scans"
bandit -r stateset_agents -f json -o "$BANDIT_REPORT_PATH" || true
"$PYTHON_BIN" - <<PY
import json
import traceback
import sys
from pathlib import Path

path = Path("$BANDIT_REPORT_PATH")
if not path.exists() or not path.read_text().strip():
    print('Bandit report not generated')
    sys.exit(1)

try:
    report = json.loads(path.read_text())
except json.JSONDecodeError as exc:
    print(f"Bandit report JSON decode failed: {exc}")
    traceback.print_exc()
    sys.exit(1)

results = []
if isinstance(report, dict):
    results = report.get('results', [])
elif isinstance(report, list):
    results = report
else:
    print(f"Unexpected Bandit report type: {type(report).__name__}")
    sys.exit(1)
high_findings = [
    item
    for item in results
    if str(item.get('issue_severity', '')).upper() in {'MEDIUM', 'HIGH', 'CRITICAL'}
]
if high_findings:
    print(f'Bandit: failing due to {len(high_findings)} medium/high/critical findings')
    for finding in high_findings[:5]:
        print(f" - {finding.get('filename')}:{finding.get('line_number')} "
              f"{finding.get('test_id')} {finding.get('issue_text')}")
    sys.exit(1)
print('Bandit: no medium/high/critical findings')
PY
safety check --json > "$SAFETY_REPORT_PATH" || true
"$PYTHON_BIN" - <<PY
import json
import traceback
import sys
from pathlib import Path

path = Path("$SAFETY_REPORT_PATH")
if not path.exists() or not path.read_text().strip():
    print("Safety report not generated")
    sys.exit(1)
try:
    payload = json.loads(path.read_text())
except json.JSONDecodeError as exc:
    print(f'Safety JSON decode failed: {exc}')
    traceback.print_exc()
    sys.exit(1)

vulnerabilities = (
    payload.get('vulnerabilities')
    if isinstance(payload, dict)
    else payload
)
if not vulnerabilities:
    vulnerabilities = []
high_vulns = [
    item
    for item in vulnerabilities
    if str(item.get('severity', '')).upper() in {'HIGH', 'CRITICAL'}
]
if high_vulns:
    print(f'Safety: failing due to {len(high_vulns)} high/critical vulnerabilities')
    for vuln in high_vulns[:5]:
        pkg = vuln.get('package_name') or vuln.get('package')
        vuln_id = vuln.get('id') or vuln.get('cve') or vuln.get('vulnerability_id')
        print(f' - {pkg}: {vuln_id}')
    sys.exit(1)
print('Safety: no high/critical vulnerabilities reported')
PY

printf "\n[5/8] Building package...\n"
CURRENT_STEP="build"
if [ -d dist ]; then
  rm -rf dist
fi
"$PYTHON_BIN" -m build --no-isolation

printf "\n[6/8] Verifying built distribution metadata...\n"
CURRENT_STEP="twine_check"
"$PYTHON_BIN" -m twine check dist/*

printf "\n[7/8] Running package smoke test...\n"
CURRENT_STEP="smoke_test"
"$PYTHON_BIN" -c "import stateset_agents, stateset_agents.api; print(stateset_agents.__version__)"

printf "\n[8/8] Verifying working tree is clean...\n"
CURRENT_STEP="working_tree_clean"
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Commit before releasing."
  git status --short
  exit 1
fi

CURRENT_STEP="publish-readiness-complete"
printf "\nPublish readiness checks passed.\n"
