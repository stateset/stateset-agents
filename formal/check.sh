#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tla_jar="${1:-${TLA2TOOLS_JAR:-}}"
with_python="${2:-}"

if [[ -z "$tla_jar" || ! -f "$tla_jar" || $# -gt 2 ||
      ( -n "$with_python" && "$with_python" != "--with-python" ) ]]; then
    echo "Usage: formal/check.sh /path/to/tla2tools.jar [--with-python]" >&2
    exit 2
fi

command -v lean >/dev/null
command -v java >/dev/null

lean "$repo_root/formal/lean/Objective.lean"
lean "$repo_root/formal/lean/Reward.lean"

run_tlc() {
    local model="$1"
    local config="$2"
    local output
    output="$(mktemp)"
    if ! java -XX:+UseParallelGC -cp "$tla_jar" tlc2.TLC -deadlock \
        -metadir "$(mktemp -d)" -config "$config" "$model" >"$output" 2>&1; then
        cat "$output" >&2
        return 1
    fi
    if command -v rg >/dev/null 2>&1; then
        rg 'Model checking completed|distinct states found' "$output"
    else
        grep -E 'Model checking completed|distinct states found' "$output"
    fi
}

cd "$repo_root/formal/tla"
run_tlc RolloutControl.tla RolloutControl.cfg
run_tlc AsyncRuntime.tla AsyncRuntime.cfg
run_tlc AutoResearch.tla AutoResearch.cfg
run_tlc AutoResearch.tla AutoResearchMin.cfg
run_tlc CheckpointSwap.tla CheckpointSwap.cfg
run_tlc ResearchCommit.tla ResearchCommit.cfg

if [[ "$with_python" == "--with-python" ]]; then
    cd "$repo_root"
    python_bin="${PYTHON:-python3}"
    "$python_bin" -m pytest -q -o addopts='' \
        tests/unit/test_formal_rollout_refinement.py \
        tests/unit/test_async_rollouts.py \
        tests/unit/test_distributed_rollouts.py \
        tests/unit/test_policy_artifacts.py \
        tests/unit/test_async_runtime.py \
        tests/unit/test_objectives_properties.py \
        tests/unit/test_auto_research.py \
        tests/unit/test_formal_numeric_refinement.py
fi
