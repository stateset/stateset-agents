#!/usr/bin/env bash
# Smoke-test the GPU-free getting-started examples.
# Verifies a fresh `pip install stateset-agents` reaches a usable state.
#
# Usage:
#   examples/getting_started/smoke.sh
#
# Exits non-zero on the first failure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "==> Python: $($PYTHON --version)"
echo "==> stateset-agents version:"
"$PYTHON" -c "import stateset_agents; print('   ', stateset_agents.__version__)"
echo

echo "==> 01 hello_stub"
"$PYTHON" "$HERE/01_hello_stub.py"
echo

echo "==> 02 custom_reward"
"$PYTHON" "$HERE/02_custom_reward.py"
echo

echo "==> 04 llm_judge_eval --stub"
"$PYTHON" "$HERE/04_llm_judge_eval.py" --stub
echo

echo "==> 06 multi_turn_episode"
"$PYTHON" "$HERE/06_multi_turn_episode.py"
echo

echo "==> 07 tool_calling"
"$PYTHON" "$HERE/07_tool_calling.py"
echo

echo "==> 08 eval_driven_loop"
"$PYTHON" "$HERE/08_eval_driven_loop.py"
echo

echo "==> 09 curate_dataset"
"$PYTHON" "$HERE/09_curate_dataset.py"
echo

echo "==> 10 scenario_testing"
"$PYTHON" "$HERE/10_scenario_testing.py"
echo

echo "✓ All GPU-free getting-started examples passed."
echo
echo "GPU examples (not run here):"
echo "  03_first_finetune.py     — needs A100, ~10 min, real training"
echo "  04_llm_judge_eval.py     — needs A100, ~3 min, loads Qwen2.5-1.5B-Instruct"
echo "  05_serve_agent.py        — interactive (starts a FastAPI server)"
