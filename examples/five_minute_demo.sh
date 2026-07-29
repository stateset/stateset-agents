#!/usr/bin/env bash
# The five-minute win: pip install -> ingest -> improve -> curated training set.
#
# Runs fully offline, no GPU, no API key. Simulates a user who already has
# conversation logs (OpenAI chat-completions format) from an agent they
# built elsewhere and wants to grade + curate them with stateset-agents.
#
# Usage:
#   examples/five_minute_demo.sh
#
# Requires only: pip install stateset-agents   (no extras)
# Exits non-zero on the first failure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "==> Python: $($PYTHON --version)"
echo "==> stateset-agents version:"
"$PYTHON" -c "import stateset_agents; print('   ', stateset_agents.__version__)"
echo

WORKDIR="$(mktemp -d -t five-minute-demo.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

LOGS="$WORKDIR/logs.jsonl"
TRANSCRIPTS_DIR="$WORKDIR/transcripts"
IMPROVED_DIR="$WORKDIR/improved"

echo "==> Working directory: $WORKDIR"
echo

echo "==> 1/4  Writing 3 sample conversation logs (OpenAI chat-completions format)"
echo "         one weak reply, two strong replies — customer-support flavored."

# Line 1: weak — terse, no politeness, no acknowledgement of the order/refund.
# Lines 2-3: strong — polite, on-topic, well within the 10-120 word brand-voice band.
cat > "$LOGS" <<'JSONL'
{"messages": [{"role": "user", "content": "My order hasn't arrived yet, can you help?"}, {"role": "assistant", "content": "No."}]}
{"messages": [{"role": "user", "content": "I need a refund for my order, it arrived damaged."}, {"role": "assistant", "content": "I'm sorry to hear your order arrived damaged — thank you for letting us know. I'd be glad to help with your refund right away. Please share your order number and I'll process it for you today, no need to send the item back first."}]}
{"messages": [{"role": "user", "content": "Can you check on the status of my order? It's been a week."}, {"role": "assistant", "content": "Of course, happy to help! I understand waiting a week is frustrating. Please share your order number and I'll look up the latest shipping status for you right now, and follow up with tracking details."}]}
JSONL

echo "         wrote $LOGS"
echo

echo "==> 2/4  stateset-agents ingest --format openai"
"$PYTHON" -m stateset_agents.cli ingest \
  --format openai \
  --input "$LOGS" \
  --output "$TRANSCRIPTS_DIR"
echo

echo "==> 3/4  stateset-agents improve run --reward customer_support"
"$PYTHON" -m stateset_agents.cli improve run \
  --transcripts "$TRANSCRIPTS_DIR" \
  --reward customer_support \
  --output "$IMPROVED_DIR"
echo

echo "==> 4/4  Summary"
"$PYTHON" -c "
import json
from pathlib import Path

improved = Path('$IMPROVED_DIR')
summary = json.loads((improved / 'improve_summary.json').read_text())
print(f\"   transcripts graded : {summary['transcript_count']}\")
print(f\"   assistant turns    : {summary['assistant_turn_count']}\")
print(f\"   mean score         : {summary['mean_score']:.3f}\")
print(f\"   threshold          : {summary['threshold']}\")
print(f\"   curated examples   : {summary['curated_count']}\")
print()
print('   per-transcript scores:')
for t in summary['transcripts']:
    print(f\"     {t['name']:<20} mean={t['mean_score']:.3f}  above_threshold={t['above_threshold']}\")
"
echo

echo "==> Artifacts:"
echo "    curated dataset : $IMPROVED_DIR/curated.jsonl"
echo "    full summary    : $IMPROVED_DIR/improve_summary.json"
echo "    what's next     : $IMPROVED_DIR/next_steps.md"
echo
echo "    (this run used a temp dir that is deleted on exit; re-run any of the"
echo "     three commands above with --output pointing somewhere permanent to"
echo "     keep the artifacts around)"
echo

echo "✓ Five-minute win complete: pip install -> graded report -> curated training set."
echo
echo "Next: wire this up as an MCP server so any MCP client (Claude Code, Claude"
echo "Desktop, etc.) can grade/curate/retrain for you:"
echo
echo "    claude mcp add stateset-agents -- stateset-agents mcp"
echo
echo "See docs/MCP_SERVER.md for the full tool list."
