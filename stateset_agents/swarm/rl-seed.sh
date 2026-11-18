cat > "$scratch/spec.json" <<EOF
{
  "run_id": "$task",
  "arm_id": "workflow-grpo",
  "seed": $parameter,
  "params": {"updates": 20},
  "workload": {"kind": "workflow_regression", "device": "cpu"}
}
EOF
python -m stateset_agents.fanout.worker --spec "$scratch/spec.json" \
  --output-dir "$scratch/output" --stage-dir "$scratch"
python - "$scratch/output/result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
metrics = result["metrics"]
if result["status"] != "succeeded" or metrics["optimizer_updates"] < 1 or metrics["reload_verified"] != 1:
    raise SystemExit("Workflow did not verify an optimizer update and checkpoint reload")
PY
cp "$scratch/output/result.json" "$out/result.json"
cp "$scratch/output/workflow/result.json" "$out/workflow.json"
