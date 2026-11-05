"""Design one 2B conclude neuron, then screen it on training-only practice data.

This pilot never generates final-test answers. Passing its fixed gate only
permits a later independent evaluation; it does not establish generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import neuron_job as nj  # noqa: E402
import pod_job as pj  # noqa: E402
import practice_job as practice  # noqa: E402
import thinking_budget_job as tb  # noqa: E402


def promotion_gate(rows: list[dict]) -> dict:
    """Require improvement over both base modes, ablation, and random control."""
    scores = {row["name"]: row for row in rows}
    if set(scores) != {"base", "wait", "candidate", "control"} or len(rows) != 4:
        raise ValueError("need exactly one base, wait, candidate, and control row")
    candidate, base = scores["candidate"], scores["base"]
    checks = {
        "beats_base_thinking": candidate["select_think_strict"]
        > base["select_think_strict"],
        "beats_base_nothinking": candidate["select_think_strict"]
        > base["select_nothink_strict"],
        "beats_wait_ablation": candidate["select_think_strict"]
        > scores["wait"]["select_think_strict"],
        "beats_matched_control": candidate["select_think_strict"]
        > scores["control"]["select_think_strict"],
        "preserves_nothinking_accuracy": candidate["select_nothink_strict"]
        >= base["select_nothink_strict"] - 0.01 - 1e-9,
        "limits_stray_closes": candidate["select_nothink_stray"]
        <= base["select_nothink_stray"] + 0.02 + 1e-9,
    }
    return {"eligible_for_final_test": all(checks.values()), "checks": checks}


def main(argv: list[str] | None = None) -> None:
    """Run the prespecified pilot and retain its complete evidence."""
    from stateset_agents.experimental.weight_space import EditSpec, TokenRowScale

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen3.5-2B"
    protocol = {
        "model": model,
        "revision": pj.REVISIONS[model],
        "design": {
            "gsm8k_train": 256,
            "trace_tokens": 3072,
            "nothink_train": 256,
            "prose_negatives": 200,
            "write_norm": 3.0,
            "gain": 4.0,
            "lambda": 100.0,
        },
        "practice": {
            "gsm8k_train": 200,
            "math_train": 200,
            "tokens": 8192,
            "conditions": ["base", "wait", "candidate", "control"],
            "thinking_modes": [True, False],
        },
        "final_test_generation": False,
        "gate": "strictly beat base in both modes, wait-only and random-write controls; "
        "thinking-off accuracy drop <= 1pp and stray-close rise <= 2pp",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    }
    (out / "pilot_protocol.json").write_text(json.dumps(protocol, indent=2))
    nj.main(
        [
            "--model",
            model,
            "--out",
            str(out),
            "--generate-traces",
            "--trace-tokens",
            "3072",
            "--nothink-train",
            "256",
            "--prose-negatives",
            "200",
            "--save-captures",
            "--design-only",
            "--tag",
            "_2b_pilot",
        ]
    )
    base_path, wait_path = out / "base_spec.json", out / "wait_spec.json"
    base_path.write_text(EditSpec(name="base-2b", edits=()).to_json())
    wait_path.write_text(
        EditSpec(
            name="wait-only-2b", edits=(TokenRowScale(ids=nj.WAIT_IDS, scale=0.5),)
        ).to_json()
    )
    practice_out = out / "practice"
    practice.main(
        [
            "--model",
            model,
            "--out",
            str(practice_out),
            "--practice-n",
            "100",
            "--practice-tokens",
            "8192",
            "--spec",
            f"base={base_path}",
            "--spec",
            f"wait={wait_path}",
            "--spec",
            f"candidate={out / 'conclude_neuron_2b_pilot_spec.json'}",
            "--spec",
            f"control={out / 'conclude_neuron_2b_pilot_control_spec.json'}",
        ]
    )
    summary = json.loads((practice_out / "practice_summary.json").read_text())
    generations = {
        (name, mode): nj.read_jsonl(practice_out / f"gen_pr_{name}_{mode}.jsonl.gz")
        for name in ("base", "wait", "candidate", "control")
        for mode in ("think", "nothink")
    }
    result = {
        "protocol": protocol,
        "rows": summary["rows"],
        "gate": promotion_gate(summary["rows"]),
        "description": {
            f"{name}_{mode}": tb.describe(rows, mode == "think")
            for (name, mode), rows in generations.items()
        },
        "candidate_vs_base_off": tb.compare(
            generations["base", "nothink"],
            generations["candidate", "think"],
            False,
            True,
        ),
        "candidate_vs_control": tb.compare(
            generations["control", "think"],
            generations["candidate", "think"],
            True,
            True,
        ),
    }
    (out / "pilot_summary.json").write_text(json.dumps(result, indent=2))
    pj.log(event="pilot_done", **result["gate"])


if __name__ == "__main__":
    main()
