"""Audit a completed 2B pilot from saved replies, edit specs, and source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import neuron_job as nj
import onpolicy_job as op
import pod_job as pj
import thinking_budget_job as tb
from conclude_pilot_job import promotion_gate

from stateset_agents.experimental.weight_space import (
    EditSpec,
    NeuronRewrite,
    clean_finish,
)


def require(condition: bool, message: str) -> None:
    """Fail an evidence check even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Regrade every reply and reproduce the frozen selection decision."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="downloaded /workspace/out directory")
    a = ap.parse_args()
    root = a.root
    report = json.loads((root / "pilot_summary.json").read_text())
    protocol = json.loads((root / "pilot_protocol.json").read_text())
    versions = json.loads((root / "runtime_versions.json").read_text())
    for name in (
        "math-verify",
        "latex2sympy2_extended",
        "antlr4-python3-runtime",
        "sympy",
        "mpmath",
    ):
        require(version(name) == versions[name], f"grader version differs: {name}")
    require(report["protocol"] == protocol, "summary protocol differs")
    for name, digest in protocol["source_sha256"].items():
        source = Path(__file__).parent / name
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == digest,
            f"analysis source differs from executed source: {name}",
        )
    model = protocol["model"]
    require(protocol["revision"] == pj.REVISIONS[model], "model revision differs")
    design = nj.read_jsonl(root / "gen_base_trace_2b_pilot.jsonl.gz")
    nj.validate_design_source(design, model)
    design_ids = {r["id"] for r in design}
    require(len(design_ids) == 256, "design trace count differs")
    negatives = nj.read_jsonl(root / "gen_nothink_train_2b_pilot.jsonl.gz")
    nj.validate_design_source(negatives, model)
    require(
        {r["id"] for r in negatives} == design_ids and len(negatives) == 256,
        "thinking-off design traces use different problems",
    )
    require(all(not r["thinking"] for r in negatives), "wrong negative-trace mode")
    filenames = {
        "base": "base_spec.json",
        "wait": "wait_spec.json",
        "candidate": "conclude_neuron_2b_pilot_spec.json",
        "control": "conclude_neuron_2b_pilot_control_spec.json",
    }
    specs = {
        name: EditSpec.from_json((root / filename).read_text())
        for name, filename in filenames.items()
    }
    require(not specs["base"].edits, "base spec contains edits")
    candidate, control = specs["candidate"].edits, specs["control"].edits
    require(len(candidate) == len(control) == 2, "unexpected edit count")
    require(
        candidate[0] == control[0] and specs["wait"].edits == (candidate[0],),
        "token-only ablation or random control differs",
    )
    rewrite, random_write = candidate[1], control[1]
    require(
        isinstance(rewrite, NeuronRewrite) and isinstance(random_write, NeuronRewrite),
        "need neuron rewrites",
    )
    # Runtime narrowing is explicit so the check remains active under -O.
    if not isinstance(rewrite, NeuronRewrite) or not isinstance(
        random_write, NeuronRewrite
    ):
        raise ValueError("need neuron rewrites")
    require(
        (rewrite.layer, rewrite.neuron, rewrite.gate, rewrite.up)
        == (
            random_write.layer,
            random_write.neuron,
            random_write.gate,
            random_write.up,
        ),
        "random control changes the read side",
    )
    require(
        math.isclose(
            math.hypot(*rewrite.down), math.hypot(*random_write.down), rel_tol=1e-6
        ),
        "random write norm differs",
    )
    require(rewrite.down != random_write.down, "random control equals candidate")
    generations = {}
    lenient_discrepancies = []
    lenient_regrade_accuracy = {}
    for name in filenames:
        for mode in ("think", "nothink"):
            rows = nj.read_jsonl(root / "practice" / f"gen_pr_{name}_{mode}.jsonl.gz")
            identities = [(r["id"], r["kind"], r["gold"]) for r in rows]
            ids = {r["id"] for r in rows}
            require(
                len(rows) == len(ids) == 400, f"wrong practice count: {name}/{mode}"
            )
            require(not ids & design_ids, "design/practice overlap")
            require(
                all(i.startswith(("gsm8k-train-", "mathtrain-")) for i in ids),
                "practice contains a non-training ID",
            )
            if generations:
                reference = generations["base", "think"]
                require(
                    identities == [(r["id"], r["kind"], r["gold"]) for r in reference],
                    "conditions use different problems or gold answers",
                )
            if name != "base":
                require(
                    [r["prompt_ids"] for r in rows]
                    == [r["prompt_ids"] for r in generations["base", mode]],
                    "conditions use different prompts",
                )
            regraded_correct = 0
            for row in rows:
                require(row["thinking"] == (mode == "think"), "thinking mode differs")
                require(
                    row["tokens"] == len(row["gen_ids"]) <= 8192, "token cap differs"
                )
                graded = pj.grade(row, thinking=mode == "think")
                regraded_correct += bool(graded["correct"])
                clean = clean_finish(row, thinking=mode == "think")
                if graded["correct"] != row["correct"]:
                    lenient_discrepancies.append(
                        {
                            "condition": f"{name}_{mode}",
                            "id": row["id"],
                            "finish": row["finish"],
                            "tokens": row["tokens"],
                            "clean_finish": clean,
                            "saved_correct": row["correct"],
                            "regraded_correct": graded["correct"],
                        }
                    )
                require(
                    (bool(graded["correct"]) and clean)
                    == (bool(row["correct"]) and clean),
                    f"strict answer regrading differs: {name}/{mode}/{row['id']}",
                )
            lenient_regrade_accuracy[f"{name}_{mode}"] = regraded_correct / len(rows)
            generations[name, mode] = rows
    recomputed = []
    by_name = {r["name"]: r for r in report["rows"]}
    for name in filenames:
        scores = op.select_scores(
            generations[name, "think"], generations[name, "nothink"]
        )
        require(
            all(by_name[name][key] == value for key, value in scores.items()),
            f"aggregate scores differ: {name}",
        )
        require(
            by_name[name]["spec_hash"] == specs[name].content_hash(),
            "spec hash differs",
        )
        recomputed.append({"name": name, **scores})
        for mode in ("think", "nothink"):
            require(
                tb.describe(generations[name, mode], mode == "think")
                == report["description"][f"{name}_{mode}"],
                "budget description differs",
            )
    require(promotion_gate(recomputed) == report["gate"], "promotion decision differs")
    for field, name, mode in (
        ("candidate_vs_base_off", "base", "nothink"),
        ("candidate_vs_control", "control", "think"),
    ):
        paired = tb.compare(
            generations[name, mode],
            generations["candidate", "think"],
            mode == "think",
            True,
        )
        # JSON encodes the bootstrap interval's tuple as a list.
        require(
            json.loads(json.dumps(paired)) == report[field], "paired statistics differ"
        )
    result = {
        "practice_problems": 400,
        "conditions": 8,
        "regraded_replies": 3200,
        "all_sources_match": True,
        "all_spec_hashes_match": True,
        "control_read_vectors_and_write_norm_match": True,
        "design_practice_ids_disjoint": True,
        "all_practice_ids_from_training": True,
        "all_conditions_have_same_problems_and_prompts": True,
        "all_lenient_answer_grades_reproduced": not lenient_discrepancies,
        "lenient_grade_discrepancies": lenient_discrepancies,
        "lenient_regrade_accuracy": lenient_regrade_accuracy,
        "all_strict_answer_grades_reproduced": True,
        "all_recorded_aggregates_reproduced": True,
        "paired_statistics_reproduced_from_recorded_grades": True,
        "promotion_reproduced": True,
        "gate": report["gate"],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
