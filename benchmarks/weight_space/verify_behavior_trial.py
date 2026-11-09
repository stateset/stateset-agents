"""Reload the pinned LM and independently replay every retained edit and gate.

No session, editor, or experiment implementation is imported. The verifier uses
direct model forwards and its own probability/loss calculations. It confirms
the declared layer target, full-model effects and decisions, not semantic quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decision(metrics: dict, limits: dict) -> tuple[bool, list[str]]:
    """Recompute the numerical acceptance policy from recorded limits."""
    failed = []
    for metric, limit in (
        ("mean_kl", "max_mean_kl"),
        ("max_token_kl", "max_token_kl"),
        ("nll_increase", "max_nll_increase"),
    ):
        if not math.isfinite(metrics[metric]) or metrics[metric] > limits[limit]:
            failed.append(metric)
    if metrics["top1_agreement"] < limits["min_top1_agreement"]:
        failed.append("top1_agreement")
    if (
        limits["min_target_gain"] is not None
        and metrics["mean_target_gain"] < limits["min_target_gain"]
    ):
        failed.append("mean_target_gain")
    if (
        limits["min_each_target_gain"] is not None
        and min(metrics["target_gains"]) < limits["min_each_target_gain"]
    ):
        failed.append("each_target_gain")
    return not failed, failed


def compare(tokens, before, after, target_before, target_after) -> dict:
    """Scalar-position oracle; no aggregation helpers from the implementation."""
    kl, agreement, loss0, loss1 = [], [], [], []
    for ids, raw0, raw1 in zip(tokens, before, after, strict=True):
        for position, (a, b) in enumerate(zip(raw0, raw1, strict=True)):
            lp, lq = a.double().log_softmax(0), b.double().log_softmax(0)
            kl.append(max(0.0, float(torch.dot(lp.exp(), lp - lq))))
            agreement.append(int(a.argmax() == b.argmax()))
            if position + 1 < len(ids):
                loss0.append(-float(lp[ids[position + 1]]))
                loss1.append(-float(lq[ids[position + 1]]))
    gain = target_after - target_before
    return {
        "locality_tokens": len(kl),
        "observed_tokens": len(loss0),
        "mean_kl": sum(kl) / len(kl),
        "max_token_kl": max(kl),
        "top1_agreement": sum(agreement) / len(agreement),
        "base_nll": sum(loss0) / len(loss0),
        "edited_nll": sum(loss1) / len(loss1),
        "nll_increase": (sum(loss1) - sum(loss0)) / len(loss0),
        "target_gains": [gain],
        "mean_target_gain": gain,
    }


@torch.inference_mode()
def verify(root: Path, model_path: Path) -> dict:
    """Replay all four edits from identical base weights with the pinned model."""
    torch.set_num_threads(4)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    required = {
        "protocol.json",
        "trials.json",
        "summary.json",
        "token_ids.json",
        "edits.safetensors",
    }
    require(required <= manifest.keys(), "manifest omits required artifacts")
    for name, expected in manifest.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "artifact escapes root")
        require(digest(path) == expected, f"hash mismatch: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    require(protocol["kind"] == "stateset-whole-model-trial-v1", "wrong protocol")
    require(
        protocol["revision"]
        == model_path.name
        == "c1899de289a04d12100db370d81485cdf75e47ca",
        "wrong checkpoint revision",
    )
    require(protocol["module"] == "model.layers.27.mlp.down_proj", "wrong module")
    require(
        protocol["arms"]
        == [
            ["sham", 0.0, 100.0],
            ["unweighted", 0.5, 0.0],
            ["weighted", 0.5, 100.0],
            ["large_weighted", 8.0, 100.0],
        ],
        "condition set changed",
    )
    require(
        protocol["limits"]
        == {
            "max_mean_kl": 0.001,
            "max_token_kl": 0.01,
            "max_nll_increase": 0.005,
            "min_top1_agreement": 0.95,
            "min_target_gain": 0.01,
            "min_each_target_gain": 0.01,
        },
        "acceptance limits changed",
    )
    for name, expected in protocol["checkpoint_hashes"].items():
        require(
            digest(model_path / name) == expected, f"model/tokenizer mismatch: {name}"
        )
    for name, expected in protocol["sources"].items():
        require(
            manifest.get(f"sources/{name}") == expected, "source not bound to protocol"
        )
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - local snapshot directory, local_files_only
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - local snapshot directory, local_files_only
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    encoded = {
        name: [
            list(tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in protocol[name]
        ]
        for name in ("target", "protected", "calibration", "validation", "confirmation")
    }
    require(
        encoded == json.loads((root / "token_ids.json").read_text()),
        "tokenization changed",
    )
    seen = set()
    for name in ("protected", "calibration", "validation", "confirmation"):
        cases = set(map(tuple, encoded[name]))
        require(
            len(cases) == len(encoded[name]) and cases.isdisjoint(seen),
            "evaluation cases overlap",
        )
        seen |= cases
    layer = model.get_submodule(protocol["module"])
    original = layer.weight.detach().clone()
    tensors = load_file(str(root / "edits.safetensors"))

    def forward(ids):
        tokens = torch.tensor([ids])
        logits = (
            model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
            )
            .logits[0]
            .float()
            .cpu()
        )
        require(bool(torch.isfinite(logits).all()), "nonfinite replay logits")
        return logits

    def target_score():
        p, c = encoded["target"]
        logits = forward(p + c)
        return sum(
            float(logits[len(p) + i - 1].double().log_softmax(0)[token])
            for i, token in enumerate(c)
        ) / len(c)

    def capture(ids):
        seen = []
        handle = layer.register_forward_pre_hook(
            lambda _, args: seen.append(args[0].detach().clone())
        )
        try:
            x = torch.tensor([ids])
            model(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                logits_to_keep=1,
                use_cache=False,
            )
        finally:
            handle.remove()
        require(len(seen) == 1, "unexpected layer invocation count")
        return seen[0][0]

    key = capture(encoded["target"][0])[-1:]
    require(torch.equal(key, tensors["key"]), "captured edit key mismatch")
    for name in ("protected", "calibration"):
        rows = torch.cat([capture(ids) for ids in encoded[name]])
        require(torch.equal(rows, tensors[name]), f"captured {name} inputs mismatch")
    direction = (
        model.get_output_embeddings().weight[encoded["target"][1][0]]
        * model.model.norm.weight
    )
    direction /= direction.norm()
    require(torch.equal(direction, tensors["direction"]), "write direction mismatch")
    require(
        torch.equal(key @ original.T, tensors["baseline_output"]),
        "base layer output mismatch",
    )

    before = {
        name: [forward(ids) for ids in encoded[name]]
        for name in ("validation", "confirmation")
    }
    target_before = target_score()
    trials = json.loads((root / "trials.json").read_text())
    require(
        [t["arm"] for t in trials] == [a[0] for a in protocol["arms"]],
        "trial coverage changed",
    )
    decisions = []
    for trial, (name, scale, _) in zip(trials, protocol["arms"], strict=True):
        update = tensors[f"{name}.B"] @ tensors[f"{name}.A"]
        require(
            update.shape == original.shape and bool(torch.isfinite(update).all()),
            "invalid edit",
        )
        if name == "sham":
            require(bool((update == 0).all()), "sham is not a zero update")
        layer.weight.copy_(original + update)
        desired = tensors["baseline_output"] + scale * tensors["direction"]
        require(
            torch.allclose(
                tensors["key"] @ layer.weight.T, desired, atol=0.002, rtol=1e-5
            ),
            "layer target failed",
        )
        target_after = target_score()
        for split in ("validation", "confirmation"):
            actual = compare(
                encoded[split],
                before[split],
                [forward(ids) for ids in encoded[split]],
                target_before,
                target_after,
            )
            for metric, value in actual.items():
                recorded = trial[split]["metrics"][metric]
                if metric == "target_gains":
                    value, recorded = value[0], recorded[0]
                require(
                    math.isclose(value, recorded, rel_tol=1e-5, abs_tol=1e-7),
                    f"metric mismatch: {name}/{split}/{metric}",
                )
            accepted, violations = decision(actual, trial[split]["limits"])
            require(
                accepted == trial[split]["accepted"]
                and violations == trial[split]["violations"],
                "acceptance decision mismatch",
            )
            if split == "validation":
                require(
                    trial[split]["limits"] == protocol["limits"],
                    "trial limits differ from protocol",
                )
                decisions.append(
                    {"arm": name, "accepted": accepted, "violations": violations}
                )
        layer.weight.copy_(original)
        require(
            torch.equal(layer.weight, original) and trial["rollback_exact"],
            "rollback mismatch",
        )
        LOGGER.info("replayed %s", name)
    summary = json.loads((root / "summary.json").read_text())
    require(
        summary["trials"] == trials and summary["all_rollbacks_exact"],
        "summary mismatch",
    )
    require(
        summary["budget"]["used"] == 19 and len(summary["budget"]["ledger"]) == 19,
        "measurement budget mismatch",
    )
    return {
        "passed": True,
        "trials": 4,
        "model_reloaded": protocol["model"],
        "decisions": decisions,
        "scope": "full-vocabulary token replay on declared prompts; no semantic-quality claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = verify(args.root, args.model)
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s", result)
