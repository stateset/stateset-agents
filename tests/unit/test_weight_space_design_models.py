"""Model-specific conclude-neuron designs and training-only contrast boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

import neuron_job as nj  # noqa: E402
import offline_design as od  # noqa: E402
import pod_job as pj  # noqa: E402
from build_answer_contrast import THINK_END, build_contrast  # noqa: E402


class CharacterTokenizer:
    """Small deterministic tokenizer with the real thinking-close sentinel."""

    def __len__(self) -> int:
        return 128

    def decode(self, ids: list[int]) -> str:
        return "".join("</think>" if i == THINK_END else chr(i) for i in ids)

    def encode(self, text: str) -> list[int]:
        parts = text.split("</think>")
        ids: list[int] = []
        for i, part in enumerate(parts):
            if i:
                ids.append(THINK_END)
            ids.extend(map(ord, part))
        return ids


def trace(text: str, index: int = 0, **extra) -> dict:
    return {
        "id": f"gsm8k-train-{index}",
        "gold": "42",
        "prompt_ids": [80],
        "gen_ids": CharacterTokenizer().encode(text),
        "text": text,
        **extra,
    }


def training(n: int = 1) -> list[dict]:
    return [
        {"id": f"gsm8k-train-{i}", "gold": "42", "question": "What is six times seven?"}
        for i in range(n)
    ]


PREFIX = "Consider the calculation carefully.\nWork through the steps.\n"


def test_contrast_uses_only_answers_before_close() -> None:
    rows = [
        trace(PREFIX + "Result = 42\nCheck.\n</think>Answer is 42.", 0),
        trace(PREFIX + "</think>The answer is 42.\n", 1),
        trace(PREFIX + "Result = 42.5\n", 2),
    ]
    result = build_contrast(CharacterTokenizer(), rows, training(3), rows)
    assert [m["id"] for m in result["meta"]] == ["gsm8k-train-0"]
    assert THINK_END not in result["answer"][0]
    assert result["post"] and all(THINK_END in p for p in result["post"])


@pytest.mark.parametrize("post", [False, True])
def test_contrast_rejects_heldout_rows_even_for_post_close_negatives(
    post: bool,
) -> None:
    row = trace(PREFIX + "Result = 42\n", id="gsm8k-test-0")
    with pytest.raises(ValueError, match="training split"):
        build_contrast(
            CharacterTokenizer(),
            [] if post else [row],
            training(),
            [row] if post else [],
        )


def test_contrast_rejects_changed_gold_and_duplicate_sources() -> None:
    row = trace(PREFIX + "Result = 42\n")
    with pytest.raises(ValueError, match="gold differs"):
        build_contrast(CharacterTokenizer(), [{**row, "gold": "43"}], training())
    with pytest.raises(ValueError, match="duplicate"):
        build_contrast(CharacterTokenizer(), [row, row], training())


def test_contrast_excludes_an_answer_already_in_the_question() -> None:
    rows = [trace(PREFIX + "Result = 42\n")]
    problems = [{**training()[0], "question": "What is 42 plus zero?"}]
    assert not build_contrast(CharacterTokenizer(), rows, problems)["answer"]


@pytest.mark.parametrize("extra", [{"thinking": False}, {"prompt_ids": [THINK_END]}])
def test_thinking_off_replies_cannot_supply_positives(extra: dict) -> None:
    with pytest.raises(ValueError, match="thinking enabled"):
        build_contrast(
            CharacterTokenizer(), [trace(PREFIX + "Result = 42\n", **extra)], training()
        )


def test_newline_positives_do_not_cross_the_thinking_boundary() -> None:
    tok = CharacterTokenizer()
    rows = [trace(PREFIX + "Result = 42\n</think>Answer.\nNext line.\n")]
    contrast = build_contrast(tok, rows, training())
    prefixes, labels, groups = nj.contrast_sets(tok, contrast, rows)
    positives = [p for p, y in zip(prefixes, labels, strict=True) if y > 0]
    assert len(positives) == 1
    assert THINK_END not in positives[0]
    assert groups.count("answer_newline") == 1

    mpos, _, _ = nj.math_contrast(tok, rows, {ord("\n")})
    assert mpos == positives


def test_old_answer_after_close_metadata_cannot_make_positive_contexts() -> None:
    row = trace(PREFIX + "</think>Answer = 42\nLater.\n")
    contrast = {
        "meta": [{"id": row["id"], "answer_token": len(row["gen_ids"]) - 8}],
        "post": [],
        "answer": [],
    }
    assert nj.contrast_sets(CharacterTokenizer(), contrast, [row]) == ([], [], [])


@pytest.mark.parametrize("model", pj.CONCLUDE_MODELS)
def test_design_source_checks_exact_model_and_revision(model: str) -> None:
    identity = {"model": model, "revision": pj.REVISIONS[model]}
    nj.validate_design_source([identity], model)
    for wrong in ({**identity, "revision": "main"}, {**identity, "model": "other"}):
        with pytest.raises(ValueError, match="design input must identify"):
            nj.validate_design_source([wrong], model)


def test_legacy_design_inputs_are_only_accepted_for_original_model() -> None:
    nj.validate_design_source([{}], nj.MODEL)
    with pytest.raises(ValueError, match="design input must identify"):
        nj.validate_design_source([{}], "Qwen/Qwen3.5-2B")


def test_generation_retains_model_provenance_for_new_design_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_name = "Qwen/Qwen3.5-2B"
    data, out = tmp_path / "input.jsonl", tmp_path / "output.jsonl.gz"
    pj.write_jsonl(data, [{**training()[0], "kind": "gsm8k"}])
    tok = SimpleNamespace(apply_chat_template=lambda *args, **kw: "prompt")
    configs = []

    class FakeLLM:
        def __init__(self, **kwargs):
            configs.append(kwargs)

        def generate(self, prompts, params):
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[2, 3],
                            text="</think>\\boxed{42}",
                            finish_reason="stop",
                        )
                    ],
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kw: tok)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw),
    )
    monkeypatch.setattr(pj, "_kill_children", lambda: None)
    monkeypatch.setattr(pj.os, "_exit", lambda code: None)
    pj.cmd_generate(
        argparse.Namespace(
            model=model_name,
            data=str(data),
            out=str(out),
            label="test",
            max_tokens=32,
            no_thinking=False,
        )
    )
    rows = pj.read_jsonl(out)
    nj.validate_design_source(rows, model_name)
    assert rows[0]["thinking"] is True and rows[0]["correct"] is True
    assert configs[0]["revision"] == pj.REVISIONS[model_name]


@pytest.mark.parametrize("model_name", pj.CONCLUDE_MODELS)
@pytest.mark.parametrize("design_only", [True, False])
def test_design_only_uses_selected_model_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, design_only: bool
) -> None:
    """Exercise real fitting/serialization with small tensors and fake provider I/O."""
    import torch

    from stateset_agents.experimental import weight_space as ws
    from stateset_agents.experimental.weight_space import checkpoints

    monkeypatch.setattr(pj, "WORK", tmp_path / "work")
    monkeypatch.setattr(pj, "MODEL", pj.MODEL)  # restore the CLI's global selection
    problems = {
        "trace": training(4),
        "select": training(2),
        "heldout": [{**r, "id": f"gsm8k-test-{i}"} for i, r in enumerate(training(2))],
    }
    monkeypatch.setattr(pj, "load_problems", lambda *args: problems)
    tok = CharacterTokenizer()
    identity = {"model": model_name, "revision": pj.REVISIONS[model_name]}
    calls = []

    def generate(label, model, rows, max_tokens, out, thinking=True):
        design_trace = label.startswith(("base_trace", "nothink_train"))
        if design_trace or "_base" in label:
            assert model == model_name
        else:
            assert model.startswith(str(pj.WORK / "ckpt" / "spec_"))
        if design_trace:
            assert rows and all(r in problems["trace"] for r in rows)
        calls.append((label, thinking))
        result = [
            trace(
                PREFIX + "Result = 42\nCheck the result.\n",
                i,
                **identity,
                id=r["id"],
                thinking=thinking,
                kind="gsm8k",
                correct=True,
                finish="stop",
            )
            for i, r in enumerate(rows)
        ]
        if not design_trace:
            for row in result:
                row["text"] += "</think>42"
        pj.write_jsonl(out / f"gen_{label}.jsonl.gz", result)
        return result

    monkeypatch.setattr(pj, "generate", generate)
    hidden, inter = (6, 10) if model_name == nj.MODEL else (8, 12)
    generator = torch.Generator().manual_seed(32)
    weights = {
        "gate_proj": torch.randn(inter, hidden, generator=generator),
        "up_proj": torch.randn(inter, hidden, generator=generator),
        "down_proj": torch.randn(hidden, inter, generator=generator),
    }
    weights["up_proj"][0] = 0
    weights["up_proj"][0, -1] = 1
    embed = torch.randn(128, hidden, generator=generator)

    class FakeModel:
        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def get_submodule(self, name):
            weight = (
                torch.zeros(hidden)
                if name == "model.norm"
                else weights[name.rsplit(".", 1)[1]]
            )
            return SimpleNamespace(weight=weight)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=embed)

    loaded = []

    def load_model(name, *, revision, dtype):
        loaded.append((name, revision))
        assert dtype == torch.bfloat16
        return FakeModel()

    def load_tokenizer(name, *, revision):
        loaded.append((name, revision))
        return tok

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        ),
    )

    def capture(model, modules, prefixes):
        x = torch.randn(len(prefixes), hidden, generator=generator) * 0.1
        x[:, -1] = 1
        x[:, 0] = torch.tensor(
            [2.0 if "= 42\n" in tok.decode(p) else -2.0 for p in prefixes]
        )
        return {m: x @ weights[m.rsplit(".", 1)[1]].T for m in modules}

    monkeypatch.setattr(nj, "capture_last", capture)
    real_direction = ws.logit_direction

    def direction(embed, ids, norm, **kw):
        assert ids == [nj.THINK_END]
        return real_direction(embed, [1], norm, **kw)

    monkeypatch.setattr(ws, "logit_direction", direction)
    resolved = []
    verified = []

    def resolve(model, revision):
        resolved.append((model, revision))
        return tmp_path / "base"

    def apply(spec, base, dst):
        assert base == tmp_path / "base"
        return {"spec_hash": spec.content_hash(), "changed_weights": 1}

    monkeypatch.setattr(checkpoints, "resolve_checkpoint", resolve)
    monkeypatch.setattr(checkpoints, "apply_spec_to_checkpoint", apply)
    monkeypatch.setattr(
        checkpoints, "verify_checkpoint", lambda s, b, d: verified.append(d)
    )
    side_calls = []
    monkeypatch.setitem(
        sys.modules, "side_effect_job", SimpleNamespace(main=side_calls.append)
    )
    out = tmp_path / "out"
    nj.main(
        [
            "--model",
            model_name,
            "--generate-traces",
            "--nothink-train",
            "2",
            "--save-captures",
            "--tag",
            "_test",
            "--out",
            str(out),
        ]
        + (["--design-only"] if design_only else ["--side-effects"])
    )
    assert loaded == [(model_name, identity["revision"])] * 2
    expected_calls = [("base_trace_test", True), ("nothink_train_test", False)]
    if not design_only:
        expected_calls += [
            ("spec_neuron_test_select", True),
            ("spec_neuron_test_heldout", True),
            ("spec_base_test_heldout", True),
            ("spec_control_test_heldout", True),
        ]
        assert resolved == [(model_name, identity["revision"])]
        assert len(verified) == 2
        assert side_calls[0][:2] == ["--model", model_name]
    else:
        assert not resolved and not verified and not side_calls
    assert calls == expected_calls
    cap = torch.load(out / "captures_test.pt", weights_only=True)
    summary = json.loads((out / "neuron_test_summary.json").read_text())
    contrast = json.loads((out / "answer_contrast_test.json").read_text())
    for artifact in (cap, summary, contrast):
        assert {k: artifact[k] for k in identity} == identity
    assert summary.get("design_only", False) is design_only
    spec = ws.EditSpec.from_json((out / "conclude_neuron_test_spec.json").read_text())
    control = ws.EditSpec.from_json(
        (out / "conclude_neuron_test_control_spec.json").read_text()
    )
    assert len(spec.edits[1].gate) == hidden
    assert spec.edits[1].gate == control.edits[1].gate
    assert spec.edits[1].down != control.edits[1].down
    assert spec.content_hash() == summary["spec_hash"]
    assert cap["gate_acts"].shape[1] == inter

    if model_name != nj.MODEL:

        def load_layer(layer, model):
            assert layer == 23 and model == model_name
            return {**weights, "embed": embed, "norm": torch.zeros(hidden)}

        monkeypatch.setattr(od, "load_layer", load_layer)
        offline_out = tmp_path / "offline"
        od.main(
            ["--captures", str(out / "captures_test.pt"), "--out", str(offline_out)]
        )
        offline_report = json.loads((offline_out / "design.json").read_text())
        assert {k: offline_report[k] for k in identity} == identity
        offline_spec = ws.EditSpec.from_json(
            (offline_out / "conclude_neuron_spec.json").read_text()
        )
        assert len(offline_spec.edits[1].gate) == hidden
        assert 0 <= offline_spec.edits[1].neuron < inter


def test_offline_design_rejects_cross_model_captures_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    path = tmp_path / "captures.pt"
    torch.save({"model": nj.MODEL, "revision": nj.REVISION}, path)
    monkeypatch.setattr(od, "load_layer", lambda *args: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="design input must identify"):
        od.main(
            [
                "--captures",
                str(path),
                "--out",
                str(tmp_path),
                "--model",
                "Qwen/Qwen3.5-2B",
            ]
        )
