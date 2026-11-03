"""Tests for checkpoint-level weight edits and the ``stateset-agents weight-edit`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from stateset_agents.cli import app  # noqa: E402
from stateset_agents.experimental.weight_space.checkpoints import (  # noqa: E402
    apply_spec_to_checkpoint,
    norm_convention,
    verify_checkpoint,
)
from stateset_agents.experimental.weight_space.edits import (  # noqa: E402
    EditSpec,
    NeuronColumnScale,
    SpecMismatch,
    TokenRowScale,
)

runner = CliRunner()

EMBED = "model.embed_tokens.weight"
DOWN0 = "model.layers.0.mlp.down_proj.weight"
SPEC = EditSpec(
    name="w5t15",
    edits=(
        TokenRowScale(ids=(1,), scale=0.5),
        NeuronColumnScale(layer=0, neurons=(2,), scale=0.0),
    ),
)


def tiny_checkpoint(root: Path) -> Path:
    """Two shards + an index, like a sharded Hub checkpoint."""
    root.mkdir()
    g = torch.Generator().manual_seed(0)
    safetensors_torch.save_file(
        {EMBED: torch.randn(6, 4, generator=g)},
        str(root / "model-00001-of-00002.safetensors"),
    )
    safetensors_torch.save_file(
        {
            DOWN0: torch.randn(4, 5, generator=g),
            "model.norm.weight": torch.ones(4),
        },
        str(root / "model-00002-of-00002.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    EMBED: "model-00001-of-00002.safetensors",
                    DOWN0: "model-00002-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (root / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    return root


# --------------------------------------------------------------- checkpoint I/O


def test_apply_to_checkpoint_patches_shards_and_writes_a_manifest(
    tmp_path: Path,
) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    out = tmp_path / "edited"
    manifest = apply_spec_to_checkpoint(SPEC, base, out)

    assert manifest["spec_hash"] == SPEC.content_hash()
    assert manifest["changed_weights"] == 4 + 4  # one 4-wide row + one 4-tall column
    assert json.loads((out / "weight_space_spec.json").read_text()) == manifest
    assert (out / "config.json").read_text() == (base / "config.json").read_text()

    before = safetensors_torch.load_file(str(base / "model-00001-of-00002.safetensors"))
    after = safetensors_torch.load_file(str(out / "model-00001-of-00002.safetensors"))
    assert torch.allclose(after[EMBED][1], before[EMBED][1] * 0.5)
    verify_checkpoint(SPEC, base, out)  # no raise


def test_verify_checkpoint_catches_tampering_in_an_untouched_shard(
    tmp_path: Path,
) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    out = tmp_path / "edited"
    apply_spec_to_checkpoint(SPEC, base, out)
    shard = out / "model-00002-of-00002.safetensors"
    tensors = safetensors_torch.load_file(str(shard))
    tensors["model.norm.weight"][0] = 2.0
    safetensors_torch.save_file(tensors, str(shard))
    with pytest.raises(SpecMismatch, match="model.norm.weight"):
        verify_checkpoint(SPEC, base, out)


def test_apply_to_checkpoint_refuses_to_overwrite(tmp_path: Path) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    (tmp_path / "edited").mkdir()
    with pytest.raises(FileExistsError):
        apply_spec_to_checkpoint(SPEC, base, tmp_path / "edited")


# --------------------------------------------------------------- CLI


def test_cli_apply_then_verify(tmp_path: Path) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(SPEC.to_json())
    out = tmp_path / "edited"

    applied = runner.invoke(
        app,
        [
            "weight-edit",
            "apply",
            str(spec_path),
            "--base",
            str(base),
            "--out",
            str(out),
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["spec_hash"] == SPEC.content_hash()

    verified = runner.invoke(
        app,
        [
            "weight-edit",
            "verify",
            str(spec_path),
            "--base",
            str(base),
            "--edited",
            str(out),
        ],
    )
    assert verified.exit_code == 0, verified.output
    assert "verified" in verified.output


def test_cli_verify_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(SPEC.to_json())
    other = EditSpec(name="other", edits=(TokenRowScale(ids=(3,), scale=0.5),))
    out = tmp_path / "edited"
    apply_spec_to_checkpoint(other, base, out)

    result = runner.invoke(
        app,
        [
            "weight-edit",
            "verify",
            str(spec_path),
            "--base",
            str(base),
            "--edited",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "MISMATCH" in result.output


def test_cli_control_draws_targets_inside_the_checkpoint(tmp_path: Path) -> None:
    base = tiny_checkpoint(tmp_path / "base")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(SPEC.to_json())
    result = runner.invoke(
        app,
        ["weight-edit", "control", str(spec_path), "--base", str(base), "--seed", "3"],
    )
    assert result.exit_code == 0, result.output
    control = EditSpec.from_json(result.output)
    assert control.name == "control:w5t15"
    tok, col = control.edits
    assert isinstance(tok, TokenRowScale) and all(
        0 <= i < 6 and i != 1 for i in tok.ids
    )
    assert isinstance(col, NeuronColumnScale) and all(0 <= j < 5 for j in col.neurons)
    assert 2 not in col.neurons


def test_cli_compare_reports_paired_stats(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    edit = tmp_path / "edit.jsonl"
    base.write_text(
        "".join(json.dumps({"id": i, "correct": i < 1}) + "\n" for i in range(8))
    )
    edit.write_text(
        "".join(json.dumps({"id": i, "correct": i < 7}) + "\n" for i in range(8))
    )
    result = runner.invoke(app, ["weight-edit", "compare", str(base), str(edit)])
    assert result.exit_code == 0, result.output
    stats = json.loads(result.output)
    assert (stats["n"], stats["gained"], stats["lost"]) == (8, 6, 0)
    assert stats["mcnemar_p"] == pytest.approx(2 / 64)


def test_cli_compare_strict_discounts_inline_end_of_thinking(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    edit = tmp_path / "edit.jsonl"
    clean = {"finish": "stop", "text": "work\n</think>\n\\boxed{1}"}
    inline = {"finish": "length", "text": "x = 1</think> keeps reasoning..."}
    base.write_text(
        "".join(
            json.dumps({"id": i, "correct": False, **clean}) + "\n" for i in range(4)
        )
    )
    edit.write_text(
        json.dumps({"id": 0, "correct": True, **clean})
        + "\n"
        + "".join(
            json.dumps({"id": i, "correct": True, **inline}) + "\n" for i in (1, 2, 3)
        )
    )
    lenient = runner.invoke(app, ["weight-edit", "compare", str(base), str(edit)])
    strict = runner.invoke(
        app, ["weight-edit", "compare", str(base), str(edit), "--strict"]
    )
    assert lenient.exit_code == 0 and strict.exit_code == 0, strict.output
    assert json.loads(lenient.output)["gained"] == 4
    assert json.loads(strict.output)["gained"] == 1


def test_cli_atlas_finds_the_neuron_that_writes_a_token(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    g = torch.Generator().manual_seed(1)
    embed = torch.randn(6, 4, generator=g)
    down = torch.randn(4, 5, generator=g) * 0.01
    down[:, 3] = embed[2] * 5.0  # neuron 3 writes exactly toward token 2
    safetensors_torch.save_file(
        {EMBED: embed, DOWN0: down, "model.norm.weight": torch.ones(4)},
        str(root / "model.safetensors"),
    )
    result = runner.invoke(
        app,
        ["weight-edit", "atlas", "--base", str(root), "--token-id", "2", "--top", "1"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["norm"] == "plain"  # no config.json -> x * w
    (best,) = report["writers"]
    assert (best["layer"], best["neuron"]) == (0, 3)
    assert best["alignment"] == pytest.approx(1.0, abs=1e-5)
    assert best["promotes"][0] == 2


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}},
            "zero-centered",
        ),
        ({"model_type": "gemma3_text"}, "zero-centered"),
        ({"model_type": "qwen3"}, "plain"),
        ({"model_type": "gemma3n"}, "plain"),
        (None, "plain"),
    ],
)
def test_norm_convention_comes_from_the_model_family(
    tmp_path: Path, config: dict | None, expected: str
) -> None:
    # Weight values cannot tell the conventions apart: Qwen3.5 (1 + w) and Qwen3 (w)
    # both ship final-norm weights averaging ~3.5.
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    assert norm_convention(tmp_path) == expected
