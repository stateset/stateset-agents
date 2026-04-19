import json
import subprocess
import sys

import pytest

from scripts.render_qwen3_5_27b_helm_values import build_values_from_manifest


def _run_render(manifest_path: str, *extra_args: str) -> str:
    return subprocess.check_output(
        [
            sys.executable,
            "scripts/render_qwen3_5_27b_helm_values.py",
            "--manifest",
            manifest_path,
            *extra_args,
        ],
        text=True,
    )


def test_render_values_from_manifest_finetuned(tmp_path):
    manifest = {
        "model": "Qwen/Qwen3.5-27B",
        "merged_model_dir": "/models/qwen3-5-27b/merged",
        "recommended": {
            "tensor_parallel_size": 8,
            "max_model_len": 262144,
            "trust_remote_code": True,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "enable_auto_tool_choice": True,
            "language_model_only": True,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path))

    assert 'INFERENCE_DEFAULT_MODEL: "Qwen/Qwen3.5-27B"' in rendered
    assert (
        'INFERENCE_MODEL_MAP: "{\\"Qwen/Qwen3.5-27B\\": \\"/models/qwen3-5-27b/merged\\"}"'
        in rendered
    )
    assert 'modelPath: "/models/qwen3-5-27b/merged"' in rendered
    assert "--reasoning-parser=qwen3" in rendered
    assert "--tool-call-parser=qwen3_coder" in rendered
    assert "--language-model-only" in rendered
    assert "--max-model-len=262144" in rendered


def test_render_values_from_manifest_without_tool_choice(tmp_path):
    manifest = {
        "model": "Qwen/Qwen3.5-27B",
        "merged_model_dir": None,
        "recommended": {
            "tensor_parallel_size": 8,
            "max_model_len": 262144,
            "enable_auto_tool_choice": False,
            "tool_call_parser": None,
            "language_model_only": False,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path), "--no-stream-usage")

    assert 'INFERENCE_STREAM_INCLUDE_USAGE: "false"' in rendered
    assert "--enable-auto-tool-choice" not in rendered
    assert "--tool-call-parser=qwen3_coder" not in rendered
    assert "--language-model-only" not in rendered
    assert "modelPath:" not in rendered


def test_render_values_from_manifest_with_gcs_uri(tmp_path):
    manifest = {
        "model": "Qwen/Qwen3.5-27B",
        "merged_model_dir": "/models/qwen3-5-27b/runs/run-123/merged",
        "recommended": {"tensor_parallel_size": 8, "max_model_len": 262144},
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(
        str(path),
        "--gcs-uri",
        "gs://stateset-models/qwen3-5-27b/runs/run-123/merged",
    )

    assert "modelSync:" in rendered
    assert (
        'gcsUri: "gs://stateset-models/qwen3-5-27b/runs/run-123/merged"' in rendered
    )
    assert 'localDir: "/models/qwen3-5-27b/runs/run-123/merged"' in rendered


def test_build_values_rejects_missing_model():
    with pytest.raises(ValueError, match="manifest.model"):
        build_values_from_manifest({})


def test_build_values_override_model_path_wins_over_manifest():
    values = build_values_from_manifest(
        {
            "model": "Qwen/Qwen3.5-27B",
            "merged_model_dir": "/stale/path",
            "recommended": {"tensor_parallel_size": 8},
        },
        override_model_path="/fresh/path",
    )
    assert values["vllm"]["modelPath"] == "/fresh/path"


def test_build_values_language_model_only_default_true():
    values = build_values_from_manifest(
        {"model": "Qwen/Qwen3.5-27B", "recommended": {}}
    )
    assert "--language-model-only" in values["vllm"]["args"]


def test_cli_rejects_non_object_manifest(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1,2,3]")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_qwen3_5_27b_helm_values.py",
            "--manifest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "JSON object" in result.stderr
