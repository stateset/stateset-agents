import json
import subprocess
import sys

import pytest

from scripts.render_glm5_1_helm_values import build_values_from_manifest


def _run_render(manifest_path: str, *extra_args: str) -> str:
    return subprocess.check_output(
        [
            sys.executable,
            "scripts/render_glm5_1_helm_values.py",
            "--manifest",
            manifest_path,
            *extra_args,
        ],
        text=True,
    )


def test_render_values_from_manifest_finetuned(tmp_path):
    manifest = {
        "model": "zai-org/GLM-5.1",
        "merged_model_dir": "/models/glm5-1/merged",
        "recommended": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "max_model_len": 131072,
            "trust_remote_code": True,
            "reasoning_parser": "glm45",
            "tool_call_parser": "glm45",
            "enable_auto_tool_choice": True,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path))

    assert 'INFERENCE_DEFAULT_MODEL: "zai-org/GLM-5.1"' in rendered
    assert (
        'INFERENCE_MODEL_MAP: "{\\"zai-org/GLM-5.1\\": \\"/models/glm5-1/merged\\"}"'
        in rendered
    )
    assert 'modelPath: "/models/glm5-1/merged"' in rendered
    assert "--reasoning-parser=glm45" in rendered
    assert "--tool-call-parser=glm45" in rendered
    assert "--tensor-parallel-size=8" in rendered
    assert "--pipeline-parallel-size=2" in rendered
    assert "--max-model-len=131072" in rendered


def test_render_values_from_manifest_fp8(tmp_path):
    manifest = {
        "model": "your-org/GLM-5.1-FP8",
        "merged_model_dir": None,
        "recommended": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "max_model_len": 131072,
            "quantization": "fp8",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "glm45",
            "reasoning_parser": "glm45",
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path))

    assert 'modelId: "your-org/GLM-5.1-FP8"' in rendered
    assert "--quantization=fp8" in rendered
    assert "--tensor-parallel-size=8" in rendered
    # Pipeline parallel of 1 should not emit the flag.
    assert "--pipeline-parallel-size" not in rendered
    assert "modelPath:" not in rendered


def test_render_values_from_manifest_without_tool_choice(tmp_path):
    manifest = {
        "model": "zai-org/GLM-5.1",
        "merged_model_dir": None,
        "recommended": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "max_model_len": 131072,
            "enable_auto_tool_choice": False,
            "tool_call_parser": None,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path), "--no-stream-usage")

    assert 'INFERENCE_STREAM_INCLUDE_USAGE: "false"' in rendered
    assert "--enable-auto-tool-choice" not in rendered
    assert "--tool-call-parser=glm45" not in rendered
    assert "modelPath:" not in rendered


def test_render_values_from_manifest_with_gcs_uri(tmp_path):
    manifest = {
        "model": "zai-org/GLM-5.1",
        "merged_model_dir": "/models/glm5-1/runs/run-123/merged",
        "recommended": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "max_model_len": 131072,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(
        str(path),
        "--gcs-uri",
        "gs://YOUR_BUCKET/glm5-1/runs/YOUR_RUN_ID/merged",
    )

    assert "modelSync:" in rendered
    assert (
        'gcsUri: "gs://YOUR_BUCKET/glm5-1/runs/YOUR_RUN_ID/merged"' in rendered
    )
    assert 'localDir: "/models/glm5-1/runs/run-123/merged"' in rendered


def test_build_values_rejects_missing_model():
    with pytest.raises(ValueError, match="manifest.model"):
        build_values_from_manifest({})


def test_build_values_rejects_blank_model():
    with pytest.raises(ValueError, match="manifest.model"):
        build_values_from_manifest({"model": "   "})


def test_build_values_override_model_path_wins_over_manifest():
    values = build_values_from_manifest(
        {
            "model": "zai-org/GLM-5.1",
            "merged_model_dir": "/stale/path",
            "recommended": {"tensor_parallel_size": 8},
        },
        override_model_path="/fresh/path",
    )
    assert values["vllm"]["modelPath"] == "/fresh/path"
    assert values["api"]["env"]["INFERENCE_MODEL_MAP"] == (
        '{"zai-org/GLM-5.1": "/fresh/path"}'
    )


def test_build_values_gcs_uri_flips_finetuned_without_merged_dir():
    values = build_values_from_manifest(
        {
            "model": "zai-org/GLM-5.1",
            "merged_model_dir": None,
            "recommended": {"tensor_parallel_size": 8},
        },
        model_gcs_uri="gs://bucket/run-x/merged",
    )
    assert values["vllm"]["finetuned"] is True
    assert values["vllm"]["modelPath"] == "/models/glm5-1/merged"
    assert values["vllm"]["modelSync"]["gcsUri"] == "gs://bucket/run-x/merged"


def test_build_values_malformed_recommended_falls_back_to_defaults():
    values = build_values_from_manifest(
        {
            "model": "zai-org/GLM-5.1",
            "merged_model_dir": None,
            "recommended": "not-a-dict",
        }
    )
    args = values["vllm"]["args"]
    assert "--tensor-parallel-size=8" in args
    assert "--max-model-len=131072" in args
    assert "--reasoning-parser=glm45" in args


def test_build_values_quantization_emits_flag():
    values = build_values_from_manifest(
        {
            "model": "your-org/GLM-5.1-FP8",
            "recommended": {"quantization": "fp8", "pipeline_parallel_size": 1},
        }
    )
    assert "--quantization=fp8" in values["vllm"]["args"]
    assert "--pipeline-parallel-size" not in " ".join(values["vllm"]["args"])


def test_cli_rejects_missing_manifest(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_glm5_1_helm_values.py",
            "--manifest",
            str(tmp_path / "does-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "manifest not found" in result.stderr


def test_cli_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_glm5_1_helm_values.py",
            "--manifest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not valid JSON" in result.stderr
