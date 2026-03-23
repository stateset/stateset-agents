import json
import subprocess
import sys


def _run_render(manifest_path: str, *extra_args: str) -> str:
    return subprocess.check_output(
        [
            sys.executable,
            "scripts/render_kimi_k25_helm_values.py",
            "--manifest",
            manifest_path,
            *extra_args,
        ],
        text=True,
    )


def test_render_values_from_manifest_finetuned(tmp_path):
    manifest = {
        "model": "moonshotai/Kimi-K2.5",
        "merged_model_dir": "/models/kimi-k25/merged",
        "recommended": {
            "tensor_parallel_size": 8,
            "max_model_len": 256000,
            "trust_remote_code": True,
        },
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path))

    assert 'INFERENCE_DEFAULT_MODEL: "moonshotai/Kimi-K2.5"' in rendered
    assert 'INFERENCE_STREAM_INCLUDE_USAGE: "true"' in rendered
    assert (
        'INFERENCE_MODEL_MAP: "{\\"moonshotai/Kimi-K2.5\\": \\"/models/kimi-k25/merged\\"}"'
        in rendered
    )
    assert "modelPath:" in rendered
    assert 'modelPath: "/models/kimi-k25/merged"' in rendered
    assert "--tool-call-parser=kimi_k2" in rendered
    assert "--max-model-len=256000" in rendered


def test_render_values_from_manifest_base_model(tmp_path):
    manifest = {
        "model": "moonshotai/Kimi-K2.5",
        "merged_model_dir": None,
        "recommended": {"tensor_parallel_size": 8, "max_model_len": 256000},
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(str(path), "--no-stream-usage")

    assert 'INFERENCE_DEFAULT_MODEL: "moonshotai/Kimi-K2.5"' in rendered
    assert 'INFERENCE_STREAM_INCLUDE_USAGE: "false"' in rendered
    assert "modelPath:" not in rendered


def test_render_values_from_manifest_with_gcs_uri(tmp_path):
    manifest = {
        "model": "moonshotai/Kimi-K2.5",
        "merged_model_dir": "/models/kimi-k25/runs/run-123/merged",
        "recommended": {"tensor_parallel_size": 8, "max_model_len": 256000},
    }
    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))

    rendered = _run_render(
        str(path),
        "--gcs-uri",
        "gs://stateset-models/kimi-k25/runs/run-123/merged",
    )

    assert "modelSync:" in rendered
    assert 'gcsUri: "gs://stateset-models/kimi-k25/runs/run-123/merged"' in rendered
    assert 'localDir: "/models/kimi-k25/runs/run-123/merged"' in rendered
