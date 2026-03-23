#!/usr/bin/env python
"""
Render a Helm values override for deploying Kimi-K2.5 via this repo's Helm chart.

This script consumes the `serving_manifest.json` written by:
  `examples/finetune_kimi_k25_gspo.py`

Usage:
  python scripts/render_kimi_k25_helm_values.py --manifest /models/kimi-k25/serving_manifest.json > values.generated.yaml

Then deploy:
  helm upgrade --install stateset-agents deployment/helm/stateset-agents \
    --namespace stateset-agents \
    -f values.generated.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from collections.abc import Sequence

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _yaml_scalar(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # JSON strings are valid YAML scalars and handle escaping correctly.
        return json.dumps(value)
    raise TypeError(f"Unsupported scalar type: {type(value)!r}")


def _yaml_dump(value: JsonValue, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_dump(child, indent=indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_dump(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def build_values_from_manifest(
    manifest: dict[str, Any],
    *,
    override_model_path: str | None = None,
    model_gcs_uri: str | None = None,
    include_stream_usage: bool = True,
) -> dict[str, Any]:
    model_id = manifest.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("manifest.model must be a non-empty string")

    merged_model_dir = manifest.get("merged_model_dir")
    finetuned = isinstance(merged_model_dir, str) and bool(merged_model_dir.strip())
    if model_gcs_uri:
        finetuned = True

    recommended = manifest.get("recommended") or {}
    if not isinstance(recommended, dict):
        recommended = {}

    tensor_parallel = recommended.get("tensor_parallel_size", 8)
    max_model_len = recommended.get("max_model_len", 256000)
    trust_remote_code = recommended.get("trust_remote_code", True)

    args = [
        f"--tensor-parallel-size={tensor_parallel}",
        "--mm-encoder-tp-mode=data",
        "--tool-call-parser=kimi_k2",
        "--reasoning-parser=kimi_k2",
        f"--max-model-len={max_model_len}",
        "--gpu-memory-utilization=0.90",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--host=0.0.0.0",
        "--port=8000",
    ]
    if trust_remote_code:
        args.insert(len(args) - 2, "--trust-remote-code")

    values: dict[str, Any] = {
        "api": {
            "env": {
                "INFERENCE_DEFAULT_MODEL": model_id,
                "INFERENCE_STREAM_INCLUDE_USAGE": "true"
                if include_stream_usage
                else "false",
            }
        },
        "vllm": {
            "modelId": model_id,
            "finetuned": finetuned,
            "args": args,
        },
    }

    if finetuned:
        model_path = override_model_path or str(merged_model_dir or "")
        if not model_path:
            model_path = "/models/kimi-k25/merged"
        if not model_path:
            raise ValueError(
                "manifest.merged_model_dir is missing. Provide --model-path."
            )
        values["vllm"]["modelPath"] = model_path

        # Keep the public model name stable even when vLLM is loading from a local path.
        values["api"]["env"]["INFERENCE_MODEL_MAP"] = json.dumps({model_id: model_path})

        if model_gcs_uri:
            values["vllm"]["modelSync"] = {
                "enabled": True,
                "gcsUri": model_gcs_uri,
                "localDir": model_path,
            }

    return values


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("manifest must be a JSON object")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Helm values overrides from Kimi-K2.5 serving_manifest.json",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to serving_manifest.json produced by the Kimi-K2.5 finetune script.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override vLLM model path (useful when moving weights into a PVC).",
    )
    parser.add_argument(
        "--no-stream-usage",
        action="store_true",
        help="Disable backend streaming usage (gateway won't request stream_options.include_usage).",
    )
    parser.add_argument(
        "--gcs-uri",
        default=None,
        help=(
            "Optional GCS URI to sync model artifacts from at pod startup "
            "(e.g. gs://bucket/kimi-k25/runs/<run>/merged)."
        ),
    )

    args = parser.parse_args(argv)

    manifest = _read_json(Path(args.manifest))
    values = build_values_from_manifest(
        manifest,
        override_model_path=args.model_path,
        model_gcs_uri=args.gcs_uri,
        include_stream_usage=not args.no_stream_usage,
    )
    sys.stdout.write("\n".join(_yaml_dump(values)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
