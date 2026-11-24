"""Unit and structural tests for Google Colab Ultra helpers, cloud='colab' blueprints,
the v1.0 GPU + River evidence capture runner, and the 4 flagship v1.0 Colab notebooks.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.capture_v1_gpu_river_evidence import (
    capture_evidence,
    render_markdown_report,
    run_cuda_chunked_vram_benchmark,
)
from stateset_agents.training.rl_blueprint import (
    RLBlueprintConfig,
    build_cloud_launch_plan,
)
from stateset_agents.utils.colab import (
    COLAB_NOTEBOOKS,
    COLAB_PROFILES,
    colab_notebook_url,
    detect_colab_runtime,
    get_colab_secret,
    recommend_colab_profile,
    resolve_checkpoint_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("gpu_name", "vram_gb", "expected_tier"),
    [
        ("NVIDIA H100 80GB HBM3", 80.0, "h100_80gb"),
        ("NVIDIA A100-SXM4-80GB", 80.0, "a100_80gb"),
        ("NVIDIA A100-SXM4-40GB", 40.0, "a100_40gb"),
        ("NVIDIA L4", 24.0, "l4_24gb"),
        ("Tesla T4", 15.0, "t4_16gb"),
        ("", 0.0, "cpu_fallback"),
    ],
)
def test_recommend_colab_profile_matches_gpu_tier(
    gpu_name: str, vram_gb: float, expected_tier: str
) -> None:
    profile = recommend_colab_profile(gpu_name=gpu_name, vram_gb=vram_gb)
    assert profile.tier == expected_tier
    assert profile.tier in COLAB_PROFILES
    assert profile.chunked_loss_chunk_size >= 32


def test_detect_colab_runtime_without_cuda_probe_is_torch_free() -> None:
    info = detect_colab_runtime(probe_cuda=False)
    assert info["cuda_available"] is False
    assert info["profile"]["tier"] == "cpu_fallback"
    assert isinstance(info["python_version"], str)
    assert isinstance(info["river_python_compatible"], bool)


def test_colab_helpers_secret_url_and_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STATESET_TEST_SECRET", "  tok_123  ")
    assert get_colab_secret("STATESET_TEST_SECRET") == "tok_123"
    assert get_colab_secret("STATESET_MISSING_SECRET", default="fallback") == "fallback"

    url = colab_notebook_url("open_weight_rl")
    assert url.startswith("https://colab.research.google.com/github/")
    assert url.endswith("/notebooks/open_weight_rl_colab_gpu.ipynb")

    ckpt_dir = resolve_checkpoint_dir("unit-run", fallback_root=tmp_path)
    assert ckpt_dir == tmp_path / "unit-run"
    assert ckpt_dir.is_dir()


def test_rl_blueprint_supports_colab_ultra_target(tmp_path: Path) -> None:
    cfg = RLBlueprintConfig(
        name="colab-ultra-qwen",
        model="Qwen/Qwen3-8B",
        algorithm="gspo",
        cloud="colab",
        gpu_type="A100-80GB",
    )
    plan = build_cloud_launch_plan(cfg, workspace_root=tmp_path)
    dispatch = plan["dispatch"]
    assert dispatch["provider"] == "colab"
    assert dispatch["plan_tier"] == "google-ai-ultra"
    assert dispatch["gpu_profile"]["tier"] == "a100_80gb"
    assert dispatch["colab_url"].endswith("open_weight_rl_colab_gpu.ipynb")
    assert "capture_v1_gpu_river_evidence.py" in dispatch["evidence_command"]


def test_colab_module_has_zero_eager_torch_imports() -> None:
    code = (
        "import sys; import stateset_agents.utils.colab; "
        "assert 'torch' not in sys.modules, 'torch eagerly imported'"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_capture_v1_gpu_river_evidence_offline_end_to_end(tmp_path: Path) -> None:
    vram = run_cuda_chunked_vram_benchmark(
        batch_size=2, seq_len=64, hidden_dim=32, vocab_size=256, chunk_size=16
    )
    assert vram["numerical_equivalence"]["exact_match"] is True

    evidence = capture_evidence(
        output_dir=tmp_path,
        skip_hf_model=True,
        river_mode="offline",
    )
    assert evidence["schema_version"] == 1
    assert len(evidence["evidence_sha256"]) == 64
    assert evidence["three_seed_ablation"]["summary"]["all_seeds_improved"] is True
    assert (
        evidence["three_seed_ablation"]["six_domain_verifiable_eval"]["all_verified"]
        is True
    )
    assert evidence["river_ai"]["mode"] == "offline"
    assert (tmp_path / "v1_ga_evidence.json").is_file()
    assert (tmp_path / "v1_ga_evidence.md").is_file()

    md_text = render_markdown_report(evidence)
    assert "Full-Logit vs. Chunked / Triton LM-Head Benchmark" in md_text
    assert "Six-Domain Enterprise Sandbox Pool" in md_text
    assert "River AI `0.11` Verification" in md_text


@pytest.mark.parametrize("notebook_file", sorted(COLAB_NOTEBOOKS.values()))
def test_flagship_v1_colab_notebooks_are_valid_nbformat_and_python_ast(
    notebook_file: str,
) -> None:
    nb_path = REPO_ROOT / "notebooks" / notebook_file
    assert nb_path.is_file(), f"Missing flagship Colab notebook: {nb_path}"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 5

    # First markdown cell must expose an Open-in-Colab badge targeting this exact file
    first_src = "".join(nb["cells"][0]["source"])
    assert "colab.research.google.com" in first_src
    assert notebook_file in first_src

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 3
    for idx, cell in enumerate(code_cells):
        raw = "".join(cell["source"])
        assert "rv_live_" not in raw and "sk-ant-" not in raw
        assert (
            "from benchmarks" not in raw and "import benchmarks" not in raw
        ), f"{notebook_file} must only import packaged stateset_agents.* modules (PyPI wheel safe)"
        cleaned = "\n".join(
            "pass" if line.lstrip().startswith(("!", "%")) else line
            for line in raw.splitlines()
        )
        ast.parse(cleaned, filename=f"{notebook_file}#code{idx}")

    readme = (REPO_ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    assert notebook_file in readme


def test_run_river_verification_live_wires_serving_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.remote import river_runtime, river_serving
    from stateset_agents.utils.colab import run_river_verification

    monkeypatch.setenv("RIVER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        river_runtime,
        "river_preflight",
        lambda models=None, live=False, timeout_seconds=20.0: {
            "ok": True,
            "live": live,
            "models": list(models or []),
        },
    )
    monkeypatch.setattr(
        river_runtime,
        "inspect_server_capabilities",
        lambda client, model_name=None: {
            "supported_models": [model_name or "Qwen/Qwen3.6-35B-A3B-FP8"],
            "features": ["rl", "deployments"],
        },
    )

    class _FakeManager:
        def __init__(self, *, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.closed = False

        def _get_client(self) -> object:
            return object()

        def list_deployments(self) -> list[dict[str, str]]:
            return [{"id": "dep-1"}]

        def close(self) -> None:
            self.closed = True

    class _FakeInferenceClient:
        def __init__(
            self,
            checkpoint: str,
            *,
            base_model: str = "",
            api_key: str | None = None,
        ) -> None:
            self.checkpoint = checkpoint
            self.base_model = base_model
            self.api_key = api_key

        def chat(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int = 512,
            temperature: float = 0.0,
        ) -> dict[str, str]:
            assert messages and max_tokens == 48 and temperature == 0.0
            return {"text": "Ready for v1.0 GA."}

        def close(self) -> None:
            return None

    monkeypatch.setattr(river_serving, "RiverDeploymentManager", _FakeManager)
    monkeypatch.setattr(river_serving, "RiverInferenceClient", _FakeInferenceClient)

    res = run_river_verification(
        mode="live",
        base_model="Qwen/Qwen3.6-35B-A3B-FP8",
        checkpoint_uri="river://ckpt/demo",
    )
    assert res["mode"] == "live"
    assert res["existing_deployments_count"] == 1
    assert res["checkpoint_chat_verified"] is True
    assert res["checkpoint_chat_preview"] == "Ready for v1.0 GA."


def test_weight_studio_colab_notebook_endpoints_live(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stateset_agents.weight_studio import create_studio_app

    app = create_studio_app(workspace_root=tmp_path)
    client = TestClient(app)

    assert client.get("/api/weight-studio/health").status_code == 200
    pool_resp = client.post(
        "/api/weight-studio/rl/sandbox-pool",
        json={
            "domains": [
                "commerce",
                "coding",
                "sql",
                "rest_api",
                "docker_code",
                "web_agent",
            ],
            "include_failure_pairs": True,
        },
    )
    assert pool_resp.status_code == 200
    assert pool_resp.json()["episode_count"] >= 6

    telem_resp = client.post(
        "/api/weight-studio/rl/telemetry",
        json={
            "run_name": "colab-ultra-gspo",
            "step": 1,
            "mean_reward": 0.57,
            "entropy": 1.12,
            "kl": 0.018,
            "guard_verdict": "HEALTHY",
        },
    )
    assert telem_resp.status_code == 200

    mcp_resp = client.post(
        "/api/weight-studio/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert mcp_resp.status_code == 200
    assert len(mcp_resp.json()["result"]["tools"]) >= 30

    auto_resp = client.post(
        "/api/weight-studio/autopilot/run",
        json={"model_id": "preset:deepseek-r1-1.5b", "reward_fn": "customer_support"},
    )
    assert auto_resp.status_code == 200
    assert (auto_resp.json().get("canary_gate") or {}).get("decision") in {
        "PROMOTE",
        "HOLD",
        "ROLLBACK",
    }

    html_resp = client.post(
        "/api/weight-studio/report/html",
        json={
            "model_id": "preset:deepseek-r1-1.5b",
            "include_eval": True,
            "include_sensitivity": True,
        },
    )
    assert html_resp.status_code == 200
    assert "<html" in html_resp.text.lower()
