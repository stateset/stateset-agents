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
