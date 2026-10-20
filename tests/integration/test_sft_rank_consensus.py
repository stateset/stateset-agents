"""Real two-rank Gloo agreement and rejection; no weights or GPU training."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from stateset_agents.training.distributed_preflight import (
    fingerprint_encoded_rows,
    sft_preflight_identity,
    verify_sft_rank_consensus,
)

pytestmark = pytest.mark.integration
_COMPONENTS = (
    "model",
    "revision",
    "training_config",
    "dataset",
    "tokenizer",
    "encoded_data",
    "runtime",
    "implementation",
)


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    if sys.platform.startswith("linux"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=20),
    )
    results = []
    from importlib.metadata import version

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    from stateset_agents.data.text_sft import load_text_sft_snapshot
    from stateset_agents.training.frontier_sft import encode_frontier_rows

    data_path = root / f"input-{rank}.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    snapshot = load_text_sft_snapshot(str(data_path))
    engine = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "user:": 3,
                "question": 4,
                "assistant:": 5,
                "answer": 6,
            },
            unk_token="[UNK]",
        )
    )
    engine.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=engine, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for m in messages %}{{m.role + ': ' + m.content + ' [EOS] '}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}"
    encoded = encode_frontier_rows(tokenizer, snapshot.train, 32)
    identity = sft_preflight_identity(
        model_id="local/random-tiny",
        revision="a" * 40,
        training_config={"seed": 42, "learning_rate": 1e-5},
        dataset=snapshot.describe(),
        tokenizer=tokenizer,
        encoded_data={
            "protocol": "jsonl-input-labels-v1",
            "train": fingerprint_encoded_rows(encoded),
            "validation": None,
        },
        runtime_versions={name: version(name) for name in ["torch", "transformers"]},
        implementation_modules=[
            "stateset_agents.data.text_sft",
            "stateset_agents.training.frontier_sft",
            "stateset_agents.core.chat_masking",
        ],
    )
    scenarios = [
        "agree",
        *_COMPONENTS,
        "missing_field",
        "extra_field",
        "nonfinite",
        "wrong_world_size",
        "builder_failure",
        "missing_local_source",
        "overflow_world_size",
        "agree_after_failures",
    ]
    try:
        for scenario in scenarios:
            local = copy.deepcopy(identity)
            expected_world = 2
            if rank == 1:
                if scenario in _COMPONENTS:
                    local[scenario] = {"sensitive_value": "DO_NOT_TRANSMIT"}
                elif scenario == "missing_field":
                    local.pop("model")
                elif scenario == "extra_field":
                    local["unknown"] = True
                elif scenario == "nonfinite":
                    local["training_config"] = {"learning_rate": float("nan")}
                elif scenario == "wrong_world_size":
                    expected_world = 3
                elif scenario == "overflow_world_size":
                    expected_world = 2**70
            try:

                def identity_factory(case=scenario, inputs=local):
                    if case == "builder_failure" and rank == 1:
                        raise ValueError("local tokenizer serialization failed")
                    if case == "missing_local_source" and rank == 1:
                        raise OSError("source unavailable on one node")
                    return inputs

                receipt = verify_sft_rank_consensus(
                    identity_factory, world_size=expected_world
                )
            except RuntimeError as exc:
                message = str(exc)
                assert scenario not in {"agree", "agree_after_failures"}, message
                assert "DO_NOT_TRANSMIT" not in message
                if scenario in _COMPONENTS:
                    assert (
                        message
                        == "SFT preflight disagreement across ranks: " + scenario
                    )
                elif scenario == "wrong_world_size":
                    assert "WORLD_SIZE differs" in message
                else:
                    assert "Invalid SFT rank identity on ranks [1]" in message
                results.append(
                    {"scenario": scenario, "rejected": True, "error": message}
                )
            else:
                assert scenario in {"agree", "agree_after_failures"}, scenario
                assert receipt["verified"] is True
                assert receipt["backend"] == "gloo" and receipt["world_size"] == 2
                results.append(
                    {"scenario": scenario, "rejected": False, "receipt": receipt}
                )
        (root / f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def test_real_ranks_reject_different_inputs_and_remain_usable(tmp_path):
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires the CPU Gloo backend")
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()),), nprocs=2, join=False
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank SFT consensus exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = json.loads((tmp_path / "rank-0.json").read_text())
    second = json.loads((tmp_path / "rank-1.json").read_text())
    assert first == second
    assert len(first) == 17
    assert sum(row["rejected"] for row in first) == 15
    assert first[0]["receipt"] == first[-1]["receipt"]
    destination = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        (output / "sft-rank-consensus.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "backend": "gloo",
                    "world_size": 2,
                    "scope": "two real local CPU ranks; fixed-size tensor collectives; no FSDP, GPU or model-weight loading",
                    "ranks": {"0": first, "1": second},
                },
                indent=2,
            )
            + "\n"
        )
