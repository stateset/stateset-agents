"""GSPO rollouts sample with the trainer config's sampling settings.

``train_with_gspo`` rebuilt the agent's generation config from ``AgentConfig``
defaults (temperature 0.8, top_k 50, repetition_penalty 1.1), so the
configured ``GSPOConfig.temperature``/``top_p`` never reached generation and
the v5 shootout's native leg sampled differently from its TRL leg.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import make_agent, make_model, make_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_trainer_sampling_settings_drive_the_agent_generation_config():
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_entrypoints import apply_rollout_sampling

    agent = make_agent(make_model(0), make_tokenizer())
    config = GSPOConfig(
        model_name="gpt2", temperature=0.7, top_p=0.95, top_k=0, repetition_penalty=1.0
    )
    apply_rollout_sampling(agent, config)
    assert (agent.config.temperature, agent.config.top_p) == (0.7, 0.95)
    assert agent.config.repetition_penalty == 1.0
    generation = agent.generation_config
    assert generation.temperature == 0.7 and generation.top_p == 0.95
    assert generation.top_k in (0, None) and generation.repetition_penalty == 1.0


def _script():
    spec = importlib.util.spec_from_file_location(
        "run_phase0_benchmark_sampling", ROOT / "scripts" / "run_phase0_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_shootout_leg_records_and_uses_the_protocol_sampling():
    config = json.loads(
        (ROOT / "benchmarks" / "shootout_manifest_v5.json").read_text()
    )["config"]
    script = _script()
    sampling = script.build_algorithm_config("gspo", config)["sampling"]
    assert sampling == {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": 0,
        "repetition_penalty": 1.0,
        "stop_sequences": [],
    }
    agent_config = script.native_agent_config("gpt2", None, 256, sampling)
    assert agent_config.temperature == config["temperature"]
    assert agent_config.top_p == config["top_p"]
    assert agent_config.top_k in (0, None)
    assert agent_config.repetition_penalty == 1.0
    assert agent_config.stop_sequences == []
