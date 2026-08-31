# Training backend protocol

StateSet's backend protocol separates **experiment meaning** from **engine
execution**. An experiment fixes the algorithm, model and dataset revisions,
seed, environment, reward, requirements, and algorithm configuration. A
backend may translate those fields into its native configuration, but it must
return the same canonical experiment digest.

This is the foundation for delegating StateSet experiments to specialized
engines such as TRL, verl, NeMo RL, and OpenRLHF without silently changing the
task or reward. Executable, fail-closed adapters exist for verl, NeMo RL, and
OpenRLHF, but they remain **live-conformance pending** and must not be described
as performance-equivalent until their retained GPU gates pass.

## Contract

```python
from pathlib import Path

from stateset_agents.training import (
    BackendCapabilities,
    CommandTrainingBackend,
    TrainingExperiment,
)

experiment = TrainingExperiment(
    algorithm="grpo",
    model="Qwen/Qwen3.5-0.8B",
    model_revision="IMMUTABLE_MODEL_REVISION",
    dataset_uri="s3://bucket/train.jsonl",
    dataset_sha256="...64 lowercase hexadecimal characters...",
    output_dir=Path("outputs/verl-run"),
    seed=42,
    task="customer-support-v1",
    config={"learning_rate": 5e-6, "max_steps": 100},
    environment={"name": "customer-support", "version": 1},
    reward={"name": "composite", "version": 3},
    requirements=frozenset({"distributed", "multi_turn", "tool_use"}),
)

backend = CommandTrainingBackend(
    name="verl",
    version="PINNED_INSTALLED_VERSION",
    capabilities=BackendCapabilities(
        algorithms=frozenset({"grpo", "gspo", "dapo"}),
        features=frozenset(
            {"distributed", "multi_turn", "tool_use", "async_rollouts"}
        ),
    ),
    command=[
        "python",
        "my_verl_adapter.py",
        "--request",
        "{request}",
        "--result",
        "{result}",
        "--output-dir",
        "{output_dir}",
    ],
)

result = backend.run(experiment)
```

Command arguments are passed directly to `subprocess` without a shell. The
three placeholders must each be a complete argument. Credentials belong in
the process environment; request/config/metadata mappings reject common secret
field names before files are written.

## Adapter result

The adapter must write this JSON object to `{result}`:

```json
{
  "protocol_version": 1,
  "backend": "verl",
  "backend_version": "PINNED_INSTALLED_VERSION",
  "experiment_sha256": "DIGEST_FROM_REQUEST",
  "artifact_uri": "/absolute/path/inside/output-dir/artifact",
  "metrics": {
    "samples_per_second": 1.5,
    "eval_score_final": 0.4,
    "cost_usd": 2.3
  },
  "metadata": {}
}
```

StateSet rejects mismatched engine identity/version, non-finite metrics,
experiment digest drift, missing artifacts, empty artifact directories, and
local artifacts outside `output_dir`. Remote artifact URIs are allowed for
engines whose durable checkpoint lives in object or provider storage.

## Capability safety

An experiment declares semantic requirements from:

- `async_rollouts`
- `distributed`
- `multi_turn`
- `multimodal`
- `tool_use`

Execution is rejected before launch if the selected backend does not declare
every requirement or the requested algorithm. This prevents a multi-turn run
from quietly becoming a single-turn approximation merely because a backend
cannot represent its environment.

## Adapter acceptance gate

An engine adapter is complete only after it:

1. applies every canonical field or rejects the experiment;
2. proves its installed version and experiment digest in the result;
3. emits a non-empty, reusable policy artifact;
4. passes a one-seed conformance preflight on real GPU hardware;
5. passes the matched three-seed framework comparison;
6. retains failures and costs alongside successes.

The benchmark-specific evidence schema remains stricter than this execution
contract; see [`BENCHMARKS.md`](BENCHMARKS.md).

For a real-GPU execution check, use the strict
[`backend_conformance.py`](../benchmarks/backend_conformance.py) runner with a
completed copy of
[`backend_conformance_manifest.example.json`](../benchmarks/backend_conformance_manifest.example.json).
It binds the backend result to the manifest and experiment digests, exact GPU
identities, runtime, wall time, and artifact hash. Success is adapter
conformance only; it is not comparative performance evidence.

Manifest schema v3 makes the execution envelope part of that immutable input:
provider and tier, digest-pinned container image, exact GPU name/count and disk,
workload timeout, total billable pod lifetime, and a finite positive dollar
ceiling. The local runner refuses GPU or timeout drift before invoking an
engine. `benchmarks/runpod_backend_conformance.py` supplies provider enforcement:
its default mode is a zero-cost catalog plan, while provisioning additionally
requires `--execute` plus an exact `--confirm-max-cost-usd`. It rechecks the
authoritative whole-pod price immediately after allocation and terminates on
drift, arms a remote lifetime watchdog, maintains a crash-recovery lease, and
records measured pod lifetime and estimated cost alongside downloaded evidence.

Conformance artifacts use paths relative to their evidence document. A complete
backend output directory can therefore be moved off the worker and revalidated
without preserving the worker's absolute filesystem layout. Once all engines
have run, `benchmarks/backend_conformance_suite.py` fails closed unless exactly
one artifact-valid NeMo RL, OpenRLHF, and verl record shares the same harness,
StateSet version, provider/GPU/time/cost envelope, algorithm, model revision,
seed, and task. The suite report
hashes the exact evidence documents; it does not claim benchmark parity.

## OpenRLHF adapter

StateSet includes an executable, version-pinned adapter for OpenRLHF's current
dotted-argument CLI:

```python
from stateset_agents.training import openrlhf_backend

backend = openrlhf_backend(version="0.10.2")
result = backend.run(experiment)
```

The engine remains an optional installation and is never imported while
listing StateSet backends. The adapter currently supports PPO, GRPO, and GSPO.
It verifies local dataset bytes, resolves remote models at an immutable commit,
and requires content hashes for Python reward and agent functions. It rejects
unknown configuration fields, mutable remote model revisions, unpinned reward
services/models, and unsupported environment semantics before launching
OpenRLHF. The adapter also verifies the installed OpenRLHF version and requires
a non-empty reusable model artifact after training.

The supported canonical config keys are:

- `learning_rate`, `train_batch_size`, `train_micro_batch_size`
- `rollout_batch_size`, `rollout_micro_batch_size`, `samples_per_prompt`
- `max_epochs`, `max_samples`, `max_length`, `max_new_tokens`
- `zero_stage`, `dtype`, `num_nodes`, `gpus_per_node`
- `vllm_num_engines`, `vllm_tensor_parallel_size`
- `kl_coefficient`, `temperature`, `top_p`
- `input_key`, `label_key`, `max_images_per_prompt`
- `apply_chat_template`, `packing_samples`, `gradient_checkpointing`
- `colocate_all`, `deterministic`

Real GPU conformance and matched three-seed benchmark evidence are still
required before claiming performance parity with OpenRLHF.

## verl adapter

The executable verl adapter targets the current Hydra-based
`verl.trainer.main_ppo` entrypoint:

```python
from stateset_agents.training import verl_backend

backend = verl_backend(version="PINNED_INSTALLED_VERSION")
result = backend.run(experiment)
```

The initial supported surface is intentionally narrow: PPO and GRPO with an
explicit `vllm` or `sglang` rollout engine, distributed execution, optional
multimodal data, a local content-addressed Parquet dataset, and a content-pinned
Python reward function. StateSet disables verl's implicit validation, W&B, and
checkpoint-resume defaults; propagates the canonical seed across data, actor,
reference, critic, and rollout workers; and forces a final checkpoint into the
experiment artifact directory.

Supported canonical config keys are:

- `learning_rate`, `train_batch_size`, `ppo_mini_batch_size`
- `ppo_micro_batch_size_per_gpu`, `rollout_samples`
- `max_prompt_length`, `max_response_length`
- `total_epochs`, `total_training_steps`
- `num_nodes`, `gpus_per_node`, `tensor_parallel_size`
- `gpu_memory_utilization`, `temperature`, `top_p`
- `prompt_key`, `image_key`, `rollout_engine`
- `kl_reward_coefficient`, `kl_loss_coefficient`
- `gradient_checkpointing`, `remove_padding`, `dynamic_batching`, `deterministic`

Multi-turn, tool-use, and asynchronous verl modes remain rejected until their
tool/agent-loop configuration files and staleness semantics are represented in
the backend protocol. Live GPU conformance and matched three-seed evidence also
remain open gates.

## NeMo RL adapter

The executable NeMo RL adapter targets NVIDIA's official Hydra-style
`examples/run_grpo.py` launcher:

```python
from stateset_agents.training import nemo_rl_backend

backend = nemo_rl_backend(version="PINNED_INSTALLED_VERSION")
result = backend.run(experiment)
```

NeMo RL is optional and is not imported while listing StateSet backends. The
engine environment must use NeMo RL's supported Python version and retain the
source checkout associated with the installed `nemo-rl` distribution because
the official launcher and recipe live under `examples/`. The exact installed
version is checked before execution.

The initial surface is deliberately narrow: GRPO, local JSON/JSONL response
data, the built-in single-turn `math` environment, the `hf_math_verify` reward,
vLLM generation, and optional distributed execution. A canonical request must
declare these exact semantics:

```python
environment={"type": "single_turn", "name": "math"}
reward={
    "type": "nemo_builtin",
    "name": "math",
    "implementation": "hf_math_verify",
}
```

Supported canonical config keys are:

- `learning_rate`, `train_global_batch_size`, `train_micro_batch_size`
- `num_prompts_per_step`, `num_generations_per_prompt`
- `max_num_steps`, `max_num_epochs`, `max_total_sequence_length`
- `num_nodes`, `gpus_per_node`, `tensor_parallel_size`
- `gpu_memory_utilization`, `temperature`, `top_p`, `top_k`, `precision`
- `input_key`, `output_key`, `environment_workers`, `generation_backend`
- `activation_checkpointing`, `colocated`, `shuffle`

The adapter verifies dataset bytes, resolves models at immutable revisions,
disables validation and external loggers, pins the seed for GRPO and dataset
splitting, forces checkpoint creation inside the result artifact, and rejects
unknown fields. `generation_backend`, `num_generations_per_prompt`, and the
cost-bounding `max_num_steps` must be explicit. PPO, custom
rewards/environments, multi-turn, multimodal, and
asynchronous modes remain fail-closed until their complete semantics are
represented and tested. Live GPU conformance and matched three-seed evidence
remain open gates.
