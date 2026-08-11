# StateSet Agents CLI Reference

Use `stateset-agents --help` to see the current runtime command list.

## Commands

### `stateset-agents version`

Show the installed version and runtime details.

```bash
stateset-agents version
stateset-agents version --json
```

### `stateset-agents train`

Run a lightweight training demo by default with `--stub`, or launch the configured training flow.

```bash
stateset-agents train
stateset-agents train --stub
stateset-agents train --config ./stateset_agents.yaml --episodes 10 --profile balanced
stateset-agents train --stub --dry-run
```

#### Options

- `--config PATH`: YAML or JSON config file.
- `--episodes INTEGER`: Override number of episodes (must be > 0).
- `--save PATH`: Optional checkpoint output directory.
- `--dry-run / --no-dry-run`: Validate configuration and print guidance.
- `--stub`: Run a fast stub flow with no external model downloads.
- `--profile [balanced|speed|quality]`: Training profile.

### `stateset-agents train-remote`

Run the SFT job from `improve` on local or rented GPU compute. Picks up where
`improve` leaves off — it consumes `curated.jsonl` and writes a trained adapter.

```bash
# On this machine
stateset-agents train-remote --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-0.8B

# On rented GPUs
stateset-agents train-remote --provider modal --gpu A100 \
    --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B

# Or on RunPod (GPU names are RunPod's own, e.g. "NVIDIA RTX A4000")
export RUNPOD_API_KEY=...
stateset-agents train-remote --provider runpod --gpu "NVIDIA RTX A4000" \
    --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B

# See the plan without training (works with no GPU)
stateset-agents train-remote --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-0.8B --dry-run
```

The job itself is `stateset_agents.training.sft` — identical whichever provider
runs it (`scripts/sft_from_curated.py` is a thin CLI over the same code).
Remote runs install a pinned published `stateset-agents[training]` rather than
syncing your working tree, so a remote run is reproducible; the tradeoff is
that testing an unreleased change remotely needs a dev release.

A remote run succeeds only if it actually produces an adapter: a container
that exits cleanly having written nothing is reported as a failure, not a
success with an empty output directory.

#### Options

- `--dataset PATH`: Chat-format JSONL to train on (required).
- `--base-model TEXT`: Hugging Face base model (required).
- `--provider [local|modal|runpod]`: Where to run. Default `local`.
- `--output-dir PATH`: Adapter output directory. Default `outputs/sft_v1`.
- `--num-epochs`, `--lora-r`, `--lora-alpha`, `--learning-rate`,
  `--max-length`, `--per-device-batch-size`,
  `--gradient-accumulation-steps`: Passed through to the training script.
- `--gpu TEXT`: GPU type to request (remote only). Default `A10G`.
- `--timeout INTEGER`: Job timeout in seconds. Default `3600`.
- `--package-version TEXT`: Version installed remotely. Defaults to the
  running version.
- `--dry-run`: Print the training plan without training.

#### Providers

| Provider | Needs | Transport | Notes |
|---|---|---|---|
| `local` | a GPU on this machine | none | Verified end-to-end |
| `runpod` | `RUNPOD_API_KEY`, an SSH keypair, `ssh`/`scp` on PATH | SSH/SCP to a rented pod | **Verified end-to-end on live hardware** (RTX A4000, Qwen3.5-0.8B, ~5 min). GPU names are RunPod's own (`"NVIDIA RTX A4000"`) |
| `modal` | `pip install "stateset-agents[modal]"` | Modal Volume | Transport **not** yet verified against a live account |

RunPod creates the pod with TCP 22 exposed and your public key
(`~/.ssh/id_ed25519.pub` or `id_rsa.pub`) injected, copies the dataset in,
runs the job, copies the adapter back, and **terminates the pod on every exit
path** — including failures and timeouts — so nothing keeps billing. No
network volume is created, so there is no storage cost after the run.

To test an unreleased change on real hardware, point the RunPod executor at a
locally built wheel instead of PyPI (the pinned version cannot resolve before
it is published):

```python
RunPodExecutor(wheel=Path("dist/stateset_agents-0.20.0-py3-none-any.whl"))
```

### `stateset-agents qwen3-5-0-8b`

Preview or run the dedicated starter path for `Qwen/Qwen3.5-0.8B`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents qwen3-5-0-8b
stateset-agents qwen3-5-0-8b --json-output
stateset-agents qwen3-5-0-8b --starter-profile memory --json-output
stateset-agents qwen3-5-0-8b --list-profiles --json-output
stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json
stateset-agents qwen3-5-0-8b --config ./qwen3_5_0_8b.json --no-dry-run
stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Qwen starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`Qwen/Qwen3.5-0.8B-Base` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents kimi-k2-6`

Preview or run the dedicated starter path for `moonshotai/Kimi-K2.6`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents kimi-k2-6
stateset-agents kimi-k2-6 --json-output
stateset-agents kimi-k2-6 --starter-profile memory --json-output
stateset-agents kimi-k2-6 --list-profiles --json-output
stateset-agents kimi-k2-6 --write-config ./kimi_k2_6.json
stateset-agents kimi-k2-6 --config ./kimi_k2_6.json --no-dry-run
stateset-agents kimi-k2-6 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Kimi starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`moonshotai/Kimi-K2.6` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents kimi-k3`

Preview or run the dedicated starter path for `moonshotai/Kimi-K3`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.
Note: the `moonshotai/Kimi-K3` ID is provisional — HF weights are not yet published (as of 2026-07-16).

```bash
stateset-agents kimi-k3
stateset-agents kimi-k3 --json-output
stateset-agents kimi-k3 --starter-profile memory --json-output
stateset-agents kimi-k3 --list-profiles --json-output
stateset-agents kimi-k3 --write-config ./kimi_k3.json
stateset-agents kimi-k3 --config ./kimi_k3.json --no-dry-run
stateset-agents kimi-k3 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Kimi starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`moonshotai/Kimi-K3` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents muse-glimmer`

Preview or run the dedicated starter path for `meta-models/Muse-Glimmer-30B`,
Meta's open agentic model (Aug 2026; dense 30B, 131K ctx, Apache-2.0).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents muse-glimmer
stateset-agents muse-glimmer --json-output
stateset-agents muse-glimmer --starter-profile memory --json-output
stateset-agents muse-glimmer --list-profiles --json-output
stateset-agents muse-glimmer --write-config ./muse_glimmer.json
stateset-agents muse-glimmer --config ./muse_glimmer.json --no-dry-run
stateset-agents muse-glimmer --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Muse Glimmer starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`meta-models/Muse-Glimmer-30B` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents nemotron-3-5`

Preview or run the dedicated starter path for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
NVIDIA's hybrid Mamba-2/attention/MoE open model (Aug 2026; 30B total / ~3B active, 256K ctx, OpenMDW-1.1).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents nemotron-3-5
stateset-agents nemotron-3-5 --json-output
stateset-agents nemotron-3-5 --starter-profile memory --json-output
stateset-agents nemotron-3-5 --list-profiles --json-output
stateset-agents nemotron-3-5 --write-config ./nemotron_3_5.json
stateset-agents nemotron-3-5 --config ./nemotron_3_5.json --no-dry-run
stateset-agents nemotron-3-5 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Nemotron 3.5 starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents validate-config`

Validate a training config without running training.

```bash
stateset-agents validate-config --config ./stateset_agents.json
stateset-agents validate-config --config ./stateset_agents.yaml --strict --json-output
stateset-agents validate-config --config ./stateset_agents.yaml --fail-on-warnings
```

Options:

- `--config PATH`: YAML or JSON config path.
- `--strict`: Exit non-zero when validation errors are found.
- `--fail-on-warnings`: Exit non-zero when validation warnings are found.
- `--json-output`: Emit machine-readable result with `valid`, `errors`, and `warnings`.

### `stateset-agents serve`

Run the API gateway (`stateset_agents.api.main`) with Uvicorn.

```bash
stateset-agents serve
stateset-agents serve --host 0.0.0.0 --port 8000 --reload
stateset-agents serve --dry-run
```

#### Options

- `--host TEXT`: Bind host.
- `--port INTEGER`: Bind port.
- `--reload`: Enable auto-reload (development).
- `--dry-run`: Preview startup command without launching the server.

### `stateset-agents doctor`

Check common runtime dependencies.

```bash
stateset-agents doctor
stateset-agents doctor --strict
stateset-agents doctor --json-output
stateset-agents doctor --strict --json-output
```

`--strict` exits with non-zero status if required dependencies are missing.
`--json-output` writes a JSON payload with `required_dependencies` and `optional_dependencies`.

### `stateset-agents evaluate`

Run a single message through a checkpointed agent.

```bash
stateset-agents evaluate --checkpoint ./checkpoints/agent --message "Hello"
stateset-agents evaluate --dry-run --message "Hello"
```

### `stateset-agents init`

Generate a starter config (`yaml` default, `json` optional).

```bash
stateset-agents init
stateset-agents init --path ./stateset_agents.yaml --format json
stateset-agents init --path ./stateset_agents.yaml --overwrite --format yaml
stateset-agents init --preset qwen3-5-0-8b --path ./qwen3_5_0_8b.json --format json
stateset-agents init --preset qwen3-5-0-8b --starter-profile memory --path ./qwen3_5_0_8b_memory.json --format json
```

Options:

- `--path PATH`: Output config path.
- `--overwrite`: Replace an existing file.
- `--format [yaml|json]`: Output file format.
- `--preset [default|qwen3-5-0-8b|kimi-k2-6|kimi-k3|gemma-4-31b|muse-glimmer|nemotron-3-5]`: Starter config preset.
- `--task TEXT`: Task preset for model-specific starter configs.
- `--starter-profile TEXT`: Starter profile for model-specific starter configs.

Aliases:

- `stateset-agents init-config` is equivalent to `stateset-agents init`.

### `stateset-agents advanced`

Experimental command bundle for advanced workflows:

- `debug`
- `profile`
- `validate`
- `progress`
- `tree`

This group is loaded only when optional advanced CLI dependencies are available.

### `stateset-agents preflight`

Run dependency and (optional) config checks together.

```bash
stateset-agents preflight
stateset-agents preflight --config ./stateset_agents.yaml
stateset-agents preflight --config ./stateset_agents.json --strict --json-output
```

Options:

- `--config PATH`: Validate this config as part of the preflight.
- `--strict`: Fail on missing required dependencies.
- `--fail-on-warnings`: Fail when validation warnings are present.
- `--json-output`: Return JSON payload for automation.

### `stateset-agents publish-check`

Run a preflight check plus import smoke checks before publishing.

```bash
stateset-agents publish-check
stateset-agents publish-check --config ./stateset_agents.yaml
stateset-agents publish-check --config ./stateset_agents.yaml --strict --json-output
stateset-agents publish-check --config ./stateset_agents.yaml --fail-on-warnings --json-output
```

Options:

- `--config PATH`: Validate this config as part of publish checks.
- `--strict`: Fail when required dependencies or required imports are missing.
- `--fail-on-warnings`: Fail when validation warnings are present.
- `--json-output`: Return JSON payload with dependency/import/config status.

### `stateset-agents chat`

Open an interactive REPL against an in-process agent.

```bash
stateset-agents chat
```

#### Options

- `--model, -m TEXT`: HF model name or stub://<id> for the in-process REPL.
- `--checkpoint, -c TEXT`: Path to a LoRA adapter to load on top of --model.
- `--system TEXT`: Optional system prompt prepended to every conversation.
- `--max-new-tokens INTEGER`: Generation length cap per response.
- `--history TEXT`: Path to a JSONL file to APPEND each turn (one JSON object per line). Capture interesting conversations to replay or grade later with `make grade-transcript`.
- `--replay TEXT`: Path to a JSONL transcript to replay as initial conversation context. Useful for resuming a debugging session.
- `--grade TEXT`: Score each assistant turn live with the named reward function. Options: gsm8k, customer_support, tool_calling. Mismatches between intuition and score surface reward-function bugs.

### `stateset-agents fine-tune`

Fine-tune from a curated JSONL in one command.

```bash
stateset-agents fine-tune CURATED
```

#### Options

- `--base-model, -m TEXT`: HF base model to fine-tune.
- `--output-dir, -o TEXT`: Where the LoRA adapter is saved.
- `--min-score FLOAT`: Drop curated examples below this score before SFT.
- `--num-epochs, -e INTEGER`: Training epochs.
- `--lora-r INTEGER`: LoRA rank.
- `--dry-run`: Print the training plan without running it (forced when no GPU).

### `stateset-agents improve`

Run the grade -> curate -> retrain loop as a single command.

```bash
stateset-agents improve ACTION
```

#### Options

- `--transcripts TEXT`: For --format transcripts: a directory of transcript JSONL files (one conversation per file, {'role','content'} per line — the shape `stateset-agents chat --history` writes). For --format openai/langchain: the single source log file to ingest first.
- `--reward TEXT`: Reward function: gsm8k, customer_support, or tool_calling (rule-based, no API key required).
- `--output, -o TEXT`: Output directory for curated data + reports.
- `--threshold FLOAT`: Minimum score for curation (default: 0.7).
- `--format, -f TEXT`: Input format: 'transcripts' (already chat-history JSONL), 'openai', or 'langchain' (ingested first via stateset_agents.data.trajectory_ingest).

### `stateset-agents ingest`

Convert third-party conversation logs into graded-history JSONL.

```bash
stateset-agents ingest
```

#### Options

- `--format, -f TEXT`: Source log format: 'openai' (chat-completions messages JSONL) or 'langchain' (LangChain/LangGraph message-dump JSON).
- `--input, -i TEXT`: Path to the source log file. For --format openai: JSONL, one conversation per line ({'messages': [...]} or a bare message list). For --format langchain: a single JSON file (see stateset_agents.data.trajectory_ingest docstring for supported shapes).
- `--output, -o TEXT`: Output path. If it ends in .jsonl, all conversations are concatenated into one graded-history JSONL file (turns from different conversations are separated by a blank line — note the grader treats such a file as ONE transcript; use directory mode to grade conversations separately). Otherwise it is treated as a directory and one <output>/conversation_<N>.jsonl file is written per conversation — feed any of them to `python scripts/grade_transcript.py --history <file>`.

### `stateset-agents mcp`

Run the StateSet Agents MCP server (stdio transport by default).

```bash
stateset-agents mcp
```

#### Options

- `--transport TEXT`: MCP transport to serve over (default: stdio).

### `stateset-agents auto-research`

Run the autonomous research loop to optimize agent training.

```bash
stateset-agents auto-research
```

#### Options

- `--config, -c TEXT`: Path to auto-research config file (YAML/JSON).
- `--max-experiments, -n INTEGER`: Maximum experiments to run (0 = unlimited).
- `--time-budget, -t INTEGER`: Wall-clock seconds per experiment.
- `--proposer, -p TEXT`: Proposer strategy: perturbation, smart, adaptive, random, grid, bayesian, llm.
- `--algorithm, -a TEXT`: Training algorithm: gspo, grpo, dapo, vapo.
- `--output-dir, -o TEXT`: Directory for results and checkpoints.
- `--search-space, -s TEXT`: Search space: grpo, auto_research, quick, reward, model, multi_algorithm, full.
- `--improvement-patience INTEGER`: Stop after this many consecutive non-improvements (0 = disabled).
- `--max-wall-clock INTEGER`: Total wall-clock budget in seconds (0 = unlimited).
- `--wandb`: Log experiments to Weights & Biases.
- `--wandb-project TEXT`: W&B project name.
- `--stub`: Run with stub model for testing the loop without GPU.
- `--dry-run`: Validate config and show plan without running.

### `stateset-agents benchmark`

Run and aggregate Phase 0 / whitepaper-v1 benchmarks.

Subcommands:

- `aggregate`: Aggregate all *.json results in a directory into summary.md + summary.csv.
- `phase0`: Run a single Phase 0 benchmark and emit a schema-compliant JSON result.
- `plot`: Generate publication figures from aggregated benchmark results.
- `smoke`: Quick end-to-end smoke test of the GSM8K benchmark pipeline (no training).

```bash
stateset-agents benchmark --help
```

### `stateset-agents recipe`

Open a cookbook recipe in $PAGER, or `list` them all.

```bash
stateset-agents recipe NAME
```

### `stateset-agents starter`

Scaffold a fork-and-go fine-tuning project.

```bash
stateset-agents starter TEMPLATE OUTPUT
```

#### Options

- `--name, -n TEXT`: Project name (defaults to the basename of the output directory).
- `--force, -f`: Overwrite an existing non-empty directory.
- `--client-name TEXT`: Client name (slugified) — patches output_dir paths and the W&B project name throughout the scaffold.

### `stateset-agents tour`

Open the platform tour — the one document that walks the full developer journey.

```bash
stateset-agents tour
```

### `stateset-agents init-config`

Alias for `init`.

```bash
stateset-agents init-config
```

#### Options

- `--path TEXT`: Path for a starter config
- `--overwrite`: Overwrite existing file
- `--format, -f TEXT`: Output format: yaml or json
- `--preset TEXT`: Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, or nemotron-3-5
- `--task TEXT`: Task preset for model-specific starter presets.
- `--starter-profile TEXT`: Starter profile for model-specific starter presets.

### `stateset-agents gemma-4-31b`

Preview or run the dedicated Gemma 4 31B GSPO starter path.

```bash
stateset-agents gemma-4-31b
```

#### Options

- `--config, -c TEXT`: Path to a Gemma 4 31B starter config file (JSON/YAML).
- `--task TEXT`: Task preset for the Gemma 4 31B starter path.
- `--starter-profile TEXT`: Starter profile: balanced, memory, or quality.
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name. For post-training, use google/gemma-4-31B-it.
- `--use-lora / --no-lora`: Override LoRA usage. Defaults come from --starter-profile.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization. Defaults come from --starter-profile.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization. Defaults come from --starter-profile.
- `--output-dir TEXT`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count for the starter run.
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config TEXT`: Write the resolved Gemma starter config to JSON/YAML and exit.
- `--dry-run / --no-dry-run`: Preview the resolved config instead of loading a model.
- `--json, --json-output`: Output machine-readable JSON.

## Exit behavior

- Non-zero exit codes indicate command failures (e.g., missing modules or invalid input).
- Use `--dry-run` modes to inspect intended behavior before running heavy operations.

## Troubleshooting

- If `version`/`doctor`/`serve` fail to import modules, install optional extras as needed:
  - API extras for serving (`fastapi`, `uvicorn`)
  - Rich tooling for `advanced` workflows (`rich`, `ipython`)
- If config loading fails, check file path, extension, and YAML/JSON syntax.
