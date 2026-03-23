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
- `--preset [default|qwen3-5-0-8b]`: Starter config preset.
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

## Exit behavior

- Non-zero exit codes indicate command failures (e.g., missing modules or invalid input).
- Use `--dry-run` modes to inspect intended behavior before running heavy operations.

## Troubleshooting

- If `version`/`doctor`/`serve` fail to import modules, install optional extras as needed:
  - API extras for serving (`fastapi`, `uvicorn`)
  - Rich tooling for `advanced` workflows (`rich`, `ipython`)
- If config loading fails, check file path, extension, and YAML/JSON syntax.
