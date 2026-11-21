# Native Local Engine: Run, Finetune, and Edit with stateset-agents

Status: MVP implemented alongside this design. Decisions recorded here came
from an explicit scope choice: design plus MVP code, a dual-engine strategy,
and a laptop-inference / GPU-training hardware split.

## Problem

Models such as `thinkingmachines/Inkling-2-Mini` break the library's single
assumption that the installed Transformers runtime can load every
checkpoint. On the reference laptop (32 GiB RAM, Intel integrated graphics,
no CUDA, Transformers 5.14.1 in the project venv):

- Loading the checkpoint through the native agent fails before any weights
  move: `ValueError: model type inkling2_loop … Transformers does not
  recognize this architecture`, raised through
  `stateset_agents/core/transformers_compat.py:load_generation_model`.
- The same model runs locally through llama.cpp from a mainline-layout GGUF
  quant: an observed single-turn generation produced the correct answer at
  8.1 prompt tok/s and 3.7 generation tok/s with ~12.1 GiB peak RSS, and the
  stateset API gateway service answered through a local `llama-server`.
- Finetuning a 26.4B NVFP4 checkpoint needs safetensors, autograd, and CUDA
  memory. None of that exists in llama.cpp or on this laptop.

One engine cannot honestly cover all three jobs on this hardware. GGUF is an
inference format with no autograd. A dequantized PyTorch copy of a 26.4B
model needs about 52.8 GB of BF16 weights alone. The design therefore keeps
two engines with one canonical artifact flow.

## Decision

Dual engine, safetensors canonical:

```text
                         GPU / remote host
  HF checkpoint (safetensors, canonical)
        |  GSPO/SFT starters (existing)      weight edits (this MVP +
        |  stateset_agents.training.*         experimental.weight_space)
        v                                          |
  adapter / edited checkpoint  -- LoRA merge / delta (weight_edit.py)
        |
        |  convert_hf_to_gguf (+ optional llama-quantize)
        v                                    laptop / edge host
  GGUF artifact  -------------------------->  llama-server (llama.cpp)
                                                     |
                       stateset Agents  <--- rollout backend (token ids
                       API gateway     <--- OpenAI HTTP  + log-probs)
```

- Canonical weights are always the Hugging Face safetensors checkpoint.
  Finetuning and weight edits read and write that format only.
- GGUF is a derived, disposable serving artifact. It is regenerated, never
  edited.
- Local inference integrates at two existing seams: the
  `MultiTurnAgent.set_rollout_backend` contract (exact prompt/response token
  ids plus per-token log-probs, as returned by
  `training.vllm_backend.GenerationResult`) and the API gateway's HTTP
  inference service (`POST {base_url}/v1/chat/completions`).

## Components (MVP)

### `stateset_agents/training/llama_cpp_backend.py`

Standard-library-only client and process manager for llama.cpp:

- `LlamaCppServer` owns one `llama-server` child process bound to loopback
  by default, with health polling and deterministic shutdown.
- `LlamaCppClient` speaks `/tokenize`, `/completion` (with `n_probs` and
  `return_tokens`), and `/v1/chat/completions` over HTTP.
- `LlamaCppRolloutBackend.generate_with_logprobs` adapts completions to the
  existing rollout contract, returning `GenerationResult`-shaped values
  with finite, length-matched token ids and log-probs. It deliberately has
  no `sync_weights`: a GGUF file cannot accept live PyTorch weights, so
  `MultiTurnAgent.sync_rollout_backend` correctly marks it stale during
  training. It is for local inference, evaluation, and frozen-policy
  rollouts.

Binary discovery order is the explicit `binary` setting, the
`LLAMA_SERVER_BIN` environment variable, then `llama-server` on `PATH`.
Nothing is installed or put on `PATH` implicitly.

On CUDA hosts, the `inkling2-speed` branch of
[realugbun/llama.cpp](https://github.com/realugbun/llama.cpp) is a
drop-in faster binary: upstream master plus Inkling-2 graph cleanups
(normalized depth states built once per block) and fused CUDA kernels for
the model's glue op chains, reporting 2.0–4.1× prefill and +37–39%
decode on an RTX 5060 Ti with perplexity KLD at most 0.0022 against
upstream. Because the backend is binary-agnostic, using it needs no code
change — point `LLAMA_SERVER_BIN` at that build. Its depth-states commit
is backend-agnostic and may also help CPU builds, but that is unverified
on the reference laptop; the fused kernels are CUDA-only. Any
GGUF-parity gate we add should reuse its method: `llama-bench`
throughput plus perplexity KLD against the reference build, not text
spot checks alone. The fork is inference-only and changes nothing about
the dual-engine decision.

### API gateway backend `llamacpp`

`InferenceConfig` accepts `INFERENCE_BACKEND=llamacpp` with
`INFERENCE_BACKEND_URL=http://127.0.0.1:<port>`. llama-server implements
the same `/v1/chat/completions` and `/health` contract the gateway already
proxies for vLLM; the distinct name keeps metrics and logs honest.

### `stateset_agents/training/weight_edit.py`

Checkpoint-level tensor surgery that never requires a model class, so it
also works for architectures the installed Transformers cannot load:

- `inspect_checkpoint` inventories tensor names, shapes, dtypes, and shards
  from `model.safetensors.index.json` or a single-file header.
- `merge_lora_adapter` folds a PEFT adapter (`adapter_config.json` +
  safetensors `lora_A`/`lora_B` pairs) into a base checkpoint directory,
  one shard at a time, copying untouched shards byte-for-byte.
- `apply_safetensors_delta` adds a scaled safetensors delta file to the
  named tensors it contains.
- Every edit refuses in-place output, fails on unknown tensors or shape
  mismatches, and writes `weight_edit_manifest.json` recording the
  operation, each changed tensor, its delta L2/max, and the SHA-256 of the
  edited tensor bytes.

Fact-style teaching edits (EditSpec/`weight-edit teach`) remain owned by
the in-flight `stateset_agents/experimental/weight_space` work on
`feat/weight-space-side-effects` and are not re-implemented here. Its specs
produce safetensors checkpoints, so they compose with this design through
the same export step.

### `stateset_agents/training/gguf_export.py`

Wraps llama.cpp's own `convert_hf_to_gguf.py` — the same converter that
produced the verified mainline Inkling GGUFs — instead of growing a native
GGUF writer:

- `build_convert_command` / `build_quantize_command` construct the exact
  argv, so callers can log or inspect a conversion before running it.
- `convert_to_gguf` validates that the converter exists under
  `LLAMA_CPP_DIR` (or an explicit path), supports a dry run, and runs the
  converter as a subprocess with checked exit status.
- Quantization stays an explicit second step via `llama-quantize`. Small
  quants trade quality for fit; the only locally verified Inkling quant to
  date on the reference laptop is `UD-IQ3_XXS` (about 11 GiB, 3.06 bpw).

### End-to-end flows

Run locally (no GPU):

```python
from stateset_agents.training.llama_cpp_backend import (
    LlamaCppRolloutBackend, LlamaCppServer,
)

server = LlamaCppServer("/path/Inkling-2-Mini-UD-IQ3_XXS.gguf")
with server:
    backend = LlamaCppRolloutBackend.from_server(server)
    agent.set_rollout_backend(backend)
    client_reply = server.client.chat(
        [{"role": "user", "content": "What is 84 * 3 / 2?"}]
    )
```

The rollout backend consumes prompts the caller has already rendered (the
same contract as the vLLM backend), so trainers should render with the
policy tokenizer's chat template. For direct conversation, prefer
`LlamaCppClient.chat` or the API gateway, where llama-server applies the
GGUF chat template; raw completions from a stub-rendered prompt can run
past the intended turn because no chat stop tokens are present.

Finetune, then serve locally: run the existing starter on a GPU host
(`stateset-agents inkling-2-mini --no-dry-run`), fold the adapter with
`merge_lora_adapter`, export with `convert_to_gguf`, and point
`LlamaCppServer` or the gateway at the result.

Edit, then verify locally: apply a weight edit to safetensors, check the
manifest, export a GGUF, and run the same prompts before/after through the
rollout backend so the comparison uses one sampler.

## Contracts and failure behavior

- Rollout results must echo the exact prompt, keep response token ids and
  log-probs length-matched and finite, and map llama.cpp stop reasons
  (`eos`, `word` to `stop`; `limit` to `length`; anything else to
  `abort`). Malformed server payloads raise instead of being padded.
- The server manager binds `127.0.0.1` unless told otherwise, never
  downloads models, and terminates only the child it started.
- Weight edits are copy-on-write directory transforms. Partial failures
  leave the base checkpoint untouched because shards are written into a
  fresh output directory; callers should delete incomplete outputs.
- Conversion is lossy only at the explicit quantize step. Parity between a
  safetensors checkpoint and its GGUF export is a sampling comparison
  (same prompts, temperature 0), never a byte comparison.

## Hardware matrix

| Job | This laptop (32 GiB, no CUDA) | GPU host |
| --- | --- | --- |
| GGUF chat / gateway / rollouts | Yes (verified for Inkling IQ3) | Yes |
| HF native agent inference | Only architectures Transformers knows | Yes |
| GSPO/SFT finetuning | Dry-run and config only | Yes |
| Tensor-level weight edits | Small checkpoints; shard-bounded RAM | Yes |
| LoRA merge / GGUF export | Small and mid checkpoints | Yes |

## Non-goals

- Training inside ggml, or a stateset-owned tensor runtime.
- A vendored PyTorch `Inkling2Loop` implementation. If upstream
  Transformers ships one, the native HF path starts working with no API
  change and this design keeps the GGUF path for weak hardware.
- Vision/audio inputs over GGUF. The MVP is text-only; `mmproj` support
  belongs with llama.cpp's mtmd surface as a later phase.
- Automatic model download or llama.cpp installation.

## Later phases (not in the MVP)

1. A `stateset-agents run-local` CLI over `LlamaCppServer`.
2. Landing `experimental/weight_space` and pointing its checkpoint
   application at the same export/verify loop.
3. Frozen-policy rollout refresh: re-export GGUF per N optimizer steps
   with explicit staleness accounting for slow CPU evaluation loops.
4. mtmd (vision) support and streaming chat through the gateway.
