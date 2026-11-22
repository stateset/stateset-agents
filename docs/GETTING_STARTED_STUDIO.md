# Getting Started with StateSet 3D Weight Studio

Explore neural network weights in real-time WebGL2 3D, perform non-destructive surgical weight edits, extract and bake activation steering vectors, trace causal attention circuits, run live PyTorch micro-finetuning, and export SHA-256 verified `.safetensors` checkpoints—all from your browser.

```
local checkpoints (.safetensors / LoRA / sharded HF models)
        │
        ▼
  stateset-agents studio --port 8765 --open
        │
        ├──▶ 5 Interactive 3D Topologies (Architecture, Microscope 128², SVD, Trajectory 4D, Circuit 3D)
        ├──▶ Non-Destructive Weight Surgery (Wanda Prune, Head Surgery, ROME, TIES/DARE/SLERP, 3D Brush)
        ├──▶ Mechanistic Probing (PyTorch Hooks, Logit Lens, Causal Patching, CAA Steering + Baking)
        ├──▶ Live RL Finetuning Workbench (GSPO/GRPO/DAPO, Dataset Traces, VRAM Budget, Live SSE)
        └──▶ Pre-Export 6-Case Eval Gate (GO/WARN/BLOCK) ──▶ Verified .safetensors + Edit Recipe JSON
```

For the full API, shader pipeline, and architecture reference, see [`docs/WEIGHT_STUDIO.md`](WEIGHT_STUDIO.md).

---

## Prerequisites

- **Python 3.10+** with `stateset-agents` installed and the `[api]` extra (`fastapi`, `uvicorn`, `safetensors`, `numpy`).
- Optional: `torch` + `transformers` for live PyTorch micro-finetuning and genuine `AutoModelForCausalLM` forward-hook probing on local checkpoints.
- Any modern desktop browser with **WebGL2** enabled (Chrome, Edge, Firefox, Safari). The Studio UI has **zero external CDN dependencies** and works completely offline.

```bash
pip install -e ".[api]"
```

---

## Launching the Studio

You can start the Weight Studio in three ways:

### 1. CLI Command (Recommended)

```bash
stateset-agents studio --host 127.0.0.1 --port 8765 --workspace . --open
```

### 2. Python Module

```bash
python -m stateset_agents.weight_studio --host 127.0.0.1 --port 8765 --workspace .
```

### 3. Mounted on the Main StateSet FastAPI Gateway

When running the standard API server (`python -m stateset_agents.api.main`), the Studio is automatically mounted at:

- **Browser UI:** `http://127.0.0.1:8000/studio`
- **Studio API:** `http://127.0.0.1:8000/api/weight-studio/*`

### 4. Runnable Python Quick Start (Headless or Browser)

Try the end-to-end programmatic workflow (inspection → CAA steering → Wanda pruning + weight bake → causal circuit discovery → 6-case eval gate → recipe export) in 2 seconds with zero GPU required:

```bash
python examples/weight_studio_quickstart.py
# Or run the workflow and immediately open the interactive 3D browser UI:
python examples/weight_studio_quickstart.py --serve --port 8765 --open
```

> **Running on a remote GPU box?** Keep `--host 127.0.0.1` on the server and forward the port over SSH:
> ```bash
> ssh -L 8765:127.0.0.1:8765 user@your-gpu-host
> ```
> Then open `http://127.0.0.1:8765/studio` in your local browser.

---

## 5-Minute Hands-On Walkthrough

### Step 1 — Load a Model or Architecture Twin

When the Studio opens at `http://127.0.0.1:8765/studio`, it automatically scans your workspace (depth-bounded, lazy 8-byte `.safetensors` header parsing) for:

- **LoRA adapters** (`adapter_model.safetensors`, including multi-step checkpoint trajectories under `checkpoint-*` or `step-*`)
- **Full local checkpoints** (`model.safetensors` or multi-shard `model.safetensors.index.json`)
- **Built-in StateSet Architecture Twins** (available immediately even in an empty workspace so you can explore Qwen, Llama, and hybrid linear-attention stacks right away)

Use the **Active Model / Checkpoint** dropdown in the top bar to switch models, and click **↻** at any time to rescan newly saved checkpoints.

- **Navigate the 3D bench:** Left-drag to orbit, Right-drag (or `Alt`/`Ctrl` + drag) to pan, Scroll to zoom, and use `ISO` / `TOP` / `FRONT` / `SIDE` / `FIT` in the bottom-left camera dock.
- **Customize optics:** Switch between **4 Studio Lighting Rigs** (`Obsidian Lab`, `Copper Forge`, `Deep Photonics`, `Cleanroom White`), **7 Scientific Colormaps** (`Signed (±)`, `Magnitude |W|`, `Edit Delta ΔW`, `Wanda/Fisher`, `Sign Conflict`, `Thermal Heat`, `Outlier Z-Score`), and toggle `REFL` (planar reflections), `SHDW` (soft shadow maps), `SSAO`, `DOF` (bokeh depth-of-field), `RAILS`, and `RULER`.

---

### Step 2 — Inspect a Weight Matrix in `Microscope 128²`

1. Click any tensor in the **Architecture Tree** (left rail) or click a 3D module on the wafer chassis to focus it.
2. **Double-click** the tensor (or press `2` / click **`Microscope 128²`**) to open its 3D crystal matrix view.
3. Switch the resolution selector from `64² (4K)` to **`128² (16.4K Bin)`** to stream high-resolution tensors over the binary `Float32Array` endpoint (`/api/weight-studio/models/tensor.bin`).
4. Toggle **`Heads`** to display translucent GQA attention-head partition planes (`H0`, `H1`, …) and **`Ghost`** to overlay the pre-edit silhouette after surgical changes.
5. Click **`✦ 3D Brush`** (or hold `Shift` + drag across the crystal terrain) to paint a rectangular matrix patch, then click **`⊕ Crop`** to deep-zoom directly into that sub-region.

---

### Step 3 — Perform Non-Destructive Weight Surgery & Split A|B Comparison

All edits are stored in an **in-memory non-destructive overlay**—your original `.safetensors` files on disk are never modified.

In the right workbench **`Surgery`** tab:

1. Choose your **Target Scope** (`Selected Tensor Only`, `Entire Selected Layer`, `All Self-Attention Weights`, `All Linear-Attention Weights`, `All MLP Weights`, or `All Model Parameters`).
2. Pick a surgical operation:
   - **Scale:** Multiply weights by a factor `α`.
   - **Prune:** Choose **Wanda + Fisher Importance (Smart)** or Magnitude Percentile pruning. (Tip: switch the topbar colormap to `Map: Wanda/Fisher` first to see which crystals are critical vs. prunable.)
   - **Clamp:** Clip extreme outliers at percentile `P80–P99.9`.
   - **SVD Rank:** Keep top-`k` singular modes (or switch to **`SVD Spectral`** mode with key `3` to preview rank truncation live in 3D before committing).
   - **Head Surgery:** Ablate, scale, prune, or sign-invert a specific GQA attention head (`H0..Hn`).
   - **ROME Rank-1:** Apply a rank-1 associative key-value update with automated 12-probe protected-prompt cosine drift verification.
   - **3D Brush:** Zero, scale, invert, or clamp the exact matrix patch you painted on the 3D stage.
   - **TIES/SLERP:** Merge with a second checkpoint using **TIES-Merging**, **DARE**, **SLERP**, or Linear blending. Click **`✦ Visualize 3D Sign-Conflict Map (A vs B)`** first to inspect opposing weight signs in blazing magenta vs. aligned emerald.
3. Click **`Apply Surgical Edit`**. Watch the 3D crystal terrain liquid-morph smoothly, and use **`↶ Undo`**, **`↷ Redo`**, or individual **`ON`/`OFF`** toggles in the **Edit Recipe & Lineage Stack** to A/B test any edit.
4. While in **`Microscope 128²`**, click **`◧ Split A|B`** (or press `S`) and drag the vertical glass wipe curtain left and right to compare the **Unedited Base** terrain against the **Live Edited** terrain in real time.

---

### Step 4 — Probe, Steer & Trace 3D Causal Circuits (`Use Model` & `Circuit 3D`)

Switch to the **`Use Model`** tab in the right workbench:

1. **Run Inference & Logit Lens:** Click one of the built-in StateSet scenarios (*Refund Policy*, *Structured Ledger Call*, *Multi-Step Policy Reasoning*, *Policy Boundary Check*) or type your own prompt, then click **`⚡ Generate & Fire 3D Wave`**.
   - Local Hugging Face causal LM checkpoints automatically run genuine PyTorch `AutoModelForCausalLM` forward passes with `register_forward_hook`.
   - Optionally expand **Live vLLM / OpenAI Endpoint (`serve-remote`)** to send completions to a live OpenAI-compatible server.
   - Click any output token to inspect its **Top-6 Token Logit Lens**, entropy, KL shift vs. base, and StateSet multi-axis reward bars.
   - Click any row in the **Layer-by-Layer Logit Lens (`h_ℓ · W_U`)** to jump the 3D camera directly to that transformer layer.
2. **Causal Activation Patching:** Pick a layer and intervention (`Ablate`, `Amplify 2.2×`, `Dampen 0.25×`, `Invert (−h)`), click **`+ Patch`**, and re-run inference to measure the exact causal shift against the unpatched base run.
3. **Contrastive Activation Steering (CAA / RepE) & Weight Baking:**
   - Select a preset in **Activation Steering** (e.g., *Empathetic Resolution*, *Strict Policy Guard*, *Structured JSON Tool-Use*, *Verifiable Stepwise Math*) or edit the positive/negative contrast prompts.
   - Click **`Extract`** to compute the contrastive steering vector `v_ℓ = h_ℓ(pos) - h_ℓ(neg)` and automatically locate the optimal injection layer.
   - Drag **`Live Steering Strength α`** (`-2.0` to `+2.0`) to steer inference live without modifying weights, or click **`✦ Bake Steering Into Weights`** to turn the steering vector into an undoable, exportable rank-1 weight edit.
4. **Discover 3D Causal Circuits (`Circuit 3D`):** Click **`✦ Circuit`** (or press `5`). The viewport switches to **`Circuit 3D`**, plotting prompt token monoliths, glowing ruby-gold causal attention head pillars, MLP cores, an output unembedding prism, and animated 3D Bezier attention-flow ribbons. Click **`Ablate`** next to any ranked node in **Causal Circuit Nodes** to immediately test ablating that circuit layer.

---

### Step 5 — Configure & Run Live Micro-Finetuning (`Finetune`)

Switch to the **`Finetune`** tab:

1. Select an RL objective (**GSPO**, **GRPO**, **DAPO**, **GEPO**, **VAPO**, **PPO**, or **SFT**) and adjust learning rate, steps, clip ratio `ε`, KL coefficient `β`, and group size `G`.
2. Click **Target Modules** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, …) to highlight and isolate the trainable LoRA modules live inside the 3D architecture stack.
3. Inspect local **Curated Dataset & Reward Traces** (`.jsonl`) and verify the live **VRAM, Workspace & Provider Budget** bar (including `finetune_workspace` schema validation and hourly cost estimates).
4. Click **`▶ Start Live Finetune`** to execute real PyTorch AdamW micro-updates streamed over Server-Sent Events (SSE), firing amber backprop pulses across the 3D layers as weights morph in real time.
5. Ready for full-scale GPU training? Expand **`Export Validated StateSet CLI Command`** and copy the exact `stateset-agents` CLI invocation.

---

### Step 6 — Regression Gate, Verified Export & Reproducible Recipes

1. Click **`Export`** in the top bar to open the **Eval Gate & Verified `.safetensors` Export** modal.
2. Click **`⚡ Run 6-Case Regression Eval Gate`** to benchmark Base vs. Edited across local workspace JSONL traces and StateSet commerce/tool/RLVR/safety scenarios. Review the **`GO`** / **`WARN`** / **`BLOCK`** verdict and per-case reward deltas before writing weights.
3. Enter an export directory name, keep **`Run merge_adapter key-remap & effect probe`** checked if exporting a LoRA adapter, and click **`Write & Verify SHA-256 Checkpoint`**.
   - Exports are written immutably under `outputs/weight_studio_exports/<name>/` alongside `edit_manifest.json` (with exact reload fidelity verification and SHA-256 digest) and `merge_probe.json`.
4. **Share or Replay Edit Pipelines:** In the **Surgery** tab, click **`↓ JSON`** to download a portable `.studio-recipe.json` of your active edit stack, or **`↑ Replay`** to deterministically replay a recipe onto another checkpoint.
5. **Capture Showcase Media:** Click **`📷`** for a crisp PNG snapshot, **`⏺`** to record a 60 FPS **360° Turntable `.webm` video**, or **`⛶`** (press `H`) for full-bleed **Cinema Zen Mode**.

---

## Keyboard Shortcuts & Controls

| Shortcut / Action | Behavior |
| :--- | :--- |
| `1` – `5` | Switch 3D topology: `1` Architecture, `2` Microscope 128², `3` SVD Spectral, `4` Trajectory 4D, `5` Circuit 3D |
| `Space` | Toggle smooth auto-orbit |
| `S` | Toggle real-time **Split A\|B** 3D wipe curtain (in Microscope mode) |
| `H` | Toggle **Cinema Zen Mode** (full-screen 3D viewport) |
| Left-Drag | Orbit 3D camera |
| Right-Drag / `Alt`+Drag | Pan 3D camera |
| Scroll Wheel | Smooth spring-damped zoom |
| Double-Click Tensor | Jump straight into `Microscope 128²` for that tensor |
| `Shift` + Left-Drag | Paint a **3D Surgical Brush** selection patch in Microscope mode |

---

## Troubleshooting

- **No local checkpoints listed:** Ensure your `.safetensors` files live under the `--workspace` directory (within depth 4, e.g., `outputs/my_run/adapter_model.safetensors`), then click **↻** next to the model selector. Built-in StateSet Architecture Twins are always available even without local files.
- **Browser shows WebGL2 error:** Enable hardware acceleration in your browser settings (`chrome://settings/system`) or ensure WebGL2 is supported by your GPU driver.
- **Live PyTorch hooks show `WEIGHT PROBE` instead of `PYTORCH HOOKS`:** Genuine `AutoModelForCausalLM` hooks activate automatically for local full causal LM checkpoints (`config.json` + `model.safetensors` under 400 MB) when `torch` and `transformers` are installed. For larger models or LoRA adapters, the Studio falls back seamlessly to its fast weight-matrix probe—or connect a live vLLM endpoint via the **Optional Live vLLM / OpenAI Endpoint** drawer.
- **Port 8765 already in use:** Pass `--port 8766` (or any free port) to `stateset-agents studio`.
