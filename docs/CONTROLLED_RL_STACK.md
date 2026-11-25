# Controlled Multi-Model Closed-Universe RL Stack

This guide documents the controlled closed-universe legal/enterprise RL stack built into `stateset-agents` (`ClientMatterSandbox`, `HierarchicalRubricReward`, `dr_cispo` Rank-64 LoRA optimization, 5-model `River` + `Tinker` + **StateSet Sandboxes** infrastructure, and the anchor hyperparameter sweep & Appendix exporter).

## 1. Environments — Closed-Universe Client Matter Sandboxes

Each task (`ClientMatterTask` in [client_matter_sandbox.py](file:///home/dom/stateset-agents/stateset_agents/core/client_matter_sandbox.py)) places an agent inside a self-contained client matter workspace containing both **key controlling documents** (governing agreements, amendments, counsel correspondence, disclosure schedules) and **peripheral distractor documents** (unrelated policies, superseded LOIs, administrative schedules).

The agent operates through six confined sandbox tools (registered as the `"client_matter"` domain in `MultiDomainSandboxPool` and `river_mcp_sandbox_factory("client_matter")`):

| Tool | Purpose |
| --- | --- |
| `list_matter_files` | Lists flat filenames, categories, and byte/line counts without leaking contents |
| `search_matter` | Case-insensitive keyword/regex search across all documents in the matter |
| `read_document` | Confined line-ranged read (rejects `..`, path separators, and out-of-matter files) |
| `draft_work_product` | Creates or appends to the work-product deliverable (`deliverable.md`) |
| `read_draft` | Inspects the current deliverable revision |
| `signal_complete` | Terminates the episode cleanly (episodes also stop at `max_steps`) |

## 2. Reward — Hierarchical Issue-Group Binary Rubric (`HierarchicalRubricReward`)

Implemented in [rubric_reward.py](file:///home/dom/stateset-agents/stateset_agents/rewards/rubric_reward.py):

- An LLM judge (`LLMRubricJudge`, fail-closed JSON verdict parsing at `T=0.0`) or offline verifier (`DeterministicRubricJudge`) marks every expert rubric criterion strictly binary **pass (`1`) / fail (`0`)**.
- Reward is the weighted sum of three terms with **explicitly zero length penalty** (`length_penalty = 0.0`, enforced by `RubricRewardConfig`):

$$
R = w_{\text{criteria}} \cdot \frac{\text{criteria passed}}{\text{total criteria}}
  + w_{\text{groups}} \cdot \frac{\text{issue groups passed in full}}{\text{total issue groups}}
  + w_{\text{all}} \cdot \mathbb{I}[\text{all criteria passed}]
$$

Default weights: `criteria_weight=0.50`, `issue_group_weight=0.35`, `all_pass_bonus=0.15`, `length_penalty=0.0`.

## 3. Optimization — Rank-64 LoRA + CISPO + Dr. GRPO (`dr_cispo`)

Every model trains a **rank-64 LoRA** adapter (`alpha=128`) using the **`dr_cispo`** objective preset ([objectives.py](file:///home/dom/stateset-agents/stateset_agents/training/objectives.py) and [chunked_rl_loss.py](file:///home/dom/stateset-agents/stateset_agents/training/chunked_rl_loss.py)):

- **Advantage normalization (Dr. GRPO):** $A_i = R_i - \bar{R}_g$ (reward minus rollout group mean, unscaled — no std division).
- **Loss normalization (Dr. GRPO):** Constant-denominator token-sum normalization (`seq_sum_const` / River `normalization="dr_grpo"`), dividing by $B \cdot L_{\max}$ to eliminate response-length bias.
- **Asynchronous importance weight clipping (CISPO):** Per-token detached clipped importance ratios $\operatorname{sg}(\operatorname{clip}(\rho_t, 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}})) \cdot A_t \cdot \log \pi_\theta(y_t \mid x, y_{<t})$ plus optional async rollout cap (`tis_clip_max` / `eps_max`).

## 4. Infrastructure & Controls

Configured in [controlled_study.py](file:///home/dom/stateset-agents/stateset_agents/training/controlled_study.py):

| Model | Checkpoint ID | Training Backend | Rollout Backend | Harness | Notes |
| --- | --- | --- | --- | --- | --- |
| **Muse Spark Open** | `meta-models/Muse-Spark-Open-816B-A42B` | `river` | `stateset_sandboxes` | `lab_default` | Managed RL + StateSet Sandboxes |
| **Nemotron 3.5 Super VL** | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` | `river` | `stateset_sandboxes` | `lab_default` | **Sweep anchor** (LR & group size) |
| **Beam 1.0** | `beam/Beam-1.0` | `river` | `stateset_sandboxes` | `lab_default` | Transferred hyperparameters |
| **GLM-5.3** | `zai-org/GLM-5.3` | `river` | `stateset_sandboxes` | `lab_default` | Transferred hyperparameters |
| **Inkling 2 Small** | `thinkingmachines/Inkling-2-Small` | `tinker` | `stateset_sandboxes` | `lab_default` | Supported fine-tuning before public weight release |

- **Hyperparameter Sweep & Transfer:** `run_anchor_hyperparameter_sweep()` sweeps `(learning_rate, rollout_group_size)` strictly on **Nemotron 3.5 Super VL** over standard ranges (`(3e-5, 1e-4, 3e-4)` × `(4, 8, 16)`), and `build_controlled_run_plans()` applies the selected pair verbatim across all 5 models.
- **Control Drift Guard:** `validate_study_controls()` fails closed if any model differs on environments, execution limit, rubric weights, zero length penalty, judge settings, LoRA rank, objective preset, rollout backend, harness, learning rate, or rollout group size.
- **Appendix Export:** `export_study_bundle()` writes a SHA-256 attested `study_manifest.json` and publication-ready `appendix_training_config.md`.

## Quickstart

```bash
python examples/controlled_legal_rl_stack.py
```
